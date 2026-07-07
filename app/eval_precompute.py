"""
대시보드용 사전 평가 스크립트 (무거운 계산은 여기서 한 번, 대시보드는 표시만).

test 셋 전체를 한 번 돌려서 아래를 생성한다:
  1) outputs/pred_masks/<case>.png   : 분할 예측 마스크(0/255, 원본 해상도)
  2) outputs/gt_masks/<case>.png     : YOLO 폴리곤 GT 를 래스터화한 마스크(0/255)
  3) outputs/eval_meta.json          : 이미지별 지표(TP/FP/FN·per-image dice 등) +
                                       중증도/증상 예측 + 전체 pooled 지표

지표는 pooled(픽셀 누적) 방식이 기본이다. 이미지별 Dice 를 단순 평균하면 작은 병변
한 장이 전체를 크게 흔들어 왜곡되므로, 전체 이미지의 TP/FP/FN 픽셀을 먼저 합친 뒤
한 번에 계산한다(대시보드의 '누적'도 이 tp/fp/fn 을 더해서 만든다).

실행:
  python3 app/eval_precompute.py                       # 기본값(lesion_merged test)
  python3 app/eval_precompute.py --split test --thr 0.5 --fg_thresh 0.05
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent   # app/ 상위 = repo 루트

# 결과 변형(variant): 분할 모델 + 테스트 데이터셋만 다르고 분류/화면은 공유한다.
#   face   : HRNet-w18 + U-Net,      dataset_face_strat_merged
#   lesion : U-Net++ (EfficientNet-b3), dataset_lesion_merged
VARIANTS = {
    "face": {
        "seg_ckpt": "segmentation/encoder_decoder/runs/hrnet18_unet/checkpoint_best.pth",
        "data": "dataset_face_strat_merged",
    },
    "lesion": {
        "seg_ckpt": "segmentation/encoder_decoder/runs/unetpp_effb3_lesion/checkpoint_best.pth",
        "data": "dataset_lesion_merged",
    },
}
# 분류 : MobileNetV4-conv-medium 멀티태스크 (atopy 원본 전체로 학습, 두 변형 공유)
CLS_CKPT = ROOT / "classification/mobilenet/runs/cls_mobilenetv4_conv_medium_atopy/best.pt"

# 학습 입력 크기의 폴백값. 실제로는 체크포인트 args 의 imgsz 를 우선 사용한다.
SEG_IMGSZ = 512
CLS_IMGSZ = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 분류 태스크 (classification/dataset.py 의 TASKS 와 동일, 순서형 등급)
TASKS = {
    "severity":        ["Clear", "Almost Clear", "Mild", "Moderate", "Severe"],
    "erythema":        ["None", "Mild", "Moderate", "Severe"],
    "papulation":      ["None", "Mild", "Moderate", "Severe"],
    "excoriation":     ["None", "Mild", "Moderate", "Severe"],
    "lichenification": ["None", "Mild", "Moderate", "Severe"],
}
NUM_CLASSES = {t: len(v) for t, v in TASKS.items()}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def polygons_to_mask(label_path: Path, w: int, h: int) -> np.ndarray:
    """YOLO 폴리곤(정규화 좌표) 여러 개를 하나의 0/1 마스크로 래스터화."""
    from PIL import ImageDraw
    mask = Image.new("L", (w, h), 0)
    drw = ImageDraw.Draw(mask)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            c = list(map(float, parts[1:]))
            pts = [(c[i] * w, c[i + 1] * h) for i in range(0, len(c) - 1, 2)]
            drw.polygon(pts, fill=1)
    return np.array(mask, dtype=np.uint8)


def preprocess(img: Image.Image, size: int, device) -> torch.Tensor:
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img).unsqueeze(0).to(device)


def load_seg(ckpt_path, device):
    """(seg, seg_imgsz) 반환. 인코더/디코더/입력크기는 체크포인트 args 에서 읽는다."""
    if not ckpt_path.exists():
        print(f"[warn] 분할 가중치 없음: {ckpt_path}")
        return None, SEG_IMGSZ
    seg_mod = _load_module(ROOT / "segmentation/encoder_decoder/model.py", "seg_model_def")
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt.get("args", {})
    seg_imgsz = int(a.get("imgsz", SEG_IMGSZ))
    seg = seg_mod.build_model(
        encoder_name=a.get("encoder", "tu-hrnet_w18"),
        encoder_weights=None,
        decoder=a.get("decoder", "unet"),
        num_classes=1, in_channels=3,
    )
    seg.load_state_dict(ckpt["model"])
    seg.eval().to(device)
    print(f"[seg] {a.get('encoder')} + {a.get('decoder')} · imgsz={seg_imgsz}")
    return seg, seg_imgsz


def load_cls(device):
    """(cls, cls_imgsz) 반환. 두 변형이 공유하는 분류기."""
    if not CLS_CKPT.exists():
        print(f"[warn] 분류 가중치 없음: {CLS_CKPT}")
        return None, CLS_IMGSZ
    cls_mod = _load_module(ROOT / "classification/mobilenet/model.py", "cls_model_def")
    ckpt = torch.load(CLS_CKPT, map_location=device)
    a = ckpt.get("args", {})
    cls_imgsz = int(a.get("imgsz", CLS_IMGSZ))
    backbone = ckpt.get("model_name", "mobilenetv4_conv_medium")
    cls = cls_mod.MultiTaskNet(
        backbone, NUM_CLASSES,
        embed_dim=a.get("embed_dim", 512), dropout=a.get("dropout", 0.3),
    )
    cls.load_state_dict(ckpt["model"])
    cls.eval().to(device)
    print(f"[cls] {backbone} · imgsz={cls_imgsz}")
    return cls, cls_imgsz


@torch.no_grad()
def infer_seg(seg, img: Image.Image, device, thr: float, imgsz: int):
    """원본 해상도 0/1 예측 마스크 반환."""
    W, H = img.size
    x = preprocess(img, imgsz, device)
    prob = torch.sigmoid(seg(x))
    prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    prob = prob[0, 0].cpu().numpy()
    return (prob >= thr).astype(np.uint8)


@torch.no_grad()
def infer_cls(cls, img: Image.Image, device, imgsz: int):
    x = preprocess(img, imgsz, device)
    out = cls(x)
    res = {}
    for t, names in TASKS.items():
        probs = F.softmax(out[t][0], dim=0).cpu().numpy()
        idx = int(probs.argmax())
        res[t] = {"label": names[idx], "idx": idx, "conf": float(probs[idx]),
                  "probs": [round(float(p), 4) for p in probs], "names": names}
    return res


def process_variant(name, seg_ckpt, data, out, split, thr, fg_thresh,
                    device, cls, cls_imgsz, label_index):
    """한 변형(face/lesion) 을 평가해 out/<...> 에 마스크와 eval_meta.json 저장."""
    img_dir = data / "images" / split
    lbl_dir = data / "labels" / split
    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"이미지 없음: {img_dir}")

    pred_dir = out / "pred_masks"
    gt_dir = out / "gt_masks"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    seg, seg_imgsz = load_seg(seg_ckpt, device)
    print(f"[{name}] {len(img_paths)}장 · {data.name} · device={device} · thr={thr}")

    images = []
    tot_tp = tot_fp = tot_fn = 0
    eps = 1e-7
    for i, ip in enumerate(img_paths):
        case = ip.stem
        img = Image.open(ip).convert("RGB")
        W, H = img.size

        gt = polygons_to_mask(lbl_dir / f"{case}.txt", W, H)      # (H,W) 0/1
        rec = {
            "case": case,
            "image": str(ip.relative_to(ROOT)),
            "width": W, "height": H,
            "fg_gt": round(float(gt.mean()), 5),
        }

        if seg is not None:
            pred = infer_seg(seg, img, device, thr, seg_imgsz)    # (H,W) 0/1
            g = gt.astype(bool)
            p = pred.astype(bool)
            tp = int(np.logical_and(p, g).sum())
            fp = int(np.logical_and(p, ~g).sum())
            fn = int(np.logical_and(~p, g).sum())
            tot_tp += tp; tot_fp += fp; tot_fn += fn
            rec.update({
                "tp": tp, "fp": fp, "fn": fn,
                "fg_pred": round(float(pred.mean()), 5),
                "per_image": {
                    "dice": round((2 * tp) / (2 * tp + fp + fn + eps), 4),
                    "iou": round(tp / (tp + fp + fn + eps), 4),
                    "precision": round(tp / (tp + fp + eps), 4),
                    "recall": round(tp / (tp + fn + eps), 4),
                },
                "pred_mask": f"pred_masks/{case}.png",
            })
            Image.fromarray((pred * 255).astype(np.uint8)).save(pred_dir / f"{case}.png")

        Image.fromarray((gt * 255).astype(np.uint8)).save(gt_dir / f"{case}.png")
        rec["gt_mask"] = f"gt_masks/{case}.png"
        rec["group"] = "sparse" if rec["fg_gt"] < fg_thresh else "dense"

        if cls is not None:
            rec["cls"] = infer_cls(cls, img, device, cls_imgsz)
            gtlab = label_index.get(case)
            if gtlab:
                for t, idx in gtlab.items():
                    if t in rec["cls"]:
                        rec["cls"][t]["gt_idx"] = int(idx)
                        rec["cls"][t]["gt_label"] = TASKS[t][idx]

        images.append(rec)
        if (i + 1) % 10 == 0 or i + 1 == len(img_paths):
            print(f"  {i + 1}/{len(img_paths)}")

    pooled = {
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "dice": round((2 * tot_tp) / (2 * tot_tp + tot_fp + tot_fn + eps), 4),
        "f1": round((2 * tot_tp) / (2 * tot_tp + tot_fp + tot_fn + eps), 4),
        "iou": round(tot_tp / (tot_tp + tot_fp + tot_fn + eps), 4),
        "precision": round(tot_tp / (tot_tp + tot_fp + eps), 4),
        "recall": round(tot_tp / (tot_tp + tot_fn + eps), 4),
    }
    meta = {
        "variant": name,
        "dataset": str(data.relative_to(ROOT)) if data.is_relative_to(ROOT) else str(data),
        "split": split,
        "seg_ckpt": str(seg_ckpt.relative_to(ROOT)) if seg is not None else None,
        "cls_ckpt": str(CLS_CKPT.relative_to(ROOT)) if cls is not None else None,
        "thr": thr,
        "fg_thresh": fg_thresh,
        "n_images": len(images),
        "tasks": TASKS,
        "pooled": pooled,
        "images": images,
    }
    (out / "eval_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{name}] pooled 지표")
    if seg is not None:
        for k in ("f1", "precision", "recall", "iou", "dice"):
            print(f"  {k:10s}: {pooled[k]:.4f}")
    print(f"저장 완료 → {out / 'eval_meta.json'}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["face", "lesion", "all"], default="all",
                    help="평가할 결과 변형(기본 all: face·lesion 모두)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "outputs"),
                    help="결과 저장 루트(기본 app/outputs). 변형별 하위 폴더(out/face, out/lesion)에 저장")
    ap.add_argument("--thr", type=float, default=0.5, help="분할 이진화 임계값")
    ap.add_argument("--fg_thresh", type=float, default=0.05,
                    help="GT 전경비율 < 이 값이면 sparse(희소), 이상이면 dense(밀집)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    base_out = Path(args.out)

    # 분류기(+정답 인덱스)는 두 변형이 공유하므로 한 번만 로드한다.
    cls, cls_imgsz = load_cls(device)
    label_index = {}
    atopy_dir = ROOT / "atopy"
    if cls is not None and atopy_dir.exists():
        ds_mod = _load_module(ROOT / "classification/dataset.py", "cls_dataset_def")
        label_index = ds_mod.build_label_index(atopy_dir)
        print(f"[gt] 분류 정답 인덱스 {len(label_index)}개")

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in variants:
        cfg = VARIANTS[name]
        process_variant(
            name, ROOT / cfg["seg_ckpt"], ROOT / cfg["data"], base_out / name,
            args.split, args.thr, args.fg_thresh,
            device, cls, cls_imgsz, label_index,
        )


if __name__ == "__main__":
    main()
