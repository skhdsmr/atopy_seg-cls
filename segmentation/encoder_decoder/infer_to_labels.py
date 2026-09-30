"""seg 체크포인트로 데이터셋 전체를 추론해 '예측 마스크'를 새 폴리곤 라벨로 저장.

GT 라벨을 복사하지 않고, 모델 예측 마스크를 cv2.findContours 로 폴리곤화해서
기존과 동일한 YOLO-seg .txt 형식("class x1 y1 x2 y2 ...", 정규화 좌표) 으로
새 라벨을 만든다.

추론 규칙은 app/app.py 의 _seg_model / infer_seg 와 동일하게 맞췄다:
  - 순수 state_dict(deeplab 등)면 파일명(`DeepLabV3Plus_resnet50_best.pth`)에서
    arch/encoder 를 유추하고, smp.create_model 로 구성.
  - 헤드 출력 채널로 클래스 수 감지: 1=sigmoid, 2+=softmax(전경=1-배경확률).
  - Resize((imgsz,imgsz)) + ImageNet 정규화, 원본 해상도로 복원 후 thr(0.5).

출력 구조(원본과 동일):
    <OUT>/images/{split}/{case}.png   (원본 이미지 복사)
    <OUT>/labels/{split}/{case}.txt   (예측 마스크 -> 폴리곤 라벨)

사용 예:
  python3 infer_to_labels.py \
      --ckpt runs/deeplab/DeepLabV3Plus_resnet50_best.pth \
      --data ../../dataset_face_new --out ../../dataset_deeplab
"""
import argparse
import shutil
import sys
from pathlib import Path

# --- GPU torch 부트스트랩 (infer_all_splits.py 와 동일) ------------------
_LOCAL = "/home/work/.local/lib/python3.12/site-packages"
_DIST = "/usr/local/lib/python3.12/dist-packages"
sys.path = [p for p in sys.path if p not in (_LOCAL,)]
if _DIST not in sys.path:
    sys.path.append(_DIST)
sys.path.append(_LOCAL)

import cv2                               # noqa: E402
import numpy as np                       # noqa: E402
import segmentation_models_pytorch as smp   # noqa: E402
import torch                             # noqa: E402
import torch.nn.functional as F          # noqa: E402
from PIL import Image                    # noqa: E402
from torchvision import transforms as T  # noqa: E402

ROOT = Path(__file__).resolve().parent

SPLITS = ["train", "val", "test"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# app.py _SMP_ARCH 와 동일한 정규화 테이블
_SMP_ARCH = {
    "unet": "unet", "unetpp": "unetplusplus", "unetplusplus": "unetplusplus",
    "manet": "manet", "deeplabv3": "deeplabv3", "deeplabv3plus": "deeplabv3plus",
    "fpn": "fpn", "pspnet": "pspnet", "linknet": "linknet", "pan": "pan",
    "segformer": "segformer", "upernet": "upernet",
}


def _norm_arch(name):
    return _SMP_ARCH.get(str(name).lower().replace("+", "plus").replace("-", ""),
                         str(name).lower())


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if ckpt and all(hasattr(v, "shape") for v in list(ckpt.values())[:5]):
            return ckpt
    return ckpt


def _seg_out_classes(sd):
    for k in ("segmentation_head.0.weight", "final.weight", "head.weight"):
        if k in sd:
            return int(sd[k].shape[0])
    return 1


def _resolve_smp_arch_encoder(a, ckpt_path, sd):
    if a.get("encoder") and a.get("decoder"):
        return _norm_arch(a["decoder"]), a["encoder"]
    toks = Path(ckpt_path).stem.split("_")
    if toks and toks[-1] in ("best", "last"):
        toks = toks[:-1]
    if len(toks) >= 2:
        return _norm_arch(toks[0]), "_".join(toks[1:])
    raise ValueError(f"인코더/디코더를 알 수 없습니다: {ckpt_path}")


def load_seg_model(ckpt_path):
    """app.py _seg_model 과 동일 규칙. 반환 (model, imgsz, nclass)."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    a = (ckpt.get("args") if isinstance(ckpt, dict) else None) or {}
    sd = _extract_state_dict(ckpt)
    nclass = _seg_out_classes(sd)
    arch, encoder = _resolve_smp_arch_encoder(a, ckpt_path, sd)
    imgsz = int(a.get("imgsz", 512))
    seg = smp.create_model(arch, encoder_name=encoder, encoder_weights=None,
                           in_channels=3, classes=nclass or 1)
    seg.load_state_dict(sd, strict=False)
    seg.eval().to(DEVICE)
    return seg, imgsz, (nclass or 1)


def infer_mask(seg, imgsz, nclass, pil, thr=0.5):
    """app.py infer_seg 와 동일. 원본 해상도 0/1 마스크 반환."""
    W, H = pil.size
    tf = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    x = tf(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = seg(x)
        if nclass <= 1:
            prob = torch.sigmoid(logits)
        else:
            prob = 1 - torch.softmax(logits, dim=1)[:, 0:1]
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
        prob = prob[0, 0].cpu().numpy()
    return (prob >= thr).astype(np.uint8)


def mask_to_polygons(mask, min_area=16, eps_frac=0.002):
    """0/1 마스크(H,W) -> 정규화 폴리곤 리스트 [[x1,y1,x2,y2,...], ...]."""
    h, w = mask.shape
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = eps_frac * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        coords = []
        for x, y in ap:
            coords.append(min(max(x / w, 0.0), 1.0))
            coords.append(min(max(y / h, 0.0), 1.0))
        polys.append(coords)
    return polys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "runs/deeplab/DeepLabV3Plus_resnet50_best.pth"))
    p.add_argument("--data", default=str(ROOT.parent.parent / "dataset_face_new"))
    p.add_argument("--out", default=str(ROOT.parent.parent / "dataset_deeplab"))
    p.add_argument("--imgsz", type=int, default=None,
                   help="추론 크기. 미지정 시 체크포인트/기본값(512) 사용")
    p.add_argument("--thr", type=float, default=0.5, help="전경 임계값")
    p.add_argument("--min-area", type=float, default=16,
                   help="이 픽셀면적 미만의 예측 조각은 라벨에서 제외")
    return p.parse_args()


def main():
    args = parse_args()
    CKPT = Path(args.ckpt)
    DATA = Path(args.data)
    OUT = Path(args.out)

    seg, imgsz, nclass = load_seg_model(CKPT)
    if args.imgsz:
        imgsz = args.imgsz
    print(f"[ckpt] {CKPT}")
    print(f"[model] classes={nclass} ({'softmax' if nclass > 1 else 'sigmoid'})  "
          f"imgsz={imgsz}  device={DEVICE}")
    print(f"[data] {DATA}\n[out ] {OUT}\n[thr] {args.thr}  [min-area] {args.min_area}")

    total = 0
    for split in SPLITS:
        img_dir = DATA / "images" / split
        img_paths = sorted(img_dir.glob("*.png"))
        if not img_paths:
            print(f"[warn] 이미지 없음, 건너뜀: {img_dir}")
            continue

        out_img = OUT / "images" / split
        out_lbl = OUT / "labels" / split
        for d in (out_img, out_lbl):
            d.mkdir(parents=True, exist_ok=True)

        n_empty = 0
        for i, ip in enumerate(img_paths):
            case = ip.stem
            pil = Image.open(ip).convert("RGB")
            mask = infer_mask(seg, imgsz, nclass, pil, thr=args.thr)

            polys = mask_to_polygons(mask, min_area=args.min_area)
            if not polys:
                n_empty += 1
            lines = ["0 " + " ".join(f"{v:.6f}" for v in poly) for poly in polys]
            (out_lbl / f"{case}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            shutil.copy2(ip, out_img / ip.name)

            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i+1}/{len(img_paths)} done")

        total += len(img_paths)
        print(f"=== {split}: n={len(img_paths)}  빈 라벨(예측 없음)={n_empty} ===")

    print(f"\n[done] {total}장 추론 완료 -> {OUT}")


if __name__ == "__main__":
    main()
