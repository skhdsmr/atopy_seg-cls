"""seg 체크포인트로 임의 데이터셋의 train/val/test 전체를 추론.

- 학습/평가와 동일한 전처리(Resize imgsz + ImageNet 정규화, thr 0.5)로 추론하되,
  예측 마스크는 각 이미지의 '원본 해상도'로 복원해서 저장한다.
  (logits 를 원본 크기로 bilinear 업샘플 후 sigmoid>0.5 -> 경계가 덜 계단짐)
- 출력은 원본 데이터셋과 동일한 구조로 새 폴더에 저장:
    <OUT>/images/{split}/{case}.png   (원본 이미지 복사)
    <OUT>/labels/{split}/{case}.txt   (원본 라벨 복사)
    <OUT>/pred/{split}/{case}.png     (예측 영역을 옅은 빨강으로 오버레이;
                                       --side-by-side 면 [원본 | 오버레이] 2장을 가로로 붙여 저장.
                                       app.py overlay() 와 동일: 빨강 alpha=0.45, 원본 해상도)
- per-image Dice/IoU 는 원본 해상도의 GT 마스크 기준으로 계산해 콘솔 요약.

사용 예:
  python3 infer_all_splits.py                                    # face + hrnet(기본)
  python3 infer_all_splits.py --ckpt runs/unetpp_effb3_lesion/checkpoint_best.pth \
      --data ../../dataset_lesion_merged --out ../../dataset_lesion_merged_pred \
      --imgsz 512 --side-by-side
"""
import argparse
import shutil
import sys
from pathlib import Path

# --- GPU torch 부트스트랩 (eval_dice_by_ratio.py 와 동일) ------------------
# torch import 전에 경로를 고쳐야 dist-packages 의 GPU torch 가 잡힌다.
_LOCAL = "/home/work/.local/lib/python3.12/site-packages"
_DIST = "/usr/local/lib/python3.12/dist-packages"
sys.path = [p for p in sys.path if p not in (_LOCAL,)]
if _DIST not in sys.path:
    sys.path.append(_DIST)
sys.path.append(_LOCAL)

import numpy as np                       # noqa: E402
import torch                             # noqa: E402
import torch.nn.functional as F          # noqa: E402
from PIL import Image                    # noqa: E402

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent
sys.path.insert(0, str(SHARED))

from augment import _MEAN, _STD          # noqa: E402
from dataset import polygons_to_mask     # noqa: E402
from model import build_model            # noqa: E402

SPLITS = ["train", "val", "test"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MEAN_T = torch.tensor(_MEAN).view(3, 1, 1)
_STD_T = torch.tensor(_STD).view(3, 1, 1)
OVERLAY_ALPHA = 0.45   # app.py overlay() 기본값과 동일
SEP_PX = 4             # side-by-side 두 장 사이 흰색 구분선 폭


def overlay_pred(im_rgb, pred):
    """원본 RGB (H,W,3) 위 예측 영역(pred bool)을 옅은 빨강으로 반투명 합성.
    app.py overlay() 와 동일한 규칙(빨강=예측, alpha=0.45)."""
    out = im_rgb.astype(np.float32)
    m = pred[..., None]
    red = np.zeros_like(out); red[..., 0] = 255
    out = np.where(m, (1 - OVERLAY_ALPHA) * out + OVERLAY_ALPHA * red, out)
    return out.clip(0, 255).astype(np.uint8)


def side_by_side(left, right):
    """[원본 | 오버레이] 두 RGB 이미지를 가로로 붙여 한 장으로."""
    h = left.shape[0]
    sep = np.full((h, SEP_PX, 3), 255, dtype=np.uint8)
    return np.concatenate([left, sep, right], axis=1)


def preprocess(im_rgb, imgsz):
    """원본 RGB uint8 (H,W,3) -> (1,3,imgsz,imgsz) 정규화 텐서."""
    t = torch.from_numpy(im_rgb.transpose(2, 0, 1)).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(imgsz, imgsz),
                      mode="bilinear", align_corners=False)[0]
    t = (t - _MEAN_T) / _STD_T
    return t.unsqueeze(0)


def dice_iou(pred, gt):
    """pred, gt: (H,W) bool. 원본 해상도 기준 Dice/IoU."""
    p = pred.astype(bool); g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return 1.0, 1.0
    dice = 2 * inter / (ps + gs + 1e-9)
    union = np.logical_or(p, g).sum()
    iou = inter / (union + 1e-9)
    return float(dice), float(iou)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "runs/hrnet18_unet/checkpoint_best.pth"))
    p.add_argument("--data", default=str(ROOT.parent.parent / "dataset_face_strat_merged"))
    p.add_argument("--out", default=None,
                   help="미지정 시 <data>_pred 로 자동 설정")
    p.add_argument("--imgsz", type=int, default=None,
                   help="추론 크기. 미지정 시 체크포인트의 imgsz 사용")
    p.add_argument("--side-by-side", action="store_true",
                   help="pred 를 [원본 | 오버레이] 2장을 가로로 붙여 저장")
    p.add_argument("--max-dice", type=float, default=None,
                   help="설정 시 per-image Dice 가 이 값 이하인 케이스만 OUT 에 저장 "
                        "(어려운 케이스만 추리기). 예: 0.7")
    return p.parse_args()


def main():
    args = parse_args()
    CKPT = Path(args.ckpt)
    DATA = Path(args.data)
    OUT = Path(args.out) if args.out else DATA.parent / f"{DATA.name}_pred"

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    IMGSZ = args.imgsz or a["imgsz"]
    print(f"[ckpt] {CKPT}")
    print(f"[ckpt] epoch={ck['epoch']} val={ck['val']}")
    print(f"[ckpt] encoder={a['encoder']} decoder={a['decoder']} "
          f"imgsz={a['imgsz']} crop={a.get('crop_size', '-')}")
    print(f"[data] {DATA}\n[out ] {OUT}")
    print(f"[device] {DEVICE}   [infer size] {IMGSZ}   "
          f"[side-by-side] {args.side_by_side}")

    model = build_model(
        encoder_name=a["encoder"], encoder_weights=None, decoder=a["decoder"],
        decoder_attention=None if a.get("decoder_attention") in ("none", None) else a["decoder_attention"],
        num_classes=1, in_channels=3)
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()

    for split in SPLITS:
        img_dir = DATA / "images" / split
        lbl_dir = DATA / "labels" / split
        img_paths = sorted(img_dir.glob("*.png"))
        if not img_paths:
            print(f"[warn] 이미지 없음, 건너뜀: {img_dir}")
            continue

        out_img = OUT / "images" / split
        out_lbl = OUT / "labels" / split
        out_pred = OUT / "pred" / split
        for d in (out_img, out_lbl, out_pred):
            d.mkdir(parents=True, exist_ok=True)

        dices, ious = [], []
        n_scanned = 0
        for i, ip in enumerate(img_paths):
            case = ip.stem
            im = np.asarray(Image.open(ip).convert("RGB"))    # (H,W,3)
            h, w = im.shape[:2]
            lbl_path = lbl_dir / f"{case}.txt"
            gt = polygons_to_mask(lbl_path, w, h)             # (H,W) 0/1

            with torch.no_grad():
                x = preprocess(im, IMGSZ).to(DEVICE)
                logits = model(x)                             # (1,1,IMGSZ,IMGSZ)
                # 원본 해상도로 복원 후 임계 -> 경계 품질 유지
                logits = F.interpolate(logits.float(), size=(h, w),
                                       mode="bilinear", align_corners=False)
                pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()

            d, io = dice_iou(pred, gt)
            n_scanned += 1
            # --max-dice 필터: 쉬운(성능 좋은) 케이스는 저장하지 않고 건너뜀
            if args.max_dice is not None and d > args.max_dice:
                if (i + 1) % 50 == 0:
                    print(f"  [{split}] {i+1}/{len(img_paths)} scanned "
                          f"(kept {len(dices)})")
                continue

            ov = overlay_pred(im, pred)
            out_arr = side_by_side(im, ov) if args.side_by_side else ov
            Image.fromarray(out_arr).save(out_pred / f"{case}.png")
            # 원본 이미지/라벨도 함께 저장(원 데이터셋과 동일 구조)
            shutil.copy2(ip, out_img / ip.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / lbl_path.name)

            dices.append(d); ious.append(io)
            if (i + 1) % 50 == 0:
                print(f"  [{split}] {i+1}/{len(img_paths)} done "
                      f"(kept {len(dices)})")

        if args.max_dice is not None:
            print(f"=== {split}: scanned={n_scanned}  "
                  f"kept(Dice<={args.max_dice})={len(dices)}  "
                  f"mean Dice(kept)={np.mean(dices) if dices else 0:.4f}  "
                  f"mean IoU(kept)={np.mean(ious) if ious else 0:.4f} ===")
        else:
            print(f"=== {split}: n={len(dices)}  "
                  f"mean Dice={np.mean(dices):.4f}  mean IoU={np.mean(ious):.4f} ===")

    print(f"\n[done] 결과 저장: {OUT}")


if __name__ == "__main__":
    main()
