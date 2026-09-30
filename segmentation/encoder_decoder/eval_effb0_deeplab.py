"""effb0_unetpp_deeplab_* 체크포인트(128~512)를 dataset_deeplab 로 평가.

- 각 체크포인트는 학습 시 imgsz 로 리사이즈 추론하되, GT 는 원본 해상도 기준으로
  Dice/IoU/Recall/Precision 을 계산(infer_all_splits.py 와 동일 규칙, thr 0.5).
- 저장 없이 지표만 출력. 기본 split=test.
"""
import argparse
import sys
from pathlib import Path

_LOCAL = "/home/work/.local/lib/python3.12/site-packages"
_DIST = "/usr/local/lib/python3.12/dist-packages"
sys.path = [p for p in sys.path if p != _LOCAL]
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MEAN_T = torch.tensor(_MEAN).view(3, 1, 1)
_STD_T = torch.tensor(_STD).view(3, 1, 1)


def preprocess(im_rgb, imgsz):
    t = torch.from_numpy(im_rgb.transpose(2, 0, 1)).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(imgsz, imgsz),
                      mode="bilinear", align_corners=False)[0]
    t = (t - _MEAN_T) / _STD_T
    return t.unsqueeze(0)


def scores(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    ps, gs = p.sum(), g.sum()
    union = np.logical_or(p, g).sum()
    if ps == 0 and gs == 0:
        return 1.0, 1.0, 1.0, 1.0
    dice = 2 * inter / (ps + gs + 1e-9)
    iou = inter / (union + 1e-9)
    recall = inter / (gs + 1e-9)
    precision = inter / (ps + 1e-9)
    return float(dice), float(iou), float(recall), float(precision)


@torch.no_grad()
def eval_ckpt(ckpt_path, data, split, imgsz_override=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    imgsz = imgsz_override or a["imgsz"]
    model = build_model(
        encoder_name=a["encoder"], encoder_weights=None, decoder=a["decoder"],
        decoder_attention=None if a.get("decoder_attention") in ("none", None) else a["decoder_attention"],
        num_classes=1, in_channels=3)
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()

    img_dir = Path(data) / "images" / split
    lbl_dir = Path(data) / "labels" / split
    img_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

    agg = np.zeros(4)
    n = 0
    for ip in img_paths:
        im = np.asarray(Image.open(ip).convert("RGB"))
        h, w = im.shape[:2]
        gt = polygons_to_mask(lbl_dir / f"{ip.stem}.txt", w, h)
        x = preprocess(im, imgsz).to(DEVICE)
        logits = model(x)
        logits = F.interpolate(logits.float(), size=(h, w),
                               mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
        agg += np.array(scores(pred, gt))
        n += 1
    return agg / max(n, 1), n, imgsz, ck["epoch"], ck["val"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT.parent.parent / "dataset_deeplab"))
    p.add_argument("--split", default="test")
    p.add_argument("--sizes", default="128,192,224,256,320,384,512")
    args = p.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    print(f"[data] {args.data}  split={args.split}  device={DEVICE}")
    print(f"[eval] effb0_unetpp_deeplab_* : {sizes}\n")
    header = f"{'model':<28}{'imgsz':>6}{'ep':>4}{'Dice':>9}{'IoU':>9}{'Recall':>9}{'Prec':>9}"
    print(header)
    print("-" * len(header))
    rows = []
    for s in sizes:
        ckpt = ROOT / f"runs/effb0_unetpp_deeplab_{s}/checkpoint_best.pth"
        if not ckpt.exists():
            print(f"[warn] 없음: {ckpt}")
            continue
        (dice, iou, rec, prec), n, imgsz, ep, val = eval_ckpt(ckpt, args.data, args.split)
        name = f"effb0_unetpp_deeplab_{s}"
        print(f"{name:<28}{imgsz:>6}{ep:>4}{dice:>9.4f}{iou:>9.4f}{rec:>9.4f}{prec:>9.4f}", flush=True)
        rows.append((name, dice, iou, rec, prec))

    if rows:
        best = max(rows, key=lambda r: r[1])
        print("-" * len(header))
        print(f"[best Dice] {best[0]}  Dice={best[1]:.4f}  IoU={best[2]:.4f}  (n={n} imgs, split={args.split})")


if __name__ == "__main__":
    main()
