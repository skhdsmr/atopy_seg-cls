"""
threshold_sweep.py

학습된 best 체크포인트로 val 확률맵을 한 번만 계산해 캐시하고,
임계값(threshold)만 바꿔가며 Dice/IoU/Recall/Precision 을 훑는다.
- 기본 0.5 대비 최적 임계값에서 Dice 가 얼마나 오르는지 확인.
- val 에서 고른 최적 임계값을 test 에도 그대로 적용해 낙관 편향 없는 추정치도 출력.

사용:
  python3 threshold_sweep.py --run unext/runs/unext_face_bcedice_merged --data ../../dataset_face_merged
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / "unext"))  # unext/archs(UNeXt). 공유모듈은 이 파일과 같은 segmentation/
from archs import UNext
from augment import build_val_tf
from dataset import AtopySegDataset

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def cache_probs(model, loader, device):
    """각 이미지의 sigmoid 확률맵과 타깃을 (flatten) 리스트로 캐시."""
    model.eval()
    probs, tgts = [], []
    for img, mask, _ in loader:
        p = torch.sigmoid(model(img.to(device))).cpu()   # (B,1,H,W)
        for i in range(p.size(0)):
            probs.append(p[i, 0].reshape(-1))
            tgts.append((mask[i, 0] > 0.5).float().reshape(-1))
    return probs, tgts


def scores_at(probs, tgts, thr, smooth=1e-5):
    """캐시된 확률맵에 임계값 thr 적용 -> per-image 평균 Dice/IoU/R/P."""
    d = i = r = pr = 0.0
    n = len(probs)
    for p, t in zip(probs, tgts):
        pred = (p > thr).float()
        tp = (pred * t).sum()
        fp = (pred * (1 - t)).sum()
        fn = ((1 - pred) * t).sum()
        d += ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).item()
        i += ((tp + smooth) / (tp + fp + fn + smooth)).item()
        r += ((tp + smooth) / (tp + fn + smooth)).item()
        pr += ((tp + smooth) / (tp + fp + smooth)).item()
    return d / n, i / n, r / n, pr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="unext/runs/unext_face_bcedice_merged")
    ap.add_argument("--data", default="../../dataset_face_merged")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / args.run / "checkpoint_best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[ckpt] {ckpt_path}  (epoch {ckpt.get('epoch','?')}, "
          f"저장시 val Dice@0.5={ckpt.get('val',{}).get('dice',float('nan')):.4f})")

    model = UNext(num_classes=1, input_channels=3, img_size=args.imgsz).to(device)
    model.load_state_dict(ckpt["model"])

    val_tf = build_val_tf(img_size=args.imgsz)
    def loader(split):
        ds = AtopySegDataset(root=args.data, split=split, transform=val_tf)
        return DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    vp, vt = cache_probs(model, loader("val"), device)
    tp_, tt = cache_probs(model, loader("test"), device)
    print(f"[data] val={len(vp)}장 test={len(tp_)}장  (from {args.data})\n")

    thrs = np.round(np.arange(args.lo, args.hi + 1e-9, args.step), 3)
    print(f"{'thr':>5} | {'Dice':>7} {'IoU':>7} {'Recall':>7} {'Prec':>7}   (VAL)")
    print("-" * 46)
    best_thr, best_dice = 0.5, -1
    rows = {}
    for thr in thrs:
        d, i, r, p = scores_at(vp, vt, float(thr))
        rows[float(thr)] = (d, i, r, p)
        mark = ""
        if abs(thr - 0.5) < 1e-9:
            mark = "  <- 기본 0.5"
        if d > best_dice:
            best_dice, best_thr = d, float(thr)
        print(f"{thr:>5.2f} | {d:>7.4f} {i:>7.4f} {r:>7.4f} {p:>7.4f}{mark}")

    d05 = rows[0.5][0]
    print("-" * 46)
    print(f"[VAL] 기본 thr=0.50  Dice={d05:.4f}")
    print(f"[VAL] 최적 thr={best_thr:.2f}  Dice={best_dice:.4f}  "
          f"(+{best_dice - d05:+.4f}, {100*(best_dice-d05)/max(d05,1e-9):+.1f}%)")

    # 낙관 편향 없는 추정: val 최적 thr 을 test 에 적용
    td_05 = scores_at(tp_, tt, 0.5)
    td_best = scores_at(tp_, tt, best_thr)
    print()
    print(f"[TEST] thr=0.50       Dice={td_05[0]:.4f} IoU={td_05[1]:.4f} "
          f"R={td_05[2]:.4f} P={td_05[3]:.4f}")
    print(f"[TEST] thr={best_thr:.2f}(val최적) Dice={td_best[0]:.4f} IoU={td_best[1]:.4f} "
          f"R={td_best[2]:.4f} P={td_best[3]:.4f}  (+{td_best[0]-td_05[0]:+.4f})")


if __name__ == "__main__":
    main()
