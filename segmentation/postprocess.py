"""
오탐 억제 ①임계값 튜닝 + ②최소면적 제거 (재학습 없음).

절차:
  A. 진단 - thr=0.5 에서 오탐(FP) 픽셀의 확률 분포를 봐서
     '저신뢰 오탐(임계값으로 잡힘)' vs '고신뢰 오탐(재학습 필요)' 판별.
  B. 임계값 스윕 - val 에서 micro Dice 최대가 되는 threshold 탐색.
  C. 최소면적 스윕 - 위 threshold 에서 작은 조각 제거 크기 탐색.
  D. val 로 고른 (thr, min_area) 를 test 에 적용해 정직하게 검증.
  E. before/after 시각화 저장.

지표는 micro Dice(픽셀 합산) 중심 + precision/recall. (per-image 평균은 빈 마스크 왜곡)
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / "unext"))  # unext/archs(UNeXt). 공유모듈은 이 파일과 같은 segmentation/
from archs import UNext
from augment import build_val_tf
from dataset import AtopySegDataset
from visualize import denorm, overlay

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "dataset_face"))
    p.add_argument("--ckpt", default=str(ROOT / "unext/runs/unext_face/checkpoint_best.pth"))
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(ROOT / "unext/runs/unext_face/viz"))
    return p.parse_args()


def collect(model, ds, dev):
    """각 이미지의 확률맵(float16)과 GT(bool), 표시용 원본을 모아 반환."""
    probs, gts, imgs, cases = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img_t, msk_t, case = ds[i]
            logit = model(img_t.unsqueeze(0).to(dev))
            prob = torch.sigmoid(logit)[0, 0].cpu().numpy().astype(np.float16)
            probs.append(prob)
            gts.append(msk_t[0].numpy() > 0.5)
            imgs.append(denorm(img_t))
            cases.append(case)
    return probs, gts, imgs, cases


def remove_small(mask, min_area):
    if min_area <= 0:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def micro_scores(probs, gts, thr, min_area=0):
    """픽셀 합산 기반 Dice/Precision/Recall."""
    TP = FP = FN = 0
    for prob, g in zip(probs, gts):
        p = prob.astype(np.float32) > thr
        if min_area > 0:
            p = remove_small(p, min_area) > 0
        TP += np.logical_and(p, g).sum()
        FP += np.logical_and(p, ~g).sum()
        FN += np.logical_and(~p, g).sum()
    eps = 1e-6
    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    prec = (TP + eps) / (TP + FP + eps)
    rec = (TP + eps) / (TP + FN + eps)
    return dice, prec, rec, int(FP)


def main():
    a = parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    model = UNext(num_classes=1, img_size=a.imgsz).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])

    val = collect(model, AtopySegDataset(a.data, "val", build_val_tf(a.imgsz)), dev)
    test = collect(model, AtopySegDataset(a.data, "test", build_val_tf(a.imgsz)), dev)
    vp, vg, vi, vc = val
    tp_, tg, ti, tc = test

    # ---------- A. 오탐 신뢰도 진단 (thr=0.5) ----------
    fp_probs = []
    for prob, g in zip(vp, vg):
        p = prob.astype(np.float32)
        fp_mask = (p > 0.5) & (~g)
        fp_probs.append(p[fp_mask])
    fp_probs = np.concatenate(fp_probs) if fp_probs else np.array([0.5])
    frac_low = (fp_probs < 0.7).mean()
    print("========== A. 오탐 신뢰도 진단 (thr=0.5, val) ==========")
    print(f" 오탐 픽셀 확률  중앙값={np.median(fp_probs):.3f}  "
          f"75%={np.percentile(fp_probs,75):.3f}  90%={np.percentile(fp_probs,90):.3f}")
    print(f" 오탐 중 확률<0.7 비율 = {frac_low*100:.0f}%  "
          f"-> {'대부분 저신뢰: 임계값으로 잘 잡힘' if frac_low>0.6 else '고신뢰 오탐 많음: 재학습 고려'}")

    # ---------- B. 임계값 스윕 (val) ----------
    thrs = np.round(np.arange(0.30, 0.91, 0.05), 2)
    curve = [micro_scores(vp, vg, t) for t in thrs]
    dices = [c[0] for c in curve]
    best_i = int(np.argmax(dices))
    best_thr = float(thrs[best_i])
    base = micro_scores(vp, vg, 0.5)
    print("\n========== B. 임계값 스윕 (val, micro Dice) ==========")
    print(" thr  Dice   Prec   Rec    FP")
    for t, (d, pr, rc, fp) in zip(thrs, curve):
        star = "  <= best" if abs(t - best_thr) < 1e-6 else ""
        print(f" {t:.2f} {d:.3f}  {pr:.3f}  {rc:.3f}  {fp:>9d}{star}")
    print(f" baseline thr=0.50: Dice={base[0]:.3f} Prec={base[1]:.3f} Rec={base[2]:.3f} FP={base[3]}")
    print(f" best     thr={best_thr:.2f}: Dice={dices[best_i]:.3f}")

    # ---------- C. 최소면적 스윕 (val, best_thr) ----------
    areas = [0, 50, 100, 200, 400, 800, 1600]
    print(f"\n========== C. 최소면적 제거 스윕 (val, thr={best_thr:.2f}) ==========")
    print(" min_area  Dice   Prec   Rec    FP")
    ma_res = []
    for ma in areas:
        d, pr, rc, fp = micro_scores(vp, vg, best_thr, ma)
        ma_res.append((ma, d, pr, rc, fp))
        print(f" {ma:>7d}  {d:.3f}  {pr:.3f}  {rc:.3f}  {fp:>9d}")
    best_ma = max(ma_res, key=lambda r: r[1])[0]
    print(f" -> best min_area={best_ma}")

    # ---------- D. val 로 고른 설정을 test 에 적용 ----------
    print("\n========== D. 최종 검증 (val 로 고른 설정을 test 에 적용) ==========")
    print(f" 선택: threshold={best_thr:.2f}, min_area={best_ma}")
    for name, (P, G) in [("val", (vp, vg)), ("test", (tp_, tg))]:
        b = micro_scores(P, G, 0.5, 0)
        t = micro_scores(P, G, best_thr, best_ma)
        print(f" [{name}] baseline(0.5): Dice={b[0]:.3f} P={b[1]:.3f} R={b[2]:.3f} FP={b[3]}"
              f"  ->  tuned: Dice={t[0]:.3f} P={t[1]:.3f} R={t[2]:.3f} FP={t[3]}"
              f"   (FP {100*(1-t[3]/max(b[3],1)):.0f}%↓, Dice {t[0]-b[0]:+.3f})")

    # ---------- E. 스윕 곡선 + before/after 시각화 ----------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thrs, dices, "o-", label="micro Dice")
    ax.plot(thrs, [c[1] for c in curve], "s--", label="Precision")
    ax.plot(thrs, [c[2] for c in curve], "^--", label="Recall")
    ax.axvline(best_thr, color="k", ls=":", lw=1, label=f"best thr={best_thr:.2f}")
    ax.set_xlabel("threshold"); ax.set_ylabel("score"); ax.set_title("Threshold sweep (val)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p1 = out_dir / "threshold_sweep.png"; plt.savefig(p1, dpi=100); plt.close()
    print(f"\n[saved] {p1}")

    # before/after: 빈 마스크 오탐 케이스 위주 4개
    idx_empty = [i for i, g in enumerate(vg) if g.sum() == 0][:4]
    idx = idx_empty if idx_empty else list(range(4))
    fig, axes = plt.subplots(len(idx), 4, figsize=(14, 3.4 * len(idx)))
    if len(idx) == 1:
        axes = axes[None, :]
    titles = ["Image", "GT", "before (thr0.5)", f"after (thr{best_thr:.2f}+area{best_ma})"]
    for r, i in enumerate(idx):
        p = vp[i].astype(np.float32)
        pred_b = (p > 0.5)
        pred_a = remove_small(p > best_thr, best_ma) > 0
        panels = [vi[i], vg[i].astype(np.uint8) * 255,
                  overlay(vi[i], vg[i].astype(np.uint8), pred_b.astype(np.uint8)),
                  overlay(vi[i], vg[i].astype(np.uint8), pred_a.astype(np.uint8))]
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel, cmap="gray" if c == 1 else None)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=11)
        axes[r, 0].set_ylabel(vc[i], fontsize=9)
    plt.tight_layout()
    p2 = out_dir / "postprocess_before_after.png"; plt.savefig(p2, dpi=90, bbox_inches="tight"); plt.close()
    print(f"[saved] {p2}")


if __name__ == "__main__":
    main()
