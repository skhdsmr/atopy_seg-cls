"""
경계 허용 tau(px) 를 눈으로 확인 — GT vs 예측 경계 + tau 허용 밴드 오버레이.

각 이미지 3열:
  1) 원본
  2) GT(초록 채움) + 예측 경계(빨강 선)  -> 두 경계가 실제로 얼마나 벌어졌나
  3) tau 허용 구역 = dilate(GT, tau)(노랑 반투명) + GT 경계(초록) + 예측 경계(빨강)
     -> 빨강(예측)이 노랑(허용구역) 안에 들면 'tau 이내로 맞음'

좌상단에 tau px 길이의 스케일 바를 그려 절대 크기 감을 준다(512px 기준).

실행: python3 viz_tolerance.py --tau 15 --n 4
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
from visualize import denorm
from boundary_dice import tol_dice

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "dataset_face"))
    p.add_argument("--split", default="test")
    p.add_argument("--ckpt", default=str(ROOT / "unext/runs/unext_face/checkpoint_best.pth"))
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--tau", type=int, default=15, help="허용 반경(px)")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--n", type=int, default=4, help="샘플 수(중앙값 부근)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(ROOT / "unext/runs/unext_face/viz"))
    return p.parse_args()


def blend(img, mask, color, a=0.45):
    o = img.copy()
    o[mask] = (( 1 - a) * o[mask] + a * np.array(color)).astype(np.uint8)
    return o


def draw_contour(img, mask, color, thick=2):
    o = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, color, thick)
    return o


def scale_bar(img, length, label):
    o = img.copy()
    x0, y0 = 15, 25
    cv2.line(o, (x0, y0), (x0 + length, y0), (255, 255, 255), 3)
    cv2.line(o, (x0, y0), (x0 + length, y0), (0, 0, 0), 1)
    cv2.putText(o, label, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return o


def main():
    a = parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    ds = AtopySegDataset(a.data, a.split, build_val_tf(a.imgsz))
    model = UNext(num_classes=1, img_size=a.imgsz).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, msk, case = ds[i]
            gt = msk[0].numpy() > 0.5
            if gt.sum() == 0:
                continue
            pred = torch.sigmoid(model(img.unsqueeze(0).to(dev)))[0, 0].cpu().numpy() > a.thr
            d0 = tol_dice(pred, gt, 0)
            dt = tol_dice(pred, gt, a.tau)
            rows.append((d0, case, denorm(img), gt, pred, dt))
    rows.sort(key=lambda r: r[0])
    mid = len(rows) // 2
    k = a.n // 2
    pick = rows[mid - k:mid + (a.n - k)]

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.tau + 1, 2 * a.tau + 1))
    fig, axes = plt.subplots(len(pick), 3, figsize=(11, 3.7 * len(pick)))
    if len(pick) == 1:
        axes = axes[None, :]
    titles = ["Image", "GT(초록채움) + Pred(빨강선)",
              f"허용구역 dilate(GT,{a.tau}px)=노랑 + Pred=빨강"]
    for r, (d0, case, im, gt, pred, dt) in enumerate(pick):
        band = cv2.dilate(gt.astype(np.uint8), kern) > 0
        p1 = scale_bar(im, a.tau, f"{a.tau}px")
        p2 = draw_contour(blend(im, gt, (0, 200, 0)), pred, (230, 0, 0), 2)
        p3 = blend(im, band, (240, 220, 0), 0.35)
        p3 = draw_contour(p3, gt, (0, 200, 0), 2)
        p3 = draw_contour(p3, pred, (230, 0, 0), 2)
        for c, pan in enumerate([p1, p2, p3]):
            axes[r, c].imshow(pan); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(f"{case}\nDice τ0={d0:.2f}  τ{a.tau}={dt:.2f}", fontsize=8)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    fig.suptitle(f"경계 허용 τ={a.tau}px  (입력 {a.imgsz}px 기준 ≈ 폭의 {100*a.tau/a.imgsz:.1f}%)",
                 fontsize=12)
    plt.tight_layout()
    out = out_dir / f"tolerance_tau{a.tau}.png"
    plt.savefig(out, dpi=95, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")
    print(f" 표시 샘플 τ0 Dice: {[round(r[0],2) for r in pick]}  ->  τ{a.tau}: {[round(r[5],2) for r in pick]}")


if __name__ == "__main__":
    main()
