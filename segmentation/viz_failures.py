"""
viz_failures.py — 잔차 오류(경계 슬롭이 아닌 진짜 미탐/오탐)가 큰 케이스 시각화.

각 병변 이미지에 대해:
  - 예측(thr=0.5)과 GT 를 비교
  - TP=초록, FN(놓친 병변)=빨강, FP(헛예측)=파랑 으로 원본 위에 반투명 오버레이
  - 경계 허용 Dice@tau 로 정렬 -> 큰 τ 로도 안 메워지는(=진짜 오류) 케이스를 상위로
각 케이스에 dice / tol@15 / 미탐율(FN/GT) / 헛예측율(FP/pred) 를 표기.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / "unext"))  # unext/archs(UNeXt). 공유모듈은 이 파일과 같은 segmentation/
from archs import UNext
from augment import build_val_tf
from dataset import AtopySegDataset

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def tol_dice(pred, gt, tau):
    ps, gs = pred.sum(), gt.sum()
    if ps + gs == 0:
        return 1.0
    if tau == 0:
        return (2 * (pred & gt).sum()) / (ps + gs + 1e-6)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tau + 1, 2 * tau + 1))
    gd = cv2.dilate(gt.astype(np.uint8), k) > 0
    pd = cv2.dilate(pred.astype(np.uint8), k) > 0
    return ((pred & gd).sum() + (gt & pd).sum()) / (ps + gs + 1e-6)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--data", default=str(REPO / "dataset_face_merged"))
    a_.add_argument("--split", default="val")
    a_.add_argument("--ckpt", default=str(ROOT / "unext/runs/unext_face_bcedice_merged/checkpoint_best.pth"))
    a_.add_argument("--imgsz", type=int, default=512)
    a_.add_argument("--thr", type=float, default=0.5)
    a_.add_argument("--topk", type=int, default=6)
    a_.add_argument("--out", default=str(ROOT / "unext/merge_review/failures.png"))
    a_.add_argument("--device", default="cuda")
    a = a_.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ds = AtopySegDataset(a.data, a.split, build_val_tf(a.imgsz))
    model = UNext(num_classes=1, img_size=a.imgsz).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])

    S = a.imgsz
    recs = []
    with torch.no_grad():
        for i in range(len(ds)):
            img_t, msk, _ = ds[i]
            gt = msk[0].numpy() > 0.5
            if gt.sum() == 0:
                continue
            logit = model(img_t.unsqueeze(0).to(dev))
            pred = torch.sigmoid(logit)[0, 0].cpu().numpy() > a.thr
            tp = (pred & gt).sum()
            fp = (pred & ~gt).sum()
            fn = (~pred & gt).sum()
            dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
            t15 = tol_dice(pred, gt, 15)
            recs.append({
                "path": ds.img_paths[i], "gt": gt, "pred": pred,
                "dice": dice, "t15": t15,
                "miss": fn / (gt.sum() + 1e-6),        # 미탐율
                "halluc": fp / (pred.sum() + 1e-6),    # 헛예측율
            })

    # τ=15 로도 안 메워지는(=경계 슬롭 아닌 진짜 오류) 순으로 정렬
    recs.sort(key=lambda r: r["t15"])
    picks = recs[:a.topk]

    tiles = []
    for r in picks:
        img = cv2.cvtColor(np.array(Image.open(r["path"]).convert("RGB")), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (S, S))
        gt, pred = r["gt"], r["pred"]
        ov = img.copy()
        ov[gt & pred] = (0, 200, 0)        # TP 초록
        ov[gt & ~pred] = (0, 0, 255)       # FN 미탐 빨강
        ov[~gt & pred] = (255, 60, 0)      # FP 헛예측 파랑
        vis = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
        txt = (f"{r['path'].stem[:16]}  Dice={r['dice']:.2f} t15={r['t15']:.2f}")
        txt2 = (f"miss(FN)={r['miss']*100:.0f}%  halluc(FP)={r['halluc']*100:.0f}%")
        cv2.rectangle(vis, (0, 0), (S, 62), (0, 0, 0), -1)
        cv2.putText(vis, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, txt2, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 220, 255), 2)
        tiles.append(vis)

    # 범례 바
    legend = np.zeros((60, S * min(3, len(tiles)), 3), np.uint8)
    for x, (c, lab) in enumerate([((0, 200, 0), "TP(green)"),
                                  ((0, 0, 255), "FN miss(red)"),
                                  ((255, 60, 0), "FP halluc(blue)")]):
        cv2.putText(legend, lab, (20 + x * 260, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, c, 2)

    cols = 3
    rows_img = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows_img.append(np.hstack(row))
    grid = np.vstack(rows_img)
    grid = np.vstack([legend if legend.shape[1] == grid.shape[1]
                      else cv2.resize(legend, (grid.shape[1], 60)), grid])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, grid)
    print(f"[{a.split}] 병변 {len(recs)}장 중 τ15 최저 {len(picks)}장 -> {a.out}")
    for r in picks:
        print(f"  {r['path'].stem:22s} Dice={r['dice']:.3f} t15={r['t15']:.3f} "
              f"miss={r['miss']*100:4.0f}% halluc={r['halluc']*100:4.0f}%")


if __name__ == "__main__":
    main()
