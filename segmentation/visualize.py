"""
학습된 UNeXt 예측 vs GT 시각화 (0.50 정체가 모델 탓인지 라벨 탓인지 진단용).

- best 체크포인트를 로드해 val(기본) 전체에 대해 per-sample Dice 계산.
- Dice 상위/중위/하위 샘플을 골라 [원본 | GT | 예측 | 오버레이] 격자 PNG 저장.
- 오버레이 색: 초록=GT만(놓침/FN), 빨강=예측만(오탐/FP), 노랑=겹침(정답/TP).
  => 노랑이 많고 경계만 어긋나면 '라벨/경계 한계', 초록/빨강이 크게 어긋나면 '모델 한계'.

사용: python3 visualize.py --split val --n 8
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
from augment import build_val_tf, _MEAN, _STD
from dataset import AtopySegDataset
from metrics import seg_scores

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "dataset_face"))
    p.add_argument("--split", default="val", help="val 또는 test")
    p.add_argument("--ckpt", default=str(ROOT / "unext/runs/unext_face/checkpoint_best.pth"))
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--n", type=int, default=8, help="시각화할 샘플 수(상/중/하 골고루)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(ROOT / "unext/runs/unext_face/viz"))
    return p.parse_args()


def denorm(img_t):
    """정규화 이미지 텐서 -> 0~255 RGB uint8 (표시용)."""
    x = img_t.numpy().transpose(1, 2, 0) * np.array(_STD) + np.array(_MEAN)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def overlay(img, gt, pred):
    """초록=FN(GT만), 빨강=FP(예측만), 노랑=TP(겹침)."""
    o = img.copy()
    tp = (gt == 1) & (pred == 1)
    fn = (gt == 1) & (pred == 0)
    fp = (gt == 0) & (pred == 1)
    o[fn] = (0.4 * o[fn] + 0.6 * np.array([0, 200, 0])).astype(np.uint8)     # 초록
    o[fp] = (0.4 * o[fp] + 0.6 * np.array([220, 0, 0])).astype(np.uint8)     # 빨강
    o[tp] = (0.4 * o[tp] + 0.6 * np.array([230, 230, 0])).astype(np.uint8)   # 노랑
    # GT 경계선(흰색)으로 정답 윤곽 강조
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, (255, 255, 255), 1)
    return o


def main():
    a = parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    ds = AtopySegDataset(root=a.data, split=a.split, transform=build_val_tf(a.imgsz))
    model = UNext(num_classes=1, img_size=a.imgsz).to(dev).eval()
    ckpt = torch.load(a.ckpt, map_location=dev)
    model.load_state_dict(ckpt["model"])
    print(f"[ckpt] epoch={ckpt.get('epoch')} val={ckpt.get('val')}")

    # per-sample Dice
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            img_t, msk_t, case = ds[i]
            logit = model(img_t.unsqueeze(0).to(dev))
            d = seg_scores(logit, msk_t.unsqueeze(0).to(dev))["dice"]
            pred = (torch.sigmoid(logit)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            rows.append((d, case, denorm(img_t), msk_t[0].numpy().astype(np.uint8), pred))
    rows.sort(key=lambda r: r[0])
    dices = np.array([r[0] for r in rows])
    print(f"[{a.split}] n={len(rows)}  Dice  mean={dices.mean():.3f}  "
          f"median={np.median(dices):.3f}  min={dices.min():.3f}  max={dices.max():.3f}")

    # 하위 n/2 + 상위 n/2 선택 (실패/성공 대비)
    k = a.n // 2
    picked = rows[:k] + rows[-k:]

    fig, axes = plt.subplots(len(picked), 4, figsize=(14, 3.4 * len(picked)))
    if len(picked) == 1:
        axes = axes[None, :]
    col_titles = ["Image", "GT", "Pred", "Overlay (G=miss R=false Y=hit)"]
    for r, (d, case, img, gt, pred) in enumerate(picked):
        panels = [img, gt * 255, pred * 255, overlay(img, gt, pred)]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray" if c in (1, 2) else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
        axes[r, 0].set_ylabel(f"{case}\nDice={d:.3f}", fontsize=9, rotation=0,
                              labelpad=55, va="center")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    plt.tight_layout()
    out = out_dir / f"pred_vs_gt_{a.split}.png"
    plt.savefig(out, dpi=90, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
