"""
경계 허용(boundary-tolerant) Dice 측정 — "0.8이 어디서 나왔나" 재현용.

방법:
  - best 체크포인트로 각 이미지를 예측(threshold 0.5), 빈 마스크(무병변) 이미지는 제외.
  - GT와 예측을 각각 반경 tau(px) 만큼 dilate 한 뒤, 서로 tau 이내에 있으면 '맞음'으로 카운트.
      tolerant_dice = (|pred ∩ dilate(gt,tau)| + |gt ∩ dilate(pred,tau)|) / (|pred| + |gt|)
    tau=0 이면 표준 Dice 와 동일. tau가 커질수록 경계 오차를 눈감아 줌.
  - 이건 표준 Dice가 아니라 NSD 계열의 '경계 허용' 진단 지표다.
    (모델이 위치는 맞히고, 격차는 홍반 경계 모호함 때문임을 보이는 용도)

실행:
  python3 boundary_dice.py                       # test, tau=0,2,5,10,15
  python3 boundary_dice.py --split val --taus 0 5 10 20
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent / "unext"))  # unext/archs(UNeXt). 공유모듈은 이 파일과 같은 segmentation/
from archs import UNext
from augment import build_val_tf
from dataset import AtopySegDataset

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "dataset_face"))
    p.add_argument("--split", default="test", help="test 또는 val")
    p.add_argument("--ckpt", default=str(ROOT / "unext/runs/unext_face/checkpoint_best.pth"))
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--thr", type=float, default=0.5, help="이진화 임계값")
    p.add_argument("--taus", type=int, nargs="+", default=[0, 2, 5, 10, 15],
                   help="경계 허용 반경(px) 목록")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def tol_dice(pred, gt, tau):
    """pred, gt: bool (H,W). tau=0 이면 표준 Dice."""
    ps, gs = pred.sum(), gt.sum()
    if ps + gs == 0:
        return 1.0
    if tau == 0:
        return (2 * (pred & gt).sum()) / (ps + gs + 1e-6)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tau + 1, 2 * tau + 1))
    gd = cv2.dilate(gt.astype(np.uint8), k) > 0
    pd = cv2.dilate(pred.astype(np.uint8), k) > 0
    ph = (pred & gd).sum()      # τ 이내에 GT가 있는 예측 픽셀
    gh = (gt & pd).sum()        # τ 이내에 예측이 있는 GT 픽셀
    return (ph + gh) / (ps + gs + 1e-6)


def main():
    a = parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    ds = AtopySegDataset(a.data, a.split, build_val_tf(a.imgsz))
    model = UNext(num_classes=1, img_size=a.imgsz).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])

    acc = {t: [] for t in a.taus}
    n_lesion = 0
    with torch.no_grad():
        for i in range(len(ds)):
            img, msk, _ = ds[i]
            gt = msk[0].numpy() > 0.5
            if gt.sum() == 0:            # 무병변 이미지 제외
                continue
            n_lesion += 1
            logit = model(img.unsqueeze(0).to(dev))
            pred = torch.sigmoid(logit)[0, 0].cpu().numpy() > a.thr
            for t in a.taus:
                acc[t].append(tol_dice(pred, gt, t))

    print(f"[{a.split}] 병변 이미지 {n_lesion}장, threshold={a.thr}, ckpt={Path(a.ckpt).name}")
    print(" tau(px)   Dice")
    for t in a.taus:
        tag = "  <- 표준 Dice" if t == 0 else ""
        print(f"   {t:>3}    {np.mean(acc[t]):.3f}{tag}")


if __name__ == "__main__":
    main()
