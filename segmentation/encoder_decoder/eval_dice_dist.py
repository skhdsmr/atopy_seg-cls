"""effb0+UNet++ 512 체크포인트로 test split 평가 -> Dice 분포 + 저성능 구간 FP/FN 분해.

두 가지 질문에 답한다:
  1) Dice 가 이미지별로 어떻게 퍼져 있나 (평균 하나로는 안 보이는 꼬리)
  2) Dice < 0.7 인 이미지들은 '더 칠해서'(FP) 틀렸나 '덜 칠해서'(FN) 틀렸나

학습과 동일 전처리(resize 512 + ImageNet 정규화, thr 0.5)를 쓴다. 지표는
metrics.seg_scores 와 같은 정의지만, 여기선 배치 평균이 아니라 이미지별
tp/fp/fn 원자료가 필요해서 직접 센다.

빈 GT(병변 없음) 이미지는 Dice 가 정의되지 않는다(smooth 때문에 pred 도 비면
1.0, 조금이라도 칠하면 0.0 으로 튄다). 분포 표에서 분리해 따로 보고한다.

실행: run.sh 와 동일하게 ../../.venv-train/bin/python 으로 부를 것
      (eval_dice_by_ratio.py 의 sys.path 부트스트랩은 시스템 python3 전용이라
       venv 에서 쓰면 dist-packages 의 깨진 onnxscript 를 끌고 온다).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent
sys.path.insert(0, str(SHARED))

from augment import build_val_tf          # noqa: E402
from dataset import AtopySegDataset       # noqa: E402
from model import build_model             # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="effb0_unetpp_512")
p.add_argument("--split", default="test")
p.add_argument("--thr", type=float, default=0.5)
p.add_argument("--low", type=float, default=0.7, help="저성능 기준 Dice")
args = p.parse_args()

CKPT = ROOT / "runs" / args.run / "checkpoint_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck["args"]
DATA = (ROOT / a["project"]).parent.parent / Path(a["data"]).name
if not DATA.exists():
    DATA = ROOT.parent.parent / Path(a["data"]).name
IMGSZ = a["imgsz"]

print(f"[ckpt] {args.run}  epoch={ck['epoch']}")
print(f"[ckpt] encoder={a['encoder']} decoder={a['decoder']} imgsz={IMGSZ} loss={a['loss']}")
print(f"[ckpt] val(학습중 기록) = { {k: round(v, 4) for k, v in ck['val'].items()} }")
print(f"[data] {DATA}  split={args.split}  thr={args.thr}")

model = build_model(encoder_name=a["encoder"], encoder_weights=None,
                    decoder=a["decoder"],
                    decoder_attention=None if a["decoder_attention"] == "none" else a["decoder_attention"],
                    num_classes=1, in_channels=3)
model.load_state_dict(ck["model"])
model.to(DEVICE).eval()

ds = AtopySegDataset(root=str(DATA), split=args.split, transform=build_val_tf(img_size=IMGSZ))
print(f"[data] {args.split} images = {len(ds)}\n")

recs = []  # case, gt_px, tp, fp, fn
with torch.no_grad():
    for i in range(len(ds)):
        img, mask, case = ds[i]
        logit = model(img.unsqueeze(0).to(DEVICE))
        pred = (torch.sigmoid(logit.float()) > args.thr).float().view(-1)
        t = (mask.to(DEVICE) > 0.5).float().view(-1)
        tp = (pred * t).sum().item()
        fp = (pred * (1 - t)).sum().item()
        fn = ((1 - pred) * t).sum().item()
        recs.append((case, t.sum().item(), tp, fp, fn))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} done")

npx = float(IMGSZ * IMGSZ)
empty = [r for r in recs if r[1] == 0]
pos = [r for r in recs if r[1] > 0]


def dice(r):
    _, _, tp, fp, fn = r
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")


d = np.array([dice(r) for r in pos])

print(f"\n{'='*74}\nDice 분포 — {args.split} (병변 있는 이미지 {len(pos)}장)\n{'='*74}")
print(f"  mean {d.mean():.4f}   std {d.std():.4f}")
qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
print("  " + "  ".join(f"p{q}={np.percentile(d, q):.3f}" for q in qs))

print(f"\n{'구간':>12s} {'장수':>6s} {'비율':>7s}  {'누적':>7s}  히스토그램")
edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
cum = 0
for lo, hi in zip(edges[:-1], edges[1:]):
    n = int(((d >= lo) & (d < hi)).sum())
    cum += n
    lab = f"{lo:.1f}–{min(hi,1.0):.1f}"
    bar = "█" * round(40 * n / max(len(d), 1))
    print(f"{lab:>12s} {n:6d} {100*n/len(d):6.1f}% {100*cum/len(d):6.1f}%  {bar}")

if empty:
    ok = sum(1 for r in empty if r[3] == 0)
    fp_px = np.array([r[3] for r in empty])
    print(f"\n  [별도] GT 빈 이미지 {len(empty)}장 (Dice 미정의): "
          f"완전정답(FP=0) {ok}장 / 오검출 {len(empty)-ok}장, "
          f"오검출 FP 면적 중앙값 {np.median(fp_px)/npx*100:.3f}%")

# ---- Dice < low 구간의 FP/FN 분해 ----
low = [r for r in pos if dice(r) < args.low]
hi_ = [r for r in pos if dice(r) >= args.low]
print(f"\n{'='*74}\nDice < {args.low} 구간의 오차 분해 (FP=과검출, FN=미검출)\n{'='*74}")
print(f"  대상 {len(low)}장 / 전체 {len(pos)}장 = {100*len(low)/len(pos):.1f}%")

if low:
    TP = sum(r[2] for r in low); FP = sum(r[3] for r in low); FN = sum(r[4] for r in low)
    print(f"\n  [화소 총합 기준] 전체 오차 화소 중")
    print(f"    FP(과검출) {FP/(FP+FN)*100:5.1f}%   FN(미검출) {FN/(FP+FN)*100:5.1f}%   FP/FN = {FP/max(FN,1):.2f}")
    print(f"    micro recall {TP/max(TP+FN,1):.3f}  precision {TP/max(TP+FP,1):.3f}")

    # 이미지 단위: 각 장이 FP 우세인지 FN 우세인지 (총합은 큰 이미지가 좌우한다)
    fpd = sum(1 for r in low if r[3] > r[4])
    fnd = sum(1 for r in low if r[4] > r[3])
    print(f"\n  [이미지 단위] FP 우세 {fpd}장({100*fpd/len(low):.1f}%) / "
          f"FN 우세 {fnd}장({100*fnd/len(low):.1f}%) / 동률 {len(low)-fpd-fnd}장")

    r_img = np.array([r[3] / max(r[3] + r[4], 1) for r in low])
    print(f"  이미지별 FP비율(FP/(FP+FN)) 중앙값 {np.median(r_img):.3f}  "
          f"평균 {r_img.mean():.3f}")
    print(f"  GT 면적 중앙값 {np.median([r[1] for r in low])/npx*100:.2f}%  "
          f"(Dice>={args.low} 군은 {np.median([r[1] for r in hi_])/npx*100:.2f}%)")

    print(f"\n  {'case':<38s} {'dice':>6s} {'GT%':>6s} {'FP%':>6s} {'FN%':>6s} {'우세':>5s}")
    for r in sorted(low, key=dice)[:15]:
        case, gt, tp, fp, fn = r
        print(f"  {case[:38]:<38s} {dice(r):6.3f} {100*gt/npx:6.2f} "
              f"{100*fp/npx:6.2f} {100*fn/npx:6.2f} {'FP' if fp > fn else 'FN':>5s}")
    if len(low) > 15:
        print(f"  ... 외 {len(low)-15}장 (전체는 아래 npy)")

out = ROOT / "runs" / args.run / f"per_image_{args.split}.npy"
np.save(out, np.array([(r[1], r[2], r[3], r[4]) for r in recs], dtype=np.float64))
print(f"\n[저장] {out}  (열: gt_px, tp, fp, fn / 행 순서 = dataset 순서)")
