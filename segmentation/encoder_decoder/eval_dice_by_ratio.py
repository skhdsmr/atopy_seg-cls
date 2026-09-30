"""HRNet 체크포인트로 test split 평가 -> polygon 커버리지 비율 구간별 Dice.

학습(train.py)과 동일한 전처리(resize 1024 + ImageNet 정규화, thr 0.5)로
이미지별 Dice/IoU 를 구하고, 각 이미지의 GT 마스크 전경 비율(=모든 polygon 을
합쳐 래스터화한 면적 / 이미지 면적)로 구간을 나눠 평균 Dice 를 집계한다.
"""
import sys

# --- GPU torch 부트스트랩 -------------------------------------------------
# 기본 sys.path 는 .local(torch 2.2.2+cpu) 이 stdlib/dist 보다 앞이라 CPU torch 가
# 먼저 잡힌다. 순서를 [stdlib -> dist-packages(GPU torch 2.10) -> .local(나머지 패키지)]
# 로 바꿔 GPU torch 를 쓰되, enum34 셰도잉(dist 를 맨 앞에 두면 stdlib enum 을 가림)은 피한다.
_LOCAL = "/home/work/.local/lib/python3.12/site-packages"
_DIST = "/usr/local/lib/python3.12/dist-packages"
sys.path = [p for p in sys.path if p not in (_LOCAL,)]
if _DIST not in sys.path:
    sys.path.append(_DIST)
sys.path.append(_LOCAL)  # torch 뒤에 두어 smp/timm/albumentations 만 여기서 로드

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent
sys.path.insert(0, str(SHARED))

from augment import build_val_tf          # noqa: E402
from dataset import AtopySegDataset       # noqa: E402
from metrics import seg_scores            # noqa: E402
from model import build_model             # noqa: E402

CKPT = ROOT / "runs" / "hrnet18_unet" / "checkpoint_best.pth"
DATA = (ROOT.parent.parent / "dataset_face_strat_merged")
IMGSZ = 1024
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck["args"]
print(f"[ckpt] epoch={ck['epoch']} val={ck['val']}")
print(f"[ckpt] encoder={a['encoder']} decoder={a['decoder']} imgsz={a['imgsz']} crop={a['crop_size']}")

model = build_model(encoder_name=a["encoder"], encoder_weights=None,
                    decoder=a["decoder"],
                    decoder_attention=None if a["decoder_attention"] == "none" else a["decoder_attention"],
                    num_classes=1, in_channels=3)
model.load_state_dict(ck["model"])
model.to(DEVICE).eval()

val_tf = build_val_tf(img_size=IMGSZ)
ds = AtopySegDataset(root=str(DATA), split="test", transform=val_tf)
print(f"[data] test images = {len(ds)}")

rows = []  # (case, gt_ratio_%, dice, iou)
with torch.no_grad():
    for i in range(len(ds)):
        img, mask, case = ds[i]
        logits = model(img.unsqueeze(0).to(DEVICE))
        s = seg_scores(logits.float(), mask.unsqueeze(0).to(DEVICE))
        # GT 커버리지 비율: resize 후 마스크 기준(모델이 실제로 채점된 마스크와 동일)
        ratio = mask.mean().item() * 100.0
        rows.append((case, ratio, s["dice"], s["iou"]))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ds)} done")

rows.sort(key=lambda r: r[1])
np.save(ROOT / "runs" / "hrnet18_unet" / "per_image_dice.npy",
        np.array([(r[1], r[2], r[3]) for r in rows], dtype=np.float64))

# ---- 구간 정의 (커버리지 %) ----
bins = [(-1e-9, 1e-6, "0% (병변없음)"),
        (1e-6, 2, "0–2%"),
        (2, 5, "2–5%"),
        (5, 10, "5–10%"),
        (10, 15, "10–15%"),
        (15, 20, "15–20%"),
        (20, 30, "20–30%"),
        (30, 1e9, ">30%")]

overall_d = np.mean([r[2] for r in rows])
overall_i = np.mean([r[3] for r in rows])
print("\n=== 전체 test ===")
print(f"images={len(rows)}  mean Dice={overall_d:.4f}  mean IoU={overall_i:.4f}")

print("\n=== polygon 커버리지 비율 구간별 Dice ===")
print(f"{'구간':14s} {'이미지':>5s} {'평균Dice':>9s} {'중앙Dice':>9s} {'평균IoU':>8s}")
print("-" * 52)
for lo, hi, lab in bins:
    sel = [r for r in rows if lo < r[1] <= hi] if lo > -1 else [r for r in rows if r[1] <= hi]
    if not sel:
        print(f"{lab:14s} {0:5d} {'-':>9s} {'-':>9s} {'-':>8s}")
        continue
    d = [r[2] for r in sel]; io = [r[3] for r in sel]
    print(f"{lab:14s} {len(sel):5d} {np.mean(d):9.4f} {np.median(d):9.4f} {np.mean(io):8.4f}")
