"""dataset_lesion_merged_pred/pred/* (side-by-side 오버레이)에서 예측 마스크를 역추출하고,
후보 체크포인트들을 재추론해 어느 모델이 그 pred 를 만들었는지 IoU 로 특정한다."""
import sys
from pathlib import Path

_L = "/home/work/.local/lib/python3.12/site-packages"
_D = "/usr/local/lib/python3.12/dist-packages"
sys.path = [p for p in sys.path if p != _L]
sys.path.append(_D); sys.path.append(_L)

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from infer_to_labels import load_seg_model, infer_mask   # noqa: E402

REPO = ROOT.parent.parent
PRED = REPO / "dataset_lesion_merged_pred"
SPLIT = "test"
SEP = 4  # side_by_side 구분선 폭

CANDIDATES = [
    ("effb0_128", "runs/effb0_unetpp_deeplab_128/checkpoint_best.pth"),
    ("effb0_224", "runs/effb0_unetpp_deeplab_224/checkpoint_best.pth"),
    ("effb0_256", "runs/effb0_unetpp_deeplab_256/checkpoint_best.pth"),
    ("effb0_384", "runs/effb0_unetpp_deeplab_384/checkpoint_best.pth"),
    ("effb0_512", "runs/effb0_unetpp_deeplab_512/checkpoint_best.pth"),
    ("hrnet18_deeplab", "runs/hrnet18_deeplab/checkpoint_best.pth"),
    ("hrnet18_new", "runs/hrnet18_new/checkpoint_best.pth"),
    ("hrnet18_new_0.3", "runs/hrnet18_new_0.3/checkpoint_best.pth"),
    ("deeplabv3plus_r50", "runs/deeplab/DeepLabV3Plus_resnet50_best.pth"),
]


def extract_pred_masks():
    """저장된 side-by-side pred 에서 [원본|오버레이] -> 예측 bool 마스크 복원."""
    out = {}
    for pp in sorted((PRED / "pred" / SPLIT).glob("*.png")):
        arr = np.asarray(Image.open(pp).convert("RGB")).astype(np.int16)
        w = arr.shape[1]
        half = (w - SEP) // 2
        left = arr[:, :half]                 # 원본
        right = arr[:, half + SEP:]          # 오버레이
        diff = np.abs(right - left).sum(axis=2)
        out[pp.stem] = diff > 12             # 오버레이가 원본과 다른 곳 = 예측
    return out


def main():
    gt_pred = extract_pred_masks()
    cases = list(gt_pred.keys())
    fg = np.mean([m.mean() for m in gt_pred.values()])
    print(f"[extract] {len(cases)}장 pred 마스크 복원 (평균 전경비 {fg*100:.2f}%)\n")
    print(f"{'candidate':<20}{'imgsz':>6}{'meanIoU':>10}{'meanDice':>10}")
    print("-" * 46)

    results = []
    for name, rel in CANDIDATES:
        ckpt = ROOT / rel
        if not ckpt.exists():
            print(f"{name:<20}{'--':>6}{'(없음)':>10}"); continue
        try:
            seg, imgsz, nclass = load_seg_model(ckpt)
        except Exception as e:
            print(f"{name:<20} 로드 실패: {e}"); continue
        ious, dices = [], []
        for c in cases:
            ip = PRED / "images" / SPLIT / f"{c}.png"
            pil = Image.open(ip).convert("RGB")
            pm = infer_mask(seg, imgsz, nclass, pil, thr=0.5).astype(bool)
            gm = gt_pred[c]
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            s1, s2 = pm.sum(), gm.sum()
            iou = 1.0 if union == 0 else inter / union
            dice = 1.0 if (s1 + s2) == 0 else 2 * inter / (s1 + s2)
            ious.append(iou); dices.append(dice)
        mi, md = float(np.mean(ious)), float(np.mean(dices))
        results.append((name, imgsz, mi, md))
        print(f"{name:<20}{imgsz:>6}{mi:>10.4f}{md:>10.4f}", flush=True)
        del seg; torch.cuda.empty_cache()

    print("-" * 46)
    best = max(results, key=lambda r: r[2])
    print(f"\n[최적 일치] {best[0]}  (imgsz={best[1]})  meanIoU={best[2]:.4f}  meanDice={best[3]:.4f}")
    print("IoU~1.0 이면 그 모델이 pred 를 만든 것으로 확정.")


if __name__ == "__main__":
    main()
