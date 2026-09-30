"""seg 체크포인트로 atopy_face 900장을 추론해, 분류 입력용 'atopy_seg' 데이터셋 생성.

원본 이미지에 세그 예측 마스크(병변 전경)를 적용해서, 분류 모델이 병변 영역에만
집중하도록 만든 입력을 만든다. 출력 구조는 atopy_face 와 동일(자립형: labels.csv 재사용)
이므로 classification/mobile/train.py 의 --data 만 바꿔 그대로 학습 가능.

    <OUT>/labels.csv            (SRC/labels.csv 를 그대로 복사; stem 매칭)
    <OUT>/{train,val,test}/<stem>.png

전처리/추론은 infer_all_splits.py 와 동일:
    Resize(imgsz) + ImageNet 정규화 -> logits -> 원본 해상도로 bilinear 복원 -> sigmoid>0.5.

마스킹 방식(--mode):
    mask : 배경을 검정(0)으로 (기본, 배경 완전 제거)
    soft : 배경을 --soft-bg 비율로 어둡게 (문맥 일부 보존; 예 0.3)
    bbox : 마스크 bounding box 로 크롭(+--pad 여유). 크기는 가변(로더가 Resize).
마스크가 비면(전경 0px) 원본을 그대로 저장(정보 손실 방지).

사용 예 (run.sh 와 동일 env 로 실행: make_atopy_seg.sh 래퍼 권장):
    python3 make_atopy_seg.py \
        --ckpt runs/effb0_unetpp_deeplab_512/checkpoint_best.pth \
        --src ../../atopy_face --out ../../atopy_seg --mode mask --dilate 4
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent
sys.path.insert(0, str(SHARED))

from augment import _MEAN, _STD          # noqa: E402
from model import build_model            # noqa: E402

SPLITS = ["train", "val", "test"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MEAN_T = torch.tensor(_MEAN).view(3, 1, 1)
_STD_T = torch.tensor(_STD).view(3, 1, 1)


def preprocess(im_rgb, imgsz):
    """원본 RGB uint8 (H,W,3) -> (1,3,imgsz,imgsz) 정규화 텐서. infer_all_splits 와 동일."""
    t = torch.from_numpy(im_rgb.transpose(2, 0, 1)).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(imgsz, imgsz),
                      mode="bilinear", align_corners=False)[0]
    t = (t - _MEAN_T) / _STD_T
    return t.unsqueeze(0)


def dilate(mask_bool, px):
    """이진 마스크를 px 만큼 팽창(병변 경계 문맥 보존). torch maxpool 사용(cv2 불필요)."""
    if px <= 0:
        return mask_bool
    k = 2 * px + 1
    m = torch.from_numpy(mask_bool.astype(np.float32))[None, None]
    m = F.max_pool2d(m, kernel_size=k, stride=1, padding=px)
    return m[0, 0].numpy() > 0.5


def apply_mask(im, mask, mode, soft_bg, pad):
    """im (H,W,3) uint8, mask (H,W) bool -> 마스킹된 이미지(uint8)."""
    if not mask.any():                                 # 전경 없음 -> 원본 유지
        return im
    if mode == "bbox":
        ys, xs = np.where(mask)
        h, w = mask.shape
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
        return im[y0:y1, x0:x1]
    m = mask[..., None].astype(np.float32)
    if mode == "soft":
        out = im.astype(np.float32) * (soft_bg + (1.0 - soft_bg) * m)
    else:                                              # mode == "mask": 배경 = 0
        out = im.astype(np.float32) * m
    return out.clip(0, 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "runs/effb0_unetpp_deeplab_512/checkpoint_best.pth"))
    p.add_argument("--src", default=str(ROOT.parent.parent / "atopy_face"),
                   help="분류용 소스(라벨매칭된 원본). {train,val,test}/*.png + labels.csv")
    p.add_argument("--out", default=str(ROOT.parent.parent / "atopy_seg"))
    p.add_argument("--imgsz", type=int, default=None, help="추론 크기. 미지정 시 ckpt 값")
    p.add_argument("--mode", choices=["mask", "soft", "bbox"], default="mask")
    p.add_argument("--soft-bg", type=float, default=0.3, help="soft 모드에서 배경 유지 비율")
    p.add_argument("--dilate", type=int, default=4, help="마스크 팽창 px(경계 문맥 보존). 0=끔")
    p.add_argument("--pad", type=int, default=16, help="bbox 모드 크롭 여유 px")
    return p.parse_args()


def main():
    args = parse_args()
    CKPT, SRC, OUT = Path(args.ckpt), Path(args.src), Path(args.out)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    IMGSZ = args.imgsz or a["imgsz"]
    print(f"[ckpt] {CKPT}\n[ckpt] epoch={ck['epoch']} val={ck['val']}")
    print(f"[ckpt] encoder={a['encoder']} decoder={a['decoder']} imgsz={a['imgsz']}")
    print(f"[src ] {SRC}\n[out ] {OUT}")
    print(f"[cfg ] mode={args.mode} soft_bg={args.soft_bg} dilate={args.dilate} "
          f"pad={args.pad}  infer_size={IMGSZ}  device={DEVICE}")

    model = build_model(
        encoder_name=a["encoder"], encoder_weights=None, decoder=a["decoder"],
        decoder_attention=None if a.get("decoder_attention") in ("none", None) else a["decoder_attention"],
        num_classes=1, in_channels=3)
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()

    OUT.mkdir(parents=True, exist_ok=True)
    src_csv = SRC / "labels.csv"
    if src_csv.exists():
        shutil.copy2(src_csv, OUT / "labels.csv")
        print(f"[label] labels.csv 복사 완료")
    else:
        print(f"[warn] {src_csv} 없음 -> labels.csv 미복사(수동 준비 필요)")

    grand, empty = 0, 0
    for split in SPLITS:
        src_dir = SRC / split
        img_paths = sorted(src_dir.glob("*.png"))
        if not img_paths:
            print(f"[warn] 이미지 없음, 건너뜀: {src_dir}")
            continue
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ratios = []
        for i, ip in enumerate(img_paths):
            im = np.asarray(Image.open(ip).convert("RGB"))    # (H,W,3)
            h, w = im.shape[:2]
            with torch.no_grad():
                x = preprocess(im, IMGSZ).to(DEVICE)
                logits = model(x)                             # (1,1,imgsz,imgsz)
                logits = F.interpolate(logits.float(), size=(h, w),
                                       mode="bilinear", align_corners=False)
                mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
            mask = dilate(mask, args.dilate)
            fg = float(mask.mean())
            ratios.append(fg)
            if fg == 0.0:
                empty += 1
            out = apply_mask(im, mask, args.mode, args.soft_bg, args.pad)
            Image.fromarray(out).save(out_dir / ip.name)
            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i+1}/{len(img_paths)} 완료")
        grand += len(img_paths)
        print(f"=== {split}: n={len(img_paths)}  "
              f"평균 전경비율={np.mean(ratios):.3f}  전경0장={sum(r == 0 for r in ratios)} ===")

    print(f"\n[done] 총 {grand}장 -> {OUT}  (전경0장 총 {empty}: 원본 그대로 저장됨)")
    print(f"학습:  cd ../../classification/mobile && "
          f"DATA={OUT} bash run_sweep.sh   (또는 train.py --data {OUT})")


if __name__ == "__main__":
    main()
