"""분할 ckpt로 atopy_face 전 이미지의 병변 마스크를 1회 추론해 PNG로 저장.

세 crop 실험(bbox / mil / twostream)이 공유하는 전처리. 원본 이미지는 atopy_face 를
그대로 쓰고, 여기서 만든 '이진 마스크'만 별도로 저장한다 → 학습 때 seg 추론 반복 없이
빠르게 crop/bbox/연결요소 계산. (make_atopy_seg.py 는 마스킹된 '이미지'를 저장하지만,
crop 실험은 마스킹하지 않은 원본 + 마스크 좌표가 필요하므로 마스크만 따로 뽑는다.)

    <OUT>/{train,val,test}/<stem>.png   # 0/255 grayscale, 원본 해상도
    <OUT>/labels.csv                    # atopy_face/labels.csv 복사(편의)

전처리/추론은 make_atopy_seg.py 와 동일:
    Resize(imgsz) + ImageNet 정규화 -> logits -> 원본 해상도 bilinear 복원 -> sigmoid>0.5.

사용 예 (run.sh 가 감싸서 호출; 직접 실행 시 .venv-train 사용):
    python3 precompute_masks.py \
        --ckpt ../segmentation/encoder_decoder/runs/effb0_unetpp_deeplab_512/checkpoint_best.pth \
        --src ../atopy_face --out ../atopy_crop_masks --dilate 4
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
OGW = ROOT.parent
SEG = OGW / "segmentation"
SEG_ED = SEG / "encoder_decoder"
sys.path.insert(0, str(SEG_ED))            # model.build_model
sys.path.insert(0, str(SEG))               # augment._MEAN/_STD

from augment import _MEAN, _STD            # noqa: E402  (segmentation/augment.py)
from model import build_model              # noqa: E402  (segmentation/encoder_decoder/model.py)

SPLITS = ["train", "val", "test"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MEAN_T = torch.tensor(_MEAN).view(3, 1, 1)
_STD_T = torch.tensor(_STD).view(3, 1, 1)


def preprocess(im_rgb, imgsz):
    """원본 RGB uint8 (H,W,3) -> (1,3,imgsz,imgsz) 정규화 텐서 (make_atopy_seg 와 동일)."""
    t = torch.from_numpy(im_rgb.transpose(2, 0, 1)).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(imgsz, imgsz),
                      mode="bilinear", align_corners=False)[0]
    t = (t - _MEAN_T) / _STD_T
    return t.unsqueeze(0)


def dilate(mask_bool, px):
    """이진 마스크 px 팽창(경계 문맥 보존). torch maxpool(cv2 불필요)."""
    if px <= 0:
        return mask_bool
    k = 2 * px + 1
    m = torch.from_numpy(mask_bool.astype(np.float32))[None, None]
    m = F.max_pool2d(m, kernel_size=k, stride=1, padding=px)
    return m[0, 0].numpy() > 0.5


def parse_args():
    p = argparse.ArgumentParser(description="atopy_face 병변 마스크 사전계산")
    p.add_argument("--ckpt", default=str(
        SEG_ED / "runs/effb0_unetpp_deeplab_512/checkpoint_best.pth"))
    p.add_argument("--src", default=str(OGW / "atopy_face"),
                   help="원본 소스(라벨매칭). {train,val,test}/*.png + labels.csv")
    p.add_argument("--out", default=str(OGW / "atopy_crop_masks"))
    p.add_argument("--imgsz", type=int, default=None, help="추론 크기(미지정 시 ckpt 값)")
    p.add_argument("--dilate", type=int, default=4, help="마스크 팽창 px(경계 문맥). 0=끔")
    return p.parse_args()


def main():
    args = parse_args()
    CKPT, SRC, OUT = Path(args.ckpt), Path(args.src), Path(args.out)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck["args"]
    IMGSZ = args.imgsz or a["imgsz"]
    print(f"[ckpt] {CKPT}\n[ckpt] epoch={ck['epoch']} val={ck.get('val')}")
    print(f"[ckpt] encoder={a['encoder']} decoder={a['decoder']} imgsz={a['imgsz']}")
    print(f"[cfg ] infer_size={IMGSZ} dilate={args.dilate} device={DEVICE}")

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
        print("[label] labels.csv 복사 완료")
    else:
        print(f"[warn] {src_csv} 없음 -> labels.csv 미복사")

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
            im = np.asarray(Image.open(ip).convert("RGB"))
            h, w = im.shape[:2]
            with torch.no_grad():
                x = preprocess(im, IMGSZ).to(DEVICE)
                logits = model(x)
                logits = F.interpolate(logits.float(), size=(h, w),
                                       mode="bilinear", align_corners=False)
                mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
            mask = dilate(mask, args.dilate)
            fg = float(mask.mean())
            ratios.append(fg)
            if fg == 0.0:
                empty += 1
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_dir / ip.name)
            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i+1}/{len(img_paths)} 완료")
        grand += len(img_paths)
        print(f"=== {split}: n={len(img_paths)}  평균 전경비율={np.mean(ratios):.3f}  "
              f"전경0장={sum(r == 0 for r in ratios)} ===")

    print(f"\n[done] 총 {grand}장 마스크 -> {OUT}  (전경0장 {empty}: 학습 시 전체이미지로 폴백)")


if __name__ == "__main__":
    main()
