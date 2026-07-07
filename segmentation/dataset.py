"""
UNeXt 용 데이터셋 (아토피 병변, semantic seg).

- 기존 dataset_* (YOLO instance-seg 폴리곤 구조)를 그대로 재사용한다.
  images/{split}/*.png, labels/{split}/*.txt
- 라벨 폴리곤(여러 개)을 하나의 0/1 마스크로 래스터화 -> semantic 마스크로 사용
  (nnU-Net 변환과 동일한 규칙: 모든 폴리곤 fill=1, 단일 클래스).
- resize/정규화/증강은 이 파일이 하지 않고 '주입받은 transform'(augment.py)이 담당한다.
  => 증강 로직이 한 곳(augment.py)에만 있고, 이 Dataset 은 모델/실험과 무관하게 재사용됨.
- val/test split 은 이미 split_all.py 에서 atopy VL/VS 이름으로 고정됨.

반환:
  image: FloatTensor (3, H, W)  (transform 의 Normalize+ToTensorV2 결과)
  mask : FloatTensor (1, H, W), 0/1
  case : 파일 stem (평가/저장용)
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def polygons_to_mask(label_path, w, h):
    """YOLO 폴리곤(정규화 좌표) 여러 개를 하나의 0/1 마스크로 래스터화."""
    mask = Image.new("L", (w, h), 0)
    drw = ImageDraw.Draw(mask)
    p = Path(label_path)
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:          # class + 최소 3점(=6좌표)
                continue
            c = list(map(float, parts[1:]))
            pts = [(c[i] * w, c[i + 1] * h) for i in range(0, len(c) - 1, 2)]
            drw.polygon(pts, fill=1)
    return np.array(mask, dtype=np.uint8)


class AtopySegDataset(Dataset):
    def __init__(self, root, split, transform, img_size=512):
        """transform: albumentations Compose (augment.py 의 build_train_tf/build_val_tf).
        img_size 는 transform=None 인 fallback 경로에서만 쓰인다."""
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        img_dir = self.root / "images" / split
        self.img_paths = sorted(img_dir.glob("*.png"))
        if not self.img_paths:
            raise FileNotFoundError(f"이미지 없음: {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        case = img_path.stem
        lbl_path = self.root / "labels" / self.split / f"{case}.txt"

        # 원본 크기로 로드 (resize 는 transform 이 담당)
        im = np.asarray(Image.open(img_path).convert("RGB"))     # (H, W, 3) uint8
        h, w = im.shape[:2]
        mask = polygons_to_mask(lbl_path, w, h)                  # (H, W) 0/1 uint8

        if self.transform is not None:
            out = self.transform(image=im, mask=mask)            # image+mask 동시 변형
            img_t = out["image"]                                 # (3, H, W) float
            msk_t = out["mask"]                                  # (H, W) tensor
            msk_t = (msk_t > 0.5).float().unsqueeze(0)           # (1, H, W) 0/1
        else:
            # fallback: transform 미지정 시 최소 전처리(resize + /255)
            s = self.img_size
            im_r = np.asarray(Image.fromarray(im).resize((s, s), Image.BILINEAR))
            mk_r = np.asarray(Image.fromarray(mask).resize((s, s), Image.NEAREST))
            img_t = torch.from_numpy(im_r.transpose(2, 0, 1).astype(np.float32) / 255.0)
            msk_t = torch.from_numpy((mk_r > 0.5)[None].astype(np.float32))

        return img_t, msk_t, case
