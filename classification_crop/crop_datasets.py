"""세 crop 실험용 데이터셋 — 공통: 원본 이미지(atopy_face) + 사전계산 마스크(precompute_masks.py).

셋 다 '마스킹(배경 0)'을 하지 않는다. 이미 마스킹 전체이미지(atopy_seg)가 성능이 나빴기
때문 — 배경을 지우면 병변-정상피부 대비와 텍스처가 사라져 중증도 판단에 불리하다.
대신 마스크는 '어디를 볼지(bbox/연결요소)'에만 쓰고, 크롭은 원본 픽셀 그대로 가져온다.

  ① BBoxCropDataset  : 마스크 union bbox + margin 크롭 1장           -> (C,H,W)
  ② MILBagDataset    : 마스크 연결요소별 crop 여러 장(가변) = bag     -> (N,C,H,W)
  ③ TwoStreamDataset : 전체이미지(global) + union bbox 크롭(local)    -> ((C,H,W),(C,H,W))

라벨은 classification_crop/dataset.py 의 CSV 인덱스(이미지 단위 등급)를 재사용.
마스크가 비면(전경 0px) → 전체 이미지로 폴백(정보 손실 방지).
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from dataset import TASK_NAMES


# ----------------------------------------------------------------------------- 공통 유틸
def _bbox_from_mask(mask, margin):
    """mask (H,W) bool -> (x0,y0,x1,y1) union bbox, 각 변을 margin(비율)만큼 확장.
    빈 마스크면 None."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))


def _crop(im, box):
    """im PIL.Image, box (x0,y0,x1,y1) 또는 None(전체)."""
    return im if box is None else im.crop(box)


def _load_mask(mask_path, size):
    """마스크 PNG -> bool ndarray (H,W). 없으면 전부 False(전체 폴백)."""
    if mask_path is None or not Path(mask_path).exists():
        return np.zeros((size[1], size[0]), dtype=bool)
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    return m


def _with_src(labels, batch):
    """다중출처 theta(논문 Eq.12)용 출처 인덱스. label_index 에 '_src' 가 주입된 경우에만
    실린다 — 없으면 아무 일도 하지 않으므로 기존 런은 그대로다."""
    if batch and "_src" in batch[0][1]:
        labels["_src"] = torch.tensor([b[1]["_src"] for b in batch])
    return labels


class _CropBase(Dataset):
    """(image_path, mask_path, labels) 목록 구성. image_dir/mask_dir 는 같은 stem 매칭."""
    def __init__(self, image_dir, mask_dir, label_index, transform=None):
        self.transform = transform
        self.samples = []
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir) if mask_dir else None
        skipped = 0
        for img_path in sorted(image_dir.glob("*.png")):
            lb = label_index.get(img_path.stem)
            if lb is None:
                skipped += 1
                continue
            mp = (mask_dir / img_path.name) if mask_dir else None
            self.samples.append((img_path, mp, lb))
        if skipped:
            print(f"[dataset] 건너뜀(라벨없음): {skipped}")
        print(f"[dataset] 샘플 {len(self.samples)}개  ({type(self).__name__})")

    def __len__(self):
        return len(self.samples)


# ----------------------------------------------------------------------------- ① bbox+margin
class BBoxCropDataset(_CropBase):
    """union bbox + margin 크롭 1장(마스킹 X). 기존 MultiTaskNet 그대로 사용."""
    def __init__(self, *a, margin=0.15, **kw):
        super().__init__(*a, **kw)
        self.margin = margin

    def __getitem__(self, i):
        img_path, mask_path, lb = self.samples[i]
        im = Image.open(img_path).convert("RGB")
        mask = _load_mask(mask_path, im.size)
        crop = _crop(im, _bbox_from_mask(mask, self.margin))
        if self.transform:
            crop = self.transform(crop)
        return crop, lb


# ----------------------------------------------------------------------------- ② MIL bag
class MILBagDataset(_CropBase):
    """마스크 연결요소별 crop = 가변 길이 bag. bag 단위 라벨(이미지 등급).

    - 각 연결요소 bbox+margin 크롭(마스킹 X)을 하나의 instance 로.
    - min_area_frac: 이미지 면적 대비 이보다 작은 요소는 노이즈로 제외.
    - max_instances: 면적 큰 순 상위 N개만(과다 요소 방지). 초과분 로그 없이 절단하지 않도록
      실제 개수도 함께 반환하려면 __getitem__ 변경; 여기선 상한만 둔다.
    - 요소가 하나도 없으면 전체 이미지 1장을 bag 으로 폴백.
    """
    def __init__(self, *a, margin=0.15, min_area_frac=0.003, max_instances=8, **kw):
        super().__init__(*a, **kw)
        self.margin = margin
        self.min_area_frac = min_area_frac
        self.max_instances = max_instances

    def _boxes(self, mask, area):
        lab, n = ndimage.label(mask)
        if n == 0:
            return []
        comps = []
        min_area = self.min_area_frac * area
        for k in range(1, n + 1):
            comp = lab == k
            if comp.sum() < min_area:
                continue
            comps.append((int(comp.sum()), _bbox_from_mask(comp, self.margin)))
        comps.sort(key=lambda c: -c[0])                 # 면적 큰 순
        return [b for _, b in comps[:self.max_instances]]

    def __getitem__(self, i):
        img_path, mask_path, lb = self.samples[i]
        im = Image.open(img_path).convert("RGB")
        mask = _load_mask(mask_path, im.size)
        boxes = self._boxes(mask, im.size[0] * im.size[1])
        if not boxes:                                   # 폴백: 전체 이미지 1장
            boxes = [None]
        crops = [self.transform(_crop(im, b)) if self.transform else _crop(im, b)
                 for b in boxes]
        bag = torch.stack(crops)                        # (N, C, H, W)
        return bag, lb


def collate_mil(batch):
    """가변 길이 bag 을 flat 하게 이어 붙임(패딩 X) + 길이. MILNet 이 split 으로 복원."""
    lengths = torch.tensor([b[0].shape[0] for b in batch])
    flat = torch.cat([b[0] for b in batch], dim=0)      # (sumN, C, H, W)
    labels = {t: torch.tensor([b[1][t] for b in batch]) for t in TASK_NAMES}
    return (flat, lengths), _with_src(labels, batch)


# ----------------------------------------------------------------------------- ③ two-stream
class TwoStreamDataset(_CropBase):
    """전체 이미지(global: 분포/범위) + union bbox 크롭(local: 텍스처). 둘 다 마스킹 X."""
    def __init__(self, *a, margin=0.15, **kw):
        super().__init__(*a, **kw)
        self.margin = margin

    def __getitem__(self, i):
        img_path, mask_path, lb = self.samples[i]
        im = Image.open(img_path).convert("RGB")
        mask = _load_mask(mask_path, im.size)
        local = _crop(im, _bbox_from_mask(mask, self.margin))
        if self.transform:
            g = self.transform(im)
            l = self.transform(local)
        else:
            g, l = im, local
        return (g, l), lb


def collate_twostream(batch):
    g = torch.stack([b[0][0] for b in batch])
    l = torch.stack([b[0][1] for b in batch])
    labels = {t: torch.tensor([b[1][t] for b in batch]) for t in TASK_NAMES}
    return (g, l), _with_src(labels, batch)


def collate_single(batch):
    """bbox 실험: 단일 이미지 텐서."""
    imgs = torch.stack([b[0] for b in batch])
    labels = {t: torch.tensor([b[1][t] for b in batch]) for t in TASK_NAMES}
    return imgs, _with_src(labels, batch)
