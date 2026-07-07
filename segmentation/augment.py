"""
증강 파이프라인 정의 (온라인 증강, albumentations 2.x).

- 이 파일이 증강의 '단일 소스'다. 학습 스크립트는 여기서 함수만 import 해서 쓴다.
- 아토피(얼굴 피부 병변) segmentation 원칙:
    * 촬영 변동(각도/거리/조명/폰카메라/압축)은 흉내 내되,
    * 진단 신호(홍반=붉은기, 질감)는 보존 -> 색조(hue)·채도는 '약하게'.
- image + mask 동시 변형. 기하 변형은 mask_interpolation=0(nearest)·border fill=0 이
  기본이라 마스크 0/1 이 자동 보존된다(albumentations 기본값).
- val/test 는 증강 없이 Resize+Normalize 만 (평가 부풀림 방지).

강도/확률은 여기 기본값을 바꾸거나, build_train_tf(...) 인자로 조절한다.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet 표준 정규화 (from-scratch 학습이라 값 자체보다 train/val 일관성이 중요)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_train_tf(img_size=512, p_geom=0.5, p_color=0.5, p_quality=0.3, seed=None):
    """train 전용 온라인 증강 파이프라인.

    한 장이 통과할 때 평균 2~4개 변형이 겹치도록 확률을 배분(참고 원칙).
    p_geom/p_color/p_quality 로 그룹별 강도를 한 번에 조절할 수 있다.
    seed: 지정 시 '재현 가능한 랜덤'(매 호출 다르되, 같은 seed로 재실행 시 순서 동일).
          albumentations 2.x 는 자체 RNG를 쓰므로 전역 np/random 시드로는 재현 안 됨.
    """
    return A.Compose([
        A.Resize(img_size, img_size),                        # 고정 크기 (마스크 nearest 자동)

        # --- 1순위: 기하 변형 (이미지+마스크 동일 적용) ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),                               # 피부는 방향성 없음(참고 원칙 반영)
        A.Affine(scale=(0.85, 1.15),                         # 카메라 거리 차
                 translate_percent=(-0.05, 0.05),            # 위치 변동
                 rotate=(-15, 15),                           # 촬영 각도
                 border_mode=0, fill=0, fill_mask=0,         # 경계는 배경(0)으로
                 p=p_geom),
        A.ElasticTransform(alpha=1.0, sigma=50, p=0.2),      # 피부 늘어남/접힘 (U-Net 전통)

        # --- 색/밝기: 조명 편차는 넉넉히, 색조는 보수적으로 (홍반 보존) ---
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=p_color),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10,       # 화이트밸런스 편차 근사(약)
                   b_shift_limit=10, p=0.2),
        A.HueSaturationValue(hue_shift_limit=5,              # 색조는 약하게 (붉은기 파괴 방지)
                             sat_shift_limit=15,
                             val_shift_limit=10, p=0.3),

        # --- 촬영 품질 저하 모사 (앱 업로드 환경) ---
        A.ImageCompression(quality_range=(60, 100), p=p_quality),  # JPEG 압축
        A.GaussNoise(std_range=(0.02, 0.10), p=0.2),         # 센서 노이즈(약)
        A.GaussianBlur(blur_limit=0, sigma_limit=(0.2, 1.0), p=0.1),

        A.Normalize(mean=_MEAN, std=_STD),                   # 정규화는 항상 끝쪽
        ToTensorV2(),
    ], seed=seed)


def build_train_crop_tf(crop_size=512, pos_ratio=0.7,
                        p_color=0.5, p_quality=0.3, seed=None):
    """crop 기반 train 증강 (작은 병변 대응).

    - 네이티브 해상도에서 crop_size x crop_size 를 잘라낸다(리사이즈 없음 -> 진짜 픽셀).
      작은 병변이 crop 안에서 차지하는 비율이 커져 전경↑ + 경계 픽셀↑ -> Dice 유리.
    - pos_ratio: 병변 포함 crop 비율. 나머지는 랜덤 crop(배경 포함, 음성 학습).
        * 병변 crop 만 쓰면 '아무것도 없음'을 못 배워 과탐 -> 배경 crop 을 섞는다.
    - crop 이후는 build_train_tf 와 동일한 색/품질/약한 기하 증강(단, Resize/큰 이동 제외).
    - 추론은 반드시 타일드(전체 이미지)로 맞춰야 평가가 정직하다(tiling.py 참고).
    """
    return A.Compose([
        A.PadIfNeeded(crop_size, crop_size, border_mode=0),   # 혹시 crop 보다 작으면 패딩
        A.OneOf([
            A.CropNonEmptyMaskIfExists(crop_size, crop_size, p=pos_ratio),
            A.RandomCrop(crop_size, crop_size, p=1.0 - pos_ratio),
        ], p=1.0),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15),          # crop 후엔 이동은 빼고 회전/스케일만
                 border_mode=0, fill=0, fill_mask=0, p=0.3),

        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=p_color),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10,
                   b_shift_limit=10, p=0.2),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15,
                             val_shift_limit=10, p=0.3),

        A.ImageCompression(quality_range=(60, 100), p=p_quality),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.2),
        A.GaussianBlur(blur_limit=0, sigma_limit=(0.2, 1.0), p=0.1),

        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ], seed=seed)


class FgConditionalTransform:
    """전경비율(fg)에 따라 crop_tf / full_tf 를 분기하는 래퍼.

    - 희소(fg < fg_thresh) 이미지 -> crop_tf (병변-인지 네이티브 crop, 전경↑)
    - 밀집(fg >= fg_thresh) 이미지 -> full_tf (전체 리사이즈, 맥락 보존)
    - dataset.py 의 transform(image=, mask=) 인터페이스를 그대로 흉내내므로
      Dataset 수정 없이 꽂힌다. mask 는 0/1 numpy(H,W) 가정.
    - fg_thresh 는 '비율'(0~1). 예: 0.05 = 5%.
    """

    def __init__(self, crop_tf, full_tf, fg_thresh):
        self.crop_tf = crop_tf
        self.full_tf = full_tf
        self.fg_thresh = fg_thresh

    def __call__(self, image, mask, **kw):
        fg = float((mask > 0).mean())
        tf = self.crop_tf if fg < self.fg_thresh else self.full_tf
        return tf(image=image, mask=mask, **kw)

    def set_random_seed(self, s):
        for tf in (self.crop_tf, self.full_tf):
            if hasattr(tf, "set_random_seed"):
                tf.set_random_seed(s)


def build_val_tf(img_size=512, seed=None):
    """val/test 전용: 증강 없이 표준화만.

    img_size=None 이면 Resize 를 생략해 '원본 해상도' 그대로 반환한다
    (타일드 추론용 - 전체 이미지를 원해상도로 평가).
    """
    ops = []
    if img_size is not None:
        ops.append(A.Resize(img_size, img_size))
    ops += [A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()]
    return A.Compose(ops, seed=seed)
