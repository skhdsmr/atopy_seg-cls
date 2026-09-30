"""
MobileNetV4 (timm) 백본 + Neck(공유 임베딩) + Head(태스크별) 멀티태스크 분류.

- 백본은 timm 의 mobilenetv4_* (ImageNet 사전학습). num_classes=0 으로 특징 벡터만 추출.
- MobileNetV4 는 conv_head 때문에 num_features(960)와 실제 출력차원(1280)이 다르므로
  더미 forward 로 특징 차원을 안전하게 잡는다.
- convnext_tiny/model.py 와 동일한 Neck-Head 구조(공유 임베딩 + 태스크별 선형).

의존성: pip install timm  (설치됨, 1.0.27)
"""

import timm
import torch
import torch.nn as nn


def build_backbone(name, pretrained=True, spatial=False):
    """반환: (backbone, feature_dim). backbone(x) -> (B, feature_dim).
    pretrained=False: ImageNet 가중치 다운로드 생략(추론 시 체크포인트가 어차피 덮어씀).
    spatial=True: 풀링 전 특징맵 (B, C, H, W) 를 그대로 반환(CANet 의 CBAM 이 공간축을
    요구한다). 반환 feature_dim 은 두 경우 모두 채널수 C."""
    m = timm.create_model(name, pretrained=pretrained, num_classes=0,
                          global_pool="" if spatial else "avg")
    m.eval()                                   # 헤드 BatchNorm 이 batch=1 프로브에서 터지지 않게
    with torch.no_grad():
        feat = m(torch.zeros(2, 3, 224, 224)).shape[1]
    return m, feat


def make_head(din, dout, hidden=0, dropout=0.0):
    """태스크별 head. hidden=0 이면 기존과 동일한 Linear 1층, >0 이면 MLP(FCNN)."""
    if not hidden:
        return nn.Linear(din, dout)
    return nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                         nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                         nn.Linear(hidden, dout))


def build_multiscale_backbone(name, pretrained=True, out_indices=(0, 1, 2, 3)):
    """반환: (backbone, [stage 별 채널]). backbone(x) -> [stage1..stage4] 특징맵 리스트.

    timm features_only 경로. pvt_v2_b0 는 [32, 64, 160, 256] 채널에 스트라이드 [4, 8, 16, 32]
    라, 512 입력이면 [128, 64, 32, 16] 해상도가 나온다(실측). 저층 stage 를 살려 쓰는
    model_local.LocalAwareMultiTask 전용이고, 기존 build_backbone 은 건드리지 않는다."""
    m = timm.create_model(name, pretrained=pretrained, features_only=True,
                          out_indices=out_indices)
    return m, list(m.feature_info.channels())


class MultiTaskNet(nn.Module):
    def __init__(self, backbone, num_classes, embed_dim=512, dropout=0.3,
                 ordinal=False, pretrained=True, head_hidden=0):
        """backbone: timm 모델 이름(예: mobilenetv4_conv_medium).
        num_classes: {task: n_classes} dict.
        ordinal: True 면 CORN 순서형용으로 head 출력을 (K-1) 개로 만든다
                 (K개 등급을 K-1개의 '임계값 초과' 이진 예측으로; 일반 CE 면 False 로 K개).
        pretrained: 추론(체크포인트 로드)만 할 땐 False 로 두면 백본 다운로드를 생략.
        head_hidden: 0 이면 head 가 Linear 1층(기존), >0 이면 Linear-GELU-Dropout-Linear MLP."""
        super().__init__()
        self.ordinal = ordinal
        self.backbone, feat = build_backbone(backbone, pretrained=pretrained)
        self.neck = nn.Sequential(
            nn.Linear(feat, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {t: make_head(embed_dim, (c - 1) if ordinal else c, head_hidden, dropout)
             for t, c in num_classes.items()})

    def forward(self, x):
        f = self.backbone(x)
        if f.ndim > 2:
            f = f.flatten(1)
        z = self.neck(f)
        return {t: head(z) for t, head in self.heads.items()}
