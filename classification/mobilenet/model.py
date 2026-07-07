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


def build_backbone(name):
    """반환: (backbone, feature_dim). backbone(x) -> (B, feature_dim)."""
    m = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
    m.eval()                                   # 헤드 BatchNorm 이 batch=1 프로브에서 터지지 않게
    with torch.no_grad():
        feat = m(torch.zeros(2, 3, 224, 224)).shape[1]
    return m, feat


class MultiTaskNet(nn.Module):
    def __init__(self, backbone, num_classes, embed_dim=512, dropout=0.3):
        """backbone: timm 모델 이름(예: mobilenetv4_conv_medium).
        num_classes: {task: n_classes} dict."""
        super().__init__()
        self.backbone, feat = build_backbone(backbone)
        self.neck = nn.Sequential(
            nn.Linear(feat, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {t: nn.Linear(embed_dim, c) for t, c in num_classes.items()})

    def forward(self, x):
        f = self.backbone(x)
        if f.ndim > 2:
            f = f.flatten(1)
        z = self.neck(f)
        return {t: head(z) for t, head in self.heads.items()}
