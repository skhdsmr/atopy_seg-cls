"""백본 + Neck + 질환 6-way 헤드.

classification_topk/model.py(TagNet)와의 차이:
    헤드가 하나뿐이다. 질환은 배타적 단일라벨이라 Q차원 sigmoid 도, 순서형 CORN 도
    필요 없다 — K차원 로짓 하나에 softmax/CE 를 걸면 끝이다.
    백본/Neck/사전학습 규약은 TagNet 과 동일하게 유지해 결과를 나란히 비교한다.
"""
import timm
import torch
import torch.nn as nn


def build_backbone(name, pretrained=True):
    """반환: (backbone, feature_dim). classification_topk/model.py 와 동일 규약."""
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    m.eval()                                   # 헤드 BatchNorm 이 batch=1 프로브에서 터지지 않게
    with torch.no_grad():
        feat = m(torch.zeros(2, 3, 224, 224)).shape[1]
    return m, feat


class DiseaseNet(nn.Module):
    def __init__(self, backbone, num_classes, embed_dim=512, dropout=0.5, pretrained=True):
        super().__init__()
        self.backbone, feat = build_backbone(backbone, pretrained=pretrained)
        self.neck = nn.Sequential(
            nn.Linear(feat, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """반환: (B,K) 로짓. softmax 는 loss/추론 쪽에서 건다."""
        return self.head(self.embed(x))

    def embed(self, x):
        """분석/검색용 특징 벡터."""
        f = self.backbone(x)
        if f.ndim > 2:
            f = f.flatten(1)
        return self.neck(f)
