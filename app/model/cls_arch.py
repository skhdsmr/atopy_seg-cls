"""아토피 멀티태스크 '분류' 아키텍처 — app 폴더 자립형(변환 전용).

classification/mobilenet/model.py 의 MultiTaskNet 과 동일한 구조를 app 안에 복제한다.
(export 시에는 체크포인트 state_dict 를 덮어쓰므로 backbone 은 pretrained=False 로 만든다.)

구조: timm 백본(특징벡터) -> Neck(Linear+BN+GELU+Dropout, 공유 임베딩) -> Head(태스크별 Linear).
학습 태스크(순서형 등급, 인덱스=등급 오름차순)는 classification/dataset.py 와 동일.
"""
import timm
import torch
import torch.nn as nn

# 순서형 등급 라벨 (classification/dataset.py TASKS 와 동일). 인덱스 = 등급 오름차순.
TASKS = {
    "severity":        ["Clear", "Almost Clear", "Mild", "Moderate", "Severe"],  # IGA
    "erythema":        ["None", "Mild", "Moderate", "Severe"],
    "papulation":      ["None", "Mild", "Moderate", "Severe"],
    "excoriation":     ["None", "Mild", "Moderate", "Severe"],
    "lichenification": ["None", "Mild", "Moderate", "Severe"],
}
TASK_NAMES = list(TASKS.keys())              # export/앱이 쓰는 고정 순서
# 앱 표시용 한글/영문 태스크명
TASK_TITLE = {
    "severity":        ("Severity (IGA)", "중증도(IGA)"),
    "erythema":        ("Erythema", "홍반"),
    "papulation":      ("Papulation", "구진"),
    "excoriation":     ("Excoriation", "찰상"),
    "lichenification": ("Lichenification", "태선화"),
}


def build_backbone(name, pretrained=False):
    """반환: (backbone, feature_dim). backbone(x) -> (B, feature_dim)."""
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    m.eval()
    with torch.no_grad():
        feat = m(torch.zeros(2, 3, 224, 224)).shape[1]
    return m, feat


class MultiTaskNet(nn.Module):
    def __init__(self, backbone, num_classes, embed_dim=512, dropout=0.5, pretrained=False):
        """backbone: timm 모델 이름. num_classes: {task: n_classes} dict."""
        super().__init__()
        self.backbone, feat = build_backbone(backbone, pretrained=pretrained)
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


class ConcatHead(nn.Module):
    """멀티 헤드 출력을 고정 순서로 이어붙여 단일 텐서 [B, sum(sizes)] 로 반환.
    (ONNX/TFLite 는 단일 출력이 앱에서 다루기 쉬움. 앱은 offset/size 로 다시 자른다.)"""

    def __init__(self, net, task_order):
        super().__init__()
        self.net = net
        self.task_order = list(task_order)

    def forward(self, x):
        out = self.net(x)
        return torch.cat([out[t] for t in self.task_order], dim=1)
