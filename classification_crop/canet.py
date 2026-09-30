"""CANet(Li et al., TMI 2020) 의 cross-disease attention 을 5-태스크로 이식.

원 저장소: https://github.com/xmengli/CANet  (models/cbam.py, models/resnet50.py 의
`crossCBAM` 분기). 원본은 DR(5등급)/DME(3등급) 2태스크 하드코딩 + CE softmax 헤드다.

## 가져온 것 (원본 forward 와 1:1)
    x1 = avgpool(branch_bam1(x)); x2 = avgpool(branch_bam2(x))   # 태스크별 CBAM(채널+공간)
    x1 = classifier_dep1(x1);     x2 = classifier_dep2(x2)       # 태스크별 임베딩(=neck)
    out1 = classifier_specific_1(x1)                             # specific 헤드(보조, λ=0.25)
    x1_att = branch_bam3(x1[:,:,None,None])                      # 채널전용 CBAM = 게이팅 MLP
    x1 = x1 + x2_att ; x2 = x2 + x1_att                          # choice="both" 교차 주입
    x1 = classifier1(x1)                                         # joint 헤드(주, weight 1)
    return x1, x2, out1, out2
저자 손실: `loss1 + loss2 + λ(loss3 + loss4)`, λ=0.25(baseline.py:382,98).
정확도는 output[0]/output[1] = joint 헤드로 잰다(baseline.py:426).

## 이 저장소에 맞춰 바꾼 것
1. **백본**: ResNet-50 -> timm(effb0/pvtv2b0 등). CBAM 채널수를 백본 출력에서 받는다.
2. **헤드**: CE softmax -> 기존 CORN 순서형(선택지표가 QWK 라 순서형이어야 한다).
   CANet 의 기여는 헤드가 아니라 특징 융합이므로 교체해도 잃는 게 없다.
3. **neck**: 저자 `classifier_dep` 은 활성 없는 Linear 1개. 여기선 대조군 MultiTaskNet 과
   같은 Linear+BN+GELU+Dropout 을 쓴다 — 대조군과의 차이를 어텐션으로만 국한시키기 위함.
4. **2 -> 5 태스크**: 교차 엣지가 O(T^2)(방향 포함 20개)로 늘어 1400장에 과하다.
   `sampler.select_edges` 의 Cramér's V 컷(기본 0.25)으로 고른 엣지만 연결한다
   (seve-eryt .47 / seve-papu .33 / eryt-papu .32 / seve-exco .26).
   저자와 동일하게 **교차 CBAM 은 source 태스크당 1개**(branch_bam3 이 x1 을 받아
   x2 로 보내는 구조 그대로) — 엣지쌍당이 아니다.
5. **다중 이웃 합성**: 저자는 이웃이 항상 1개라 단순 덧셈. 여기선 severity 가 이웃 3개라
   합이면 스케일이 커진다 -> 기본 `mean`(이웃 1개면 저자와 완전히 동일). `sum` 도 선택 가능.
6. 이웃이 없는 태스크(lichenification: 모든 쌍 V<=.18)는 z_fused = z 로 통과한다.
   두 헤드가 같은 특징을 보게 되지만 λ 가중 보조헤드로서 정규화 역할은 남는다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_base import build_backbone


# ============================================================================= CBAM
# xmengli/CANet models/cbam.py 이식. 미사용 pool_type(lp/lse)과 시각화용 scale 반환만 제거.
class _ChannelGate(nn.Module):
    """avg/max 풀링 각각을 공유 MLP 에 통과시켜 합 -> sigmoid 채널 스케일."""

    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        hidden = max(1, gate_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, gate_channels),
        )

    def forward(self, x):
        # (B,C,1,1) 벡터 입력에서는 avg==max 라 logit 이 2배가 된다(저자 동작 그대로).
        # 초기 MLP 출력이 0 근처라 sigmoid=0.5 에서 시작하므로 포화 문제는 없다.
        avg = F.avg_pool2d(x, (x.size(2), x.size(3)))
        mx = F.max_pool2d(x, (x.size(2), x.size(3)))
        att = self.mlp(avg) + self.mlp(mx)
        return x * torch.sigmoid(att).unsqueeze(2).unsqueeze(3).expand_as(x)


class _SpatialGate(nn.Module):
    """채널축 [max, mean] -> 7x7 conv -> sigmoid 공간 스케일."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)

    def forward(self, x):
        c = torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.bn(self.conv(c)))


class CBAM(nn.Module):
    """no_spatial=True 면 채널 어텐션만(저자 branch_bam3/4 = 교차 게이팅)."""

    def __init__(self, gate_channels, reduction_ratio=16, no_spatial=False):
        super().__init__()
        self.channel = _ChannelGate(gate_channels, reduction_ratio)
        self.spatial = None if no_spatial else _SpatialGate()

    def forward(self, x):
        x = self.channel(x)
        return x if self.spatial is None else self.spatial(x)


# ============================================================================= CANet
AUX_SUFFIX = "__spec"          # forward 출력에서 specific(보조) 헤드를 구분하는 키 접미사


class CANetMultiTask(nn.Module):
    """CANet crossCBAM 을 5개 순서형 태스크로 확장.

    forward(x) -> dict:
        out[t]              joint 헤드 로짓  (주 예측 / 손실 weight 1)
        out[t + "__spec"]   specific 헤드 로짓 (보조 / 손실 weight λ)
    train.py 의 head_losses/head_predict 가 TASK_NAMES 만 순회하므로 joint 가 자동으로
    주 경로가 되고, 보조 손실은 head_losses 의 aux_lambda 에서 더해진다.

    edges: sampler.select_edges 의 [(a, b, v), ...] (무방향). 양방향으로 전개한다.
    """

    def __init__(self, backbone, num_classes, edges, *, embed_dim=512, dropout=0.3,
                 ordinal=False, pretrained=True, reduction=16, fuse="mean"):
        super().__init__()
        self.ordinal = ordinal
        self.fuse = fuse
        self.tasks = list(num_classes.keys())
        # 무방향 엣지 -> 태스크별 이웃(source) 목록. choice="both" 에 해당.
        self.nbrs = {t: [] for t in self.tasks}
        for a, b, *_ in edges:
            self.nbrs[a].append(b)
            self.nbrs[b].append(a)

        self.backbone, feat = build_backbone(backbone, pretrained=pretrained, spatial=True)
        # ① disease-specific attention: 공유 특징맵에서 태스크별 영역/채널을 고른다.
        self.spec_att = nn.ModuleDict({t: CBAM(feat, reduction) for t in self.tasks})
        # ② 태스크별 임베딩(저자 classifier_dep). 대조군과 동일한 neck 구성.
        self.neck = nn.ModuleDict({t: nn.Sequential(
            nn.Linear(feat, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ) for t in self.tasks})
        # ③ disease-dependent attention: 채널전용 CBAM, source 태스크당 1개(저자와 동일).
        #    이웃이 아무에게도 안 붙는 태스크는 모듈을 만들지 않는다.
        used = {r for q in self.tasks for r in self.nbrs[q]}
        self.cross_att = nn.ModuleDict({t: CBAM(embed_dim, reduction, no_spatial=True)
                                        for t in sorted(used)})

        def _head(c):
            return nn.Linear(embed_dim, (c - 1) if ordinal else c)
        self.head_spec = nn.ModuleDict({t: _head(c) for t, c in num_classes.items()})
        self.head_joint = nn.ModuleDict({t: _head(c) for t, c in num_classes.items()})

    def edge_desc(self):
        return ", ".join(f"{q[:4]}<-{'+'.join(r[:4] for r in n)}"
                         for q, n in self.nbrs.items() if n) or "(없음)"

    def forward(self, x):
        f = self.backbone(x)                                   # (B, C, H, W)
        z, out = {}, {}
        for t in self.tasks:
            a = self.spec_att[t](f)                            # ① 태스크별 CBAM
            v = F.adaptive_avg_pool2d(a, 1).flatten(1)
            z[t] = self.neck[t](v)                             # ② (B, embed_dim)
            out[t + AUX_SUFFIX] = self.head_spec[t](z[t])      # specific 헤드
        att = {t: self.cross_att[t](z[t].unsqueeze(-1).unsqueeze(-1)).flatten(1)
               for t in self.cross_att}                        # ③ 게이팅된 source 특징
        for t in self.tasks:
            n = self.nbrs[t]
            if n:
                inj = torch.stack([att[r] for r in n], dim=0)
                zt = z[t] + (inj.mean(0) if self.fuse == "mean" else inj.sum(0))
            else:
                zt = z[t]
            out[t] = self.head_joint[t](zt)                    # joint 헤드(주 예측)
        return out
