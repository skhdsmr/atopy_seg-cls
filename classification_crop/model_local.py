"""찰상/태선화 전용 multi-scale local branch — PVTv2 저층 stage 를 살려 쓰는 구조.

## 왜

현재 MultiTaskNet(model_base.py)은 백본을 `global_pool="avg"` 로 만들어 **마지막 stage 를
GAP 한 벡터 하나**로 5축을 전부 예측한다. PVTv2-B0 를 512 입력으로 돌리면 실제로는 이런
피라미드가 계산되는데:

    stage1  32ch  128x128 (1/4)    선형/점상 병변, 경계, 피부결
    stage2  64ch   64x64  (1/8)    국소 패턴
    stage3 160ch   32x32  (1/16)   병변 단위
    stage4 256ch   16x16  (1/32)   전역 문맥   <- 지금 이것만 쓴다

stage1/2 는 계산해 놓고 버린다. 찰상은 실제 이미지를 보면 **점상 병변과 표면 거칠기의
밀도**로 나타나고 512 에서도 겨우 보이는 크기라, 1/32 까지 내려간 특징에서는 IGA·홍반 같은
전역 정보에 희석될 수밖에 없다.

그리고 bbox crop 은 **배율이 정규화돼 있지 않다** — 같은 등급이라도 얼굴 전체가 512 에
들어간 사진(병변 1~2px)과 볼만 확대된 사진(병변 20px)이 섞여 있다. 그래서 (a) 여러 stage 를
같이 보고(multi-scale), (b) 이미지마다 local/global 비중을 다르게 두는(gated fusion) 구조가
필요하다.

## 구성

    LocalTrunk   : 지정한 저층 stage 들을 가장 낮은 해상도 쪽으로 정렬 -> concat
                   -> depthwise-separable conv 2개 -> 1채널 공간 attention -> GAP
    GatedFusion  : alpha = sigmoid(MLP([z_g, z_l])),  z = alpha*z_l + (1-alpha)*z_g
    LocalAware   : 전 태스크는 stage4 GAP -> 공유 neck. local_tasks 만 위 경로를 추가로 탄다.

head 는 기존과 같은 (K-1) CORN 로짓이라 **copula(CCNN)/IBB(IT) 는 손댈 필요가 없다**.

## 진단 (학습 후 반드시 볼 것)

`gate_stats()` 의 alpha 분포. 전 샘플에서 0.5 근처에 몰리면 게이트가 아무 일도 안 하는
것이고(= 단순 평균), 0/1 에 붙으면 한쪽을 통째로 버린 것이다. 이미지마다 퍼져야
'adaptive' 라는 주장이 성립한다 — cls_mbn 의 CFEN attention 이 균등분포로 붕괴했던 전례를
같은 방식으로 감시한다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_base import build_multiscale_backbone


def conv1x1_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def dwsep_conv(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LocalTrunk(nn.Module):
    """저층 stage -> 정렬/융합 -> 공간 attention -> (B, out_dim)."""

    def __init__(self, in_chs, dim=64, out_dim=128, att_k=7):
        super().__init__()
        self.proj = nn.ModuleList([conv1x1_bn_relu(c, dim) for c in in_chs])
        self.mix = nn.Sequential(dwsep_conv(dim * len(in_chs), out_dim),
                                 dwsep_conv(out_dim, out_dim))
        self.att = nn.Conv2d(out_dim, 1, att_k, padding=att_k // 2)
        self.last_att = None                       # 진단/시각화용(그래프 분리)

    def forward(self, feats):
        ref = feats[-1].shape[-2:]                 # 가장 낮은 해상도에 맞춰 정렬
        xs = []
        for p, f in zip(self.proj, feats):
            h = p(f)
            if h.shape[-2:] != ref:
                h = F.interpolate(h, ref, mode="bilinear", align_corners=False)
            xs.append(h)
        h = self.mix(torch.cat(xs, 1))
        a = torch.sigmoid(self.att(h))
        self.last_att = a.detach()
        return (h * a).mean(dim=(2, 3))


class GatedFusion(nn.Module):
    """이미지마다 local 을 얼마나 믿을지 정한다: z = a*proj(z_l) + (1-a)*proj(z_g)."""

    def __init__(self, g_dim, l_dim, dim):
        super().__init__()
        self.pg = nn.Linear(g_dim, dim)
        self.pl = nn.Linear(l_dim, dim)
        self.gate = nn.Sequential(nn.Linear(g_dim + l_dim, max(dim // 4, 16)), nn.GELU(),
                                  nn.Linear(max(dim // 4, 16), 1), nn.Sigmoid())
        self.last_alpha = None

    def forward(self, z_g, z_l):
        a = self.gate(torch.cat([z_g, z_l], 1))
        self.last_alpha = a.detach()
        return a * self.pl(z_l) + (1 - a) * self.pg(z_g)


class ConcatFusion(nn.Module):
    """게이트 없는 대조군(ablation M3/M4): 그냥 붙여서 선형투영."""

    def __init__(self, g_dim, l_dim, dim):
        super().__init__()
        self.proj = nn.Linear(g_dim + l_dim, dim)
        self.last_alpha = None

    def forward(self, z_g, z_l):
        return self.proj(torch.cat([z_g, z_l], 1))


class LocalAwareMultiTask(nn.Module):
    """stage4 GAP -> 공유 neck -> 전 태스크 head. local_tasks 만 LocalTrunk 를 추가로 탄다.

    local_tasks 가 여럿이면 **LocalTrunk 는 공유**하고 fusion/head 만 태스크별로 둔다.
    실데이터(atopy.csv 9,150행)에서 찰상-태선화 상관이 0.377 로 태선화의 최강 파트너라
    둘을 한 trunk 에 묶는 게 맞다(합성 데이터만 보면 0.069 라 정반대 결론이 난다).
    `--local_share 0` 으로 태스크마다 trunk 를 따로 둘 수 있다.
    """

    def __init__(self, backbone, num_classes, local_tasks, *, stages=(0, 1),
                 embed_dim=512, dropout=0.3, ordinal=False, pretrained=True,
                 local_dim=128, local_width=64, fusion="gate", share_trunk=True):
        super().__init__()
        self.ordinal = ordinal
        self.local_tasks = [t for t in local_tasks if t in num_classes]
        self.stages = tuple(stages)
        self.fusion_kind = fusion
        self.backbone, chs = build_multiscale_backbone(backbone, pretrained=pretrained)
        self.global_dim = chs[-1]

        self.neck = nn.Sequential(
            nn.Linear(self.global_dim, embed_dim), nn.BatchNorm1d(embed_dim),
            nn.GELU(), nn.Dropout(dropout))

        in_chs = [chs[i] for i in self.stages]
        if self.local_tasks and fusion != "none":
            keys = ["_shared"] if share_trunk else list(self.local_tasks)
            self.trunk = nn.ModuleDict(
                {k: LocalTrunk(in_chs, dim=local_width, out_dim=local_dim) for k in keys})
            self.share_trunk = share_trunk
            F_ = GatedFusion if fusion == "gate" else ConcatFusion
            self.fusion = nn.ModuleDict(
                {t: F_(self.global_dim, local_dim, embed_dim) for t in self.local_tasks})
            self.local_drop = nn.Dropout(dropout)
        else:
            self.trunk = self.fusion = None
            self.local_tasks = []

        self.heads = nn.ModuleDict(
            {t: nn.Linear(embed_dim, (c - 1) if ordinal else c) for t, c in num_classes.items()})

    def forward(self, x):
        feats = self.backbone(x)
        z_g = feats[-1].mean(dim=(2, 3))                       # stage4 GAP
        z_shared = self.neck(z_g)

        out = {}
        for t, head in self.heads.items():
            if self.fusion is not None and t in self.local_tasks:
                key = "_shared" if self.share_trunk else t
                z_l = self.trunk[key]([feats[i] for i in self.stages])
                z = self.local_drop(self.fusion[t](z_g, z_l))
            else:
                z = z_shared
            out[t] = head(z)
        return out

    # ------------------------------------------------------------------ 진단
    def gate_stats(self):
        """태스크별 alpha 의 (평균, 표준편차, 5/95 분위). std 가 0 에 가까우면 게이트가
        이미지를 구분하지 못하고 고정 혼합비로 굳었다는 뜻이다."""
        if self.fusion is None or self.fusion_kind != "gate":
            return None
        st = {}
        for t, f in self.fusion.items():
            a = f.last_alpha
            if a is None:
                continue
            a = a.float().flatten()
            st[t] = dict(mean=float(a.mean()), std=float(a.std()),
                         p05=float(a.quantile(0.05)), p95=float(a.quantile(0.95)))
        return st or None

    def describe(self):
        if not self.local_tasks:
            return "  local branch 없음(기저선)"
        st = ", ".join(f"S{i+1}" for i in self.stages)
        share = "공유" if self.share_trunk else "태스크별"
        return (f"  local: {'+'.join(self.local_tasks)}  stages={st}  "
                f"trunk={share}  fusion={self.fusion_kind}")
