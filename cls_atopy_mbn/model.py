"""MaMNet(Xing et al., Appl. Sci. 2024)의 MbN + SFEN + CFEN + ASPP 를 아토피 5축
(IGA + 홍반·구진·찰상·태선화) 등급 분류로 이식한 모델.

[`cls_idrid`](../cls_idrid)(논문 그대로의 2분기 DR/DME 재현)에서 출발해 대화에서 정한
세 가지를 바꿨다.

1. **분기를 태스크가 아니라 '그룹' 단위로 둔다** (`--groups sep`, 기본)

       branch_iga    <- IGA (전역 중증도)         : 논문의 DR 자리 (ASPP)
       branch_acute  <- 홍반·구진·찰상 (급성 염증) : 논문의 DME 자리 (F skip)
       branch_lich   <- 태선화 (만성 재형성)       : 신규 branch5

   근거는 실데이터 9,150행 라벨 상관이다. 태선화만 나머지 전부와 상관이 약하고
   (스피어만 0.21~0.38, IGA 와도 0.36) 나머지 셋은 서로 0.44~0.72 로 묶인다.
   5축을 평평하게 펴는 대신(`cls_mbn`) 이 구조를 모델에 박아 넣는다.

   `--groups merged` 면 태선화를 급성과 합쳐 논문과 동일한 2분기가 되고,
   `--groups flat` 이면 태스크당 1분기(5분기)다. 세 값이 곧 branch5 의 ablation 축.

2. **SFEN 을 CA(채널) 와 SA(공간) 로 쪼개 배치를 다르게 한다** (`--sfen share`, 기본)

   SA 는 출력이 1채널 H×W 맵이라 "어디를 볼지"만 정한다. 홍반·구진·찰상은 같은 병변
   영역에 같이 나타나므로 **SA 는 그룹당 하나면 충분**하고, 태선화는 굴곡부의 두꺼워진
   피부라 애초에 다른 위치를 봐야 하므로 그룹이 갈리면 SA 도 저절로 갈린다.
   반대로 "그 영역에서 무슨 특징을 볼지"(색/융기/선상결손)는 채널 선택이라 **CA 는
   태스크별**로 둔다. 비용도 이쪽이 유리하다 — SA 4갈래 비대칭 conv 가 SFEN 파라미터의
   95%(2.36M/2.50M)이고 CA 는 131K 다.

       share (기본) : CA 태스크별 + SA 그룹공유,  S_t = (F_g·CA_t(F_g)) · SA_g(F_g)
       task         : 논문 그대로 SFEN 통째로 태스크별, S_t = f2 · SA_t(f2), f2=F_g·CA_t(F_g)
       group        : SFEN 통째로 그룹별(가장 쌈),     S_g = f2 · SA_g(f2)

   `share` 의 SA 입력이 F_g(원본)인 것은 논문 Eq(4)~(7)에서 **의도적으로 벗어난 지점**이다.
   Eq(4)는 SA 입력이 CA 통과 후의 F' 인데, CA 가 태스크별이면 F' 도 태스크별이 되어
   SA 를 그룹에서 공유할 수가 없다. 그래서 SA 입력만 그룹 공통인 F_g 로 되돌렸다.

3. **CFEN 을 그룹 쌍마다 따로 둔다**(논문 원형 복원)

   논문 CFEN 은 softmax 가 없다 — `C_i = CA(F_i) ⊗ F_j`, 채널 게이트 하나가 전부다.
   상대가 하나뿐이라 "누구를 볼까"라는 질문이 없었기 때문이다. `cls_mbn` 은 5태스크로
   늘리며 태스크축 softmax 를 도입했다가 균등분포로 붕괴했다(0.20~0.31, 균등=0.25).
   여기서는 그룹이 3개라 방향 포함 6쌍뿐이라, 쌍마다 독립 CA 게이트를 두는 논문 원형을
   그대로 쓸 수 있다. 정규화 제약이 없으니 균등 붕괴 모드가 구조적으로 사라지고,
   대신 쌍별 게이트 std 가 진단 지표가 된다(`gate_stats`).

   `--cfen` 으로 쌍을 직접 고를 수 있다: `all` | `none` | `"lich<-acute,iga<-acute"`.
   데이터가 맞다면 `lich<-acute`(찰상→태선화, 0.38)는 살고 `acute<-lich`(홍반-태선화
   0.21)는 죽어야 한다.

논문 그대로 남긴 것: MbN 의 VGG16 1:1 대응(Table 1), CA 의 Eq(1)~(3), SA 의 비대칭
conv(Eq 4~7, k=9), CFEN 의 Eq(8), 수정형 ASPP(Fig.6, rate 1/3/5/7/7), 등급분류
헤드(Fig.7, GA→FC→ReLU→Dropout→FC→ReLU→FC), 그리고 **IGA 분기만 ASPP 를 받는
비대칭**(논문 3.2.3: 전역에 흩어진 병변은 전역문맥이 필요하다).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn

# torchvision vgg16_bn.features 의 블록 경계 (cls_idrid 와 동일 — 논문 Table 1 대응)
_B1_END, _B2_END = 33, 24        # Branch1 = features[:33], Branch2 = features[:24]
_BLOCK4 = (24, 33)               # conv3-512 x3 (256->512): side branch 앞단
_BLOCK5 = (34, 43)               # conv3-512 x3            : side branch 뒷단
MBN_CHANNELS = 512               # MbN 출력 채널


def _vgg_features(pretrained):
    if not pretrained:
        return vgg16_bn(weights=None).features
    try:
        return vgg16_bn(weights="IMAGENET1K_V1").features
    except Exception as e:                       # 오프라인 등 — 조용히 죽지 않게 알리고 진행
        print(f"[model] 경고: VGG16 사전학습 가중치 로드 실패({type(e).__name__}) -> 랜덤 초기화")
        return vgg16_bn(weights=None).features


def _slice(feats, lo, hi):
    return nn.Sequential(*[copy.deepcopy(m) for m in list(feats)[lo:hi]])


def _key(g, h):
    """ModuleDict 키. '.' 이 못 들어가므로 'g__h' 로 쓰고 표시만 'g<-h' 로 한다."""
    return f"{g}__{h}"


# =============================================================================
# MbN — 공유 Branch1·2 + 그룹별 side branch (논문 3.1, Table 1)
# =============================================================================
class MbN(nn.Module):
    """Branch1(=features[:33]) 과 Branch2(=features[:24]) 를 공유하고, 그룹마다
    side branch(block4+block5 사본, 가중치 별개)를 하나씩 둔다.

        F_g = Branch1(x) + up(side_g(Branch2(x)))          # 논문 F_DME/F_DR 와 동일 형태

    Table 1 검산(224 입력): Branch2 출력 28x28x256 -> conv512x3 -> pool(2,2,s2) 14x14
    -> conv512x3 -> pool(3,3,s1, 크기유지) -> Upsample x2 = 28x28x512 로 Branch1 과 일치.
    """

    def __init__(self, group_names, pretrained=True):
        super().__init__()
        feats = _vgg_features(pretrained)
        self.branch1 = _slice(feats, 0, _B1_END)                       # -> (B,512,H/8,W/8)
        self.branch2 = _slice(feats, 0, _B2_END)                       # -> (B,256,H/8,W/8)
        self.side = nn.ModuleDict({g: self._side_branch(feats) for g in group_names})

    @staticmethod
    def _side_branch(feats):
        return nn.Sequential(
            _slice(feats, *_BLOCK4),                      # conv3-512 x3
            nn.MaxPool2d(2, 2),                           # Table1: (2,2) s2
            _slice(feats, *_BLOCK5),                      # conv3-512 x3
            nn.MaxPool2d(3, stride=1, padding=1),         # Table1: (3,3) s1 -> 크기유지
            nn.Upsample(scale_factor=2, mode="nearest"),  # Keras UpSampling2D 기본값
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return {g: b1 + m(b2) for g, m in self.side.items()}

    def shared_parameters(self):
        """warmup 때 동결할 사전학습 구간(= MbN 전체)."""
        return list(self.parameters())


# =============================================================================
# CA / SA — SFEN(논문 3.2.1, Fig.4)을 둘로 분리
# =============================================================================
class ChannelAttention(nn.Module):
    """논문 Eq(1)~(2): GAP -> FC(c/r) -> ReLU -> FC(c) -> sigmoid. r=4."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, f):
        return self.fc(f)                                  # (B,C,1,1) 채널 게이트


class SpatialAttention(nn.Module):
    """논문 Eq(4)~(7): 1xk -> kx1 과 kx1 -> 1xk 두 갈래의 합을 sigmoid. k=9(Table 4).

    출력이 (B,1,H,W) 라 '어디를 볼지'만 정한다 — 그래서 그룹에서 공유할 수 있다.
    """

    def __init__(self, channels, k=9):
        super().__init__()
        pad, half = k // 2, max(channels // 2, 8)
        self.conv1 = nn.Conv2d(channels, half, (1, k), padding=(0, pad))
        self.conv3 = nn.Conv2d(half, 1, (k, 1), padding=(pad, 0))
        self.conv2 = nn.Conv2d(channels, half, (k, 1), padding=(pad, 0))
        self.conv4 = nn.Conv2d(half, 1, (1, k), padding=(0, pad))

    def forward(self, f):
        return torch.sigmoid(self.conv3(self.conv1(f)) + self.conv4(self.conv2(f)))


# =============================================================================
# CFEN — Cross-Feature Extraction (논문 3.2.2, Fig.5, Eq 8)
# =============================================================================
class CFEN(nn.Module):
    """자기 그룹의 채널 게이트를 **상대 그룹의 특징맵**에 곱한다: C_(g<-h) = CA(F_g) x F_h.

    논문 Eq(9)는 Eq(8)과 같은 식이 실린 오타라, Fig.5 와 본문에 따라 방향마다 모듈을
    따로 둔다. softmax 가 없으므로 쌍끼리 경쟁하지 않는다 — 쌍이 죽으면 그 쌍만 죽는다.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.last_gate = None                              # 진단용(그래프 분리)

    def forward(self, f_self, f_other):
        gate = self.ca(f_self)
        self.last_gate = gate.detach()
        return gate * f_other


# =============================================================================
# ASPP — 수정형 (논문 3.2.3, Fig.6): 5분기, rate 1/3/5/7/7
# =============================================================================
class ASPP(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or channels
        c = max(channels // 4, 32)                          # 5분기 concat 폭 억제용 분기폭
        self.b1 = self._conv(channels, c, 1, 1)
        self.b2 = self._conv(channels, c, 3, 3)
        self.b3 = self._conv(channels, c, 3, 5)
        self.b4 = self._conv(channels, c, 3, 7)
        self.b5_conv = self._conv(channels, c, 3, 7)        # 5번째: rate7 conv -> avgpool -> 1x1
        self.b5_proj = self._conv(c, c, 1, 1)
        self.project = self._conv(c * 5, out_channels, 1, 1)

    @staticmethod
    def _conv(cin, cout, k, rate):
        pad = 0 if k == 1 else rate
        return nn.Sequential(
            nn.Conv2d(cin, cout, k, padding=pad, dilation=rate, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, f):
        hw = f.shape[-2:]
        g = self.b5_proj(F.adaptive_avg_pool2d(self.b5_conv(f), 1))
        g = g.expand(-1, -1, *hw)
        return self.project(torch.cat([self.b1(f), self.b2(f), self.b3(f), self.b4(f), g], 1))


# =============================================================================
# 등급 분류 모듈 (논문 3.3, Fig.7): GA -> FC -> ReLU -> Dropout -> FC -> ReLU -> FC
# =============================================================================
class GradingHead(nn.Module):
    def __init__(self, cin, num_out, hidden=512, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cin, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, num_out),
        )

    def forward(self, d):
        return self.mlp(d.mean(dim=(2, 3)))                 # GA(Global Average Pooling)


# =============================================================================
# AtopyMaMNet
# =============================================================================
class AtopyMaMNet(nn.Module):
    """반환: {task: (B,K) 로짓} — ordinal=True 면 CORN 용으로 (B,K-1).

    그룹 g 의 서술자 D 는 이렇게 쌓인다(논문 3장 3단계의 비대칭을 그대로 유지):

        D_t = [F_g?] + [S_t] + [C_(g<-h) for h != g] + [G_g?]

    - F_g   : MbN 원특징(게이트 안 거친 skip). IGA 분기는 논문 D_DR 처럼 기본적으로 뺀다.
    - S_t   : 태스크별 self-feature (CA_t + SA_g)
    - C     : 활성 쌍마다 하나씩
    - G_g   : ASPP 전역특징. IGA 그룹에만.

    `iga_skip` 은 ASPP 를 껐을 때 IGA 분기에 F 를 돌려줄지 정한다:
        auto (기본) : ASPP 켜짐 -> F 제외(논문), 꺼짐 -> F 포함
                      (ASPP 축만 흔들었을 때 IGA 가 게이트 안 거친 자기 경로를 전부
                       잃어버리는 걸 막는다 — cls_idrid 의 `dr_in or c` 폴백과 같은 취지)
        f           : 항상 F 포함 (ASPP 와 F 를 둘 다 준다)
        none        : 항상 F 제외 (Table 5 식 순수 ablation)
    """

    def __init__(self, num_classes, groups, iga_group="iga", pretrained=True,
                 sfen_mode="share", sfen_k=9, reduction=4, use_sfen=True,
                 cfen_pairs="all", use_aspp=True, iga_skip="auto",
                 head_hidden=512, dropout=0.5, ordinal=False):
        super().__init__()
        self.groups = {g: list(ts) for g, ts in groups.items()}
        self.task_group = {t: g for g, ts in self.groups.items() for t in ts}
        self.group_names = list(self.groups)
        self.iga_group = iga_group if iga_group in self.groups else self.task_group["severity"]
        self.sfen_mode, self.use_sfen, self.use_aspp = sfen_mode, use_sfen, use_aspp
        self.iga_skip = iga_skip
        c = MBN_CHANNELS

        self.mbn = MbN(self.group_names, pretrained=pretrained)

        # --- SFEN: CA/SA 를 모드에 따라 태스크별/그룹별로 배치 ---
        self.ca_per_task = sfen_mode in ("share", "task")
        self.sa_per_task = sfen_mode == "task"
        if use_sfen:
            ca_keys = list(self.task_group) if self.ca_per_task else self.group_names
            sa_keys = list(self.task_group) if self.sa_per_task else self.group_names
            self.ca = nn.ModuleDict({k: ChannelAttention(c, reduction) for k in ca_keys})
            self.sa = nn.ModuleDict({k: SpatialAttention(c, sfen_k) for k in sa_keys})

        # --- CFEN: 활성 그룹 쌍(방향 포함)마다 모듈 하나 ---
        self.pairs = self._resolve_pairs(cfen_pairs)
        self.cfen = nn.ModuleDict({_key(g, h): CFEN(c, reduction) for g, h in self.pairs})

        if use_aspp:
            self.aspp = ASPP(c, c)

        # --- 그룹별 서술자 구성(F/C/G 는 그룹 공통, S 만 태스크별) ---
        self.use_f = {g: self._use_f(g) for g in self.group_names}
        self.cross_of = {g: sorted(h for gg, h in self.pairs if gg == g) for g in self.group_names}
        self.dim_of = {}
        for g in self.group_names:
            n = int(self.use_f[g]) + int(use_sfen) + len(self.cross_of[g]) \
                + int(use_aspp and g == self.iga_group)
            if n == 0:                                       # 전부 끈 대조군이면 F 만이라도
                self.use_f[g], n = True, 1
            self.dim_of[g] = c * n

        self.heads = nn.ModuleDict({
            t: GradingHead(self.dim_of[g], num_classes[t] - 1 if ordinal else num_classes[t],
                           head_hidden, dropout)
            for t, g in self.task_group.items()})

    # ---------------------------------------------------------------- 구성 helper
    def _use_f(self, g):
        if g != self.iga_group:
            return True
        if self.iga_skip == "f":
            return True
        if self.iga_skip == "none":
            return False
        return not self.use_aspp                             # auto

    def _resolve_pairs(self, spec):
        """'all' | 'none' | 'g<-h,g2<-h2' -> [(g,h), ...] (g 가 참조하는 쪽, h 가 참조되는 쪽)."""
        if isinstance(spec, (list, tuple, set)):
            wanted = list(spec)
        elif spec in (None, "none", ""):
            return []
        elif spec == "all":
            return [(g, h) for g in self.group_names for h in self.group_names if g != h]
        else:
            wanted = []
            for item in str(spec).split(","):
                item = item.strip()
                if not item:
                    continue
                if "<-" not in item:
                    raise ValueError(f"CFEN 쌍 형식 오류: {item!r} (예: 'lich<-acute')")
                g, h = (s.strip() for s in item.split("<-", 1))
                wanted.append((g, h))
        out = []
        for g, h in wanted:
            if g not in self.groups or h not in self.groups:
                raise ValueError(f"CFEN 쌍의 그룹 이름이 없다: {g}<-{h} (그룹 {self.group_names})")
            if g != h and (g, h) not in out:
                out.append((g, h))
        return out

    # ---------------------------------------------------------------- forward
    def _self_feature(self, t, g, f):
        """S_t = f2 * SA(...)  — sfen_mode 에 따라 CA/SA 키와 SA 입력이 달라진다."""
        f2 = f * self.ca[t if self.ca_per_task else g](f)
        if self.sfen_mode == "share":
            sa = self.sa[g](f)          # SA 입력을 F_g 로: 그룹 공유를 위한 의도적 이탈
        else:
            sa = self.sa[t if self.sa_per_task else g](f2)   # 논문 Eq(4): 입력은 F'
        return f2 * sa

    def forward(self, x):
        feats = self.mbn(x)                                  # {group: (B,512,h,w)}

        shared = {}                                          # 그룹 공통 항(F, C, G)
        for g in self.group_names:
            pre = [feats[g]] if self.use_f[g] else []
            post = [self.cfen[_key(g, h)](feats[g], feats[h]) for h in self.cross_of[g]]
            if self.use_aspp and g == self.iga_group:
                post.append(self.aspp(feats[g]))
            shared[g] = (pre, post)

        out = {}
        for t, g in self.task_group.items():
            pre, post = shared[g]
            parts = pre + ([self._self_feature(t, g, feats[g])] if self.use_sfen else []) + post
            out[t] = self.heads[t](torch.cat(parts, 1))
        return out

    # ---------------------------------------------------------------- 진단
    def gate_stats(self):
        """쌍별 CFEN 채널게이트의 (평균, 표준편차). std 가 0 에 가까우면 게이트가 균등하게
        붕괴해 cross-feature 가 상대 특징맵의 상수배로만 작동한다는 뜻(cls_mbn 전례).
        마지막 forward 기준이라, test set 평균은 train.py 의 collect_gate_stats 를 쓴다."""
        stats = {}
        for (g, h) in self.pairs:
            m = self.cfen[_key(g, h)]
            if m.last_gate is not None:
                stats[f"{g}<-{h}"] = (float(m.last_gate.mean()), float(m.last_gate.std()))
        return stats or None

    def describe(self):
        """구조 한 줄 요약 — 로그/README 대조용."""
        rows = []
        for g in self.group_names:
            parts = (["F"] if self.use_f[g] else []) + (["S"] if self.use_sfen else []) \
                + [f"C<-{h}" for h in self.cross_of[g]] \
                + (["G(ASPP)"] if self.use_aspp and g == self.iga_group else [])
            rows.append(f"  {g:<6}({','.join(self.groups[g])}) : D = {' + '.join(parts)}"
                        f"  [{self.dim_of[g]}ch]")
        return "\n".join(rows)
