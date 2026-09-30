"""백본 + (선택)neck + 태스크별 순서형 헤드.

백본은 timm 을 그대로 쓴다(`build_backbone` = `classification_crop/model_base.py`
와 동일한 얇은 래퍼). ogw 는 인터넷이 되는 학습 서버라 timm 의 `pretrained=True`
가 HuggingFace/GitHub 에서 바로 받아 오므로, cls_kd(오프라인판)처럼 로컬
`pretrained/<name>.pth` 를 미리 반입해 두거나 timm 소스를 벤더링할 필요가 없다.

--- 헤드 규약 ---------------------------------------------------------------
ordinal=True(=--loss corn): 태스크당 (K-1) 로짓. 이게 논문의 OCNN — 등급을 K-1 개의
'이 등급 초과?' 이진 문제로 분해하고 조건부로 학습해 누적확률의 단조성을 보장한다.
ordinal=False(=--loss ce): 태스크당 K 로짓. 순서 정보를 버리는 대조군이자, 도메인
ckpt(effunet_pvtv2b0_512.pt)의 헤드까지 이식하려면 필요한 경로다.
"""
from pathlib import Path

import timm
import torch
import torch.nn as nn

from labels import NUM_CLASSES

ROOT = Path(__file__).resolve().parent

# 짧은 이름 -> timm 모델명.
MODELS = {
    "efflite0": "tf_efficientnet_lite0",
    "effb0":    "efficientnet_b0",
    "mnv3s":    "mobilenetv3_small_100",
    "mnv4s":    "mobilenetv4_conv_small",
    "mnv4m":    "mobilenetv4_conv_medium",
    "mnv4l":    "mobilenetv4_conv_large",
    # hybrid = 뒤쪽 stage 에 MQA(attention)를 섞은 변형. 위치임베딩이 없어 512 같은
    # 큰 해상도에도 그대로 쓸 수 있다.
    "mnv4hm":   "mobilenetv4_hybrid_medium",
    # 계층형 트랜스포머 — CNN 이 놓치는 넓은/확산성 병변(태선화 등) 대비군.
    "pvtv2b0":  "pvt_v2_b0",
    "pvtv2b1":  "pvt_v2_b1",
    "pvtv2b2":  "pvt_v2_b2",
}


def build_backbone(name, pretrained=True, verbose=True, spatial=False):
    """반환: (backbone, feature_dim). backbone(x) -> (B, feature_dim).

    num_features 대신 더미 forward 로 차원을 잡는 이유: MobileNetV4 등은 conv_head
    때문에 num_features(960)와 실제 출력차원(1280)이 다르다.
    pretrained=False: from-scratch 대조군 / 추론(ckpt 가 어차피 덮어씀)용.

    spatial=True 면 global_pool 을 끄고 (B, C, H, W) 를 그대로 내보낸다. 풀링을
    바깥에서 하려는 경우다(GeM / 태스크별 어텐션). pvt_v2_b0 기준 512 입력에서
    (B, 256, 16, 16) 이 나온다.
    """
    m = timm.create_model(name, pretrained=pretrained, num_classes=0,
                          global_pool="" if spatial else "avg")
    m.eval()                                   # 헤드 BatchNorm 이 batch=2 프로브에서 안 터지게
    with torch.no_grad():
        feat = m(torch.zeros(2, 3, 224, 224)).shape[1]
    if verbose:
        print(f"[backbone] {name} (timm, pretrained={pretrained})  feat={feat}")
    return m, feat


# ----------------------------------------------------------------------------- 풀링
class GeMPool(nn.Module):
    """일반화 평균 풀링(Radenovic+ 2019).  (B,C,H,W) -> (B,C).

    GAP(p=1)와 max(p->inf) 사이를 학습으로 고른다. 전역 평균은 '넓게 퍼진 색'에는
    맞지만 '드문드문한 작은 병변'에는 맞지 않는다 — 구진 몇 개가 512x512 안에
    흩어져 있으면 평균에서 거의 사라진다. p 를 올리면 강한 위치가 살아남는다.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, x):
        # fp32 로 고정한다. p=3 거듭제곱은 fp16 에서 쉽게 넘치고(활성 50 -> 1.25e5 > 65504),
        # autocast 가 pow 를 fp32 로 올려 주는 것은 버전에 딸린 정책이라 기대지 않는다.
        p = self.p.clamp(min=1.0).float()
        return x.float().clamp(min=self.eps).pow(p).mean(dim=(-2, -1)).pow(1.0 / p)

    def extra_repr(self):
        return f"p={float(self.p):.3f}"


class TaskAttentionPool(nn.Module):
    """태스크별 공간 어텐션 풀링.  (B,C,H,W) -> {task: (B,C)}.

    5축이 같은 전역 평균 벡터 하나를 나눠 쓰는 것이 지금 구조의 병목이다. IGA/홍반은
    전역 색·범위라 평균에서 살아남지만, 구진·찰상·태선화는 국소 질감이라 평균이
    지운다(그래서 해상도를 2배로 올려도 세 축이 안 움직인다). 축마다 '어디를 볼지'를
    따로 학습하게 해서 그 평균을 걷어낸다.

    점수망은 1x1 conv 2층이라 위치마다 독립적으로 계산된다 — 공간 크기에 무관하므로
    imgsz 를 바꿔도 그대로 쓸 수 있다.
    """

    def __init__(self, dim, tasks, hidden=0):
        super().__init__()
        hidden = int(hidden) or max(32, dim // 4)
        self.tasks = list(tasks)
        self.score = nn.ModuleDict(
            {t: nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(),
                              nn.Conv2d(hidden, 1, 1))
             for t in self.tasks})

    def forward(self, x):
        # 어텐션 가중은 fp32 로 낸다. HW(512 입력이면 256~16384) 에 걸친 softmax 와
        # 가중합은 fp16 에서 정밀도를 잃고, 그 오차가 그대로 특징에 실린다.
        flat = x.float().flatten(2)                          # (B,C,HW)
        out = {}
        for t in self.tasks:
            a = self.score[t](x).float().flatten(2).softmax(dim=-1)   # (B,1,HW)
            out[t] = (flat * a).sum(dim=-1)                   # (B,C)
        return out

    def maps(self, x):
        """진단용 — 태스크별 어텐션 맵 (B,1,H,W). 붕괴(전부 균등)를 눈으로 본다."""
        H, W = x.shape[-2:]
        return {t: self.score[t](x).flatten(2).softmax(dim=-1).reshape(-1, 1, H, W)
                for t in self.tasks}


def build_pool(kind, dim, tasks, hidden=0):
    """kind: avg(백본 내장 GAP) | gem | attn. -> (모듈 또는 None, 태스크별 여부)."""
    if kind == "gem":
        return GeMPool(), False
    if kind == "attn":
        return TaskAttentionPool(dim, tasks, hidden), True
    return None, False


def _head(din, dout, hidden=0, dropout=0.0):
    """순서형 헤드. hidden=0 이면 기존과 동일한 Linear 하나."""
    if not hidden:
        return nn.Linear(din, dout)
    return nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                         nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                         nn.Linear(hidden, dout))


class ScaleEmbed(nn.Module):
    """크롭 배율 스칼라 -> (B, dim) 임베딩.

    왜 필요한가: bbox 크롭을 전부 같은 imgsz 로 리사이즈하므로 '원본에서 얼마나
    확대했는가'가 사라진다. 구진 등급은 병변이 얼마나 크고 촘촘한지가 기준이라,
    같은 픽셀 크기가 이미지마다 다른 실제 크기를 뜻하게 된다. 배율을 되돌려 준다.
    """

    def __init__(self, dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dim = int(dim)

    def forward(self, s):
        return self.net(s.reshape(-1, 1).float())


class MultiScaleTrunk(nn.Module):
    """백본 -> (g4, g3). SPEC.md §3.1 의 다중스케일 특징 추출기.

    g4 는 백본의 평소 출력(global_pool='avg')을 **그대로** 쓴다. 단일스케일 경로와
    비트 단위로 같은 값이라, multiscale on/off 대조가 'g3 를 더했는가' 하나로만
    갈린다 — 추출 방식까지 같이 바뀌면 델타를 무엇에 돌릴지 알 수 없다.

    g3 는 feature_info 의 reduction=16 스테이지에 forward hook 을 걸어 공간평균한다.
    timm 의 features_only 래퍼 대신 hook 을 직접 쓰는 이유는 이 한 곳(중간 스테이지
    하나만)이 필요해서고, features_only 래퍼는 모든 스테이지를 다 뽑는 쪽에 맞춰져
    있어 여기엔 과하다. pvt_v2_b0 기준 g3=(B,160), g4=(B,256) 으로 SPEC 의 수치와
    정확히 일치한다.

    multiscale=False 면 hook 을 아예 걸지 않고 dim3=0 을 보고한다(SPEC §9 단계 3 축소판).
    """

    def __init__(self, name, pretrained=True, multiscale=True, pool="avg"):
        super().__init__()
        # pool != 'avg' 면 백본의 GAP 을 끄고 (B,C,H,W) 를 받아 바깥에서 줄인다.
        # 공유 인자 f 는 정의상 축 공통이므로 여기서는 태스크별 어텐션을 쓰지 않는다
        # (attn 은 SpecNet 의 **잔차** r_t 에서만 쓴다 — 거기가 축별 자리다).
        self.pool_kind = pool
        self.spatial = pool != "avg"
        self.backbone, self.dim4 = build_backbone(name, pretrained=pretrained,
                                                  spatial=self.spatial)
        self.gem4 = GeMPool() if pool in ("gem", "attn") else None
        self.gem3 = GeMPool() if pool in ("gem", "attn") else None
        self.multiscale = bool(multiscale)
        self.dim3 = 0
        self._g3 = None
        self._g3_map = None
        if self.multiscale:
            entry = self._pick_stage(name)
            self.dim3 = int(entry["num_chs"])
            self.backbone.get_submodule(entry["module"]).register_forward_hook(self._hook)
            print(f"[trunk] 다중스케일 {name}: g3={self.dim3}ch @1/{entry['reduction']} "
                  f"({entry['module']})  +  g4={self.dim4}ch")

    def _pick_stage(self, name):
        """reduction=16 항목. 없으면 끝에서 두 번째(=g4 바로 앞 스케일)."""
        info = [f for f in getattr(self.backbone, "feature_info", []) if "module" in f]
        if len(info) < 2:
            raise SystemExit(
                f"[에러] {name} 에서 중간 특징을 뽑을 수 없다(feature_info 부족). "
                f"--spec_no_multiscale 로 단일스케일 축소판을 쓸 것.")
        return next((f for f in info if f.get("reduction") == 16), info[-2])

    def _hook(self, _module, _inp, out):
        # 공간축을 여기서 바로 줄인다 — (B,C,H,W) 를 붙들고 있지 않기 위해서다.
        if out.ndim != 4:
            self._g3 = out.flatten(1)
            return
        self._g3 = self.gem3(out) if self.gem3 is not None else out.mean(dim=(-1, -2))

    def forward(self, x):
        """-> (g4, g3, g4_map). g4_map 은 pool='avg' 면 None(공간 특징을 안 꺼낸다)."""
        self._g3 = None
        feat = self.backbone(x)
        g4_map = feat if feat.ndim == 4 else None
        if feat.ndim == 4:
            g4 = self.gem4(feat) if self.gem4 is not None else feat.mean(dim=(-1, -2))
        else:
            g4 = feat.flatten(1) if feat.ndim > 2 else feat
        g3 = self._g3
        self._g3 = None                        # 참조를 남기면 그래프가 에폭 내내 살아 있다
        if self.multiscale and g3 is None:
            raise RuntimeError("다중스케일 hook 이 호출되지 않았다 — 백본 구조를 확인할 것.")
        return g4, g3, g4_map


class MultiTaskNet(nn.Module):
    """단일 이미지(full 또는 bbox 크롭) -> 5개 축 등급 로짓.

    OCNN 의 구조 그 자체다: 공유 trunk(백본+neck) 하나 위에 태스크별 순서형 헤드
    5개. 헤드끼리 직접 연결된 경로는 없다 — 축 사이의 의존성은 (a) 공유 trunk 와
    (b) IBB Step2 의 조인트 층화 샘플링으로만 들어온다.
    """

    def __init__(self, backbone, embed_dim=512, dropout=0.3, ordinal=True,
                 pretrained=True, use_neck=True, pool="avg", pool_hidden=0,
                 head_hidden=0, use_scale=False, scale_dim=16):
        super().__init__()
        self.ordinal = ordinal
        self.pool_kind = pool
        # avg 는 백본 내장 GAP 을 그대로 쓴다 — 기존 런과 같은 경로를 남겨 두려는
        # 것이다. gem/attn 일 때만 공간 특징 (B,C,H,W) 를 꺼내 바깥에서 풀링한다.
        self.backbone, feat = build_backbone(backbone, pretrained=pretrained,
                                             spatial=pool != "avg")
        self.pool, self.per_task_pool = build_pool(pool, feat, list(NUM_CLASSES),
                                                   pool_hidden)
        self.use_neck = use_neck
        if use_neck:
            self.neck = nn.Sequential(
                nn.Linear(feat, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_in = embed_dim
        else:
            # 백본 특징에 헤드 직결. 도메인 ckpt 의 헤드 구조와 일치시켜 이식하기 위한 경로.
            self.neck = nn.Dropout(dropout)
            head_in = feat
        self.scale_embed = ScaleEmbed(scale_dim) if use_scale else None
        if self.scale_embed is not None:
            head_in += self.scale_embed.dim
        self.heads = nn.ModuleDict(
            {t: _head(head_in, (c - 1) if ordinal else c, head_hidden, dropout)
             for t, c in NUM_CLASSES.items()})

    def _pooled(self, x):
        """-> (B,C) 또는 {task: (B,C)}. avg 는 백본이 이미 풀링해서 내보낸다."""
        f = self.backbone(x)
        if self.pool is None:
            return f.flatten(1) if f.ndim > 2 else f
        return self.pool(f)

    def embed(self, x):
        """공유 임베딩. 태스크별 풀링(attn)이면 {task: (B,D)} 를 돌려준다."""
        f = self._pooled(x)
        if isinstance(f, dict):
            return {t: self.neck(v) for t, v in f.items()}
        return self.neck(f)

    def forward(self, x, scale=None):
        z = self.embed(x)
        out = {}
        for t, head in self.heads.items():
            zt = z[t] if isinstance(z, dict) else z
            if self.scale_embed is not None:
                if scale is None:
                    raise RuntimeError("--use_scale 인데 배치에 scale 이 없다 — "
                                       "collate 가 meta 를 넘기는지 확인할 것.")
                zt = torch.cat([zt, self.scale_embed(scale)], dim=1)
            out[t] = head(zt)
        return out


# ----------------------------------------------------------------------------- 이식
def load_pretrained(model, ckpt_path, *, strict_backbone=True, load_heads="auto",
                    verbose=True, bb_prefix="backbone."):
    """도메인 사전학습 ckpt(.pt/.pth) -> 현재 모델. 반환: 무엇이 이식됐는지 dict.

    ../weights/effunet_pvtv2b0_512.pt 는 backbone(pvt_v2_b0) + heads 만 있고
    **neck 이 없다**. 헤드는 태스크당 K 로짓(CE)이고 입력 차원이 백본 특징차원(256)이다:

        --model pvtv2b0 --no_neck --loss ce  -> 백본 + 헤드 전부 이식
        그 외 조합                            -> 백본만 이식(neck/헤드는 새로 학습)

    load_heads: "auto"(모양이 맞는 헤드만) | "never" | "always"(안 맞으면 예외)
    strict_backbone=True 면 백본이 하나도 안 붙었을 때 예외를 던진다.
    bb_prefix: 이 모델에서 백본이 사는 접두사. MultiTaskNet 은 "backbone.", SPEC 모델
    (spec.SpecNet)은 trunk 를 한 겹 더 거치므로 "trunk.backbone." 이다. ckpt 쪽 키는
    어느 경우든 "backbone." 으로 저장돼 있다.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    tgt = model.state_dict()
    took, skipped = {}, []

    bb_src = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    n_bb = 0
    for k, v in bb_src.items():
        full = f"{bb_prefix}{k}"
        if full in tgt and tgt[full].shape == v.shape:
            took[full] = v
            n_bb += 1
        elif full in tgt:
            skipped.append(f"{full} (모양 {tuple(v.shape)} != {tuple(tgt[full].shape)})")

    # 헤드는 전부 아니면 전무로 간다. 부분 이식(weight 는 안 맞고 bias 만 맞는 경우 등)은
    # 랜덤 weight 에 남의 bias 를 얹는 꼴이라 도움이 안 되고 '이식됐다'는 오해만 만든다.
    n_head = 0
    head_src = {k: v for k, v in sd.items() if k.startswith("heads.")}
    if load_heads != "never" and head_src:
        cand, mismatch = {}, []
        for k, v in head_src.items():
            if k in tgt and tgt[k].shape == v.shape:
                cand[k] = v
            else:
                cur = tuple(tgt[k].shape) if k in tgt else None
                mismatch.append(f"{k} (ckpt {tuple(v.shape)} != 모델 {cur})")
        if mismatch:
            skipped.extend(mismatch)
            if load_heads == "always":
                raise SystemExit(
                    f"[에러] --load_heads always 인데 헤드 {len(mismatch)}/{len(head_src)} "
                    f"텐서의 모양이 다르다. 헤드까지 이식하려면 --no_neck --loss ce "
                    f"--head_hidden 0 이고 백본이 ckpt 와 같아야 한다 "
                    f"(--head_hidden>0 이면 헤드가 MLP 라 키 이름부터 달라진다).\n"
                    f"       불일치: {mismatch[:4]}")
        else:
            took.update(cand)
            n_head = len(cand)

    model.load_state_dict({**tgt, **took}, strict=True)

    if strict_backbone and n_bb == 0:
        raise SystemExit(
            f"[에러] 백본 가중치가 하나도 이식되지 않았다: {ckpt_path}\n"
            f"       ckpt 백본과 --model 이 다르지 않은지 확인할 것.")

    info = {"ckpt": str(ckpt_path), "epoch": ck.get("epoch"),
            "n_backbone": n_bb, "n_backbone_src": len(bb_src),
            "n_head": n_head, "n_head_src": len(head_src)}
    if verbose:
        print(f"[weights] {ckpt_path}  (epoch={info['epoch']})")
        print(f"[weights]   백본 이식 {n_bb}/{len(bb_src)} 텐서")
        if load_heads == "never":
            print("[weights]   헤드: 이식 안 함(--load_heads never) -> 새로 학습")
        elif n_head and n_head == len(head_src):
            print(f"[weights]   헤드 이식 {n_head}/{len(head_src)} 텐서 -> 완전 이어학습")
        else:
            print("[weights]   헤드 이식 안 함(모양 불일치) -> 헤드는 새로 학습"
                  " (헤드까지 쓰려면 --no_neck --loss ce)")
        if skipped:
            print(f"[weights]   건너뜀 예시: {skipped[:3]}")
    return info
