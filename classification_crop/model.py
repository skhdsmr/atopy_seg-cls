"""crop 실험용 멀티태스크 모델 3종 — 백본/neck/head 구조는 기존 MultiTaskNet 과 동일.

  ⓪ full      : MultiTaskNet (원본 이미지 전체 — 마스크 미사용, crop 대조군)
  ① bbox      : MultiTaskNet (model_base.py, 그대로 재사용 — 단일 이미지 입력)
  ② mil       : MILNet       (연결요소 crop bag -> attention/max pooling -> 공유 임베딩)
  ③ twostream : TwoStreamNet (global+local 백본 -> feature concat -> 공유 임베딩)

full/bbox 는 --canet 으로 CANetMultiTask(canet.py) 로 교체할 수 있다.

공통 neck+head(NeckHeads)를 뽑아 세 모델이 같은 순서형(CORN)/CE head 규약을 공유한다.
head 는 ordinal=True 면 (K-1) 로짓(CORN), False 면 K 로짓(CE) — train.py 와 일치.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_base import MultiTaskNet, build_backbone   # noqa: F401  (bbox/full 실험이 재사용)
from canet import CANetMultiTask
from model_local import LocalAwareMultiTask


class NeckHeads(nn.Module):
    """공유 임베딩(neck) + 태스크별 선형 head. MultiTaskNet 과 동일 규약."""
    def __init__(self, feat, num_classes, embed_dim=512, dropout=0.3, ordinal=False):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Linear(feat, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {t: nn.Linear(embed_dim, (c - 1) if ordinal else c) for t, c in num_classes.items()})

    def forward(self, feat):
        z = self.neck(feat)
        return {t: head(z) for t, head in self.heads.items()}


# ----------------------------------------------------------------------------- ② MIL
class MILNet(nn.Module):
    """Multiple Instance Learning: instance(=연결요소 crop)별 특징 -> bag 특징 -> 등급.

    - 공유 백본으로 모든 instance 특징 추출(패딩 없이 flat 처리 후 bag 단위 split).
    - pool='attention': 게이트 없는 tanh attention(Ilse et al. 2018) 가중합.
             'max'      : instance 최대값(가장 심한 병변이 등급 결정하는 성격에 적합).
    forward((flat, lengths)) — flat: (sumN,3,H,W), lengths: (B,) 각 bag instance 수.
    """
    def __init__(self, backbone, num_classes, embed_dim=512, dropout=0.3,
                 ordinal=False, pretrained=True, pool="attention", att_dim=128):
        super().__init__()
        self.ordinal = ordinal
        self.pool = pool
        self.backbone, feat = build_backbone(backbone, pretrained=pretrained)
        if pool == "attention":
            self.attn = nn.Sequential(nn.Linear(feat, att_dim), nn.Tanh(),
                                      nn.Linear(att_dim, 1))
        self.head = NeckHeads(feat, num_classes, embed_dim, dropout, ordinal)

    def _pool_bag(self, f):
        """f: (Ni, feat) 한 bag -> (feat,)."""
        if self.pool == "max":
            return f.max(dim=0).values
        a = torch.softmax(self.attn(f), dim=0)          # (Ni,1)
        return (a * f).sum(dim=0)

    def forward(self, inputs):
        flat, lengths = inputs
        f = self.backbone(flat)
        if f.ndim > 2:
            f = f.flatten(1)
        feats = torch.split(f, lengths.tolist(), dim=0)
        bag = torch.stack([self._pool_bag(fb) for fb in feats])  # (B, feat)
        return self.head(bag)


# ----------------------------------------------------------------------------- ③ two-stream
class TwoStreamNet(nn.Module):
    """global(전체) + local(bbox crop) 두 스트림 특징을 concat 해서 등급 예측.

    share=True: 두 스트림이 백본 하나를 공유(파라미터·과적합↓, 소규모 데이터 권장).
    share=False: 스트림별 독립 백본(표현력↑, 데이터 충분할 때).
    forward((g, l)) — g,l: (B,3,H,W).
    """
    def __init__(self, backbone, num_classes, embed_dim=512, dropout=0.3,
                 ordinal=False, pretrained=True, share=True):
        super().__init__()
        self.ordinal = ordinal
        self.share = share
        self.backbone_g, feat = build_backbone(backbone, pretrained=pretrained)
        if share:
            self.backbone_l = self.backbone_g
        else:
            self.backbone_l, _ = build_backbone(backbone, pretrained=pretrained)
        self.head = NeckHeads(feat * 2, num_classes, embed_dim, dropout, ordinal)

    @staticmethod
    def _feat(bb, x):
        f = bb(x)
        return f.flatten(1) if f.ndim > 2 else f

    def forward(self, inputs):
        g, l = inputs
        fg = self._feat(self.backbone_g, g)
        fl = self._feat(self.backbone_l, l)
        return self.head(torch.cat([fg, fl], dim=1))


def build_exp_model(exp, backbone, num_classes, *, embed_dim, dropout, ordinal,
                    pretrained=True, mil_pool="attention", twostream_share=True,
                    canet_edges=None, canet_reduction=16, canet_fuse="mean",
                    local_tasks=None, local_stages=(0, 1), local_dim=128,
                    local_width=64, local_fusion="gate", local_share=True, head_hidden=0):
    """exp 에 맞는 모델 생성(train.py 에서 호출).

    canet_edges 가 주어지면(=--canet) 단일이미지 exp(full/bbox)에서 CANetMultiTask 로 교체.
    local_tasks 가 주어지면(=--local_tasks) LocalAwareMultiTask 로 교체 — 지정 태스크만
    백본 저층 stage 의 multi-scale local 특징을 추가로 본다(model_local.py).
    """
    if head_hidden and (exp not in ("bbox", "full") or local_tasks or canet_edges is not None):
        raise ValueError("--head_hidden 은 단일 이미지 기본 모델(bbox/full, canet/local 제외)만 지원")
    if exp in ("bbox", "full"):
        if local_tasks:
            return LocalAwareMultiTask(backbone, num_classes, local_tasks,
                                       stages=local_stages, embed_dim=embed_dim,
                                       dropout=dropout, ordinal=ordinal,
                                       pretrained=pretrained, local_dim=local_dim,
                                       local_width=local_width, fusion=local_fusion,
                                       share_trunk=local_share)
        if canet_edges is not None:
            return CANetMultiTask(backbone, num_classes, canet_edges, embed_dim=embed_dim,
                                  dropout=dropout, ordinal=ordinal, pretrained=pretrained,
                                  reduction=canet_reduction, fuse=canet_fuse)
        return MultiTaskNet(backbone, num_classes, embed_dim=embed_dim,
                            dropout=dropout, ordinal=ordinal, pretrained=pretrained,
                            head_hidden=head_hidden)
    if exp == "mil":
        return MILNet(backbone, num_classes, embed_dim=embed_dim, dropout=dropout,
                      ordinal=ordinal, pretrained=pretrained, pool=mil_pool)
    if exp == "twostream":
        return TwoStreamNet(backbone, num_classes, embed_dim=embed_dim, dropout=dropout,
                            ordinal=ordinal, pretrained=pretrained, share=twostream_share)
    raise ValueError(f"unknown exp: {exp}")
