"""crop 기반 아토피 멀티태스크 분류 — 세 실험 통합 트레이너.

방법1(마스킹 전체이미지=atopy_seg)이 성능이 나빴던 것에 대한 대안 3종을 --exp 로 전환:
    bbox      : 마스크 union bbox + margin 크롭 1장(마스킹 X). 유효해상도↑, 대비 보존.
    mil       : 마스크 연결요소별 crop 여러 장(bag) -> attention/max pooling. polygon 다수 정석.
    twostream : 전체(global, 분포) + bbox 크롭(local, 텍스처) 두 스트림 concat.

학습 로직(CORN 순서형 loss, Kendall uncertainty MTL, QWK 선택기준, 차등 LR, early stop)은
classification/mobile/train.py 와 동일 — dataset/model 만 exp 별로 교체한다.

사전 준비: python3 precompute_masks.py  (atopy_crop_masks/ 생성)
직접 실행:
    python3 train.py --exp bbox --model effb0
    python3 train.py --exp mil --model effb0 --mil_pool attention
    python3 train.py --exp twostream --model effb0 --twostream_share
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent            # classification_crop
OGW = ROOT.parent
sys.path.insert(0, str(ROOT))

from dataset import (label_index_from_csv, build_source_index,      # noqa: E402
                     TASK_NAMES, NUM_CLASSES)
from metrics import per_task_metrics                                # noqa: E402
from crop_datasets import (BBoxCropDataset, MILBagDataset, TwoStreamDataset,  # noqa: E402
                           collate_single, collate_mil, collate_twostream)
from model import build_exp_model                                   # noqa: E402
from canet import AUX_SUFFIX                                        # noqa: E402
from sampler import build_ibb_sampler, labels_from_dataset, select_edges  # noqa: E402
from copula import (CopulaPairwise, cdf_from_corn, node_loss, joint_decode,  # noqa: E402
                    CUT)

MODELS = {
    "efflite0": "tf_efficientnet_lite0",
    "effb0":    "efficientnet_b0",
    "mnv3s":    "mobilenetv3_small_100",
    "mnv4s":    "mobilenetv4_conv_small",
    "mnv4m":    "mobilenetv4_conv_medium",
    "pvtv2b0":  "pvt_v2_b0",
    "r50":      "resnet50",          # CANet 저자 백본
    "dn121":    "densenet121",
}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description="crop 기반 아토피 멀티태스크 분류")
    p.add_argument("--exp", choices=["full", "bbox", "mil", "twostream"], required=True,
                   help="full=원본 이미지 전체(마스크 미사용) / bbox=마스크 union bbox 크롭")
    p.add_argument("--model", choices=list(MODELS), default="effb0")
    p.add_argument("--data", default=str(OGW / "atopy_face"),
                   help="원본 이미지 루트({train,val,test}/*.png + labels.csv)")
    p.add_argument("--masks", default=str(OGW / "atopy_crop_masks"),
                   help="precompute_masks.py 산출 마스크 루트")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--head_hidden", type=int, default=0,
                   help="0=head 가 Linear 1층(기본) / >0=Linear-GELU-Dropout-Linear MLP head 은닉 차원")
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--iga_weight", type=float, default=0.5)
    p.add_argument("--loss", choices=["ce", "corn"], default="corn")
    p.add_argument("--mtl", choices=["fixed", "uncertainty"], default="uncertainty")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    # --- crop 전용 ---
    p.add_argument("--margin", type=float, default=0.15, help="bbox/crop 여유(비율)")
    p.add_argument("--min_area_frac", type=float, default=0.003,
                   help="(mil) 이미지 면적 대비 이보다 작은 연결요소 제외")
    p.add_argument("--max_instances", type=int, default=8, help="(mil) bag 최대 instance 수")
    p.add_argument("--mil_pool", choices=["attention", "max"], default="attention")
    p.add_argument("--twostream_share", action="store_true", default=False,
                   help="(twostream) 두 스트림 백본 공유(기본: 독립)")
    # --- 2-step IBB (sampler.py) ---
    p.add_argument("--ibb", action="store_true",
                   help="2-step IBB 샘플링(Step1 등급균등 + Step2 쌍-공존균등)")
    p.add_argument("--ibb_alpha", type=float, default=0.5,
                   help="역빈도 완화 지수(0=자연분포, 0.5=sqrt, 1=완전균등)")
    p.add_argument("--ibb_alpha_end", type=float, default=None,
                   help="설정 시 alpha 를 epochs 동안 이 값까지 선형 어닐(0 이면 말미에 자연분포 복귀)")
    p.add_argument("--ibb_cap", type=float, default=4.0, help="샘플별 기대 반복배수 상한")
    p.add_argument("--ibb_min_cell", type=int, default=5,
                   help="(Step2) 이 미만인 등급조합 셀은 제외")
    p.add_argument("--ibb_edge_v", type=float, default=0.25,
                   help="(Step2) Cramer's V 가 이 값 이상인 태스크 쌍만 엣지로 채택")
    p.add_argument("--ibb_step2_ratio", type=float, default=0.5,
                   help="Step2 배치 수 / 자연분포 에폭 배치 수")
    p.add_argument("--ibb_step1_mult", type=float, default=1.0,
                   help="Step1 배치 수 배율. 1.0=태스크당 1/5 에폭치, 5.0=태스크당 1 에폭치(비용 5배)")
    p.add_argument("--ibb_edge_loss", choices=["pair", "all"], default="pair",
                   help="(Step2) 쌍에 속한 두 태스크만 역전파 vs 전체 태스크")
    p.add_argument("--ibb_warmup", type=int, default=3,
                   help="앞 N 에폭은 자연분포로 학습(백본 W 워밍업 — 논문 Step1 제거 보완)")
    # --- A안: 코퓰러 pairwise 항 (copula.py) ---
    # --- CANet(교차 태스크 어텐션, canet.py) ---
    # --- 찰상/태선화 전용 multi-scale local branch (model_local.py) ---
    p.add_argument("--local_tasks", default="",
                   help="쉼표구분. 예: excoriation,lichenification (빈 값=off)")
    p.add_argument("--local_stages", default="0,1",
                   help="쓸 백본 stage 인덱스. pvtv2b0: 0=1/4(128px@512) 1=1/8 2=1/16 3=1/32")
    p.add_argument("--local_dim", type=int, default=128, help="local 특징 차원")
    p.add_argument("--local_width", type=int, default=64, help="stage 정렬 채널")
    p.add_argument("--local_fusion", choices=["gate", "concat", "none"], default="gate",
                   help="gate=adaptive local-global, concat=단순결합(대조군), none=local 끔")
    p.add_argument("--local_share", type=int, default=1,
                   help="1=local 태스크들이 trunk 공유(실데이터 찰상-태선화 상관 0.377 근거)")
    # --- 순서형 임계값 불균형 보정 ---
    p.add_argument("--thr_init", choices=["none", "bias", "exact"], default="none",
                   help="CORN 헤드 임계값을 train 주변분포로 초기화. "
                        "bias=마지막 층 bias 만, exact=weight 도 0 으로 둬 초기출력이 주변분포와 일치")
    p.add_argument("--mono_lambda", type=float, default=0.0,
                   help="iga >= max(징후) soft 제약 가중(0=off). train 99.29% 성립(위반 10행)")
    p.add_argument("--corn_pos_weight", choices=["none", "auto"], default="none",
                   help="auto: CORN 레벨별 pos_weight=neg/pos (찰상 P(y>2)는 positive 113개뿐)")
    p.add_argument("--canet", action="store_true",
                   help="CANet crossCBAM 이식(full/bbox 전용). 태스크별 CBAM + 교차 채널 어텐션")
    p.add_argument("--canet_lambda", type=float, default=0.25,
                   help="specific(보조) 헤드 손실 가중. 저자 baseline.py lambda_value 기본 0.25")
    p.add_argument("--canet_edge_v", type=float, default=0.25,
                   help="교차 어텐션을 붙일 태스크쌍의 Cramér's V 하한(IBB 와 같은 기준)")
    p.add_argument("--canet_reduction", type=int, default=16,
                   help="CBAM 채널 MLP 축소비(저자 reduction_ratio 기본 16)")
    p.add_argument("--canet_fuse", choices=["mean", "sum"], default="mean",
                   help="이웃 2개 이상일 때 주입 합성. 이웃 1개면 둘이 동일(=저자 동작)")

    p.add_argument("--pairwise", action="store_true",
                   help="엣지별 Frank 코퓰러 theta 를 추가(논문 CCNN). --loss corn 필요")
    p.add_argument("--pairwise_sources", choices=["none", "angle"], default="none",
                   help="논문 Eq(12): unary 는 공유하고 theta 만 출처별로 둔다. "
                        "angle=정면/측면(합성데이터의 맥락 축)")
    p.add_argument("--atopy_root", default="../atopy",
                   help="--pairwise_sources 가 출처를 역추적할 원본 폴더")
    p.add_argument("--pairwise_w_nodes", type=float, default=0.1,
                   help="저자 COR.w_nodes: loss = w*노드 + (1-w)*엣지. 저자 기본 0.1")
    p.add_argument("--pairwise_unary", choices=["nll", "corn"], default="nll",
                   help="nll=저자 node_potn(-log P(y=l)) / corn=기존 CORN loss")
    p.add_argument("--pairwise_cut", type=float, default=CUT,
                   help="저자 frank(cut): theta=(sigmoid(raw)-0.5)*cut -> +-cut/2")
    p.add_argument("--pairwise_shared", type=int, default=1,
                   help="저자 shared_copula. 1=엣지당 theta 스칼라, 0=엣지당 (Kr,Ks) 행렬")
    p.add_argument("--pairwise_copula", choices=["frank", "indep"], default="frank")
    p.add_argument("--pairwise_theta_lr", type=float, default=0.02,
                   help="theta 전용 lr. 주 lr(3e-4)로는 한 런에 theta 가 0.5 도 못 움직인다")
    p.add_argument("--pairwise_decode", choices=["joint", "marginal"], default="joint",
                   help="joint=1280 조합 전수 열거 MAP(논문 AD3 를 정확 추론으로 대체)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", default=None)
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_transforms(imgsz):
    train_tf = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    eval_tf = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    return train_tf, eval_tf


def make_dataset(args, split, tf):
    img_dir = Path(args.data) / split
    mask_dir = Path(args.masks) / split
    index = label_index_from_csv(Path(args.data) / "labels.csv")
    if args.pairwise_sources != "none":
        src_idx, src_names = build_source_index(args.atopy_root, args.pairwise_sources)
        miss = [k for k in index if k not in src_idx]
        if miss:
            raise SystemExit(f"--pairwise_sources {args.pairwise_sources}: 출처를 못 찾은 "
                             f"stem {len(miss)}개(예: {miss[:3]}). --atopy_root 를 확인할 것.")
        for k, v in index.items():
            v["_src"] = src_idx[k]
        print(f"[sources] {args.pairwise_sources}: " + ", ".join(
            f"{n}={sum(1 for k in index if src_idx[k] == i)}" for i, n in enumerate(src_names)))
    if args.exp == "full":
        # 마스크를 주지 않으면 _bbox_from_mask 가 None -> 원본 이미지 전체가 그대로 나온다.
        return BBoxCropDataset(img_dir, None, index, transform=tf, margin=args.margin)
    if args.exp == "bbox":
        return BBoxCropDataset(img_dir, mask_dir, index, transform=tf, margin=args.margin)
    if args.exp == "mil":
        return MILBagDataset(img_dir, mask_dir, index, transform=tf, margin=args.margin,
                             min_area_frac=args.min_area_frac, max_instances=args.max_instances)
    return TwoStreamDataset(img_dir, mask_dir, index, transform=tf, margin=args.margin)


COLLATE = {"full": collate_single, "bbox": collate_single,
           "mil": collate_mil, "twostream": collate_twostream}


def build_loaders(args):
    train_tf, eval_tf = build_transforms(args.imgsz)
    tr = make_dataset(args, "train", train_tf)
    va = make_dataset(args, "val", eval_tf)
    te = make_dataset(args, "test", eval_tf)
    collate = COLLATE[args.exp]
    # MIL 은 bag 크기가 가변이라 batch 당 총 instance 수가 달라짐 -> drop_last 로 안정화.

    def dl(ds, sh):
        return DataLoader(ds, batch_size=args.batch, shuffle=sh, num_workers=args.workers,
                          pin_memory=True, drop_last=sh, collate_fn=collate)

    edges = None
    if args.pairwise:
        # IBB Step2 가 theta_rs 를 갱신하므로 엣지 집합은 반드시 하나여야 한다.
        edges, _ = select_edges(labels_from_dataset(tr), args.ibb_edge_v)
    elif args.canet:
        # CANet 교차 어텐션을 붙일 태스크쌍(같은 Cramér's V 기준, 임계값만 별도 노브).
        edges, _ = select_edges(labels_from_dataset(tr), args.canet_edge_v)

    ibb_ld = ibb_sampler = None
    if args.ibb:
        ibb_sampler, _ = build_ibb_sampler(
            labels_from_dataset(tr), args.batch, alpha=args.ibb_alpha, cap=args.ibb_cap,
            min_cell=args.ibb_min_cell, edge_v=args.ibb_edge_v,
            step2_ratio=args.ibb_step2_ratio, step1_mult=args.ibb_step1_mult,
            edge_loss=args.ibb_edge_loss, seed=args.seed)
        # sampler 가 배치 경계에 맞춰 정확한 개수의 인덱스를 내므로 drop_last 불필요.
        ibb_ld = DataLoader(tr, batch_size=args.batch, sampler=ibb_sampler,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate)
    return tr, dl(tr, True), dl(va, False), dl(te, False), ibb_ld, ibb_sampler, edges


# ---- 입력 텐서 디바이스 이동 / 모델 호출 (exp 별 입력 구조 흡수) -----------------
def to_device(inputs, device):
    # pin_memory=True 는 plain tuple 을 list 로 바꾼다(torch<=2.2). mil/twostream 입력 대비.
    if isinstance(inputs, (tuple, list)):
        return tuple(t.to(device) for t in inputs)
    return inputs.to(device)


def forward_model(model, inputs):
    """bbox: Tensor -> forward(x). mil/twostream: tuple -> forward((..))."""
    return model(inputs)


# =============================================================================
# 아래 loss/metric/train 루프는 classification/mobile/train.py 와 동일(입력 처리만 위임).
# =============================================================================
def corn_bias_init(counts):
    """CORN 레벨별 초기 bias = logit P(y>i | y>i-1).

    주의: 누적형(GRM) 파라미터화의 -logit P(y>j) 와 **다르다**. CORN 의 로짓 i 는
    부분집합 {y > i-1} 안에서의 조건부 확률을 모델링하므로 분모가 전체가 아니다.
    그대로 GRM 식을 쓰면 상위 임계값이 크게 어긋난다.

    초기화 없이는 모든 임계값이 P=0.5 에서 출발하는데, 실제 주변분포는 예컨대
    lichenification P(y>2)=0.019 다. 초반 에폭이 이 간극을 메우는 데 낭비된다.
    """
    c = np.asarray(counts, dtype=np.float64)
    out = []
    for i in range(len(c) - 1):
        denom = c[i:].sum()                       # y > i-1 인 표본(= 이 레벨의 학습 대상)
        num = c[i + 1:].sum()                     # 그 중 y > i
        p = float(np.clip(num / max(denom, 1e-9), 1e-4, 1 - 1e-4))
        out.append(float(np.log(p / (1 - p))))
    return out


def init_corn_thresholds(model, counts, mode):
    """모델의 태스크별 마지막 Linear 헤드에 임계값 초기값을 심는다.

    exact 면 weight 도 0 으로 둔다 — 그래야 f=0 일 때 초기 출력이 주변분포와 정확히
    일치한다(마지막 층 zero-init). 가중치 gradient 는 입력이 0 이 아니므로 살아 있다.
    """
    banks = [getattr(model, n) for n in ("heads", "head_spec", "head_joint")
             if hasattr(model, n)]
    if not banks:
        print("[thr_init] 경고: 헤드 ModuleDict 을 찾지 못해 초기화를 건너뛴다.")
        return None
    inits = {t: corn_bias_init(counts[t]) for t in TASK_NAMES}
    n_set = 0
    for bank in banks:
        for t, head in bank.items():
            b = inits.get(t)
            if isinstance(head, nn.Sequential):      # MLP head: 마지막 Linear 에 심는다
                head = head[-1]
            if b is None or not isinstance(head, nn.Linear) or head.out_features != len(b):
                continue
            with torch.no_grad():
                head.bias.copy_(torch.tensor(b, dtype=head.bias.dtype))
                if mode == "exact":
                    head.weight.zero_()
            n_set += 1
    # 검증: 초기 출력이 실제로 주변분포를 재현하는지(exact 일 때만 정확히 일치)
    print(f"[thr_init] {mode}: 헤드 {n_set}개 초기화")
    for t in TASK_NAMES:
        p = 1.0 / (1.0 + np.exp(-np.asarray(inits[t])))       # 조건부 P(y>i | y>i-1)
        cum = np.cumprod(p)                                    # P(y>i)
        c = np.asarray(counts[t], dtype=np.float64)
        tgt = np.array([c[i + 1:].sum() / c.sum() for i in range(len(c) - 1)])
        print(f"  {t:<16} out=[" + " ".join(f"{x:.4f}" for x in cum) + "]  "
              f"target=[" + " ".join(f"{x:.4f}" for x in tgt) + "]")
    return inits


def mono_penalty(out, loss_type):
    """iga >= max(징후) soft 제약. 기대등급 E[y]=sum_i P(y>i) 로 비교한다.

    train 에서 99.29% 성립(위반 10행)하는, 이 5축에서 **논리적으로 유도되는 유일한**
    제약이다. 관측 안 된 등급조합을 막는 마스킹과는 다르다 — 그건 1400장에 1280칸이라
    '관측 안 됨 = 불가능'의 근거가 없어서 하면 안 된다(관측 237칸, 그중 82칸이 1회).
    """
    if loss_type == "corn":
        E = {t: torch.cumprod(torch.sigmoid(out[t].float()), dim=1).sum(1) for t in TASK_NAMES}
    else:
        E = {t: (torch.softmax(out[t].float(), 1)
                 * torch.arange(out[t].shape[1], device=out[t].device)).sum(1)
             for t in TASK_NAMES}
    signs = torch.stack([E[t] for t in TASK_NAMES if t != "severity"], 1).max(1).values
    return torch.relu(signs - E["severity"]).mean()


def corn_pos_weights(counts):
    """CORN 레벨별 pos_weight = neg/pos. counts 는 train 등급 히스토그램.

    CORN 은 레벨 i 를 부분집합 {y > i-1} 에서만 학습하므로 상위 레벨일수록 표본이 급감한다.
    찰상(dataset_all_final train): 레벨0 = 1219/1800(68%), 레벨1 = 487/1219(40%),
    레벨2 = 113/487(23%) -> 마지막 임계값만 pos_weight 3.3 이 필요하다.
    (별도 presence 헤드를 다는 것과 다르다 — 레벨0 이 곧 presence 이고 이미 전 표본으로
    학습되고 있다. 비어 있는 자리는 맨 위 임계값이다.)"""
    c = np.asarray(counts, dtype=np.float64)
    w = []
    for i in range(len(c) - 1):
        pos = c[i + 1:].sum()                 # y > i
        neg = c[i]                            # y == i (부분집합 {y > i-1} 안의 음성)
        w.append(float(neg / pos) if pos > 0 else 1.0)
    return w


def corn_loss(logits, targets, num_classes, pos_weight=None):
    total = logits.new_zeros(())
    n = 0
    for i in range(num_classes - 1):
        mask = targets > (i - 1)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        lvl_logit = logits[mask, i]
        lvl_label = (targets[mask] > i).float()
        pw = None if pos_weight is None else lvl_logit.new_tensor(pos_weight[i])
        total = total + F.binary_cross_entropy_with_logits(
            lvl_logit, lvl_label, reduction="sum", pos_weight=pw)
        n += cnt
    return total / max(n, 1)


def corn_predict(logits):
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (cum > 0.5).sum(dim=1)


class UncertaintyWeighter(nn.Module):
    def __init__(self, task_names):
        super().__init__()
        self.log_var = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in task_names})

    def forward(self, losses):
        total = 0.0
        for t, l in losses.items():
            s = self.log_var[t]
            total = total + 0.5 * torch.exp(-s) * l + 0.5 * s
        return total

    def sigmas(self):
        return {t: float(torch.exp(0.5 * s).item()) for t, s in self.log_var.items()}


def make_criterions(train_ds, use_weight, smoothing, device):
    counts = train_ds.class_counts() if hasattr(train_ds, "class_counts") else _counts(train_ds)
    crits = {}
    for t in TASK_NAMES:
        w = None
        if use_weight:
            c = torch.tensor(counts[t], dtype=torch.float)
            w = (c.sum() / (c + 1e-6))
            w = (w / w.sum() * len(c)).to(device)
        crits[t] = nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing)
    return crits


def _counts(ds):
    """crop 데이터셋은 samples=(img,mask,labels) 형태 -> 클래스 빈도 집계."""
    counts = {t: [0] * NUM_CLASSES[t] for t in TASK_NAMES}
    for _, _, lb in ds.samples:
        for t, idx in lb.items():
            if t in counts:                     # '_src'(다중출처 인덱스)는 등급이 아니다
                counts[t][idx] += 1
    return counts


CORN_PW = {}          # {task: [레벨별 pos_weight]} — main 에서 --corn_pos_weight auto 일 때 채운다


def _one_loss(out, labels, crits, loss_type, t, key):
    if loss_type == "corn":
        return corn_loss(out[key], labels[t], NUM_CLASSES[t], pos_weight=CORN_PW.get(t))
    return crits[t](out[key], labels[t])


def head_losses(out, labels, crits, loss_type, tasks=None, aux_lambda=0.0):
    """태스크별 손실. CANet 이면 joint 헤드 손실 + aux_lambda * specific 헤드 손실.

    저자 baseline.py:382 `loss1 + loss2 + λ(loss3 + loss4)` 을 태스크별로 묶은 형태 —
    이렇게 묶어야 손실 dict 이 5개로 유지돼 UncertaintyWeighter(태스크 5개로 keyed)와
    IBB 스텝별 부분갱신이 그대로 동작한다.
    """
    d = {}
    for t in (tasks or TASK_NAMES):
        l = _one_loss(out, labels, crits, loss_type, t, t)
        aux = t + AUX_SUFFIX
        if aux_lambda > 0.0 and aux in out:
            l = l + aux_lambda * _one_loss(out, labels, crits, loss_type, t, aux)
        d[t] = l
    return d


def combine_losses(losses, mtl, weighter, iga_weight):
    if mtl == "uncertainty":
        return weighter(losses)          # losses 가 부분집합이어도 동작(IBB 스텝별 갱신)
    sev = losses.get("severity")
    sym = [losses[t] for t in losses if t != "severity"]
    if sev is None:
        return sum(sym) / len(sym)
    if not sym:
        return sev
    return iga_weight * sev + (1.0 - iga_weight) * (sum(sym) / len(sym))


def head_predict(out, loss_type, pw=None, cdfs=None, decode="joint", w_nodes=0.1, src=None):
    """pw 가 있고 decode=joint 면 composite likelihood 상의 정확 MAP(전수 열거).

    src 가 있으면(다중출처 theta) 표본마다 자기 출처의 theta 로 디코딩한다 — 추론 시에도
    맥락별 의존구조를 써야 학습과 일관된다."""
    if pw is not None and decode == "joint":
        return joint_decode(cdfs, pw, NUM_CLASSES, TASK_NAMES, w_nodes, src=src)
    return {t: (corn_predict(out[t]) if loss_type == "corn" else out[t].argmax(1))
            for t in TASK_NAMES}


def run_epoch(model, loader, crits, device, optimizer=None, use_amp=False,
              iga_weight=0.5, loss_type="ce", mtl="fixed", weighter=None,
              pw=None, pw_wn=0.1, pw_unary="nll", pw_decode="joint", aux_lambda=0.0,
              mono_lambda=0.0):
    train = optimizer is not None
    model.train(train)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    tot, n = 0.0, 0
    preds = {t: [] for t in TASK_NAMES}
    trues = {t: [] for t in TASK_NAMES}
    for inputs, labels in loader:
        inputs = to_device(inputs, device)
        labels = {t: v.to(device) for t, v in labels.items()}
        src = labels.pop("_src", None)              # 다중출처 theta 인덱스(없으면 None)
        bs = labels[TASK_NAMES[0]].size(0)
        cdfs = None
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = forward_model(model, inputs)
            if pw is None:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    losses = head_losses(out, labels, crits, loss_type, aux_lambda=aux_lambda)
                    loss = combine_losses(losses, mtl, weighter, iga_weight)
            else:
                # 저자 COR._loss: w_nodes*mean(NCLL) + (1-w_nodes)*mean(NCJLL). fp32 고정.
                cdfs = {t: cdf_from_corn(out[t].float()) for t in TASK_NAMES}
                losses = (node_loss(cdfs, labels, TASK_NAMES) if pw_unary == "nll"
                          else head_losses(out, labels, crits, loss_type,
                                           aux_lambda=aux_lambda))
                loss = (pw_wn * combine_losses(losses, mtl, weighter, iga_weight)
                        + (1.0 - pw_wn) * pw.edge_loss(cdfs, labels, src=src))
            if mono_lambda > 0:
                loss = loss + mono_lambda * mono_penalty(out, loss_type)
        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        tot += loss.item() * bs; n += bs
        with torch.no_grad():
            pred = head_predict(out, loss_type, pw, cdfs, pw_decode, pw_wn, src=src)
        for t in TASK_NAMES:
            preds[t] += pred[t].cpu().tolist()
            trues[t] += labels[t].cpu().tolist()
    metrics = {t: per_task_metrics(trues[t], preds[t], NUM_CLASSES[t]) for t in TASK_NAMES}
    return tot / max(n, 1), metrics


def run_ibb_epoch(model, loader, sampler, crits, device, optimizer, use_amp=False,
                  iga_weight=0.5, loss_type="ce", mtl="fixed", weighter=None,
                  pw=None, pw_wn=0.1, pw_unary="nll", pw_decode="joint", opt_theta=None,
                  aux_lambda=0.0, mono_lambda=0.0):
    """IBB 학습 에폭 — 배치마다 sampler 가 정한 스텝/태스크의 loss 만 역전파한다.

    Step1(lv:*) 배치는 그 태스크 head 하나만 갱신한다(논문 'forall q: phi^q').
    Step2(co:*) 배치는:
      - A안(--pairwise): 주변분포를 detach 하고 그 엣지의 theta_rs '만' 갱신한다.
        논문 'forall (rs): theta^rs' 그대로 — B안에서 이 스텝이 공유 trunk 를 건드려
        조인트 층화 샘플링에 그쳤던 부분이 여기서 해소된다.
      - B안(--pairwise 없음): 그 엣지 두 태스크의 unary loss 를 역전파한다(근사).
    train metric 은 리샘플된 분포라 의미가 없어 loss 만 집계한다.
    """
    model.train(True)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # batch_phases 는 DataLoader 가 sampler 를 소비하는 시점(=iter(loader))에 채워진다.
    it = iter(loader)
    phases = list(sampler.batch_phases)
    if len(phases) != sampler.n_batches:
        raise RuntimeError(f"IBB 스케줄 불일치: {len(phases)} != {sampler.n_batches}")
    tot, n, stat = 0.0, 0, {}
    for bi, (inputs, labels) in enumerate(it):
        name, tasks = phases[bi]
        inputs = to_device(inputs, device)
        labels = {t: v.to(device) for t, v in labels.items()}
        src = labels.pop("_src", None)              # 다중출처 theta 인덱스(없으면 None)
        bs = labels[TASK_NAMES[0]].size(0)
        step2_theta = pw is not None and name.startswith("co:")
        if step2_theta:
            # 주변분포(=trunk+head) 고정, theta 만. forward 는 grad 불필요.
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                out = forward_model(model, inputs)
            cdfs = {t: cdf_from_corn(out[t].float()) for t in TASK_NAMES}
            eid = [pw.edge_index(*tasks[:2])]
            loss = (1.0 - pw_wn) * pw.edge_loss(cdfs, labels, edge_ids=eid,
                                                detach_marginals=True, src=src)
            opt_theta.zero_grad()
            loss.backward()
            opt_theta.step()
        else:
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = forward_model(model, inputs)
            if pw is not None and pw_unary == "nll":     # 저자 node_potn 을 unary 로
                cdfs = {t: cdf_from_corn(out[t].float()) for t in tasks}
                losses = node_loss(cdfs, labels, tasks)
            else:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    losses = head_losses(out, labels, crits, loss_type, tasks=tasks,
                                         aux_lambda=aux_lambda)
            loss = combine_losses(losses, mtl, weighter, iga_weight)
            if pw is not None:
                loss = pw_wn * loss
            if mono_lambda > 0:
                # 제약이라 페이즈와 무관하게 전 헤드에 건다. IBB 의 '이 배치는 이 헤드만'
                # 성질을 조금 완화하지만, 태스크 손실이 아니라 구조 제약이라 그게 맞다.
                loss = loss + mono_lambda * mono_penalty(out, loss_type)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        tot += loss.item() * bs; n += bs
        a = stat.setdefault(name, [0.0, 0]); a[0] += loss.item(); a[1] += 1
    return tot / max(n, 1), {k: v[0] / v[1] for k, v in stat.items()}


def fmt(metrics, key):
    return " ".join(f"{t[:4]}={metrics[t][key]:.3f}" for t in TASK_NAMES)


def sel_score(m):
    sev = m["severity"]["qwk"]
    sym = float(np.mean([m[t]["qwk"] for t in TASK_NAMES if t != "severity"]))
    return 0.5 * sev + 0.5 * sym


def collect_gate_stats(model, loader, device, use_amp):
    """test set 전체의 alpha 를 모아 태스크별 분포를 낸다(배치 단위 last_alpha 누적).
    cls_mbn 의 CFEN attention 이 균등분포로 붕괴했던 전례가 있어 같은 방식으로 감시한다."""
    if not hasattr(model, "gate_stats"):
        return None
    model.eval()
    acc = {}
    with torch.no_grad():
        for batch in loader:
            img = batch[0].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(img)
            if model.fusion is None:
                return None
            for t, f in model.fusion.items():
                if getattr(f, "last_alpha", None) is not None:
                    acc.setdefault(t, []).append(f.last_alpha.float().flatten().cpu())
    if not acc:
        return None
    out = {}
    for t, xs in acc.items():
        a = torch.cat(xs)
        out[t] = dict(mean=float(a.mean()), std=float(a.std()),
                      p05=float(a.quantile(0.05)), p95=float(a.quantile(0.95)))
    return out


def backbone_param_ids(model):
    """exp 무관하게 백본 파라미터 id 집합(이름에 'backbone' 포함)."""
    return {id(p) for name, p in model.named_parameters() if "backbone" in name}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 인자 검증을 out_dir 생성보다 먼저 — 거절된 조합이 빈 런 폴더를 남기지 않게.
    if args.pairwise and args.loss != "corn":
        raise SystemExit("--pairwise 는 순서형 CDF 가 필요하다 -> --loss corn 과 함께 써야 한다.")
    if args.canet and args.exp not in ("full", "bbox"):
        raise SystemExit("--canet 은 단일 이미지 입력 전용이다 -> --exp full 또는 bbox.")
    if args.canet and args.pairwise:
        # CCNN 경로는 unary 를 node_loss(cdfs) 로 계산해 specific 헤드를 무시한다.
        # 조합하려면 node_loss 쪽도 이중 헤드로 고쳐야 하므로 명시적으로 막는다.
        raise SystemExit("--canet 과 --pairwise 는 함께 쓸 수 없다(보조헤드 손실이 무시됨).")

    model_name = MODELS[args.model]
    name = args.name or f"crop_{args.exp}_{args.model}" + ("_canet" if args.canet else "")
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_ds, tr_ld, va_ld, te_ld, ibb_ld, ibb_sampler, edges = build_loaders(args)
    ordinal = args.loss == "corn"
    local_tasks = [t.strip() for t in args.local_tasks.split(",") if t.strip()]
    if local_tasks and args.local_fusion == "none":
        local_tasks = []
    for t in local_tasks:
        if t not in NUM_CLASSES:
            raise SystemExit(f"--local_tasks 에 없는 태스크: {t} (가능: {', '.join(TASK_NAMES)})")
    local_stages = tuple(int(i) for i in args.local_stages.split(",") if i.strip() != "")
    model = build_exp_model(args.exp, model_name, NUM_CLASSES, embed_dim=args.embed_dim,
                            dropout=args.dropout, ordinal=ordinal, pretrained=True,
                            mil_pool=args.mil_pool, twostream_share=args.twostream_share,
                            canet_edges=edges if args.canet else None,
                            canet_reduction=args.canet_reduction,
                            canet_fuse=args.canet_fuse,
                            local_tasks=local_tasks, local_stages=local_stages,
                            local_dim=args.local_dim, local_width=args.local_width,
                            local_fusion=args.local_fusion,
                            local_share=bool(args.local_share),
                            head_hidden=args.head_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[exp] {args.exp}  [model] {args.model} ({model_name})  params={n_params:.1f}M")
    if args.canet:
        print(f"[canet] 엣지 {len(edges)}개(V>={args.canet_edge_v}): "
              + ", ".join(f"{a[:4]}-{b[:4]}({v:.2f})" for a, b, v in edges))
        print(f"[canet] 교차 주입: {model.edge_desc()}  "
              f"lambda={args.canet_lambda} fuse={args.canet_fuse} r={args.canet_reduction}")
    if local_tasks:
        print(f"[local] {model.describe().strip()}")
    if args.thr_init != "none":
        if not ordinal:
            raise SystemExit("--thr_init 은 --loss corn 전용이다.")
        _c = tr_ds.class_counts() if hasattr(tr_ds, "class_counts") else _counts(tr_ds)
        init_corn_thresholds(model, _c, args.thr_init)
    if args.corn_pos_weight == "auto":
        if not ordinal:
            raise SystemExit("--corn_pos_weight 는 --loss corn 전용이다.")
        counts = tr_ds.class_counts() if hasattr(tr_ds, "class_counts") else _counts(tr_ds)
        for t in TASK_NAMES:
            CORN_PW[t] = corn_pos_weights(counts[t])
        print("[corn_pw] 레벨별 pos_weight(neg/pos): "
              + " | ".join(f"{t[:4]} " + ",".join(f"{w:.2f}" for w in CORN_PW[t])
                           for t in TASK_NAMES))

    crits = make_criterions(tr_ds, args.class_weight, args.label_smoothing, device)
    weighter = UncertaintyWeighter(TASK_NAMES).to(device) if args.mtl == "uncertainty" else None
    pw = opt_theta = None
    if args.pairwise:
        n_src, src_nm = 1, None
        if args.pairwise_sources != "none":
            _, src_nm = build_source_index(args.atopy_root, args.pairwise_sources)
            n_src = len(src_nm)
            if not args.pairwise_shared:
                raise SystemExit("--pairwise_sources 는 --pairwise_shared 1 에서만 쓸 수 있다.")
        pw = CopulaPairwise(edges, TASK_NAMES, NUM_CLASSES,
                            shared=bool(args.pairwise_shared), cut=args.pairwise_cut,
                            copula=args.pairwise_copula,
                            n_sources=n_src, source_names=src_nm).to(device)
        mode = "CCNN-IT(3-step)" if args.ibb else "CCNN(joint)"
        if args.ibb and args.ibb_step2_ratio < 1.0:
            print(f"[경고] --ibb_step2_ratio {args.ibb_step2_ratio} 은 엣지당 Step2 배치가 적어 "
                  f"theta 추정 분산이 크다(진동 확인됨). --pairwise 에는 1.0 권장.")
        print(f"[pairwise] {mode}  엣지 {len(pw.edges)}개: "
              + ", ".join(f"{a[:4]}-{b[:4]}" for a, b in pw.edges)
              + f"  w_nodes={args.pairwise_w_nodes} unary={args.pairwise_unary} "
                f"cut={args.pairwise_cut} shared={bool(args.pairwise_shared)} "
                f"theta_lr={args.pairwise_theta_lr} decode={args.pairwise_decode}")
    head_desc = "CORN(순서형)" if ordinal else f"CE{'(역빈도가중)' if args.class_weight else ''}"
    comb_desc = ("Kendall σ 자동가중" if args.mtl == "uncertainty"
                 else f"{args.iga_weight:.2f}*IGA + {1-args.iga_weight:.2f}*증상4평균")
    print(f"[loss] head={head_desc}  결합={comb_desc}  ls={args.label_smoothing}")

    bb_ids = backbone_param_ids(model)
    groups = [
        {"params": [p for p in model.parameters() if id(p) in bb_ids],
         "lr": args.lr * args.backbone_lr_scale},
        {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": args.lr},
    ]
    if weighter is not None:
        groups.append({"params": list(weighter.parameters()), "lr": args.lr, "weight_decay": 0.0})
    if pw is not None and not args.ibb:
        # CCNN(joint): theta 도 주 옵티마이저에서 함께 학습
        groups.append({"params": list(pw.parameters()),
                       "lr": args.pairwise_theta_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if pw is not None and args.ibb:
        # CCNN-IT: theta 는 Step2 전용 옵티마이저로만 갱신(논문 Alg.1 Step3)
        opt_theta = torch.optim.AdamW(pw.parameters(), lr=args.pairwise_theta_lr,
                                      weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"

    def save(path, vm, ep):
        torch.save({"model": model.state_dict(), "args": vars(args), "model_name": model_name,
                    "val_metrics": vm, "epoch": ep,
                    "weighter": weighter.state_dict() if weighter is not None else None,
                    "pairwise": pw.state_dict() if pw is not None else None,
                    "edges": pw.edges if pw is not None else None}, path)

    ep_kw = dict(loss_type=args.loss, mtl=args.mtl, weighter=weighter,
                 pw=pw, pw_wn=args.pairwise_w_nodes, pw_unary=args.pairwise_unary,
                 pw_decode=args.pairwise_decode,
                 aux_lambda=args.canet_lambda if args.canet else 0.0,
                 mono_lambda=args.mono_lambda)
    best, best_ep, no_improve = -1e9, 0, 0
    for ep in range(1, args.epochs + 1):
        use_ibb = args.ibb and ep > args.ibb_warmup
        phase_stat, tag = None, ""
        if use_ibb:
            if args.ibb_alpha_end is not None:
                span = max(1, args.epochs - args.ibb_warmup - 1)
                f = min(1.0, (ep - args.ibb_warmup - 1) / span)
                ibb_sampler.set_alpha(args.ibb_alpha + (args.ibb_alpha_end - args.ibb_alpha) * f)
            tag = f" IBB a={ibb_sampler.alpha:.2f}"
            tr_loss, phase_stat = run_ibb_epoch(model, ibb_ld, ibb_sampler, crits, device, opt,
                                                use_amp, args.iga_weight,
                                                opt_theta=opt_theta, **ep_kw)
        else:
            tag = " warmup" if args.ibb else ""
            tr_loss, _ = run_epoch(model, tr_ld, crits, device, opt, use_amp,
                                   args.iga_weight, **ep_kw)
        va_loss, vm = run_epoch(model, va_ld, crits, device, None, use_amp, args.iga_weight, **ep_kw)
        sched.step()
        score = sel_score(vm)
        mean_qwk = float(np.mean([vm[t]["qwk"] for t in TASK_NAMES]))
        print(f"[Ep {ep:>3}/{args.epochs}{tag}] tr={tr_loss:.3f} va={va_loss:.3f} "
              f"| score={score:.3f} (mean_QWK={mean_qwk:.3f})")
        if phase_stat:
            s1 = {k[3:]: v for k, v in phase_stat.items() if k.startswith("lv:")}
            s2 = {k[3:]: v for k, v in phase_stat.items() if k.startswith("co:")}
            print("    S1 : " + " ".join(f"{k[:4]}={v:.3f}" for k, v in s1.items()))
            if s2:
                print("    S2 : " + " ".join(f"{k}={v:.3f}" for k, v in s2.items()))
        print(f"    QWK: {fmt(vm,'qwk')}")
        print(f"    ±1 : {fmt(vm,'acc1')}   acc: {fmt(vm,'acc')}", flush=True)
        if pw is not None:
            if pw.n_sources > 1:
                for si, nm in enumerate(pw.source_names):
                    print(f"    th[{nm}]: " + " ".join(
                        f"{a[:4]}-{b[:4]}={v:+.2f}"
                        for (a, b), v in zip(pw.edges, pw.thetas(si))), flush=True)
            else:
                print("    th : " + " ".join(f"{a[:4]}-{b[:4]}={v:+.2f}"
                                             for (a, b), v in zip(pw.edges, pw.thetas())),
                      flush=True)
        if weighter is not None:
            sg = weighter.sigmas()
            print(f"    σ  : " + " ".join(f"{t[:4]}={sg[t]:.2f}" for t in TASK_NAMES), flush=True)
        save(out_dir / "last.pt", vm, ep)
        if score > best + args.min_delta:
            best, best_ep, no_improve = score, ep, 0
            save(out_dir / "best.pt", vm, ep)
            print(f"    -> best 갱신 (score={best:.3f})", flush=True)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[early stop] val score {args.patience}ep 개선 없음 "
                      f"(best ep{best_ep}={best:.3f}) -> ep{ep} 중단", flush=True)
                break

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    if pw is not None and ckpt.get("pairwise") is not None:
        pw.load_state_dict(ckpt["pairwise"])
    _, tm = run_epoch(model, te_ld, crits, device, None, use_amp, args.iga_weight, **ep_kw)
    test_score = sel_score(tm)
    test_qwk = float(np.mean([tm[t]["qwk"] for t in TASK_NAMES]))
    print(f"\n[TEST] (best ep{ckpt['epoch']})  score={test_score:.3f}  mean_QWK={test_qwk:.3f}")
    print(f"    QWK: {fmt(tm,'qwk')}")
    print(f"    ±1 : {fmt(tm,'acc1')}   acc: {fmt(tm,'acc')}")

    gate = collect_gate_stats(model, te_ld, device, use_amp) if local_tasks else None
    if gate:
        print("\n[local gate] alpha 분포(test). alpha=local 을 믿는 정도.")
        print("  std 가 0 에 가까우면 이미지를 구분 못하고 고정 혼합비로 굳은 것(=단순 평균),")
        print("  p05/p95 가 0/1 에 붙으면 한쪽을 통째로 버린 것. 퍼져야 adaptive 가 성립한다.")
        for t, v in gate.items():
            print(f"  {t:<16} mean={v['mean']:.3f} std={v['std']:.4f} "
                  f"p05={v['p05']:.3f} p95={v['p95']:.3f}")
    torch.save({**ckpt, "test_metrics": tm, "gate_stats": gate}, out_dir / "best.pt")
    (out_dir / "done.txt").write_text(
        f"exp={args.exp} model={args.model} imgsz={args.imgsz} epochs={args.epochs} "
        f"loss={args.loss} mtl={args.mtl} margin={args.margin} "
        f"ibb={int(args.ibb)}"
        + (f" local={'+'.join(local_tasks)} local_stages={args.local_stages} "
           f"local_fusion={args.local_fusion} local_dim={args.local_dim} "
           f"local_share={args.local_share}" if local_tasks else "")
        + (f" corn_pw={args.corn_pos_weight}" if args.corn_pos_weight != "none" else "")
        + (f" thr_init={args.thr_init}" if args.thr_init != "none" else "")
        + (f" mono_lambda={args.mono_lambda}" if args.mono_lambda > 0 else "")
        + (f" canet=1 lambda={args.canet_lambda} edge_v={args.canet_edge_v} "
           f"fuse={args.canet_fuse} reduction={args.canet_reduction}" if args.canet else "")
        + (f" ibb_alpha={args.ibb_alpha}->{args.ibb_alpha_end} cap={args.ibb_cap} "
           f"edge_v={args.ibb_edge_v} step2_ratio={args.ibb_step2_ratio} "
           f"step1_mult={args.ibb_step1_mult} "
           f"edge_loss={args.ibb_edge_loss} warmup={args.ibb_warmup}" if args.ibb else "")
        + (f" pw_sources={args.pairwise_sources}" if args.pairwise_sources != "none" else "")
        + (f" pairwise=1 w_nodes={args.pairwise_w_nodes} unary={args.pairwise_unary} "
           f"cut={args.pairwise_cut} shared={bool(args.pairwise_shared)} "
           f"decode={args.pairwise_decode} edges={len(pw.edges)}"
           if pw is not None else " pairwise=0")
        + "\n"
        f"stopped_epoch={ep} best_epoch={ckpt['epoch']} val_best_score={best:.4f} "
        f"test_score={test_score:.4f} test_mean_qwk={test_qwk:.4f}\n")
    print(f"[done] {out_dir}  val_best={best:.3f}  test_score={test_score:.3f} "
          f"(test_mean_QWK={test_qwk:.3f})")


if __name__ == "__main__":
    main()
