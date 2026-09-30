"""cls_kd 공용 유틸.

`labels.py`/`dataset.py`/`model.py`/`metrics.py`/`corn.py`는 CORN 순서형 손실,
QWK 지표, split 폴더 규약을 담은 이 폴더 자체 사본이다(다른 `cls_*` 폴더를
import 하지 않는다). 백본은 timm 을 바로 쓴다(`model.py::build_backbone`).

이 위에 cls_kd 가 얹는 것은 BAM(Born-Again Multi-task) 논문의 지식증류 조각뿐이다:

  1) make_folds.py       : train split 안에서만 5-fold 층화 분할 (teacher OOF 용)
  2) KDDataset/collate_kd: 배치에 stem 을 같이 실어 teacher 로짓을 찾을 수 있게 함
  3) corn_distill_loss   : CORN 의 조건부 이진 구조 위에서 BAM 식 블렌드
                           (λ*hard + (1-λ)*teacher_sigmoid)를 그대로 확장
  4) anneal_lambda        : teacher annealing λ 스케줄 (BAM Fig.1: 0 -> 1 선형)
  5) coherence_loss       : IGA - max(증상) 힌지 제약. 실데이터(atopy.csv, 9,150개)
                           분석에서 이 값이 98.69% 확률로 [0,2] 에 들어간다는 것을
                           student 자신의 예측에 거는 정합성 정규화로 반영한다.
  6) try_resume           : cls_sev/train.py 와 같은 last.pt(+optimizer/scheduler/
                           AMP/RNG 상태) 기반 --resume — 학습이 중간에 끊겨도
                           처음부터 다시 돌리지 않게 한다.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import corn                             # noqa: E402  (cls_kd 자체 사본, cls_sev 아님)
from labels import IGA_TASK, NUM_CLASSES, SIGN_TASKS, TASK_NAMES, TASKS  # noqa: E402
from dataset import (SevDataset, build_label_index, collate, resolve_mask_dir,  # noqa: E402
                     resolve_split_dir)
from metrics import (confusion, evaluate_all, format_confusion, format_metrics,  # noqa: E402
                     mean_qwk, per_task_metrics, selection_score)
from model import MODELS, MultiTaskNet  # noqa: E402


# ----------------------------------------------------------------------------- 데이터셋
class KDDataset(SevDataset):
    """SevDataset + stem. teacher 로짓을 stem 으로 찾아 배치에 붙이려면 필요하다.

    원본 SevDataset.__getitem__ 은 (img, labels, meta) 3-튜플만 낸다 — 다른 코드가
    그 계약에 의존할 수 있으므로 원본을 고치지 않고 서브클래스로 한 겹만 얹는다.
    """

    def __getitem__(self, i):
        img, lab, meta = super().__getitem__(i)
        stem = self.samples[i][0].stem
        return img, lab, meta, stem


def collate_kd(batch):
    imgs, labels, meta = collate([(b[0], b[1], b[2]) for b in batch])
    stems = [b[3] for b in batch]
    return imgs, labels, meta, stems


def make_full_dataset(data_root, split, transform, labels_csv=None, exp="full",
                      masks=None, margin=0.15, bbox_square=False):
    """exp="full"(기본): 원본 이미지 전체, 마스크 불필요.
    exp="bbox": 분할 마스크의 union bbox(+margin)로 크롭 — 배경은 지우지 않는다
    (마스크는 '어디를 볼지'만 정하고, 크롭된 픽셀은 원본 그대로 가져온다. 그래야
    병변-정상피부 대비/질감이 살아서 중증도 판단에 유리하다). bbox_square=True면
    긴 변 기준 정사각으로 넓혀 crop해 Resize의 종횡비 왜곡(구진이 타원이 되는 등)을
    막는다. 마스크 폴더는 resolve_mask_dir 규칙대로 자동 탐색하고(<data>/masks/
    <split> -> <data>/images/masks/<split> -> <data>/../masks/<split>), --masks
    로 직접 지정할 수도 있다.
    """
    index, info = build_label_index(data_root, labels_csv)
    img_dir = resolve_split_dir(data_root, split)
    mask_dir = None
    if exp == "bbox":
        mask_dir, cands = resolve_mask_dir(data_root, masks, split, img_dir.name)
        if mask_dir is None:
            tried = "\n         ".join(str(c) for c in cands)
            sys.exit(f"[에러] --exp bbox 인데 '{split}' 마스크 폴더를 찾지 못했다.\n"
                     f"       탐색한 경로:\n         {tried}\n"
                     f"       --masks 로 직접 지정하거나, --exp full 로 마스크 없이 돌릴 것.")
    ds = KDDataset(img_dir, mask_dir, index, transform=transform, margin=margin,
                   square=bbox_square)
    if len(ds) == 0:
        sys.exit(f"[에러] {split} 샘플이 0개. --data 경로와 labels.csv 의 identifier(과거 stem) "
                 f"컬럼이 이미지 파일명과 매칭되는지 확인할 것.")
    return ds, index, info


def add_crop_args(parser):
    """--exp/--masks/--margin/--bbox_square — train_single.py/train_multitask.py/
    train_distill.py 세 곳이 전부 write_full_dataset(exp=args.exp, masks=args.masks,
    margin=args.margin, bbox_square=args.bbox_square) 형태로 넘길 수 있도록 인자
    정의를 한 곳에 모아 둔다(세 스크립트가 각자 add_argument 를 반복하면 언젠가
    기본값이 갈라진다)."""
    parser.add_argument("--exp", choices=["full", "bbox"], default="full",
                        help="full=원본 이미지 전체(기본) | bbox=마스크 union bbox+margin 크롭"
                             "(배경을 지우지 않는다 — 마스크는 '어디를 볼지'만 정한다)")
    parser.add_argument("--masks", default="",
                        help="(bbox) 마스크 루트. 비우면 자동 탐색: <data>/masks/<split> -> "
                             "<data>/images/masks/<split> -> <data>/../masks/<split>")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="(bbox) union bbox 각 변을 이 비율만큼 바깥으로 확장")
    parser.add_argument("--bbox_square", action="store_true",
                        help="(bbox) 크롭을 긴 변 기준 정사각으로 넓힌다 — Resize의 "
                             "종횡비 왜곡(구진이 타원이 되는 등)을 막는다")


def add_mtl_args(parser):
    """--mtl/--iga_weight — train_multitask.py(B)/train_distill.py(C/D/E) 공용.

    uncertainty(기본): Kendall 불확실성 자동가중(corn.UncertaintyWeighter).
    fixed: --iga_weight*IGA손실 + (1-iga_weight)*mean(증상4종손실) 고정 배분 —
    cls_sev/ogw 계열에서 --mtl fixed 로 부르던 것과 같은 대조군이다(축마다 σ 를
    학습하지 않으니 손실 결합 자체가 무엇을 바꾸는지 보고 싶을 때 uncertainty 와
    대조하는 용도)."""
    parser.add_argument("--mtl", choices=["uncertainty", "fixed"], default="uncertainty",
                        help="5축 손실 결합 방식. uncertainty=Kendall σ 자동가중(기본) | "
                             "fixed=--iga_weight 로 IGA/증상 비중을 고정")
    parser.add_argument("--iga_weight", type=float, default=0.5,
                        help="(--mtl fixed 전용) IGA 축 비중. 나머지(1-이 값)를 증상 4종이 "
                             "평균으로 나눠 가진다")


def build_weighter(mtl, task_names, device):
    """--mtl uncertainty 면 UncertaintyWeighter, fixed 면 None(combine_losses 가
    iga_weight 고정 배분으로 대신 처리한다)."""
    if mtl == "uncertainty":
        return corn.UncertaintyWeighter(task_names).to(device)
    return None


def crop_tag(args):
    """런 이름에 붙일 crop 설정 태그. exp=full 이면 빈 문자열(기존 이름 그대로
    유지 — bbox 를 안 쓰면 이름이 안 바뀐다). exp=bbox 로 학습 방식 자체가 바뀌면
    같은 --model/--imgsz 라도 다른 폴더에 저장돼야 한다(안 그러면 full 로 만든
    결과를 bbox 런이 덮어쓴다) — cls_sev 의 "결과를 바꾸는 축은 이름에 남긴다"
    관례를 그대로 따른다."""
    if args.exp != "bbox":
        return ""
    return "_bbox" + ("_sq" if args.bbox_square else "")


def mtl_tag(args):
    """런 이름에 붙일 5축 결합 방식 태그. uncertainty(기본)면 빈 문자열 — 기존
    B/C/D/E 런 이름을 그대로 유지한다. fixed 로 바꾸면 손실 결합 자체가 달라지므로
    같은 폴더를 덮어쓰지 않게 iga_weight 값까지 이름에 남긴다."""
    if args.mtl != "fixed":
        return ""
    return f"_fixed{args.iga_weight:g}"


def keep_stems(ds, stems):
    """ds.samples 를 주어진 stem 집합으로만 제자리에서 좁힌다(5-fold teacher 학습용).

    SevDataset.samples 는 [(img_path, mask_path, rec), ...] 공개 리스트라 이렇게
    직접 필터링해도 안전하다 — dataset.py 안에서 재사용하는 다른 메서드
    (label_arrays/records/meta)도 전부 self.samples 를 그때그때 다시 읽는다.
    """
    keep = set(stems)
    before = len(ds.samples)
    ds.samples = [s for s in ds.samples if s[0].stem in keep]
    return before, len(ds.samples)


# ----------------------------------------------------------------------------- 시드 / 최적화
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr, backbone_lr_scale, weight_decay, extra_params=None):
    """cls_sev/train.py 에서 검증된 파라미터 그룹 규칙(백본 저LR + BN/bias
    no-decay)을 그대로 옮긴 것.

    A/B/C/D/E 가 전부 같은 최적화 동역학을 쓰게 하려는 것 — 그래야 조건 사이 성능차가
    최적화 세팅이 아니라 (single/multi/distill/anneal/coherence) 그 자체에서 온다.
    """
    def no_decay(_n, p):
        return p.ndim <= 1

    def groups_of(named, group_lr):
        decay = [p for n, p in named if not no_decay(n, p)]
        nd = [p for n, p in named if no_decay(n, p)]
        return [{"params": decay, "lr": group_lr, "weight_decay": weight_decay},
                {"params": nd, "lr": group_lr, "weight_decay": 0.0}]

    bb = [(n, p) for n, p in model.named_parameters() if corn.is_backbone(n)]
    hd = [(n, p) for n, p in model.named_parameters() if not corn.is_backbone(n)]
    groups = groups_of(bb, lr * backbone_lr_scale) + groups_of(hd, lr)
    if extra_params is not None:
        extra_params = list(extra_params)
        if extra_params:
            groups.append({"params": extra_params, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def cosine_warmup_lambda(warmup_epochs, total_epochs):
    """cls_sev/train.py::lr_lambda 와 동일한 스케줄(선형 warmup -> cosine decay)."""
    def f(ep):
        if warmup_epochs > 0 and ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        prog = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))
    return f


def add_sched_args(parser):
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "constant"],
                        default="cosine",
                        help="warmup 뒤 LR 모양. cosine(기본)=--cosine_epochs 에 걸쳐 감쇠 | "
                             "plateau=val loss 가 --plateau_patience epoch 정체하면 x0.5 | "
                             "constant=감쇠 없음")
    parser.add_argument("--cosine_epochs", type=int, default=0,
                        help="(cosine) 감쇠 지평. 0 이면 --epochs. 조기종료가 10~20ep 에서 걸리면 "
                             "--epochs 100 기준 cosine 은 사실상 상수 LR 이라 짧게 줄 때 의미가 있다")
    parser.add_argument("--plateau_patience", type=int, default=2,
                        help="(plateau) LR 을 줄이기 전 기다리는 epoch — 조기종료 patience 보다 "
                             "작아야 LR 감소가 실제로 한 번은 일어난다")


class EpochScheduler:
    """epoch 단위 LR 스케줄 — 공통 선형 warmup 뒤 cosine | plateau | constant.

    cosine + cosine_epochs=total 이면 cosine_warmup_lambda 를 쓴 LambdaLR 과 같은 LR 을
    낸다(생성 시점에 epoch 0 의 LR 을 바로 적용하는 것까지 동일). plateau 는 val loss 를
    받아야 하므로 step(val_loss) 로 부른다 — 조기종료와 같은 신호(raw val loss)를 쓴다.
    """

    def __init__(self, opt, kind, warmup_epochs, total_epochs, cosine_epochs=0,
                 plateau_patience=2, plateau_factor=0.5, min_factor=0.01):
        self.opt = opt
        self.base_lrs = [g["lr"] for g in opt.param_groups]
        self.kind = kind
        self.warmup = int(warmup_epochs)
        self.horizon = int(cosine_epochs) or int(total_epochs)
        self.plateau_patience = int(plateau_patience)
        self.plateau_factor = float(plateau_factor)
        self.min_factor = float(min_factor)
        self.epoch = 0
        self.plateau_scale = 1.0
        self.plateau_best = float("inf")
        self.plateau_bad = 0
        self._apply()

    def _factor(self):
        if self.warmup > 0 and self.epoch < self.warmup:
            return (self.epoch + 1) / self.warmup
        if self.kind == "cosine":
            prog = (self.epoch - self.warmup) / max(1, self.horizon - self.warmup)
            return max(self.min_factor, 0.5 * (1 + np.cos(np.pi * min(prog, 1.0))))
        if self.kind == "plateau":
            return self.plateau_scale
        return 1.0

    def _apply(self):
        f = self._factor()
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base * f

    def step(self, val_loss=None):
        self.epoch += 1
        if self.kind == "plateau" and val_loss is not None and self.epoch > self.warmup:
            if val_loss < self.plateau_best:
                self.plateau_best, self.plateau_bad = val_loss, 0
            else:
                self.plateau_bad += 1
                if self.plateau_bad >= self.plateau_patience:
                    self.plateau_scale = max(self.min_factor,
                                             self.plateau_scale * self.plateau_factor)
                    self.plateau_bad = 0
        self._apply()

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "opt"}

    def load_state_dict(self, sd):
        self.__dict__.update(sd)


# ----------------------------------------------------------------------------- KD 손실
def corn_distill_loss(student_logits, teacher_logits, targets, num_classes, lam, weight=None,
                      smoothing=0.0):
    """BAM 식 KD를 CORN 의 조건부 이진 구조 그대로 확장.

    corn.corn_loss 와 마스크/조건부 구조가 완전히 같다 — 레벨 i 의 이진 타깃이
    hard 0/1 대신 λ*hard + (1-λ)*teacher_sigmoid 로 blend 된다는 점만 다르다
    (논문 §3.2 의  ℓ(λy + (1-λ)f_teacher, f_student) 를 CORN 의 K-1개 조건부 이진
    문제 각각에 적용한 것). λ=1 이면 corn_loss 와 완전히 같다(순수 지도학습),
    λ=0 이면 순수 증류 — teacher 의 레벨별 조건부확률 P(등급>i | 등급>i-1) 을 그대로
    모사한다. teacher_logits 는 항상 detach 상태로 취급한다(증류 방향은 student만).
    smoothing 은 corn_loss 와 같이 hard 성분에만 적용한다(0/1 -> ε/2, 1-ε/2) —
    blend 전에 당겨서, λ=1(순수 지도학습)일 때 corn_loss(smoothing=...)와 완전히
    같아지게 한다.
    """
    total = student_logits.new_zeros(())
    n = 0
    for i in range(num_classes - 1):
        mask = targets > (i - 1)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        s_logit = student_logits[mask, i]
        hard = (targets[mask] > i).float()
        if smoothing > 0:
            hard = hard * (1.0 - smoothing) + 0.5 * smoothing
        t_soft = torch.sigmoid(teacher_logits[mask, i].detach().float())
        blended = lam * hard + (1.0 - lam) * t_soft
        w = weight[targets[mask]] if weight is not None else None
        total = total + F.binary_cross_entropy_with_logits(
            s_logit, blended, weight=w, reduction="sum")
        n += cnt
    return total / max(n, 1)


def anneal_lambda(step, total_steps, mode):
    """공식 BAM 구현(bam/run_classifier.py)과 정확히 같은 스케줄:

        percent_done = global_step / num_train_steps     (매 학습 스텝마다 갱신)
        labels = true_labels*percent_done + teacher_labels*(1-percent_done)

    (bam/task_specific/classification/classification_tasks.py 의
    get_prediction_module). 즉 λ 는 **에폭이 아니라 스텝 단위**로 선형 증가한다 —
    에폭당 한 번만 갱신하면 한 에폭 안의 모든 배치가 같은 λ 를 쓰는 계단식
    근사가 되어 버려 "학습 진행 내내 선형" 이라는 원 스케줄과 어긋난다.

    step 은 0-base 누적 학습 스텝. mode: none(조건 C, λ=0 고정, ablation "λ=0"과
    동일) | linear(조건 D/E, 위 공식 그대로).
    """
    if mode == "none":
        return 0.0
    if mode != "linear":
        raise ValueError(mode)
    if total_steps <= 0:
        return 1.0
    return float(min(1.0, max(0.0, step / total_steps)))


def corn_expected_value(logits):
    """CORN 로짓 -> 기대 등급 E[Y]. corn.corn_predict 와 같은 누적곱 규칙을 쓰되
    0.5 임계 이산화 대신 기대값을 써서 미분 가능한 연속량으로 만든다.

    E[Y] = sum_{i=0}^{K-2} P(Y>i),  P(Y>i) = prod_{j<=i} sigmoid(logit_j).
    """
    probs = torch.sigmoid(logits.float())
    cum = torch.cumprod(probs, dim=1)          # P(Y>i), i=0..K-2
    return cum.sum(dim=1)                       # 정수 서포트에서의 표준 항등식


def corn_class_probs(logits):
    """CORN 로짓 -> 클래스별 확률 벡터 P(Y=k), k=0..K-1(합이 1).

    P(Y>i) = prod_{j<=i} sigmoid(logit_j),  i=0..K-2, 양끝은 P(Y>-1)=1, P(Y>K-1)=0.
    P(Y=k) = P(Y>k-1) - P(Y>k). corn_distill_loss 는 이 완전한 분포가 아니라 레벨별
    조건부 sigmoid 를 그대로 blend 하지만(§4), 사람이 "teacher 가 이 이미지를 어떤
    등급으로 보는가"를 확인하려면 이 카테고리 분포로 디코딩해야 읽힌다 — eval_oof.py
    의 --show_probs 가 이걸 쓴다.
    """
    probs = torch.sigmoid(logits.float())
    cum = torch.cumprod(probs, dim=1)                              # (B, K-1)
    ones = cum.new_ones(cum.shape[0], 1)
    zeros = cum.new_zeros(cum.shape[0], 1)
    p_gt = torch.cat([ones, cum, zeros], dim=1)                    # (B, K+1): P(Y>-1..K-1)
    return (p_gt[:, :-1] - p_gt[:, 1:]).clamp(min=0.0)             # (B, K)


def coherence_loss(out, low=0.0, high=2.0):
    """g = E[IGA] - max(E[홍반/구진/찰상/태선화]) 가 [low,high] 밖이면 힌지 벌점.

    atopy.csv 9,150개 실측: diff=IGA-max(증상축) 분포가 {-1:96, 0:3211, 1:5275,
    2:544, 3:24} — 98.69%가 {0,1,2}. IGA(0~4)와 증상(0~3) 스케일이 그대로이므로
    (병합 없음) low=0/high=2 는 이 분포를 그대로 옮긴 것이다.

    정답 라벨이 아니라 **student 자신의 현재 예측**에 거는 제약이다 — teacher 나
    라벨 없이도 매 배치 걸 수 있고, 두 배치에 걸친 라벨 결측과 무관하게 축 사이
    일관성을 직접 규제한다(조건 E 전용).
    """
    iga_e = corn_expected_value(out[IGA_TASK])
    sign_e = torch.stack([corn_expected_value(out[t]) for t in SIGN_TASKS], dim=0)
    g = iga_e - sign_e.max(dim=0).values
    return (F.relu(low - g) + F.relu(g - high)).mean()


# ----------------------------------------------------------------------------- 재개(resume)
def add_resume_arg(parser):
    parser.add_argument("--resume", action="store_true",
                        help="<out_dir>/last.pt 에서 이어서 학습(파일이 없으면 처음부터). "
                             "cls_sev/train.py 와 같은 관례: best.pt 는 평가용으로 가볍게 "
                             "유지하고, optimizer/scheduler/AMP/RNG 상태는 last.pt 에만 싣는다.")


def rng_state():
    """random/numpy/torch(+CUDA) 시드 상태 스냅샷 — --resume 시 학습 궤적을 그대로
    이어가기 위함(재현성 자체가 목적이 아니라, 재개 지점에서 갑자기 분포가 바뀌는
    것을 막기 위해서다)."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(),
           "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])}


def load_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu().to(torch.uint8))
    cu = st.get("cuda") or []
    if cu and torch.cuda.is_available() and len(cu) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in cu])
    elif cu:
        print("[경고] GPU 개수가 ckpt 와 달라 CUDA RNG 는 복원하지 않는다.", file=sys.stderr)


def try_resume(ckpt_path, args, model, opt, sched, scaler, weighter, check_keys):
    """last.pt 가 있으면 이어서 학습할 상태를 전부 복원한다.

    check_keys 에 든 args 필드가 ckpt 저장 당시와 다르면 바로 에러로 죽는다 —
    예를 들어 --imgsz 나 --model 이 다른 ckpt 를 이어받으면 텐서 shape 자체가
    안 맞거나(즉시 에러) shape 은 우연히 맞아도 의미가 다른 학습이 섞여 버린다
    (조용히 넘어가면 나중에 결과가 왜 이상한지 추적하기 훨씬 어렵다).

    best/best_ep(QWK 기준, best.pt 갱신)와 no_improve/best_val_loss(val loss 기준,
    early stop 카운터)는 서로 다른 축이다 — QWK 는 갱신됐는데 val loss 는 안 좋아진
    epoch도, 그 반대도 있을 수 있어서 하나로 합치면 어느 쪽 신호로 멈췄는지 알 수
    없어진다.

    반환: (start_ep, best, best_ep, no_improve, best_val_loss). --resume 를 안
    줬거나 last.pt 가 없으면 (1, -1e9, 0, 0, inf) — 처음부터.
    """
    best, best_ep, no_improve, start_ep, best_val_loss = -1e9, 0, 0, 1, float("inf")
    if not args.resume:
        return start_ep, best, best_ep, no_improve, best_val_loss
    if not ckpt_path.is_file():
        print(f"[resume] {ckpt_path} 없음 -> 처음부터 학습한다.")
        return start_ep, best, best_ep, no_improve, best_val_loss

    rc = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prev = rc.get("args", {}) or {}
    for k in check_keys:
        if k in prev and prev[k] != getattr(args, k):
            sys.exit(f"[에러] resume 불가: --{k} 가 ckpt 와 다르다 "
                     f"(ckpt={prev[k]}, 지금={getattr(args, k)}).")

    model.load_state_dict(rc["model"])
    if weighter is not None and rc.get("weighter"):
        weighter.load_state_dict(rc["weighter"])
    start_ep = int(rc.get("epoch", 0)) + 1
    if start_ep > args.epochs:
        sys.exit(f"[에러] ckpt 가 이미 ep{start_ep - 1} 까지 돌았다(--epochs={args.epochs}). "
                 f"--epochs 를 늘릴 것.")
    if "optimizer" in rc:
        opt.load_state_dict(rc["optimizer"])
        sched.load_state_dict(rc["scheduler"])
        if scaler is not None and rc.get("scaler"):
            scaler.load_state_dict(rc["scaler"])
        best = float(rc.get("best", -1e9))
        best_ep = int(rc.get("best_ep", 0))
        no_improve = int(rc.get("no_improve", 0))
        best_val_loss = float(rc.get("best_val_loss", float("inf")))
        if rc.get("rng"):
            load_rng_state(rc["rng"])
    print(f"[resume] {ckpt_path} -> ep{start_ep} 부터 (best={best:.4f}@ep{best_ep}, "
         f"no_improve={no_improve}, best_val_loss={best_val_loss:.4f})")
    return start_ep, best, best_ep, no_improve, best_val_loss
