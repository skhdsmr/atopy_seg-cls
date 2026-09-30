"""CORN 순서형 학습에 필요한 조각만 — cls_sev/train.py 에서 실제로 쓰는 부분만 뽑아
cls_kd 안에 다시 쓴 것(SPEC/IBB 등 cls_kd 가 쓰지 않는 부분은 들고 오지 않았다).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from labels import IGA_TASK, NUM_CLASSES, TASK_NAMES

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transforms(imgsz):
    """cls_sev/train.py::build_transforms 와 동일. 수직 뒤집기는 넣지 않는다(피부
    병변 사진은 위아래가 정해진 구도다). 색 지터도 약하게만 — 홍반(붉은기)이 등급의
    핵심 단서라 색을 세게 흔들면 라벨과 어긋난다."""
    train_tf = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def corn_loss(logits, targets, num_classes, weight=None, smoothing=0.0):
    """CORN 순서형 손실 — K 등급을 K-1 개의 '이 등급 초과?' 이진 문제로 분해.

    핵심은 조건부 학습이다: i 번째 이진 문제는 '등급 > i-1' 인 표본만으로 학습한다.
    이렇게 해야 누적확률이 단조가 되고 예측이 등급 순서를 위반하지 않는다.
    weight 는 등급별 가중(K,). smoothing 은 각 이진 타깃을 0/1 -> ε/2, 1-ε/2 로 당긴다
    (인접 등급 경계가 원래 주관적인 라벨이라 과신을 줄이려는 것). 출력 형식(K-1 로짓)은
    그대로라 증류 teacher 로도 그대로 쓸 수 있다.
    """
    total = logits.new_zeros(())
    n = 0
    for i in range(num_classes - 1):
        mask = targets > (i - 1)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        lvl_logit = logits[mask, i]
        lvl_label = (targets[mask] > i).float()
        if smoothing > 0:
            lvl_label = lvl_label * (1.0 - smoothing) + 0.5 * smoothing
        w = weight[targets[mask]] if weight is not None else None
        total = total + F.binary_cross_entropy_with_logits(
            lvl_logit, lvl_label, weight=w, reduction="sum")
        n += cnt
    return total / max(n, 1)


def corn_predict(logits):
    """누적곱 규칙: P(>0)*P(>1)*... 이 0.5 를 넘는 개수 = 등급."""
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (cum > 0.5).sum(dim=1)


def head_predict(out):
    """전체 태스크의 CORN 예측. cls_kd 는 항상 ordinal(CORN)만 쓰므로 cls_sev.train
    의 head_predict 와 달리 predict 모드/CE 분기가 없다(loss_type 인자 자체가 없음)."""
    return {t: corn_predict(out[t].float()) for t in TASK_NAMES}


class UncertaintyWeighter(nn.Module):
    """Kendall et al. 동분산 불확실성 가중 — 축마다 σ 를 학습해 손실을 자동 저울질.

    5축의 난이도와 스케일이 제각각이라 고정 가중은 튜닝 지옥이 된다. weight_decay=0
    필수 — σ 를 0 쪽으로 당기면 그 축 손실이 폭주한다.
    """

    def __init__(self, task_names):
        super().__init__()
        self.log_var = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(())) for t in task_names})

    def forward(self, losses):
        total = 0.0
        for t, l in losses.items():
            s = self.log_var[t]
            total = total + 0.5 * torch.exp(-s) * l + 0.5 * s
        return total

    def sigmas(self):
        return {t: float(torch.exp(0.5 * s).item()) for t, s in self.log_var.items()}


def combine_losses(losses, weighter, iga_weight=0.5):
    """weighter 가 있으면 Kendall 가중 결합, 없으면 IGA/증상 고정 가중 결합."""
    if not losses:
        return None
    if weighter is not None:
        return weighter(losses)
    iga = losses.get(IGA_TASK)
    signs = [l for t, l in losses.items() if t != IGA_TASK]
    sign_mean = sum(signs) / len(signs) if signs else None
    if iga is None:
        return sign_mean
    if sign_mean is None:
        return iga
    return iga_weight * iga + (1.0 - iga_weight) * sign_mean


def head_losses(out, labels, device, cls_w=None, tasks=None, smoothing=0.0):
    """축별 CORN 손실. UNKNOWN(-1) 표본은 그 축에서만 제외된다."""
    cls_w = cls_w or {}
    losses = {}
    for t in (tasks or TASK_NAMES):
        y = labels[t].to(device, non_blocking=True)
        m = y >= 0
        if not bool(m.any()):
            continue
        losses[t] = corn_loss(out[t][m].float(), y[m], NUM_CLASSES[t], weight=cls_w.get(t),
                              smoothing=smoothing)
    return losses


def is_backbone(name):
    return name.startswith("backbone")


def backward_step(loss, opt, scaler, model, clip_grad):
    opt.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        # clip 은 반드시 unscale_ 뒤에. scaler 가 곱해 둔 스케일을 되돌리지 않고 자르면
        # clip_grad 값이 스케일 배수만큼 무의미해진다.
        if clip_grad and clip_grad > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        if clip_grad and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
