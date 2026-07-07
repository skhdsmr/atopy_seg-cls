"""UNeXt 손실 함수 (sigmoid 기반 단일 클래스): BCE+Dice, BCE+Tversky, BCE+Focal Tversky."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BCEDiceLoss", "BCETverskyLoss", "BCEFocalTverskyLoss"]


class BCEDiceLoss(nn.Module):
    """원 구현과 동일: BCEWithLogits + soft Dice(0.5:0.5 가중)."""

    def __init__(self, dice_w=1.0, bce_w=0.5):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        smooth = 1e-5
        pred = torch.sigmoid(logits)
        num = target.size(0)
        pred = pred.view(num, -1)
        target = target.view(num, -1)
        inter = (pred * target).sum(1)
        dice = (2.0 * inter + smooth) / (pred.sum(1) + target.sum(1) + smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_w * bce + self.dice_w * dice_loss


class BCETverskyLoss(nn.Module):
    """BCEWithLogits + soft Tversky.

    Tversky = TP / (TP + alpha*FP + beta*FN),  beta = 1 - alpha (지정 없으면).
      - alpha > beta  -> 오탐(FP)에 큰 벌점 (precision↑, 우리 과탐 문제용)
      - alpha = beta = 0.5 -> Dice 와 동일 => BCEDiceLoss(baseline)와 정확히 일치.
    BCE 항은 그대로 두어 alpha 만의 효과를 격리 측정한다(bce_w=0.5 동일).
    """

    def __init__(self, alpha=0.7, beta=None, bce_w=0.5, tversky_w=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = (1.0 - alpha) if beta is None else beta
        self.bce_w = bce_w
        self.tversky_w = tversky_w

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        smooth = 1e-5
        pred = torch.sigmoid(logits)
        num = target.size(0)
        pred = pred.view(num, -1)
        target = target.view(num, -1)
        tp = (pred * target).sum(1)
        fp = (pred * (1 - target)).sum(1)
        fn = ((1 - pred) * target).sum(1)
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        tversky_loss = 1 - tversky.mean()
        return self.bce_w * bce + self.tversky_w * tversky_loss


class BCEFocalTverskyLoss(nn.Module):
    """BCEWithLogits + soft Focal Tversky.

    Focal Tversky = ((1 - Tversky)^gamma) 를 이미지별로 계산 후 평균한다(Abraham 2019).
      - Tversky = TP / (TP + alpha*FP + beta*FN),  beta = 1 - alpha (지정 없으면).
      - gamma > 1: 이미 잘 맞은(Tversky↑) 이미지의 그래디언트를 죽이고, 잘 못 맞은
        '어려운' 이미지(작은/놓치는 병변)에 집중시킨다. gamma=1 이면 BCETverskyLoss 와 동일.
      - 방향 규칙은 Tversky 와 같다:
          alpha > beta -> 오탐(FP) 벌점(precision↑, 과탐 억제)
          alpha < beta -> 미탐(FN) 벌점(recall↑, 작은 병변 놓침 방지)  <- 작은 전경비율용
    BCE 항은 그대로 두어(bce_w=0.5) 다른 손실과 조건을 맞춘다.
    """

    def __init__(self, alpha=0.7, beta=None, gamma=1.333, bce_w=0.5, ft_w=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = (1.0 - alpha) if beta is None else beta
        self.gamma = gamma
        self.bce_w = bce_w
        self.ft_w = ft_w

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        smooth = 1e-5
        pred = torch.sigmoid(logits)
        num = target.size(0)
        pred = pred.view(num, -1)
        target = target.view(num, -1)
        tp = (pred * target).sum(1)
        fp = (pred * (1 - target)).sum(1)
        fn = ((1 - pred) * target).sum(1)
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        # 이미지별 (1 - Tversky)^gamma 후 평균 -> 어려운 이미지에 가중
        focal_tversky = ((1 - tversky).clamp_min(0) ** self.gamma).mean()
        return self.bce_w * bce + self.ft_w * focal_tversky
