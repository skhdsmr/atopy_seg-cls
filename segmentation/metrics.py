"""세그멘테이션 평가 지표 (임계값 0.5 이진화 기준)."""

import torch


@torch.no_grad()
def seg_scores(logits, target, thr=0.5, smooth=1e-5):
    """logits, target: (B,1,H,W). Dice/IoU/Recall/Precision 을 배치 평균으로 반환."""
    pred = (torch.sigmoid(logits) > thr).float()
    t = (target > 0.5).float()

    b = pred.size(0)
    pred = pred.view(b, -1)
    t = t.view(b, -1)

    tp = (pred * t).sum(1)
    fp = (pred * (1 - t)).sum(1)
    fn = ((1 - pred) * t).sum(1)

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "recall": recall.mean().item(),
        "precision": precision.mean().item(),
    }
