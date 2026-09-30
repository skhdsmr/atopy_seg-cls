"""
타일드(sliding-window) 추론 유틸 — crop 학습 모델을 '전체 이미지'로 정직하게 평가.

- crop 으로 학습하면 추론도 같은 스케일(타일)로 맞춰야 한다. 여기서는 원해상도 전체
  이미지를 tile x tile 창으로 stride 간격 슬라이딩하며 예측하고, 겹치는 부분은 확률을
  평균해 합친 뒤 전체 크기의 logit 맵을 돌려준다.
- 반환은 logit(=평균확률을 다시 logit 으로 변환) 이라 기존 metrics.seg_scores(sigmoid
  기반)에 그대로 넣을 수 있다.
- 왜 전체 이미지 평가인가: crop 조각 위에서 Dice 를 재면 배경 crop(빈 GT)이 Dice=1 로
  잡혀 점수가 뻥튀기된다. 배포와 동일하게 '전체 이미지' 기준으로 재야 정직하다.
"""

import torch


def _starts(length, tile, stride):
    """0..length-tile 를 stride 로 훑되 마지막이 경계를 덮도록 보정."""
    if length <= tile:
        return [0]
    ss = list(range(0, length - tile + 1, stride))
    if ss[-1] != length - tile:
        ss.append(length - tile)
    return ss


@torch.no_grad()
def tiled_logits(model, img, tile=512, stride=384, use_amp=False):
    """img: (1,3,H,W) 정규화된 원해상도 텐서. 반환: (1,1,H,W) logit."""
    _, _, H, W = img.shape
    device = img.device
    acc = torch.zeros((1, 1, H, W), device=device)
    cnt = torch.zeros((1, 1, H, W), device=device)
    for y in _starts(H, tile, stride):
        for x in _starts(W, tile, stride):
            patch = img[:, :, y:y + tile, x:x + tile]
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(patch).float()
            acc[:, :, y:y + tile, x:x + tile] += torch.sigmoid(out)
            cnt[:, :, y:y + tile, x:x + tile] += 1.0
    prob = (acc / cnt.clamp_min(1e-6)).clamp(1e-6, 1 - 1e-6)
    return torch.log(prob / (1 - prob))       # prob -> logit (seg_scores 가 sigmoid 적용)
