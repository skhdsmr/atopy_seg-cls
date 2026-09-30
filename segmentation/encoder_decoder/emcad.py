"""
EMCAD 디코더 (Efficient Multi-scale Convolutional Attention Decoding, CVPR 2024).

smp 에 없는 디코더라 논문/공식 구현을 이 파일에 이식했다. model.py 의 build_model 이
decoder="emcad" 일 때 이 모듈의 EMCADSegmenter 를 만든다.

구조(스케일 1/32 -> 1/4 순으로 4단계):
    CAB(채널 attention) -> SAB(공간 attention) -> MSCB(멀티스케일 depthwise conv)
    단계 사이 업샘플은 EUCB, skip 결합은 LGAG(large-kernel grouped attention gate).
각 단계에 seg head 를 달아 deep supervision 하고, 4개 logits 를 입력 해상도로 올려 합산해
(B, num_classes, H, W) 하나를 반환한다 -> 기존 손실/지표 파이프라인과 그대로 호환.

인코더는 smp 의 get_encoder 를 그대로 쓴다(=기존 --encoder 문자열 전부 사용 가능).
마지막 4개 스테이지(1/4, 1/8, 1/16, 1/32)만 사용하므로, PVTv2 처럼 1/2 스테이지가 없어
smp 가 0채널 더미를 끼워 넣는 인코더도 문제없이 동작한다(UNet++ 가 깨지는 지점).

하이퍼파라미터(expansion_factor 등)는 의도적으로 상수로 고정했다. build_model 인자로 빼면
추론 스크립트들이 체크포인트의 args 만 보고 재구성할 때 값이 어긋나 state_dict 로드가
깨질 수 있기 때문.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder

__all__ = ["EMCADSegmenter"]

# --- 고정 하이퍼파라미터 -------------------------------------------------------
# 공식 구현 기본값은 expansion_factor=6 이지만, 512px 학습에서 1/4 스테이지 활성값이
# 커져 OOM 위험이 있어 2 로 낮춰 고정했다(논문의 경량 설정과 같은 값).
_EXPANSION_FACTOR = 2
_KERNEL_SIZES = (1, 3, 5)     # MSCB 의 멀티스케일 depthwise 커널
_LGAG_KS = 3                  # LGAG large-kernel 크기
_EUCB_KS = 3                  # EUCB depthwise 업컨브 커널


def _act(name="relu6"):
    return {"relu": nn.ReLU, "relu6": nn.ReLU6, "gelu": nn.GELU}[name]()


def channel_shuffle(x, groups):
    """ShuffleNet 식 채널 셔플. groups 가 채널을 나누지 못하면 원본 반환(안전장치)."""
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(b, -1, h, w)


class MSDC(nn.Module):
    """멀티스케일 depthwise conv: 커널 크기별 depthwise 결과를 리스트로 반환."""

    def __init__(self, channels, kernel_sizes, stride=1, activation="relu6",
                 dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, k, stride, k // 2,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                _act(activation),
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        outs = []
        for dwconv in self.dwconvs:
            out = dwconv(x)
            outs.append(out)
            if not self.dw_parallel:      # cascade 모드: 이전 스케일 결과를 누적
                x = x + out
        return outs


class MSCB(nn.Module):
    """Multi-scale convolution block: 1x1 확장 -> MSDC -> 채널셔플 -> 1x1 축소 (inverted residual)."""

    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_sizes=_KERNEL_SIZES, expansion_factor=_EXPANSION_FACTOR,
                 dw_parallel=True, add=True, activation="relu6"):
        super().__init__()
        self.add = add
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        ex_channels = int(in_channels * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ex_channels),
            _act(activation),
        )
        self.msdc = MSDC(ex_channels, kernel_sizes, stride, activation,
                         dw_parallel=dw_parallel)
        combined = ex_channels if add else ex_channels * len(kernel_sizes)
        self.combined_channels = combined
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # stride/채널이 그대로일 때만 residual
        self.use_skip = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        pout = self.pconv1(x)
        msdc_outs = self.msdc(pout)
        if self.add:
            dout = msdc_outs[0]
            for o in msdc_outs[1:]:
                dout = dout + o
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        return x + out if self.use_skip else out


class CAB(nn.Module):
    """Channel attention block: avg/max pooling -> 공유 MLP -> sigmoid 게이트."""

    def __init__(self, channels, ratio=16, activation="relu"):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            _act(activation),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SAB(nn.Module):
    """Spatial attention block: 채널축 avg/max 를 concat -> 큰 커널 conv -> sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class EUCB(nn.Module):
    """Efficient up-convolution block: bilinear x2 -> depthwise conv -> 채널셔플 -> 1x1."""

    def __init__(self, in_channels, out_channels, kernel_size=_EUCB_KS,
                 activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size // 2,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            _act(activation),
        )
        self.pwc = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        return self.pwc(x)


class LGAG(nn.Module):
    """Large-kernel grouped attention gate: 업샘플 특징 g 로 skip x 를 게이팅."""

    def __init__(self, f_g, f_l, f_int, kernel_size=_LGAG_KS, activation="relu"):
        super().__init__()
        # 공식 구현은 groups=channels//2. 채널이 안 나눠떨어지는 인코더도 있어 gcd 로 안전하게.
        groups = math.gcd(math.gcd(f_g, f_l), f_int)
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size, 1, kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size, 1, kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, 1, 0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.act = _act(activation)

    def forward(self, g, x):
        return x * self.psi(self.act(self.w_g(g) + self.w_x(x)))


class EMCADDecoder(nn.Module):
    """channels: [c4(1/32), c3(1/16), c2(1/8), c1(1/4)] 순서. 4개 스케일 특징 반환."""

    def __init__(self, channels, activation="relu6"):
        super().__init__()
        c4, c3, c2, c1 = channels

        self.cab4, self.mscb4 = CAB(c4), MSCB(c4, c4, activation=activation)
        self.eucb3 = EUCB(c4, c3)
        self.lgag3 = LGAG(c3, c3, c3 // 2)
        self.cab3, self.mscb3 = CAB(c3), MSCB(c3, c3, activation=activation)
        self.eucb2 = EUCB(c3, c2)
        self.lgag2 = LGAG(c2, c2, c2 // 2)
        self.cab2, self.mscb2 = CAB(c2), MSCB(c2, c2, activation=activation)
        self.eucb1 = EUCB(c2, c1)
        self.lgag1 = LGAG(c1, c1, c1 // 2)
        self.cab1, self.mscb1 = CAB(c1), MSCB(c1, c1, activation=activation)
        self.sab = SAB()

    @staticmethod
    def _align(x, ref):
        """업샘플 결과가 skip 과 1~2px 어긋나는 경우(홀수 해상도) 보정."""
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x4, skips):
        """x4: 1/32 특징, skips: [x3(1/16), x2(1/8), x1(1/4)]."""
        x3, x2, x1 = skips

        d4 = self.cab4(x4) * x4
        d4 = self.sab(d4) * d4
        d4 = self.mscb4(d4)

        d3 = self._align(self.eucb3(d4), x3)
        d3 = d3 + self.lgag3(g=d3, x=x3)
        d3 = self.cab3(d3) * d3
        d3 = self.sab(d3) * d3
        d3 = self.mscb3(d3)

        d2 = self._align(self.eucb2(d3), x2)
        d2 = d2 + self.lgag2(g=d2, x=x2)
        d2 = self.cab2(d2) * d2
        d2 = self.sab(d2) * d2
        d2 = self.mscb2(d2)

        d1 = self._align(self.eucb1(d2), x1)
        d1 = d1 + self.lgag1(g=d1, x=x1)
        d1 = self.cab1(d1) * d1
        d1 = self.sab(d1) * d1
        d1 = self.mscb1(d1)

        return [d4, d3, d2, d1]


class EMCADSegmenter(nn.Module):
    """smp 인코더 + EMCAD 디코더. forward(x) -> (B, num_classes, H, W) logits.

    train.py 가 model.encoder 로 인코더 파라미터를 구분(차등 LR/동결)하므로
    인코더는 반드시 self.encoder 로 노출한다.
    """

    def __init__(self, encoder_name="tu-pvt_v2_b2", encoder_weights="imagenet",
                 num_classes=1, in_channels=3):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels,
                                   depth=5, weights=encoder_weights)
        # 마지막 4개 스테이지 = 1/4, 1/8, 1/16, 1/32
        c1, c2, c3, c4 = self.encoder.out_channels[-4:]
        if min(c1, c2, c3, c4) < 8:
            raise ValueError(
                f"인코더 '{encoder_name}' 의 스테이지 채널이 EMCAD 에 부적합합니다: "
                f"{self.encoder.out_channels}")
        self.decoder = EMCADDecoder([c4, c3, c2, c1])
        # deep supervision: 스케일별 seg head
        self.heads = nn.ModuleList([nn.Conv2d(c, num_classes, 1)
                                    for c in (c4, c3, c2, c1)])

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)
        x1, x2, x3, x4 = feats[-4:]
        outs = self.decoder(x4, [x3, x2, x1])
        logits = None
        for head, d in zip(self.heads, outs):
            p = F.interpolate(head(d), size=size, mode="bilinear", align_corners=False)
            logits = p if logits is None else logits + p
        return logits
