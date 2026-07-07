"""
U-Net++ (EfficientNet-b3 인코더, ImageNet 사전학습) 모델 정의.

- UNeXt 의 archs.py 를 '대체'하는 역할. 나머지 파이프라인(dataset/augment/losses/
  metrics)은 segmentation/ 의 공유 모듈을 그대로 재사용한다(단일 소스 유지).
- 인코더는 ImageNet 사전학습 -> 입력은 ImageNet 정규화가 필요하다.
  augment.py 가 이미 ImageNet mean/std 로 정규화하므로 그대로 호환된다.
- segmentation_models_pytorch(smp) 사용. 인코더/디코더를 인자로 바꿔 실험 가능.

의존성: pip install segmentation-models-pytorch  (설치돼 있음, smp 0.5.0)
"""

import segmentation_models_pytorch as smp

__all__ = ["build_model", "DECODERS"]

# 디코더 이름 -> smp 클래스
DECODERS = {
    "unetpp": smp.UnetPlusPlus,   # 기본(권장): nested skip, 경계에 유리
    "unet": smp.Unet,             # 대조군: 순정 U-Net
    "manet": smp.MAnet,           # attention 계열: 디코더에 self/채널 attention 내장
}

# decoder_attention_type("scse") 를 받는 디코더(smp 0.5.0). manet 은 자체 attention 이라 미지원.
_ATTN_CAPABLE = {"unetpp", "unet"}


def build_model(encoder_name="efficientnet-b3",
                encoder_weights="imagenet",
                decoder="unetpp",
                decoder_attention=None,
                num_classes=1,
                in_channels=3):
    """U-Net++/U-Net/MAnet 세그 모델 생성.

    encoder_name : smp 인코더 이름.
        - CNN: efficientnet-b3/b0, resnet34/50, mobilenet_v2 ...
        - HRNet(고해상도 표현을 끝까지 유지 -> 작은 병변에 강함): timm 경유
          "tu-hrnet_w18", "tu-hrnet_w32", "tu-hrnet_w48" (imagenet 가중치 자동 다운로드).
    encoder_weights: "imagenet"(사전학습) 또는 None(from-scratch).
    decoder      : "unetpp"(기본) / "unet" / "manet"(attention 디코더).
    decoder_attention: None(기본) 또는 "scse"(spatial+channel squeeze-excite 게이트).
        unet/unetpp 에서만 유효(작은 병변 영역에 집중, Attention U-Net 계열 효과).
        manet 은 자체 attention 이라 지정해도 무시된다.
    반환: nn.Module. forward(x)->(B, num_classes, H, W) logits (sigmoid 미적용).
    """
    if decoder not in DECODERS:
        raise ValueError(f"decoder 는 {list(DECODERS)} 중 하나여야 함: {decoder}")

    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    if decoder_attention:
        if decoder in _ATTN_CAPABLE:
            kwargs["decoder_attention_type"] = decoder_attention
        else:
            print(f"[model] '{decoder}' 는 decoder_attention 미지원 -> 무시 "
                  f"(요청값 {decoder_attention}). manet 은 자체 attention 사용.")
    return DECODERS[decoder](**kwargs)
