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

from emcad import EMCADSegmenter

__all__ = ["build_model", "DECODERS"]

# 디코더 이름 -> smp 클래스. "emcad" 는 smp 에 없어 emcad.py 이식본을 쓴다(아래 분기).
DECODERS = {
    "unetpp": smp.UnetPlusPlus,   # 기본(권장): nested skip, 경계에 유리
    "unet": smp.Unet,             # 대조군: 순정 U-Net
    "manet": smp.MAnet,           # attention 계열: 디코더에 self/채널 attention 내장
    "emcad": None,                # EMCAD(CVPR'24): 멀티스케일 conv attention, PVTv2 와 조합 권장
}

# decoder_attention_type("scse") 를 받는 디코더(smp 0.5.0). manet 은 자체 attention 이라 미지원.
_ATTN_CAPABLE = {"unetpp", "unet"}

# UNet++ 는 depth=5 를 요구해서, 1/2 스테이지가 없는 트랜스포머 인코더(PVTv2 등)에서는
# smp 가 끼워 넣는 0채널 더미 스테이지 때문에 conv weight 가 [0, ...] 이 되어 터진다.
# 해당 인코더는 unet / emcad 로 안내한다.
_NO_HALF_STAGE_PREFIXES = ("tu-pvt_v2", "tu-swin", "tu-mit_", "mit_b")


def build_model(encoder_name="efficientnet-b3",
                encoder_weights="imagenet",
                decoder="unetpp",
                decoder_attention=None,
                num_classes=1,
                in_channels=3):
    """U-Net++/U-Net/MAnet/EMCAD 세그 모델 생성.

    encoder_name : smp 인코더 이름.
        - CNN: efficientnet-b3/b0, resnet34/50, mobilenet_v2 ...
        - HRNet(고해상도 표현을 끝까지 유지 -> 작은 병변에 강함): timm 경유
          "tu-hrnet_w18", "tu-hrnet_w32", "tu-hrnet_w48" (imagenet 가중치 자동 다운로드).
        - 트랜스포머(1/4 부터 시작, 전역 문맥 -> 넓은 병변): "tu-pvt_v2_b0"~"tu-pvt_v2_b5".
          단 1/2 스테이지가 없어 unetpp 와는 호환되지 않는다(unet 또는 emcad 사용).
    encoder_weights: "imagenet"(사전학습) 또는 None(from-scratch).
    decoder      : "unetpp"(기본) / "unet" / "manet"(attention 디코더) / "emcad".
    decoder_attention: None(기본) 또는 "scse"(spatial+channel squeeze-excite 게이트).
        unet/unetpp 에서만 유효(작은 병변 영역에 집중, Attention U-Net 계열 효과).
        manet/emcad 는 자체 attention 이라 지정해도 무시된다.
    반환: nn.Module. forward(x)->(B, num_classes, H, W) logits (sigmoid 미적용).
    """
    if decoder not in DECODERS:
        raise ValueError(f"decoder 는 {list(DECODERS)} 중 하나여야 함: {decoder}")

    if decoder == "emcad":
        if decoder_attention:
            print(f"[model] 'emcad' 는 decoder_attention 미지원 -> 무시 "
                  f"(요청값 {decoder_attention}). 자체 CAB/SAB attention 사용.")
        return EMCADSegmenter(encoder_name=encoder_name,
                              encoder_weights=encoder_weights,
                              num_classes=num_classes, in_channels=in_channels)

    if decoder == "unetpp" and encoder_name.startswith(_NO_HALF_STAGE_PREFIXES):
        raise ValueError(
            f"인코더 '{encoder_name}' 는 1/2 해상도 스테이지가 없어 unetpp 와 호환되지 않습니다"
            f"(smp 가 0채널 더미 스테이지를 만들어 conv 가 터짐). "
            f"DECODER=unet 또는 DECODER=emcad 로 실행하세요.")

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
