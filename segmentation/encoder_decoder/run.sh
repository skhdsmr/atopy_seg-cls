#!/usr/bin/env bash
# =============================================================================
# run.sh - U-Net++ (EfficientNet-b3, ImageNet 사전학습) 세그 학습
#
# 데이터/증강/손실/지표는 segmentation/ 공유 모듈 재사용, 모델만 model.py(smp).
#
# 사용법:
#   bash run.sh                                   # 기본(dataset_lesion_merged, 100ep, 512, bs16)
#   DATA=../../dataset_face bash run.sh           # 다른 데이터셋
#   ENCODER=efficientnet-b0 bash run.sh           # 인코더 교체(CPU 배포용 경량)
#   ATT=scse NAME=effb3_scse bash run.sh          # 디코더 attention(작은 병변 집중)
#   ENCODER=tu-hrnet_w18 DECODER=unet NAME=hrnet18 bash run.sh  # HRNet(고해상도 유지)
#   DECODER=manet NAME=manet_effb3 bash run.sh    # MAnet(attention 디코더)
#   LOSS=tversky ALPHA=0.7 NAME=effb3_tv07 bash run.sh   # 과탐 억제(FP 벌점)
#   LOSS=focaltversky ALPHA=0.3 GAMMA=1.33 NAME=effb3_ft bash run.sh  # 미탐↓(작은병변) 집중
#   LOSS=focaltversky ALPHA=0.7 GAMMA=1.33 NAME=effb3_ft_fp bash run.sh  # 과탐↓ + 어려운이미지 집중
#   ENC_LR_SCALE=0.1 bash run.sh                  # 인코더 LR 낮춤(과적합 억제)
#   FREEZE=5 bash run.sh                          # 초기 5ep 인코더 동결(디코더 워밍업)
#   NOPRE=1 NAME=effb3_scratch bash run.sh        # from-scratch 대조군
#   PATIENCE=20 bash run.sh                        # early stop(val Dice 20ep 미갱신 시 종료, 기본값)
#   PATIENCE=0 bash run.sh                          # early stop 끄기(전체 epoch 학습)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- torch 환경 격리 (학습 전용 venv) ---------------------------------------
# 이 컨테이너는 전역 PYTHONPATH 와 LD_LIBRARY_PATH 가 ~/.local·시스템 torch 를 가리켜서,
# 어떤 python 으로 torch 를 import 해도 CPU 빌드 .so 가 섞여 `_OutOfMemoryError 없음`
# 에러가 난다. 학습은 CUDA 빌드 torch 를 담은 venv(.venv-train)에서 돌리되, 아래 두 변수를
# venv 기준으로 재설정해 우리 빌드만 로드되게 한다:
#   - PYTHONPATH  : 비워서 ~/.local 의 CPU torch 가 venv 를 가리지 못하게 한다.
#   - LD_LIBRARY_PATH : venv torch/lib(+ 번들 CUDA nvidia/*/lib)를 맨 앞에, 뒤엔 드라이버.
#   준비:  python3 -m venv .venv-train
#          PYTHONPATH= .venv-train/bin/pip install -r requirements-train.txt
export PYTHONPATH=""
VENV="${VENV:-$(cd ../.. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ -x "$PY" ]; then
  SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  NV_LIBS="$(printf '%s:' "$SP"/nvidia/*/lib)"
  export LD_LIBRARY_PATH="$SP/torch/lib:${NV_LIBS}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  PY=python3
  echo "경고: 학습용 venv($VENV) 를 찾지 못해 시스템 python3 로 실행합니다(torch 충돌 가능)." >&2
fi

DATA="${DATA:-../../dataset_lesion_merged_pred}"   # 학습 데이터 루트
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"        # early stop: val Dice 미갱신 N ep 시 종료. 0=끄기
BATCH="${BATCH:-8}"               # 1024 + U-Net++ 는 무거움. 물리 배치는 작게.
ACCUM="${ACCUM:-2}"              # 실효 배치 = BATCH*ACCUM (기본 8*2=16)
IMGSZ="${IMGSZ:-512}"
LR="${LR:-0.0001}"                # 사전학습 미세조정 -> UNeXt(1e-3)보다 낮음
ENC_LR_SCALE="${ENC_LR_SCALE:-1.0}"
FREEZE="${FREEZE:-0}"
DEVICE="${DEVICE:-cuda}"
NAME="${NAME:-effb0_unetpp_lesion_pred}"
ENCODER="${ENCODER:-efficientnet-b0}"   # HRNet: tu-hrnet_w18/w32/w48 (고해상도 유지->작은병변) / efficientnet-b0(경량)
DECODER="${DECODER:-unetpp}"            # unetpp / unet / manet(attention 디코더)
ATT="${ATT:-none}"                      # none / scse(디코더 attention 게이트, unet·unetpp 만)
LOSS="${LOSS:-tversky}"           # bcedice / tversky / focaltversky. alpha 는 tversky·focaltversky 에서 적용
ALPHA="${ALPHA:-0.5}"             # alpha>beta=FP벌점(과탐↓), alpha<beta=FN벌점(미탐↓, 작은병변용)
GAMMA="${GAMMA:-1}"           # focaltversky 지수(>1: 어려운 이미지 집중). LOSS=focaltversky 에서만
AUG="${AUG:-on}"
CROP="${CROP:-0}"                 # >0 이면 crop 학습 + 타일드 평가 (작은 병변용, 예: 512)
CROP_POS="${CROP_POS:-0.4}"       # 병변 포함 crop 비율(나머지 배경 crop)
CROP_FG="${CROP_FG:-0.07}"        # >0 이면 조건부 crop: fg<이값인 희소만 crop, 밀집은 full

AUG_FLAG=""; [ "$AUG" = "off" ] && AUG_FLAG="--no_aug"
PRE_FLAG=""; [ "${NOPRE:-0}" = "1" ] && PRE_FLAG="--no_pretrained"
AMP_FLAG=""; [ "${NOAMP:-0}" = "1" ] && AMP_FLAG="--no_amp"

echo "==================== U-Net++ / $ENCODER 설정 ===================="
echo " DATA=$DATA  EPOCHS=$EPOCHS  PATIENCE=$([ "$PATIENCE" -gt 0 ] && echo "$PATIENCE" || echo off)  BATCH=$BATCH x ACCUM=$ACCUM (실효 $((BATCH*ACCUM)))  IMGSZ=$IMGSZ"
echo " LR=$LR  ENC_LR_SCALE=$ENC_LR_SCALE  FREEZE=$FREEZE  LOSS=$LOSS  ALPHA=$ALPHA  GAMMA=$GAMMA  AMP=$([ "${NOAMP:-0}" = "1" ] && echo off || echo on)"
echo " DECODER=$DECODER  ATT=$ATT  AUG=$AUG  CROP=$CROP  CROP_POS=$CROP_POS  CROP_FG=$CROP_FG(조건부)"
echo " PRETRAINED=$([ "${NOPRE:-0}" = "1" ] && echo no || echo imagenet)  결과 -> runs/$NAME"
echo "================================================================"

"$PY" train.py \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch "$BATCH" \
  --accum "$ACCUM" \
  --imgsz "$IMGSZ" \
  --lr "$LR" \
  --encoder_lr_scale "$ENC_LR_SCALE" \
  --freeze_encoder "$FREEZE" \
  --device "$DEVICE" \
  --name "$NAME" \
  --encoder "$ENCODER" \
  --decoder "$DECODER" \
  --decoder_attention "$ATT" \
  --loss "$LOSS" \
  --alpha "$ALPHA" \
  --ft_gamma "$GAMMA" \
  --crop_size "$CROP" \
  --crop_pos_ratio "$CROP_POS" \
  --crop_fg_thresh "$CROP_FG" \
  $AUG_FLAG $PRE_FLAG $AMP_FLAG

echo "완료. 체크포인트/로그: runs/$NAME"
