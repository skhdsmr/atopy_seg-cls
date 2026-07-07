#!/usr/bin/env bash
# =============================================================================
# run.sh - 아토피 멀티태스크 분류 (MobileNetV4 / timm, 원본 이미지, QWK 지표)
#
# 모델명 = mobilenetv4_${VERSION}_${SIZE}
#   VERSION : conv | hybrid
#   SIZE    : small | medium | large   (hybrid 는 medium/large 만)
#
# 사용법:
#   bash run.sh                                  # 기본: conv medium, atopy 원본, 50ep
#   SOURCE=face bash run.sh                       # dataset_face 서브셋만 학습
#   SOURCE=lesion bash run.sh                     # dataset_lesion 서브셋만 학습
#   VERSION=hybrid SIZE=medium bash run.sh       # 하이브리드 medium
#   SIZE=large bash run.sh                        # conv large
#   CLASS_WEIGHT=1 bash run.sh                    # 불균형 보정
#   IMGSZ=256 bash run.sh                         # 입력 크기(사전학습 r224/r256)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VERSION="${VERSION:-conv}"             # conv | hybrid
SIZE="${SIZE:-medium}"                 # small | medium | large
VIEWS="${VIEWS:-both}"                 # front | side | both (source=atopy 일 때만)
SOURCE="${SOURCE:-face}"              # atopy | face | lesion  (이미지 소스 서브셋)
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-384}"
LR="${LR:-3e-4}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"   # 과적합 억제: 0.1~0.3 (backbone 천천히)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
DROPOUT="${DROPOUT:-0.5}"
EMBED_DIM="${EMBED_DIM:-512}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
CLASS_WEIGHT="${CLASS_WEIGHT:-0}"      # 0=off, 1=on
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"
# source=atopy 는 뷰 태그, face/lesion 은 소스명 태그
TAG="$VIEWS"; [ "$SOURCE" != "atopy" ] && TAG="$SOURCE"
NAME="${NAME:-cls_mobilenetv4_${VERSION}_${SIZE}_${TAG}}"

CW_FLAG=""; [ "$CLASS_WEIGHT" = "1" ] && CW_FLAG="--class_weight"

echo "==================== MobileNetV4 분류 설정 ===================="
echo " MODEL=mobilenetv4_${VERSION}_${SIZE}  SOURCE=$SOURCE  VIEWS=$VIEWS  IMGSZ=$IMGSZ  EMBED=$EMBED_DIM"
echo " EPOCHS=$EPOCHS  BATCH=$BATCH  LR=$LR (bb×$BACKBONE_LR_SCALE)  WD=$WEIGHT_DECAY  DROPOUT=$DROPOUT  LS=$LABEL_SMOOTHING  CLASS_WEIGHT=$CLASS_WEIGHT"
echo " 지표=QWK(주)/±1/acc   결과 -> runs/$NAME"
echo "=============================================================="

python3 train.py \
  --version "$VERSION" --size "$SIZE" --views "$VIEWS" --source "$SOURCE" \
  --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ" \
  --lr "$LR" --backbone_lr_scale "$BACKBONE_LR_SCALE" \
  --weight_decay "$WEIGHT_DECAY" --dropout "$DROPOUT" \
  --embed_dim "$EMBED_DIM" --label_smoothing "$LABEL_SMOOTHING" \
  --workers "$WORKERS" --device "$DEVICE" --name "$NAME" \
  $CW_FLAG

echo "완료. 체크포인트: runs/$NAME"
