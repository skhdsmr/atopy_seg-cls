#!/usr/bin/env bash
# =============================================================================
# run.sh - UNeXt 로 아토피 병변 semantic segmentation 학습
#
# 데이터: dataset_face (YOLO 폴리곤 -> on-the-fly 0/1 마스크). val/test 는
#         split_all.py 가 atopy VL/VS 이름으로 고정한 split 을 그대로 사용.
#
# 사용법:
#   bash run.sh                          # 기본(dataset_face, 100ep, 512, bs8, 증강 ON)
#   EPOCHS=200 BATCH=16 bash run.sh      # 하이퍼파라미터 변경
#   DATA=../../dataset_lesion bash run.sh   # 다른 데이터셋으로 학습
#   AUG=off NAME=unext_face_noaug bash run.sh   # 증강 끄고 baseline 비교
#   LOSS=tversky ALPHA=0.7 NAME=unext_tv07 bash run.sh   # Tversky(오탐 벌점)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- 하이퍼파라미터 (환경변수로 덮어쓰기 가능) -------------------------------
DATA="${DATA:-../../dataset_deeplab}"   # 학습 대상 dataset_* 루트
EPOCHS="${EPOCHS:-100}"           # 학습 epoch
BATCH="${BATCH:-8}"               # 배치 크기
IMGSZ="${IMGSZ:-512}"             # 입력 크기(32의 배수)
LR="${LR:-0.001}"                 # AdamW 초기 학습률
DEVICE="${DEVICE:-cuda}"          # cuda 또는 cpu
NAME="${NAME:-unext_face_new}"        # 실험 이름 -> runs/<NAME>
AUG="${AUG:-on}"                  # on=온라인 증강 사용, off=끄고 baseline
LOSS="${LOSS:-bcedice}"           # bcedice(baseline) 또는 tversky
ALPHA="${ALPHA:-0.5}"             # Tversky alpha(FP 벌점). beta=1-alpha
CROP="${CROP:-0}"                 # >0 이면 crop 학습 + 타일드 평가 (작은 병변용, 예: 512)
CROP_POS="${CROP_POS:-0.7}"       # 병변 포함 crop 비율

# AUG=off 면 --no_aug 플래그 전달
AUG_FLAG=""
if [ "$AUG" = "off" ]; then AUG_FLAG="--no_aug"; fi

echo "==================== UNeXt 설정 ===================="
echo " DATA=$DATA  EPOCHS=$EPOCHS  BATCH=$BATCH  IMGSZ=$IMGSZ  LR=$LR  DEVICE=$DEVICE  AUG=$AUG"
echo " LOSS=$LOSS  ALPHA=$ALPHA  CROP=$CROP  결과 -> runs/$NAME"
echo "===================================================="

python3 train_unext.py \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --lr "$LR" \
  --device "$DEVICE" \
  --name "$NAME" \
  --loss "$LOSS" \
  --alpha "$ALPHA" \
  --crop_size "$CROP" \
  --crop_pos_ratio "$CROP_POS" \
  $AUG_FLAG

echo "완료. 체크포인트/로그: runs/$NAME"
