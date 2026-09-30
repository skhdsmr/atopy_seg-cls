#!/usr/bin/env bash
# =============================================================================
# run_canet_sweep.sh — CANet(cross-disease attention) 4런 스윕
#
#   모델 2종 {effb0, pvtv2b0}  ×  입력 2종 {full(원본 전체), bbox(마스크 union bbox 크롭)}
#   × IBB 2종 {끔, 켬}  (IBB_LIST="0 1" 일 때. 기본은 끔만 -> 4런)
#
# CANet 은 xmengli/CANet 의 crossCBAM 분기를 5-태스크로 이식한 것(canet.py).
# CCNN(--pairwise) / IBB(--ibb) 와는 결합하지 않는다 — 순수 CANet 축만 본다.
#
# 대조군(이미 runs/ 에 있음, 같은 조건에서 --canet 없이 돌린 것):
#   crop_bbox_effb0_r512_b32      test_mean_qwk=0.4995
#   crop_bbox_pvtv2b0_r512_b32    test_mean_qwk=0.5297
# full 입력은 대조군이 없으므로 필요하면 CANET=0 으로 같은 스크립트를 한 번 더 돌린다.
#
# 사용법:
#   bash run_canet_sweep.sh                     # 4런 (r512, 100ep)
#   IBB_LIST="0 1" bash run_canet_sweep.sh      # 8런 (IBB on/off 축 추가)
#   MODELS_TO_RUN=effb0 bash run_canet_sweep.sh # 특정 모델만
#   EXP_LIST=full bash run_canet_sweep.sh       # 특정 입력만
#   CANET=0 SUFFIX=_base bash run_canet_sweep.sh# 어텐션 없는 대조군 4런
#   IMGSZ=224 FORCE=1 bash run_canet_sweep.sh   # 해상도 변경 / 완료된 런 재학습
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# --- 학습 venv(CUDA torch) 격리 (다른 run.sh 와 동일 패턴) ---
export PYTHONPATH=""
VENV="${VENV:-$(cd .. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ ! -x "$PY" ]; then
  echo "[에러] 학습 venv 없음: $PY" >&2; exit 1
fi

# --- 스윕 축 ---
MODELS_TO_RUN="${MODELS_TO_RUN:-effb0 pvtv2b0}"
EXP_LIST="${EXP_LIST:-full bbox}"     # full=원본 이미지 전체 / bbox=마스크 bbox 크롭
IMGSZ="${IMGSZ:-512}"
SUFFIX="${SUFFIX:-_canet}"

# --- CANet 노브 ---
CANET="${CANET:-1}"                   # 0 이면 어텐션 없는 대조군(MultiTaskNet)
CANET_LAMBDA="${CANET_LAMBDA:-0.25}"  # specific 보조헤드 가중(저자 lambda_value 기본)
CANET_EDGE_V="${CANET_EDGE_V:-0.25}"  # 교차 엣지 Cramér's V 하한 -> 4엣지
CANET_FUSE="${CANET_FUSE:-mean}"      # 이웃 2개 이상일 때 주입 합성
CANET_REDUCTION="${CANET_REDUCTION:-16}"

# --- IBB 축 (2-step 균형배치 샘플러). "0 1" 이면 대조군까지 자동 생성(런 2배) ---
IBB_LIST="${IBB_LIST:-0}"
IBB_ARGS="${IBB_ARGS:---ibb_alpha 0.5 --ibb_cap 4.0 --ibb_edge_v 0.25 --ibb_step2_ratio 0.5 --ibb_warmup 3}"

# --- 공통 고정 (기존 run_sweep.sh 와 동일 값 = 대조군과 비교 가능) ---
DATA="${DATA:-../dataset_all_final/images}"
MASKS="${MASKS:-../atopy_crop_masks}"
CKPT="${CKPT:-../segmentation/encoder_decoder/runs/effb0_unetpp_512/checkpoint_best.pth}"
LOSS="${LOSS:-corn}"
MTL="${MTL:-uncertainty}"
MARGIN="${MARGIN:-0.15}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-3e-4}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
DROPOUT="${DROPOUT:-0.5}"
EMBED_DIM="${EMBED_DIM:-512}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"

# 해상도별 기본 배치(run_sweep.sh 와 동일 표). CANet 은 태스크별 CBAM 으로 활성값이
# 늘어나므로 512 에서 OOM 이 나면 BATCH=4 로 낮춘다.
if [ -n "${BATCH:-}" ]; then B="$BATCH"; else
  case "$IMGSZ" in 512) B=8 ;; 384) B=16 ;; 256) B=24 ;; *) B=32 ;; esac
fi

CANET_ARGS=""
if [ "$CANET" = "1" ]; then
  CANET_ARGS="--canet --canet_lambda $CANET_LAMBDA --canet_edge_v $CANET_EDGE_V \
--canet_fuse $CANET_FUSE --canet_reduction $CANET_REDUCTION"
fi

# --- bbox 런이 있으면 마스크 확인 (full 만 돌리면 불필요) ---
if echo "$EXP_LIST" | grep -qw bbox; then
  if [ "${SKIP_MASKS:-0}" != "1" ] && [ ! -d "$MASKS/train" ]; then
    echo "==================== 마스크 사전계산 -> $MASKS ===================="
    "$PY" precompute_masks.py --ckpt "$CKPT" --src "$DATA" --out "$MASKS" || {
      echo "[에러] 마스크 생성 실패. seg ckpt/env 확인." >&2; exit 1; }
  else
    echo "[skip] 마스크 존재($MASKS) 또는 SKIP_MASKS=1 -> 재생성 생략"
  fi
fi

TOTAL=0
for E in $EXP_LIST; do for M in $MODELS_TO_RUN; do for I in $IBB_LIST; do TOTAL=$((TOTAL+1)); done; done; done
echo "###### CANet 스윕: 총 $TOTAL 런  (입력 [$EXP_LIST] × 모델 [$MODELS_TO_RUN] × ibb [$IBB_LIST], r$IMGSZ batch$B, ${EPOCHS}ep) ######"
echo "###### canet=$CANET lambda=$CANET_LAMBDA edge_v=$CANET_EDGE_V fuse=$CANET_FUSE | loss=$LOSS mtl=$MTL ######"
[ "${FORCE:-0}" = "1" ] && echo "(FORCE=1: 완료된 런도 재학습)"

IDX=0; RAN=0; SKIPPED=0; FAILED=""
for EXP in $EXP_LIST; do
  for M in $MODELS_TO_RUN; do
   for IB in $IBB_LIST; do
    # CANet(특징 층) × IBB(데이터 층)는 직교 축이라 그대로 곱한다.
    if [ "$IB" = "1" ]; then IBB_TAG="_ibb"; IB_ARGS="--ibb $IBB_ARGS"
    else                     IBB_TAG="";     IB_ARGS=""; fi
    NAME="crop_${EXP}_${M}_r${IMGSZ}${IBB_TAG}${SUFFIX}"
    IDX=$((IDX+1))
    if [ "${FORCE:-0}" != "1" ] && [ -f "runs/$NAME/done.txt" ]; then
      echo "[$IDX/$TOTAL] skip (완료됨): $NAME"; SKIPPED=$((SKIPPED+1)); continue
    fi
    echo ""
    echo "==================== [$IDX/$TOTAL] exp=$EXP  $M  r$IMGSZ  (batch $B) ibb=$IB ===================="
    echo " 결과 -> runs/$NAME"
    echo "======================================================================"
    mkdir -p "runs/$NAME"
    rm -f "runs/$NAME/done.txt"
    if "$PY" train.py \
        --exp "$EXP" --model "$M" --imgsz "$IMGSZ" --batch "$B" \
        --data "$DATA" --masks "$MASKS" \
        --loss "$LOSS" --mtl "$MTL" --epochs "$EPOCHS" --patience "$PATIENCE" \
        --margin "$MARGIN" \
        --lr "$LR" --backbone_lr_scale "$BACKBONE_LR_SCALE" \
        --weight_decay "$WEIGHT_DECAY" --dropout "$DROPOUT" \
        --embed_dim "$EMBED_DIM" --label_smoothing "$LABEL_SMOOTHING" \
        --workers "$WORKERS" --device "$DEVICE" --name "$NAME" $CANET_ARGS $IB_ARGS \
        2>&1 | tee "runs/$NAME/train.log"; then
      RAN=$((RAN+1))
    else
      echo "!! 실패(계속 진행): $NAME" >&2; FAILED="$FAILED $NAME"
    fi
   done
  done
done

echo ""
echo "###### 완료: 실행 $RAN / 건너뜀 $SKIPPED / 실패${FAILED:- 없음} ######"
echo "--- 결과 요약 ---"
for E in $EXP_LIST; do for M in $MODELS_TO_RUN; do for I in $IBB_LIST; do
  [ "$I" = "1" ] && T="_ibb" || T=""
  d="runs/crop_${E}_${M}_r${IMGSZ}${T}${SUFFIX}"
  printf '%-46s %s\n' "$(basename $d)" "$(grep -o 'test_mean_qwk=[0-9.]*' $d/done.txt 2>/dev/null | tail -1 || echo '(없음)')"
done; done; done
