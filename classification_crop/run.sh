#!/usr/bin/env bash
# =============================================================================
# run.sh - crop 기반 아토피 중증도 분류: 세 실험(bbox / mil / twostream) 일괄 실행
#
# 방법1(마스킹 전체이미지=atopy_seg) 대안 3종을 같은 조건에서 비교한다.
#   bbox      : 마스크 union bbox + margin 크롭(마스킹 X)
#   mil       : 연결요소별 crop bag -> attention/max pooling
#   twostream : 전체(global) + bbox 크롭(local) concat
#
# 사전 단계로 분할 ckpt 추론 마스크(atopy_crop_masks/)를 1회 생성한다(없을 때만).
#
# 사용법:
#   bash run.sh                       # 마스크 생성 후 세 실험 모두 (effb0, r224)
#   EXP=mil bash run.sh               # 특정 실험만
#   MODEL=mnv4s IMGSZ=256 bash run.sh
#   SKIP_MASKS=1 bash run.sh          # 마스크 재생성 생략(이미 있으면 자동 생략됨)
#   IBB=1 bash run.sh                 # 2-step IBB 샘플링(runs/..._ibb 로 분리 저장)
#   IBB=1 IBB_ARGS="--ibb_alpha 0.7 --ibb_alpha_end 0.0 --ibb_warmup 5" bash run.sh
#   FORCE=1 bash run.sh               # 완료된 런(done.txt)도 재학습
#   MIL_POOL=max bash run.sh          # MIL 풀링 방식
#   TWOSTREAM_SHARE=1 bash run.sh     # two-stream 백본 공유
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# --- 학습 venv(CUDA torch) 격리 (다른 run.sh 와 동일 패턴) ---
export PYTHONPATH=""
VENV="${VENV:-$(cd .. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ -x "$PY" ]; then
  SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  NV_LIBS="$(printf '%s:' "$SP"/nvidia/*/lib)"
  export LD_LIBRARY_PATH="$SP/torch/lib:${NV_LIBS}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  PY=python3
  echo "경고: 학습 venv($VENV) 없음 -> 시스템 python3." >&2
fi

# --- 설정 ---
MODEL="${MODEL:-effb0}"
IMGSZ="${IMGSZ:-224}"
DATA="${DATA:-../dataset_all_final/images}"
MASKS="${MASKS:-../atopy_crop_masks}"
CKPT="${CKPT:-../segmentation/encoder_decoder/runs/effb0_unetpp_512/checkpoint_best.pth}"
LOSS="${LOSS:-corn}"
MTL="${MTL:-uncertainty}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
MARGIN="${MARGIN:-0.15}"
MIL_POOL="${MIL_POOL:-attention}"
MIN_AREA_FRAC="${MIN_AREA_FRAC:-0.003}"
MAX_INSTANCES="${MAX_INSTANCES:-8}"
DROPOUT="${DROPOUT:-0.5}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"
# 2-step IBB (sampler.py): IBB=1 로 켠다. 세부 노브는 IBB_ARGS 로 덮어쓴다.
IBB="${IBB:-0}"
IBB_ARGS="${IBB_ARGS:---ibb_alpha 0.5 --ibb_cap 4.0 --ibb_edge_v 0.25 --ibb_step2_ratio 0.5 --ibb_warmup 3}"
IBB_FLAG=""; [ "$IBB" = "1" ] && IBB_FLAG="--ibb $IBB_ARGS"

if [ -n "${BATCH:-}" ]; then B="$BATCH"; else
  case "$IMGSZ" in 512) B=8;; 384) B=16;; 256) B=24;; *) B=32;; esac
fi
TWOSTREAM_SHARE_FLAG=""; [ "${TWOSTREAM_SHARE:-0}" = "1" ] && TWOSTREAM_SHARE_FLAG="--twostream_share"

# --- 1) 마스크 사전계산 (없을 때만) ---
if [ "${SKIP_MASKS:-0}" != "1" ] && [ ! -d "$MASKS/train" ]; then
  echo "==================== 마스크 사전계산 -> $MASKS ===================="
  "$PY" precompute_masks.py --ckpt "$CKPT" --src "$DATA" --out "$MASKS" || {
    echo "[에러] 마스크 생성 실패. seg ckpt/env 확인." >&2; exit 1; }
else
  echo "[skip] 마스크 존재($MASKS) 또는 SKIP_MASKS=1 -> 재생성 생략"
fi

# --- 로그 기록 (run_sweep.sh 와 동일 규약) ---
LOGDIR="${LOGDIR:-runs/logs}"; mkdir -p "$LOGDIR"

# --- 2) 실험 실행 ---
EXPS="${EXP:-bbox}"
for exp in $EXPS; do
  NAME="crop_${exp}_${MODEL}_r${IMGSZ}_all"; [ "$IBB" = "1" ] && NAME="${NAME}_ibb"
  if [ -f "runs/$NAME/done.txt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] 완료된 런: runs/$NAME (FORCE=1 로 재학습)"; continue
  fi
  echo "==================== [$exp] $MODEL r$IMGSZ (batch $B, ${EPOCHS}ep) ===================="
  mkdir -p "runs/$NAME"; echo " 로그 -> runs/$NAME/train.log"
  "$PY" train.py \
    --exp "$exp" --model "$MODEL" --imgsz "$IMGSZ" --batch "$B" \
    --data "$DATA" --masks "$MASKS" \
    --loss "$LOSS" --mtl "$MTL" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --margin "$MARGIN" --mil_pool "$MIL_POOL" \
    --min_area_frac "$MIN_AREA_FRAC" --max_instances "$MAX_INSTANCES" \
    --dropout "$DROPOUT" --device "$DEVICE" --workers "$WORKERS" \
    --name "$NAME" $TWOSTREAM_SHARE_FLAG $IBB_FLAG \
    2>&1 | tee "runs/$NAME/train.log" || echo "[에러] $exp 학습 실패(계속 진행)" >&2
done
echo "[all done] runs/ 결과 비교: crop_bbox_* / crop_mil_* / crop_twostream_*"
