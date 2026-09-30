#!/usr/bin/env bash
# =============================================================================
# run.sh - 질환 6-way 분류 학습 (건선/아토피/여드름/정상/주사/지루)
#
# 중증도/징후는 다루지 않는다. '어떤 질환인가'만 맞힌다.
# 사전 단계로 dataset_disease/ 를 1회 생성한다(없을 때만).
#
# test 는 제공된 Validation(새 환자 + 새 출처)이라 최종 1회만 본다.
#
# 사용법:
#   bash run.sh                        # dataset 생성 후 학습 (effb0, r224, 정면+측면)
#   ANGLE=front bash run.sh            # 정면만(1024 실사 얼굴)
#   ANGLE=side  bash run.sh            # 측면만(512 피부 접사)
#   MODEL=mnv4s IMGSZ=256 bash run.sh
#   CLASS_WEIGHT=1 bash run.sh         # 역빈도 클래스 가중(클래스 균형이라 기본 off)
#   LINK=1 bash run.sh                 # 이미지 복사 대신 심볼릭 링크(2GB+ 절약)
#   SKIP_DATA=1 bash run.sh            # dataset 재생성 생략(이미 있으면 자동 생략됨)
#   FORCE=1 bash run.sh                # 완료된 런(done.txt)도 재학습
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
IMGSZ="${IMGSZ:-512}"
DATA="${DATA:-../dataset_disease}"
SRC="${SRC:-../skin_dataset}"
ANGLE="${ANGLE:-both}"          # both | front | side
VAL_FRAC="${VAL_FRAC:-0.125}"
SELECT="${SELECT:-macro_f1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
CW_FLAG=""; [ "${CLASS_WEIGHT:-0}" = "1" ] && CW_FLAG="--class_weight"
LINK_FLAG=""; [ "${LINK:-0}" = "1" ] && LINK_FLAG="--link"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
DROPOUT="${DROPOUT:-0.5}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"

if [ -n "${BATCH:-}" ]; then B="$BATCH"; else
  case "$IMGSZ" in 512) B=8;; 384) B=16;; 256) B=24;; *) B=32;; esac
fi

# --- 1) dataset_disease 생성 (없을 때만) ---
if [ "${SKIP_DATA:-0}" != "1" ] && [ ! -f "$DATA/labels.csv" ]; then
  echo "==================== dataset_disease 생성 -> $DATA ===================="
  "$PY" make_dataset_disease.py --src "$SRC" --out "$DATA" \
    --val_frac "$VAL_FRAC" $LINK_FLAG || {
    echo "[에러] dataset 생성 실패. SRC 경로($SRC)와 다운로드 완료 여부 확인." >&2; exit 1; }
else
  echo "[skip] dataset 존재($DATA) 또는 SKIP_DATA=1 -> 재생성 생략"
fi

# --- 2) 학습 ---
NAME="${NAME:-dis_${MODEL}_r${IMGSZ}_${ANGLE}_log2}"
if [ -f "runs/$NAME/done.txt" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[skip] 완료된 런: runs/$NAME (FORCE=1 로 재학습)"; exit 0
fi
echo "==================== [disease] $MODEL r$IMGSZ angle=$ANGLE (batch $B, ${EPOCHS}ep) ===================="
"$PY" train.py \
  --model "$MODEL" --imgsz "$IMGSZ" --batch "$B" --data "$DATA" --angle "$ANGLE" \
  --select "$SELECT" --label_smoothing "$LABEL_SMOOTHING" $CW_FLAG \
  --epochs "$EPOCHS" --patience "$PATIENCE" \
  --dropout "$DROPOUT" --device "$DEVICE" --workers "$WORKERS" \
  --name "$NAME" || { echo "[에러] 학습 실패" >&2; exit 1; }

echo "[done] runs/$NAME/test_report.json 에 클래스별 P/R/F1 + 혼동행렬 + 출처별 정확도 저장됨"
