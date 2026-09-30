#!/usr/bin/env bash
# =============================================================================
# run_multitask.sh — 조건 B(표준 multi-task, 지식증류 없음) 학습
#
# 사용법:
#   bash run_multitask.sh                                # DATA 기본값(아토피 dataset_all_final)
#   DATA=/path/to/dataset bash run_multitask.sh
#   DATA=... MODEL=pvtv2b0 IMGSZ=224 bash run_multitask.sh
#   DATA=... EXP=bbox bash run_multitask.sh              # 마스크 union bbox 크롭
#   DATA=... EXP=bbox MARGIN=0.2 BBOX_SQUARE=1 bash run_multitask.sh
#   DATA=... MTL=fixed IGA_WEIGHT=0.5 bash run_multitask.sh   # Kendall 대신 고정 배분
#   DATA=... RESUME=1 bash run_multitask.sh                    # 끊긴 학습 이어서
# =============================================================================
set -uo pipefail

_abs() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
for _v in DATA PROJECT; do
  eval "_cur=\${$_v:-}"
  [ -n "$_cur" ] && eval "$_v=\$(_abs \"\$_cur\")"
done
unset _v _cur

cd "$(dirname "$0")"

export PYTHONPATH=""
export PYTHONUNBUFFERED=1   # 로그 파일로 리다이렉트해도 줄단위로 바로바로 flush
VENV="${VENV:-$(cd .. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ -x "$PY" ]; then
  SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  shopt -s nullglob
  NV_LIBS="$(printf '%s:' "$SP"/nvidia/*/lib)"
  shopt -u nullglob
  export LD_LIBRARY_PATH="$SP/torch/lib:${NV_LIBS}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  PY=python3
  echo "경고: 학습 venv($VENV) 없음 -> 시스템 python3." >&2
fi

MODEL="${MODEL:-pvtv2b0}"
IMGSZ="${IMGSZ:-224}"
DATA="${DATA:-$(cd .. && pwd)/dataset_all_final/images}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
PROJECT="${PROJECT:-$(pwd)/runs}"
EXP="${EXP:-full}"                   # full | bbox
MASKS="${MASKS:-}"
MARGIN="${MARGIN:-0.15}"
BBOX_SQUARE="${BBOX_SQUARE:-0}"
MTL="${MTL:-uncertainty}"            # uncertainty | fixed
IGA_WEIGHT="${IGA_WEIGHT:-0.5}"
RESUME="${RESUME:-0}"
EXTRA="${EXTRA:-}"

CROP_FLAGS=(--exp "$EXP" --margin "$MARGIN")
[ -n "$MASKS" ] && CROP_FLAGS+=(--masks "$MASKS")
[ "$BBOX_SQUARE" = "1" ] && CROP_FLAGS+=(--bbox_square)
RESUME_FLAG=()
[ "$RESUME" = "1" ] && RESUME_FLAG=(--resume)

"$PY" train_multitask.py --data "$DATA" --model "$MODEL" --imgsz "$IMGSZ" \
  --epochs "$EPOCHS" --patience "$PATIENCE" --batch "$BATCH" --workers "$WORKERS" \
  --device "$DEVICE" --project "$PROJECT" --mtl "$MTL" --iga_weight "$IGA_WEIGHT" \
  "${CROP_FLAGS[@]}" "${RESUME_FLAG[@]}" $EXTRA

echo "완료 -> $PROJECT"
