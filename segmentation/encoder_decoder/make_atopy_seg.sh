#!/usr/bin/env bash
# =============================================================================
# make_atopy_seg.sh - seg 체크포인트로 atopy_face 900장을 추론해 분류 입력용
#                     atopy_seg 를 생성. torch env 는 run.sh 와 동일하게 격리.
#
# 사용법:
#   bash make_atopy_seg.sh                         # 기본: effb0_unetpp_deeplab_512, mask 모드
#   MODE=soft SOFT_BG=0.3 bash make_atopy_seg.sh   # 배경 완전제거 대신 어둡게
#   MODE=bbox bash make_atopy_seg.sh               # 병변 bbox 크롭
#   CKPT=runs/xxx/checkpoint_best.pth bash make_atopy_seg.sh
#   DILATE=0 bash make_atopy_seg.sh                # 마스크 팽창 끔
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- torch env 격리 (run.sh 와 동일 패턴) ---
export PYTHONPATH=""
VENV="${VENV:-$(cd ../.. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ -x "$PY" ]; then
  SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  NV_LIBS="$(printf '%s:' "$SP"/nvidia/*/lib)"
  export LD_LIBRARY_PATH="$SP/torch/lib:${NV_LIBS}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  PY=python3
  echo "경고: 학습용 venv($VENV) 없음 -> 시스템 python3." >&2
fi

CKPT="${CKPT:-runs/effb0_unetpp_deeplab_512/checkpoint_best.pth}"
SRC="${SRC:-../../atopy_face}"
OUT="${OUT:-../../atopy_seg}"
MODE="${MODE:-mask}"
SOFT_BG="${SOFT_BG:-0.3}"
DILATE="${DILATE:-4}"
PAD="${PAD:-16}"

exec "$PY" make_atopy_seg.py \
  --ckpt "$CKPT" --src "$SRC" --out "$OUT" \
  --mode "$MODE" --soft-bg "$SOFT_BG" --dilate "$DILATE" --pad "$PAD"
