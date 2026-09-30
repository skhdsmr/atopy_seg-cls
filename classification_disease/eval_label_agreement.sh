#!/usr/bin/env bash
# =============================================================================
# eval_label_agreement.sh - 합성 이미지(GAN/SD) 라벨 일치율 측정
#
# evaluate.sh 와 동일하게 학습 venv(CUDA torch) 격리만 해 주고 파이썬으로 넘긴다.
#
# 사용법:
#   bash eval_label_agreement.sh                                   # 기본 런(front)
#   bash eval_label_agreement.sh --run dis_effb0_r512_both         # 비교용
#   bash eval_label_agreement.sh --arm gan --arm sd_ft --per_class 500
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

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

exec "$PY" eval_label_agreement.py "$@"
