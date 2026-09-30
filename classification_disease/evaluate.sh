#!/usr/bin/env bash
# =============================================================================
# evaluate.sh - 저장된 체크포인트 평가 (acc / F1 / 혼동행렬)
#
# run.sh 와 동일한 학습 venv(CUDA torch) 격리만 해 주고 evaluate.py 로 넘긴다.
# 인자는 그대로 전달된다.
#
# 사용법:
#   bash evaluate.sh --list                              # 런 목록
#   bash evaluate.sh --run dis_effb0_r512_both           # test, best.pt
#   bash evaluate.sh --run dis_effb0_r512_both --png     # 혼동행렬 PNG 도 저장
#   bash evaluate.sh --run dis_effb0_r512_both --split val --ckpt last
#   bash evaluate.sh --run all                           # 전 런 비교표
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

exec "$PY" evaluate.py "$@"
