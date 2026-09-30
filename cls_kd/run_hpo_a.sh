#!/usr/bin/env bash
# =============================================================================
# run_hpo_a.sh — 조건 A 하이퍼파라미터 탐색(pvtv2b0) : search -> confirm -> report
#
# 자세한 설계는 hpo_a.py 상단 docstring. 결과는 runs_hpo_a/ 아래:
#   leaderboard.txt   진행 중에도 trial 이 끝날 때마다 갱신되는 상위 15개(val 기준)
#   results.csv       전체 trial(축별 val/test QWK, best epoch, 설정)
#   final_report.md   confirm 까지 끝난 뒤의 최종 정리
#
# 사용법:
#   bash run_hpo_a.sh                                   # 포그라운드
#   setsid nohup bash run_hpo_a.sh > hpo_a.log 2>&1 < /dev/null & disown
#   N_TRIALS=80 N_JOBS=2 bash run_hpo_a.sh
#   중단된 뒤 같은 명령으로 다시 실행하면 study.db 에서 이어서 돈다.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=""
export PYTHONUNBUFFERED=1
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

N_TRIALS="${N_TRIALS:-60}"
N_JOBS="${N_JOBS:-2}"
WORKERS="${WORKERS:-5}"
TOP_K="${TOP_K:-3}"
SEEDS="${SEEDS:-1 2}"

echo "[hpo_a] 시작 $(date '+%F %T')  n_trials=$N_TRIALS n_jobs=$N_JOBS top_k=$TOP_K seeds=$SEEDS"
"$PY" -W ignore hpo_a.py search --n_trials "$N_TRIALS" --n_jobs "$N_JOBS" --workers "$WORKERS" || exit 1
"$PY" -W ignore hpo_a.py confirm --top_k "$TOP_K" --seeds $SEEDS --n_jobs "$N_JOBS" --workers "$WORKERS" || exit 1
"$PY" -W ignore hpo_a.py report || exit 1
echo "[hpo_a] 전체 완료 $(date '+%F %T') -> $(pwd)/runs_hpo_a/final_report.md"
