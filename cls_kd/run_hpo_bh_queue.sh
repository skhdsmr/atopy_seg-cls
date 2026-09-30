#!/usr/bin/env bash
# =============================================================================
# run_hpo_bh_queue.sh — 지금 돌고 있는 run_anneal_sweep.sh 가 끝날 때까지 기다렸다가,
# 조건 B~H 하이퍼파라미터 탐색(hpo_bh.py, 각 60 trial)을 순서대로 실행한다.
#
# 조건 A HPO(hpo_a.py)와 달리 B~H는 서로 독립된 study이고, trial 하나가 모델 1개
# 학습이라(조건 A처럼 5축 반복이 아님) 60trial x 7조건 = 420개 학습.
#
# 사용법: nohup setsid bash run_hpo_bh_queue.sh > runs_hpo_bh/logs/queue_driver.log 2>&1 &
#         disown
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=""
export PYTHONUNBUFFERED=1
VENV="$(cd .. && pwd)/.venv-train"
PY="$VENV/bin/python"
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

LOGROOT="$(pwd)/runs_hpo_bh/logs"
mkdir -p "$LOGROOT"

echo "[$(date '+%m-%d %H:%M:%S')] run_anneal_sweep.sh 종료 대기 중..."
while pgrep -f run_anneal_sweep.sh > /dev/null 2>&1; do
  sleep 60
done
echo "[$(date '+%m-%d %H:%M:%S')] anneal_epochs 스윕 종료 확인 -> 조건 B~H HPO 시작"

for cond in B C D E F G H; do
  echo "[$(date '+%m-%d %H:%M:%S')] === 조건 ${cond} 탐색 시작 (60 trial) ==="
  "$PY" -W ignore hpo_bh.py --cond "$cond" search --n_trials 60 --n_jobs 2 --workers 5 \
    >> "$LOGROOT/hpo_${cond}.log" 2>&1
  "$PY" -W ignore hpo_bh.py --cond "$cond" report >> "$LOGROOT/hpo_${cond}.log" 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] === 조건 ${cond} 탐색 완료 ==="
done

echo "[$(date '+%m-%d %H:%M:%S')] === 조건 B~H 전체 HPO(60trial x 7조건) 완료 ==="
