#!/usr/bin/env bash
# =============================================================================
# run.sh - 구진 싱글태스크 + 면적 스칼라 (dataset_rmask)
#
# 큐 전체를 setsid+nohup 로 분리 실행 -> 터미널/VSCode 를 닫아도 계속 돈다.
# SEEDS 를 하나씩 돌면서, 그 안에서 SCALAR 들을 동시에 띄우고 끝날 때까지 기다린다
# (CPU 12코어라 동시 2개가 적당). 런별 로그: logs/<런이름>.log, 큐 로그: logs/queue_<TAG>.log
#
# 사용법:
#   bash run.sh                                           # SCALAR="z none", SEEDS=42
#   TAG=dom WEIGHTS=/home/work/Code/weights/effunet_pvtv2b0_512.pt bash run.sh
#   TAG=dom_fix WEIGHTS=... EPOCHS=30 STOP_ON=qwk PATIENCE=10 EMA=0.998 SEEDS="42 1 2" bash run.sh
#   tail -f logs/queue_<TAG>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SCALAR="${SCALAR:-z none}"
SEEDS="${SEEDS:-42}"
TAG="${TAG:-base}"
MODEL="${MODEL:-pvtv2b0}"
IMGSZ="${IMGSZ:-512}"
MARGIN="${MARGIN:-0.15}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
STOP_ON="${STOP_ON:-loss}"
EMA="${EMA:-0}"
WEIGHTS="${WEIGHTS:-}"
WORKERS="${WORKERS:-5}"
mkdir -p logs

if [ "${_PAPU_WORKER:-0}" != 1 ]; then
  export SCALAR SEEDS TAG MODEL IMGSZ MARGIN EPOCHS PATIENCE STOP_ON EMA WEIGHTS WORKERS
  _PAPU_WORKER=1 setsid nohup bash "$0" > "logs/queue_${TAG}.log" 2>&1 < /dev/null &
  echo "[queue] TAG=$TAG pid=$!  log=$(pwd)/logs/queue_${TAG}.log"
  exit 0
fi

# ---- 여기부터 분리된 워커 ----
export PYTHONPATH=""
VENV="${VENV:-$(cd .. && pwd)/.venv-train}"
PY="$VENV/bin/python"
SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SP/torch/lib:$(printf '%s:' "$SP"/nvidia/*/lib)/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONUNBUFFERED=1

for seed in $SEEDS; do
  pids=()
  for s in $SCALAR; do
    name="pap_rmask_${TAG}_${s}_${MODEL}_r${IMGSZ}_s${seed}"
    extra=()
    [ -n "$WEIGHTS" ] && extra+=(--weights "$WEIGHTS")
    echo "[$(date +%T)] start $name"
    "$PY" train.py --scalar "$s" --model "$MODEL" --imgsz "$IMGSZ" --margin "$MARGIN" \
        --epochs "$EPOCHS" --patience "$PATIENCE" --stop_on "$STOP_ON" --ema "$EMA" \
        --workers "$WORKERS" --seed "$seed" --name "$name" "${extra[@]}" \
        > "logs/${name}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || echo "[warn] pid $p 실패"; done
  echo "[$(date +%T)] seed $seed 완료"
done
echo "[$(date +%T)] 큐 완료 (TAG=$TAG)"
