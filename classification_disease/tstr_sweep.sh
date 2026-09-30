#!/usr/bin/env bash
# =============================================================================
# tstr_sweep.sh — TSTR(Train Synthetic, Test Real) 4런 순차 sweep.
#
# 합성(GAN / SD-전체FT)만으로 6-way 질환 분류기를 학습하고 실제 이미지로 시험한다.
# 기준선은 이미 있는 실제 학습 런(runs/dis_effb0_r512_front, TRTR)이다.
#
#   bash tstr_sweep.sh prep     # dataset_tstr/{gan,sdft}_{A,B} 조립 (심볼릭 링크)
#   bash tstr_sweep.sh start    # 감독자 기동 — IDE/터미널을 꺼도 계속 돈다
#   bash tstr_sweep.sh watch    # 진행 상황
#   bash tstr_sweep.sh stop     # 감독자 + 학습 정지
#   bash tstr_sweep.sh report   # 결과표 (TSTR/TRTR 비율)
#
# 환경변수: EPOCHS PATIENCE BATCH WORKERS MODEL IMGSZ JOBS MAX_RETRY
#
# IDE 를 꺼도 살아남는 방법 (gan/supervise.sh 와 같은 이유)
#   nohup 만으로는 부족하다 — nohup 은 SIGHUP 만 막을 뿐, 프로세스가 IDE 터미널과
#   같은 세션(SID)에 그대로 남아서 그 세션이 정리될 때 프로세스 그룹째 신호를 받으면
#   같이 죽는다. setsid 로 자기 세션을 파야 한다.
#   `ps -o sid= -p <pid>` 가 자기 pid 와 같으면 제대로 분리된 것이다.
#
#   못 막는 것: 이 머신은 Backend.AI 컨테이너다. 세션 자체가 유휴 타임아웃이나 수동
#   종료로 내려가면 컨테이너 안의 모든 프로세스가 함께 사라진다.
#
# 중간에 죽으면
#   런 단위로 이어 붙인다 — done.txt 가 있는 런은 건너뛰고, 실패한 런은 MAX_RETRY 번
#   까지 처음부터 다시 돌린다. train.py 에 epoch 단위 resume 이 없어서 런 하나가
#   통째로 다시 도는 것이고, 한 런이 1~2시간이라 이 정도로 충분하다고 봤다.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
OGW="$(cd .. && pwd)"
RUNSDIR="$ROOT/runs"
STATE="$RUNSDIR/.tstr"
PIDFILE="$STATE/pid"
LOG="$STATE/sweep.log"

MODEL="${MODEL:-effb0}"
IMGSZ="${IMGSZ:-512}"
BATCH="${BATCH:-8}"          # 기준선(dis_effb0_r512_front)과 같아야 비교가 성립한다
EPOCHS="${EPOCHS:-100}"      # 기준선(dis_effb0_r512_front)과 같은 상한. 조기종료가 끊는다
PATIENCE="${PATIENCE:-15}"
WORKERS="${WORKERS:-10}"     # 12코어. 512px PIL 증강이 병목이라 여기서 속도가 갈린다
MAX_RETRY="${MAX_RETRY:-2}"
JOBS="${JOBS:-gan_A sdft_A gan_B sdft_B}"

mkdir -p "$STATE"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- 학습 venv(CUDA torch) 격리 — run.sh 와 동일 패턴 ---
setup_py() {
  export PYTHONPATH=""
  local VENV="${VENV:-$OGW/.venv-train}"
  PY="${PY:-$VENV/bin/python}"
  if [ -x "$PY" ]; then
    local SP NV
    SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
    NV="$(printf '%s:' "$SP"/nvidia/*/lib)"
    export LD_LIBRARY_PATH="$SP/torch/lib:${NV}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
  else
    PY=python3
    echo "경고: 학습 venv 없음 -> 시스템 python3." >&2
  fi
}

run_name() { echo "tstr_${MODEL}_r${IMGSZ}_$1"; }

case "${1:-start}" in

prep)
  setup_py
  "$PY" make_tstr_dataset.py
  ;;

start)
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "이미 감독자가 돌고 있다 (pid $(cat "$PIDFILE")). bash tstr_sweep.sh watch"; exit 0
  fi
  for j in $JOBS; do
    [ -f "$OGW/dataset_tstr/$j/labels.csv" ] || {
      echo "[에러] dataset_tstr/$j 없음. 먼저 bash tstr_sweep.sh prep" >&2; exit 1; }
  done
  MODEL=$MODEL IMGSZ=$IMGSZ BATCH=$BATCH EPOCHS=$EPOCHS PATIENCE=$PATIENCE \
    WORKERS=$WORKERS MAX_RETRY=$MAX_RETRY JOBS="$JOBS" \
    setsid nohup bash "$ROOT/tstr_sweep.sh" _run </dev/null >/dev/null 2>&1 &
  disown
  sleep 3
  echo "감독자 기동 (pid $(cat "$PIDFILE" 2>/dev/null || echo '?'))."
  echo "  진행: bash tstr_sweep.sh watch   |   결과: bash tstr_sweep.sh report"
  ;;

_run)
  # 감독자 본체. 두 개가 같은 GPU 에 붙으면 둘 다 느려지거나 OOM 이라 중복을 막는다.
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    log "중복 기동 감지. 종료."; exit 0
  fi
  echo $$ > "$PIDFILE"
  trap 'rm -f "$PIDFILE"' EXIT
  setup_py
  log "sweep 시작 (pid $$, sid $(ps -o sid= -p $$ | tr -d ' ')) jobs=[$JOBS] " \
      "model=$MODEL imgsz=$IMGSZ batch=$BATCH epochs=$EPOCHS patience=$PATIENCE"

  for job in $JOBS; do
    NAME="$(run_name "$job")"
    DATA="$OGW/dataset_tstr/$job"
    OUT="$RUNSDIR/$NAME"
    if [ -f "$OUT/done.txt" ]; then
      log "[$job] 이미 완료 -> 건너뜀 ($NAME)"; continue
    fi
    mkdir -p "$OUT"
    try=1
    while [ "$try" -le "$MAX_RETRY" ]; do
      log "[$job] 학습 시작 (시도 $try/$MAX_RETRY) -> $OUT/train.log"
      "$PY" train.py \
        --model "$MODEL" --imgsz "$IMGSZ" --batch "$BATCH" --angle front \
        --data "$DATA" --epochs "$EPOCHS" --patience "$PATIENCE" \
        --workers "$WORKERS" --name "$NAME" >> "$OUT/train.log" 2>&1
      rc=$?
      if [ "$rc" -eq 0 ] && [ -f "$OUT/done.txt" ]; then
        log "[$job] 완료 — $(tail -1 "$OUT/done.txt" | tr '\n' ' ')"
        break
      fi
      log "[$job] 실패 rc=$rc (train.log 확인)"
      try=$(( try + 1 )); sleep 30
    done
    [ -f "$OUT/done.txt" ] || { log "[$job] ${MAX_RETRY}회 실패. 다음 런으로 넘어간다."; continue; }

    # 프로토콜 B 는 test 안에 '원래 train 에서 옮긴 실제 이미지'가 섞여 있어서
    # TRTR 기준선(test 600장)과 직접 비교가 안 된다. 원본 test 600장만으로 한 번 더
    # 재기어 둔다 -> eval_test_best_front.json (--angle 을 주면 파일명이 갈린다).
    case "$job" in *_B)
      log "[$job] 원본 test 600장 재평가"
      "$PY" evaluate.py --run "$NAME" --split test --data "$OGW/dataset_disease" \
        --angle front --batch 16 --workers 4 >> "$OUT/eval_orig_test.log" 2>&1 \
        || log "[$job] 원본 test 재평가 실패 (eval_orig_test.log)"
      ;;
    esac
  done
  log "sweep 종료. bash tstr_sweep.sh report"
  ;;

watch)
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "감독자 살아있음 (pid $(cat "$PIDFILE"))"
  else
    echo "감독자 없음"
  fi
  echo "--- 런 상태 ---"
  for job in $JOBS; do
    NAME="$(run_name "$job")"; OUT="$RUNSDIR/$NAME"
    if [ -f "$OUT/done.txt" ]; then
      printf "  %-12s 완료   %s\n" "$job" "$(sed -n 2p "$OUT/done.txt")"
    elif [ -f "$OUT/train.log" ]; then
      printf "  %-12s 진행중 %s\n" "$job" "$(grep '^\[Ep' "$OUT/train.log" | tail -1)"
    else
      printf "  %-12s 대기\n" "$job"
    fi
  done
  echo "--- 감독자 로그 ---"; tail -6 "$LOG" 2>/dev/null
  ;;

stop)
  if [ -f "$PIDFILE" ]; then
    kill -TERM "$(cat "$PIDFILE")" 2>/dev/null && echo "감독자 정지"
    rm -f "$PIDFILE"
  fi
  pkill -f "train.py --model $MODEL --imgsz $IMGSZ" && echo "학습 정지" || echo "돌고 있는 학습 없음"
  ;;

report)
  setup_py
  shift || true
  "$PY" tstr_report.py "$@"
  ;;

*)
  echo "usage: bash tstr_sweep.sh {prep|start|watch|stop|report}" >&2; exit 1
  ;;
esac
