#!/usr/bin/env bash
# =============================================================================
# run_baseline_sweep.sh - CCNN-IT(a2) baseline 을 논문(Walecki+ 2017)에 맞추는 3축 스윕
#
# local branch 를 얹기 전에 baseline 을 먼저 고정한다. 세 축은 논문 대조에서 나온 것:
#
#   ① EDGE_V  0.25 -> 0.05   엣지 전수(10쌍). 현행 0.25 는 태선화 엣지를 **전부** 잘라내
#                            태선화가 사실상 OCNN 이 된다. 논문은 전 쌍에서 theta 를 추정하고
#                            (Eq 11) 낮은 연관은 theta->0 으로 자동 환원된다(3.2절).
#                            Fig.3 의 가지치기는 시각화용이다.
#   ② W_NODES 0.1 -> 0.5     Eq 11 은 unary 와 pairwise 를 동일 가중으로 더한다.
#                            0.1 은 저자 코드 기본값이고 실측에서 CCNN(0.564) < OCNN(0.597).
#   ③ SOURCES none -> angle  Eq 12: unary 공유 + theta 는 출처별. 합성데이터의 맥락 축은
#                            정면/측면이고 의존구조가 실제로 다르다(구진-태선화 0.120/0.274,
#                            중증도-태선화는 합치면 0.071 로 사라짐).
#
# 2x2x2 = 8런. 전부 a2 규약(--pairwise_shared 1 --pairwise_unary nll --ibb_step2_ratio 1.0).
#
# 사용법:
#   bash run_baseline_sweep.sh              # 8런
#   DRY=1 bash run_baseline_sweep.sh        # 무엇이 돌지만 확인
#   FORCE=1 ... 는 완료된 런도 재학습
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

DATA="${DATA:-$(cd .. && pwd)/dataset_all_final/images}"
MASKS="${MASKS:-$(cd .. && pwd)/atopy_crop_masks}"
ATOPY="${ATOPY:-$(cd .. && pwd)/atopy}"
MODEL="${MODEL:-pvtv2b0}"
IMGSZ="${IMGSZ:-512}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
WORKERS="${WORKERS:-6}"
DRY="${DRY:-0}"
LOG="sweep_baseline_$(date +%Y%m%d_%H%M%S).txt"

BASE="--exp bbox --model $MODEL --imgsz $IMGSZ --batch $BATCH --data $DATA --masks $MASKS \
      --loss corn --mtl uncertainty --epochs $EPOCHS --patience $PATIENCE --workers $WORKERS \
      --pairwise --pairwise_shared 1 --pairwise_unary nll --ibb --ibb_step2_ratio 1.0 \
      --atopy_root $ATOPY"

runs=()
for ev in 0.05 0.25; do            # ① 전수 엣지를 먼저
  for wn in 0.5 0.1; do            # ② 논문 가중을 먼저
    for sr in angle none; do       # ③ Eq12 를 먼저
      tag="e$(echo $ev | tr -d '.')_w$(echo $wn | tr -d '.')_$sr"
      runs+=("ccnnit_a2_$tag|--ibb_edge_v $ev --pairwise_w_nodes $wn --pairwise_sources $sr")
    done
  done
done

echo "==================== baseline 스윕: ${#runs[@]}런 ===================="
printf '%s\n' "${runs[@]}" | sed 's/|/   ->   /'
echo " 공통: $MODEL r$IMGSZ b$BATCH  a2(shared/nll/step2=1.0)  로그 $LOG"
echo "======================================================================"
[ "$DRY" = "1" ] && { echo "(DRY=1 이라 여기서 종료)"; exit 0; }

ok=0; fail=0
for item in "${runs[@]}"; do
  name="${item%%|*}"; extra="${item#*|}"
  echo "" | tee -a "$LOG"
  echo "######## [$((ok+fail+1))/${#runs[@]}] $name ########" | tee -a "$LOG"
  if [ -f "runs/$name/done.txt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "skip (완료됨): $name" | tee -a "$LOG"; ok=$((ok+1)); continue
  fi
  rm -f "runs/$name/done.txt"
  # shellcheck disable=SC2086
  if "$PY" train.py $BASE $extra --name "$name" 2>&1 | tee -a "$LOG"; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "!! 실패: $name" | tee -a "$LOG" >&2
  fi
done

echo "" | tee -a "$LOG"
echo "###### baseline 스윕 완료: 성공 $ok / 실패 $fail ######" | tee -a "$LOG"
for d in runs/ccnnit_a2_*/done.txt; do
  [ -f "$d" ] || continue
  echo "--- $(dirname "$d" | xargs basename)" | tee -a "$LOG"; cat "$d" | tee -a "$LOG"
done
