#!/usr/bin/env bash
# =============================================================================
# run_m5b.sh - M5(찰상 local, S1+S2, gated fusion) + ①임계값초기화 ③단조제약
#
# 접두사를 M5b_ 로 새로 뒀다. 이유: done.txt 만 보고 skip 하는 기존 스크립트가
# **코드 편집 전에 만들어진 런**(M5_exco_s12_gate_s42 — args 에 thr_init/mono_lambda 키가
# 아예 없다)을 기준선으로 재활용했다. 앞서 ccnnit_a2 가 코드 편집 전후로 .618 -> .607 로
# 바뀐 전례가 있어(원인 미규명) 버전이 섞인 채로 비교하면 안 된다.
# 여기서는 plain 을 포함해 6런 전부 지금 코드로 새로 학습한다.
#
#   plain : M5 그대로              (기준선, 현재 코드)
#   thr   : + --thr_init exact     (CORN bias = logit P(y>i|y>i-1), 마지막층 zero-init)
#   both  : + --mono_lambda        (iga >= max(징후) soft 제약)
#   -> mono 단독 효과는 both - thr 로 읽는다(6런에 맞추려 단독 조건을 뺐다).
#
# 사용법: bash run_m5b.sh | DRY=1 ... | MONO=0.3 ... | SEEDS="42 43" ...
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
  PY=python3; echo "경고: 학습 venv($VENV) 없음 -> 시스템 python3." >&2
fi

DATA="${DATA:-$(cd .. && pwd)/dataset_all_final/images}"
MASKS="${MASKS:-$(cd .. && pwd)/atopy_crop_masks}"
SEEDS="${SEEDS:-42 43}"
MONO="${MONO:-0.1}"
DRY="${DRY:-0}"
LOG="sweep_m5b_$(date +%Y%m%d_%H%M%S).txt"

COMMON="--exp bbox --model pvtv2b0 --imgsz 512 --batch 32 --data $DATA --masks $MASKS \
        --loss corn --mtl uncertainty --epochs 100 --patience 20 --workers 6 \
        --ibb --ibb_edge_v 0.25 --ibb_step2_ratio 0.5 --ibb_warmup 3 \
        --local_tasks excoriation --local_stages 0,1 --local_fusion gate"

runs=()
for s in $SEEDS; do
  runs+=("M5b_plain_s$s|--seed $s")
  runs+=("M5b_thr_s$s|--seed $s --thr_init exact")
  runs+=("M5b_both_s$s|--seed $s --thr_init exact --mono_lambda $MONO")
done

echo "==================== M5b: ${#runs[@]}런 (시드 $SEEDS, mono=$MONO) ===================="
printf '%s\n' "${runs[@]}" | sed 's/|/   ->   /'
echo " 전부 새 접두사 M5b_ -> skip 없음. 로그 $LOG"
echo "==========================================================================="
[ "$DRY" = "1" ] && { echo "(DRY=1 이라 여기서 종료)"; exit 0; }

ok=0; fail=0
for item in "${runs[@]}"; do
  name="${item%%|*}"; extra="${item#*|}"
  echo "" | tee -a "$LOG"
  echo "######## [$((ok+fail+1))/${#runs[@]}] $name ########" | tee -a "$LOG"
  rm -rf "runs/$name"
  # shellcheck disable=SC2086
  if "$PY" train.py $COMMON $extra --name "$name" 2>&1 | tee -a "$LOG"; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "!! 실패: $name" | tee -a "$LOG" >&2
  fi
done

echo "" | tee -a "$LOG"
echo "###### M5b 완료: 성공 $ok / 실패 $fail ######" | tee -a "$LOG"
for d in runs/M5b_*/done.txt; do
  [ -f "$d" ] || continue
  echo "--- $(dirname "$d" | xargs basename)" | tee -a "$LOG"; cat "$d" | tee -a "$LOG"
done
