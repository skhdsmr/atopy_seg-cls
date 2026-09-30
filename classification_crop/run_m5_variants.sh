#!/usr/bin/env bash
# =============================================================================
# run_m5_variants.sh - M5(찰상 local, S1+S2, gated fusion) 에 ①임계값초기화 ③단조제약 적용
#
# M5 를 고른 이유: 사다리에서 찰상이 M3 .481 -> M4 .501 -> M5 .541 로 단조 증가했고
#   M5 의 +0.078 은 시드 노이즈(±0.029~0.044)를 넘는 유일한 구간이었다.
#   M6(태선화 공유)은 찰상을 깎았고 M7(pos_weight)은 게이트를 붕괴시켰다(alpha mean 0.059).
#
# 축 두 개를 2x2 로, 시드 2개씩:
#   thr_init exact : CORN 헤드 bias = logit P(y>i | y>i-1), 마지막 층 weight zero-init.
#                    초기 출력이 train 주변분포와 정확히 일치한다(검증 출력 있음).
#   mono_lambda    : iga >= max(징후) soft 제약. train 99.29% 성립(위반 10행).
#
# plain/s42 는 이미 M5_exco_s12_gate_s42 로 있으므로 s43 만 새로 돌린다(결정적이라 재현됨).
#
# 사용법: bash run_m5_variants.sh | DRY=1 ... | MONO=0.3 ... | SEEDS="42 43 44" ...
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
LOG="sweep_m5_$(date +%Y%m%d_%H%M%S).txt"

COMMON="--exp bbox --model pvtv2b0 --imgsz 512 --batch 32 --data $DATA --masks $MASKS \
        --loss corn --mtl uncertainty --epochs 100 --patience 20 --workers 6 \
        --ibb --ibb_edge_v 0.25 --ibb_step2_ratio 0.5 --ibb_warmup 3 \
        --local_tasks excoriation --local_stages 0,1 --local_fusion gate"

runs=()
for s in $SEEDS; do
  runs+=("M5_exco_s12_gate_s$s|--seed $s")
  runs+=("M5_thr_s$s|--seed $s --thr_init exact")
  runs+=("M5_mono_s$s|--seed $s --mono_lambda $MONO")
  runs+=("M5_both_s$s|--seed $s --thr_init exact --mono_lambda $MONO")
done

echo "==================== M5 변형: ${#runs[@]}런 (시드 $SEEDS, mono=$MONO) ===================="
printf '%s\n' "${runs[@]}" | sed 's/|/   ->   /'
echo " 기준점: M5_exco_s12_gate_s42 (test .594 / 찰상 .541)   로그 $LOG"
echo "==============================================================================="
[ "$DRY" = "1" ] && { echo "(DRY=1 이라 여기서 종료)"; exit 0; }

ok=0; fail=0; skip=0
for item in "${runs[@]}"; do
  name="${item%%|*}"; extra="${item#*|}"
  echo "" | tee -a "$LOG"
  echo "######## [$((ok+fail+skip+1))/${#runs[@]}] $name ########" | tee -a "$LOG"
  if [ -f "runs/$name/done.txt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "skip (완료됨): $name" | tee -a "$LOG"; skip=$((skip+1)); continue
  fi
  rm -f "runs/$name/done.txt"
  # shellcheck disable=SC2086
  if "$PY" train.py $COMMON $extra --name "$name" 2>&1 | tee -a "$LOG"; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "!! 실패: $name" | tee -a "$LOG" >&2
  fi
done

echo "" | tee -a "$LOG"
echo "###### M5 변형 완료: 성공 $ok / 실패 $fail / 건너뜀 $skip ######" | tee -a "$LOG"
for d in runs/M2_ocnnit_s*/done.txt runs/M5_*/done.txt; do
  [ -f "$d" ] || continue
  echo "--- $(dirname "$d" | xargs basename)" | tee -a "$LOG"; cat "$d" | tee -a "$LOG"
done
