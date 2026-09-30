#!/usr/bin/env bash
# =============================================================================
# run_m2_baseline.sh - local branch 사다리(M3~M7)의 기준점 M2 를 시드 2개로 고정
#
# 왜 baseline 이 둘인가:
#   같은 설정·같은 시드를 다시 돌렸더니 test score 가 0.011 (mQWK 0.004) 움직였다
#   (ccnnit_a2 .618 ↔ e025_w01_none .607, best_ep 30 ↔ 79). AMP/cuDNN 비결정성이다.
#   그리고 이 프로젝트가 개선하려는 두 축에서 코퓰러가 지렛대가 아니다:
#       찰상   OCNN .419 / OCNN-IT .536 / CCNN-IT .444   <- 코퓰러가 오히려 깎는다
#       태선화 OCNN .623 / OCNN-IT .598 / CCNN-IT .574
#   그래서 M2 를 CCNN-IT 하나로 두면 local branch 의 이득을 과대평가하게 된다.
#   코퓰러 있는 쪽(ccnnit)과 없는 쪽(ocnnit) 둘 다 기준점으로 두고, 시드 2개로 노이즈
#   폭까지 같이 재둔다.
#
# 각 baseline 은 원본 런의 설정을 그대로 복제한다(step2_ratio 가 서로 다른 것도 원본 그대로).
#   ocnnit : crop_bbox_pvtv2b0_r512_ibb_b32     (pairwise 없음, step2_ratio 0.5)
#   ccnnit : crop_bbox_pvtv2b0_r512_ccnnit_a2   (pairwise, shared/nll, step2_ratio 1.0)
#
# 사용법: bash run_m2_baseline.sh   |   DRY=1 bash run_m2_baseline.sh   |   SEEDS="42 43 44" ...
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
DRY="${DRY:-0}"
LOG="sweep_m2_$(date +%Y%m%d_%H%M%S).txt"

COMMON="--exp bbox --model pvtv2b0 --imgsz 512 --batch 32 --data $DATA --masks $MASKS \
        --loss corn --mtl uncertainty --epochs 100 --patience 20 --workers 6 \
        --ibb --ibb_edge_v 0.25 --ibb_warmup 3"
OCNNIT="--ibb_step2_ratio 0.5"
CCNNIT="--ibb_step2_ratio 1.0 --pairwise --pairwise_shared 1 --pairwise_unary nll \
        --pairwise_w_nodes 0.1"

runs=()
for s in $SEEDS; do
  runs+=("M2_ocnnit_s$s|$OCNNIT --seed $s")
  runs+=("M2_ccnnit_s$s|$CCNNIT --seed $s")
done

echo "==================== M2 baseline: ${#runs[@]}런 (시드 $SEEDS) ===================="
printf '%s\n' "${runs[@]}" | sed 's/|/   ->   /'
echo " 로그 $LOG"
echo "==========================================================================="
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
  if "$PY" train.py $COMMON $extra --name "$name" 2>&1 | tee -a "$LOG"; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "!! 실패: $name" | tee -a "$LOG" >&2
  fi
done

echo "" | tee -a "$LOG"
echo "###### M2 baseline 완료: 성공 $ok / 실패 $fail ######" | tee -a "$LOG"
for d in runs/M2_*/done.txt; do
  [ -f "$d" ] || continue
  echo "--- $(dirname "$d" | xargs basename)" | tee -a "$LOG"; cat "$d" | tee -a "$LOG"
done
