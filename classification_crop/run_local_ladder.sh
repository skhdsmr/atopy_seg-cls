#!/usr/bin/env bash
# =============================================================================
# run_local_ladder.sh - 찰상/태선화 전용 local branch 사다리 (M3~M7)
#
# 기준점 M2 = M2_ocnnit_s42 (코퓰러 없음, OCNN-IT). 여기에 모듈을 하나씩 얹는다.
#   M3  찰상 local, S1 만, 단순 concat      <- "저해상 stage 를 살리면 버는가"
#   M4  + S2 추가(multi-scale)              <- "여러 배율이 필요한가" (crop 배율이 제각각)
#   M5  + gated fusion                      <- "이미지마다 local 신뢰도를 다르게 둬야 하는가"
#   M6  + 태선화도 같은 trunk 공유          <- 실데이터 찰상-태선화 0.377 근거
#   M7  + CORN 레벨별 pos_weight            <- 찰상 P(y>2) positive 113개 문제
#
# 주의(1 시드 탐색): M2 에서 찰상 QWK 의 시드간 폭이 ±0.044 였다. 여기 차이가 그보다
# 작으면 방향 참고만 하고, 살아남은 후보만 시드를 늘려 다시 재야 한다.
#
# 사용법: bash run_local_ladder.sh | DRY=1 ... | SEED=43 ... | FORCE=1 ...
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
SEED="${SEED:-42}"
DRY="${DRY:-0}"
LOG="sweep_local_$(date +%Y%m%d_%H%M%S).txt"

# M2_ocnnit 과 동일 조건(코퓰러 없음, step2_ratio 0.5)
COMMON="--exp bbox --model pvtv2b0 --imgsz 512 --batch 32 --data $DATA --masks $MASKS \
        --loss corn --mtl uncertainty --epochs 100 --patience 20 --workers 6 \
        --ibb --ibb_edge_v 0.25 --ibb_step2_ratio 0.5 --ibb_warmup 3 --seed $SEED"

runs=(
  "M3_exco_s1_concat|--local_tasks excoriation --local_stages 0 --local_fusion concat"
  "M4_exco_s12_concat|--local_tasks excoriation --local_stages 0,1 --local_fusion concat"
  "M5_exco_s12_gate|--local_tasks excoriation --local_stages 0,1 --local_fusion gate"
  "M6_excolich_s12_gate|--local_tasks excoriation,lichenification --local_stages 0,1 --local_fusion gate"
  "M7_excolich_s12_gate_pw|--local_tasks excoriation,lichenification --local_stages 0,1 --local_fusion gate --corn_pos_weight auto"
)

echo "==================== local branch 사다리: ${#runs[@]}런 (시드 $SEED) ===================="
printf '%s\n' "${runs[@]}" | sed 's/|/\n      /'
echo " 기준점: M2_ocnnit_s$SEED   로그 $LOG"
echo "=================================================================================="
[ "$DRY" = "1" ] && { echo "(DRY=1 이라 여기서 종료)"; exit 0; }

ok=0; fail=0
for item in "${runs[@]}"; do
  name="${item%%|*}_s$SEED"; extra="${item#*|}"
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
echo "###### 사다리 완료: 성공 $ok / 실패 $fail ######" | tee -a "$LOG"
for d in runs/M2_ocnnit_s$SEED/done.txt runs/M[3-7]_*_s$SEED/done.txt; do
  [ -f "$d" ] || continue
  echo "--- $(dirname "$d" | xargs basename)" | tee -a "$LOG"; cat "$d" | tee -a "$LOG"
done
