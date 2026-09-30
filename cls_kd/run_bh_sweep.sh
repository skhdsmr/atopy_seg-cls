#!/usr/bin/env bash
# =============================================================================
# run_bh_sweep.sh — 조건 A HPO(runs_hpo_a) 상위 trial 4/24/49 하이퍼파라미터로
# 조건 B~H(train_multitask.py/train_distill.py) 전체를 재현.
#
# trial 마다 완전히 독립된 파이프라인을 돈다(teacher OOF 25개 + teacher A 5개 +
# B 1개 + C~H 6개 = 37개, 3 trial 합쳐 111개 학습) — teacher 도 그 trial의
# margin/imgsz/lr 등으로 새로 만들어야 B~H 가 그 trial 조건을 온전히 재현한다.
#
# 산출물: runs_bh_sweep/trial_<N>/{A_single_*,oof_*,teacher_logits*,B_*,C_*..H_*}
#         runs_bh_sweep/logs/trial_<N>*.log
#
# 사용법: nohup bash run_bh_sweep.sh > runs_bh_sweep/logs/driver.log 2>&1 &
#         disown
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

DATA="$(cd .. && pwd)/dataset_all_final/images"
MASKS="$(cd .. && pwd)/atopy_crop_masks"
SWEEP_ROOT="$(pwd)/runs_bh_sweep"
LOGROOT="$SWEEP_ROOT/logs"
mkdir -p "$LOGROOT"

run_trial() {
  local tid="$1" imgsz="$2" lr="$3" bb="$4" wd="$5" drop="$6" warm="$7" sched_extra="$8" \
       cw="$9" ls="${10}" margin="${11}"
  local project="$SWEEP_ROOT/trial_${tid}"
  local trial_log="$LOGROOT/trial_${tid}.log"
  mkdir -p "$project"
  local extra="--lr ${lr} --backbone_lr_scale ${bb} --weight_decay ${wd} --dropout ${drop} \
--warmup_epochs ${warm} ${sched_extra} --class_weight_power ${cw} --label_smoothing ${ls}"

  echo "[$(date '+%m-%d %H:%M:%S')] === trial ${tid} 시작 (imgsz=${imgsz} margin=${margin}) ===" \
    | tee -a "$trial_log"

  echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid}: teacher(both) ===" | tee -a "$trial_log"
  DATA="$DATA" MASKS="$MASKS" MODEL=pvtv2b0 IMGSZ="$imgsz" PROJECT="$project" \
    EXP=bbox BBOX_SQUARE=1 MARGIN="$margin" MODE=both EXTRA="$extra" \
    bash run_teachers.sh >> "$LOGROOT/trial_${tid}_teachers.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid}: teacher 단계 실패 -> B~H 건너뜀 (로그: ${LOGROOT}/trial_${tid}_teachers.log)" \
      | tee -a "$trial_log"
    return 1
  fi

  echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid}: 조건 B(multitask) ===" | tee -a "$trial_log"
  DATA="$DATA" MASKS="$MASKS" MODEL=pvtv2b0 IMGSZ="$imgsz" PROJECT="$project" \
    EXP=bbox BBOX_SQUARE=1 MARGIN="$margin" EXTRA="$extra" \
    bash run_multitask.sh >> "$LOGROOT/trial_${tid}_B.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid}: 조건 B 실패 (로그: ${LOGROOT}/trial_${tid}_B.log)" \
      | tee -a "$trial_log"
  fi

  echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid}: 조건 C~H(distill, all) ===" | tee -a "$trial_log"
  DATA="$DATA" MASKS="$MASKS" MODEL=pvtv2b0 IMGSZ="$imgsz" PROJECT="$project" \
    EXP=bbox BBOX_SQUARE=1 MARGIN="$margin" COND=all EXTRA="$extra" \
    bash run_distill.sh >> "$LOGROOT/trial_${tid}_distill.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid}: 조건 C~H 일부 실패 (로그: ${LOGROOT}/trial_${tid}_distill.log)" \
      | tee -a "$trial_log"
  fi

  echo "[$(date '+%m-%d %H:%M:%S')] === trial ${tid} 완료 ===" | tee -a "$trial_log"
}

# tid  imgsz lr        bb_scale wd         drop warm  scheduler-extra                          cw   ls   margin
run_trial 4  320   1.22e-04  0.124   8.63e-03  0.5  3  "--scheduler constant"                   1.0  0.1  0.1
run_trial 24 224   2.70e-04  0.198   2.84e-03  0.4  3  "--scheduler cosine --cosine_epochs 30"   0.5  0.0  0.3
run_trial 49 224   6.38e-04  0.073   2.43e-02  0.5  4  "--scheduler cosine --cosine_epochs 100"  0.5  0.0  0.0

echo "[$(date '+%m-%d %H:%M:%S')] === 전체 스윕(trial 4/24/49 x B~H) 완료 ==="
