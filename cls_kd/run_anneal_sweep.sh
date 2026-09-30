#!/usr/bin/env bash
# =============================================================================
# run_anneal_sweep.sh — --anneal_epochs(15/20/25/30) 스윕, trial 4/24/49 조건.
#
# 배경: total_steps = len(train_loader)*args.epochs(=100 고정)로 λ(teacher
# annealing)를 정규화했는데, patience=5 조기종료가 실제로는 10~35 epoch 사이에서
# 걸려서 D/E/F/G/H 전부 λ가 0.16~0.35를 못 넘고 끝났다(runs_bh_sweep 111런 분석
# 결과). anneal_epochs를 훨씬 작은 값으로 줘서 조기종료 시점 근방에서 λ가 실제로
# 1.0(순수 hard label)에 도달하게 만드는 스윕.
#
# teacher_logits/teacher_logits_a는 이미 runs_bh_sweep/trial_<N>/에 만들어져
# 있으므로(구조 자체는 안 바뀌었다) 재사용한다 — teacher 재학습 없이 student(D/E/G/H)만
# 다시 돈다. anneal=none인 C/F는 λ 자체가 영향을 안 받으므로 재실행하지 않는다.
#
# 산출물: runs_anneal_sweep/trial_<N>_ae<AE>/{D,E,G,H}_pvtv2b0_.../test_report.json
#         runs_anneal_sweep/logs/trial_<N>_ae<AE>_<COND>.log
#
# 사용법: nohup setsid bash run_anneal_sweep.sh > runs_anneal_sweep/logs/driver.log 2>&1 &
#         disown
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

DATA="$(cd .. && pwd)/dataset_all_final/images"
MASKS="$(cd .. && pwd)/atopy_crop_masks"
BASE_ROOT="$(pwd)/runs_bh_sweep"
SWEEP_ROOT="$(pwd)/runs_anneal_sweep"
LOGROOT="$SWEEP_ROOT/logs"
mkdir -p "$LOGROOT"

run_one_cond() {
  local tid="$1" ae="$2" cond="$3" imgsz="$4" lr="$5" bb="$6" wd="$7" drop="$8" warm="$9"
  shift 9
  local sched_extra="$1" cw="$2" ls="$3" margin="$4"
  local project="$SWEEP_ROOT/trial_${tid}_ae${ae}"
  local base_trial="$BASE_ROOT/trial_${tid}"
  mkdir -p "$project"
  local extra="--lr ${lr} --backbone_lr_scale ${bb} --weight_decay ${wd} --dropout ${drop} \
--warmup_epochs ${warm} ${sched_extra} --class_weight_power ${cw} --label_smoothing ${ls} \
--anneal_epochs ${ae}"
  local logf="$LOGROOT/trial_${tid}_ae${ae}_${cond}.log"

  echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid} ae=${ae} 조건 ${cond} 시작"
  DATA="$DATA" MASKS="$MASKS" MODEL=pvtv2b0 IMGSZ="$imgsz" PROJECT="$project" \
    EXP=bbox BBOX_SQUARE=1 MARGIN="$margin" COND="$cond" \
    TEACHER_DIR="$base_trial/teacher_logits" TEACHER_DIR_A="$base_trial/teacher_logits_a" \
    EXTRA="$extra" \
    bash run_distill.sh >> "$logf" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid} ae=${ae} 조건 ${cond} 실패 (로그: $logf)"
  else
    echo "[$(date '+%m-%d %H:%M:%S')] trial ${tid} ae=${ae} 조건 ${cond} 완료"
  fi
}

run_trial_all() {
  local tid="$1"; shift
  for ae in 15 20 25 30; do
    for cond in D E G H; do
      run_one_cond "$tid" "$ae" "$cond" "$@"
    done
  done
}

# tid  imgsz lr        bb_scale wd         drop warm  scheduler-extra                          cw   ls   margin
run_trial_all 4  320   1.22e-04  0.124   8.63e-03  0.5  3  "--scheduler constant"                   1.0  0.1  0.1
run_trial_all 24 224   2.70e-04  0.198   2.84e-03  0.4  3  "--scheduler cosine --cosine_epochs 30"   0.5  0.0  0.3
run_trial_all 49 224   6.38e-04  0.073   2.43e-02  0.5  4  "--scheduler cosine --cosine_epochs 100"  0.5  0.0  0.0

echo "[$(date '+%m-%d %H:%M:%S')] === anneal_epochs 스윕(15/20/25/30 x D/E/G/H x trial 4/24/49) 전체 완료 ==="
