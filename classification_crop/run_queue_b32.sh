#!/usr/bin/env bash
# =============================================================================
# run_queue_b32.sh — batch 32 후속 스윕 전체 (앞 작업 종료 후 자동 시작)
#
# 기존 결과표(effb0/pvtv2b0)의 축을 (a) batch 32 로 통일하고 (b) CANet 에도 IBB 축을
# 붙이며 (c) resnet50/densenet121 두 백본으로 확장한다. **batch 8 축은 제외**한다.
#
#   Stage 0  CANet  b32  effb0/pvtv2b0   full+bbox × ibb{0,1}          8런
#   Stage A  OCNN / OCNN-IT   r50/dn121  bbox      (--ibb step2 0.5)   4런
#   Stage B  CCNN / CCNN-IT   r50/dn121  bbox      (--ibb step2 1.0)   4런
#   Stage C  CANet  b32  r50/dn121       full+bbox × ibb{0,1}          8런
#   --------------------------------------------------------------------
#                                                                     24런
#
# 모드별 인자는 기존 표를 만든 런의 done.txt 와 대조해 맞췄다:
#   OCNN     crop_bbox_*_r512_b32        (플래그 없음)
#   OCNN-IT  crop_bbox_*_r512_ibb_b32    ibb step2_ratio=0.5
#   CCNN     crop_bbox_*_r512_ccnn_a2    pairwise, 저자 기본값(w_nodes .1 / nll / cut 25 / shared)
#   CCNN-IT  crop_bbox_*_r512_ccnnit_a2  pairwise + ibb step2_ratio=1.0
# CANet 은 --pairwise 와 배타(train.py 가 막음)라 IBB 축만 곱한다. CANet+IBB 는
# 특징 층 × 데이터 층으로 직교하며 step2_ratio 는 OCNN-IT 와 같은 0.5 를 쓴다.
#
# 사용법:
#   bash run_queue_b32.sh                    # 앞 작업(WAIT_PID) 끝나면 24런
#   WAIT_PID=0 bash run_queue_b32.sh         # 대기 없이 즉시
#   STAGES="0" bash run_queue_b32.sh         # 특정 스테이지만
#   MODEL_NEW="r50" bash run_queue_b32.sh    # 신규 백본 하나만
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

MODEL_OLD="${MODEL_OLD:-effb0 pvtv2b0}"    # 기존 표의 두 백본 (Stage 0)
MODEL_NEW="${MODEL_NEW:-r50 dn121}"        # 신규 백본     (Stage A/B/C)
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-512}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
STAGES="${STAGES:-0 A B C}"
WAIT_PID="${WAIT_PID:-0}"                  # 이 PID 종료까지 대기(0=대기 안 함)
WAIT_MATCH="${WAIT_MATCH:-run_canet_sweep}"  # PID 재사용 오인 방지용 cmdline 검사

# IBB 인자: OCNN-IT/CANet 계열은 step2 0.5, CCNN-IT 는 theta 분산 때문에 1.0.
IBB_A="--ibb_alpha 0.5 --ibb_cap 4.0 --ibb_edge_v 0.25 --ibb_step2_ratio 0.5 --ibb_warmup 3"
IBB_B="--ibb_alpha 0.5 --ibb_cap 4.0 --ibb_edge_v 0.25 --ibb_step2_ratio 1.0 --ibb_warmup 3"

log() { echo "[$(date '+%F %T')] $*"; }
FAILED=""

# --- 1) 앞 작업 대기 -----------------------------------------------------------
if [ "$WAIT_PID" != "0" ]; then
  if kill -0 "$WAIT_PID" 2>/dev/null; then
    log "대기 시작: PID $WAIT_PID ($WAIT_MATCH) 종료까지"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
      # PID 가 재사용돼 다른 프로세스가 됐으면 대기를 끝낸다.
      if ! tr '\0' ' ' < "/proc/$WAIT_PID/cmdline" 2>/dev/null | grep -q "$WAIT_MATCH"; then
        log "PID $WAIT_PID 가 더 이상 $WAIT_MATCH 아님 -> 대기 종료"; break
      fi
      sleep 60
    done
    log "앞 작업 종료 확인 -> 후속 스윕 시작"
  else
    log "PID $WAIT_PID 이미 종료됨 -> 즉시 시작"
  fi
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/[GPU] /'

# --- 2) 스테이지 실행 헬퍼 -----------------------------------------------------
# OCNN/CCNN 계열은 run_sweep.sh(모델 변수 MODEL), CANet 은 run_canet_sweep.sh
# (모델 변수 MODELS_TO_RUN) — 변수명이 다르니 주의.
stage_sweep() {   # $1=이름  $2...=추가 env
  local nm="$1"; shift
  log "########## Stage $nm 시작 (run_sweep.sh) ##########"
  if env EXP_LIST=bbox IMGSZ_LIST="$IMGSZ" BATCH="$BATCH" MODEL="$MODEL_NEW" \
         SKIP_MASKS=1 EPOCHS="$EPOCHS" PATIENCE="$PATIENCE" "$@" bash run_sweep.sh; then
    log "########## Stage $nm 완료 ##########"
  else
    log "!! Stage $nm 실패(계속 진행)"; FAILED="$FAILED $nm"
  fi
}

stage_canet() {   # $1=이름  $2=모델목록
  local nm="$1" models="$2"
  log "########## Stage $nm 시작 (run_canet_sweep.sh, models=$models) ##########"
  if env MODELS_TO_RUN="$models" EXP_LIST="full bbox" IMGSZ="$IMGSZ" BATCH="$BATCH" \
         IBB_LIST="0 1" IBB_ARGS="$IBB_A" SKIP_MASKS=1 SUFFIX="_canet_b32" \
         EPOCHS="$EPOCHS" PATIENCE="$PATIENCE" bash run_canet_sweep.sh; then
    log "########## Stage $nm 완료 ##########"
  else
    log "!! Stage $nm 실패(계속 진행)"; FAILED="$FAILED $nm"
  fi
}

# --- 3) 실행 -------------------------------------------------------------------
echo "$STAGES" | grep -qw 0 && stage_canet 0 "$MODEL_OLD"
echo "$STAGES" | grep -qw A && stage_sweep A SUFFIX=_b32 IBB_LIST="0 1" PW_LIST=0 IBB_ARGS="$IBB_A"
echo "$STAGES" | grep -qw B && stage_sweep B SUFFIX=_b32 IBB_LIST="0 1" PW_LIST=1 IBB_ARGS="$IBB_B"
echo "$STAGES" | grep -qw C && stage_canet C "$MODEL_NEW"

# --- 4) 요약 -------------------------------------------------------------------
log "###### 후속 스윕 종료.  실패한 스테이지:${FAILED:- 없음} ######"
echo ""
echo "--- b32 결과 (test_mean_qwk) ---"
for d in runs/crop_*_b32/; do
  [ -f "$d/done.txt" ] || continue
  printf '%-48s %s\n' "$(basename "$d")" \
    "$(grep -o 'test_mean_qwk=[0-9.]*' "$d/done.txt" | tail -1)"
done | sort
