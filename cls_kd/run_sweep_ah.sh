#!/usr/bin/env bash
# =============================================================================
# run_sweep_ah.sh — 모델(pvtv2b0/pvtv2b1) x margin(0.0/0.05/0.15) 스윕으로
# 조건 A~H 전체 파이프라인(teacher OOF -> 조건 A -> 조건 B -> 조건 C~H)을
# 순차 실행한다.
#
# 고정값: --imgsz 512, --patience 5, --exp bbox --bbox_square,
#         --masks ../atopy_crop_masks(precompute_masks.py 산출 PNG 마스크,
#         dataset_all_final 과 같은 stem 규약).
#
# margin 은 crop_tag(=_bbox_sq)에 이름이 안 남으므로(model.py 의 --exp/--bbox_square
# 만 태그된다) margin 값마다 PROJECT 를 따로 둔다 — 같은 프로젝트에 두 margin 결과를
# 넣으면 teacher_logits/<task>_bbox_sq.pt 를 서로 덮어쓴다. 같은 이유로 model 마다도
# PROJECT 를 분리한다(teacher_logits 파일명에 모델 이름이 안 들어간다 — model.py 주석
# 참고 없음, collect_oof.py/collect_a.py 산출 파일명은 task+crop_tag 뿐).
#
# 한 (margin, model) 조합의 각 단계(teacher_full/teacher_oof/multitask_B/
# distill_C..H)가 끝나면 그 PROJECT 안에 .sweep_done_<stage> 마커를 남긴다.
# 스크립트가 중단된 뒤 다시 실행하면 이미 끝난 단계는 건너뛰고, 모든 하위 학습
# 호출에 --resume 도 같이 걸려 있어 그 안에서 중단된 학습(epoch 중간)도 last.pt
# 에서 이어서 계속된다.
#
# 사용법:
#   bash run_sweep_ah.sh                                      # 포그라운드(터미널 붙들림)
#   setsid nohup bash run_sweep_ah.sh > sweep_ah.log 2>&1 < /dev/null &
#   disown                                                    # 터미널/IDE 종료와 무관하게 계속 돔
#   tail -f sweep_ah.log                                      # 진행상황 확인
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

# 로그 파일로 리다이렉트하면(파이프/파일은 tty 가 아니므로) python 이 줄단위가 아니라
# 블록단위로 버퍼링해서 tail -f 가 한참 멈춰 있는 것처럼 보인다 — 매 epoch 출력이
# 바로바로 로그에 찍히도록 강제로 끈다.
export PYTHONUNBUFFERED=1

MODELS=(pvtv2b0 pvtv2b1)
MARGINS=(0.0 0.05 0.15)
MTAGS=(m0 m05 m15)

MASKS="$(cd .. && pwd)/atopy_crop_masks"
IMGSZ=512
PATIENCE=5
EXP=bbox
BBOX_SQUARE=1

marker() { printf '%s/.sweep_done_%s' "$1" "$2"; }

run_stage() {
  local project="$1" stage="$2"; shift 2
  local mk; mk="$(marker "$project" "$stage")"
  if [ -f "$mk" ]; then
    echo "[sweep] 건너뜀(완료됨): $stage -> $project"
    return 0
  fi
  echo "[sweep] === $stage 시작 -> $project ($(date '+%F %T')) ==="
  if "$@"; then
    touch "$mk"
    echo "[sweep] === $stage 완료 -> $project ($(date '+%F %T')) ==="
  else
    echo "[sweep] !! 실패: $stage -> $project ($(date '+%F %T'))" >&2
    exit 1
  fi
}

for mi in "${!MARGINS[@]}"; do
  MARGIN="${MARGINS[$mi]}"
  MTAG="${MTAGS[$mi]}"
  for MODEL in "${MODELS[@]}"; do
    PROJECT="$ROOT/runs_sweep/${MTAG}/${MODEL}"
    mkdir -p "$PROJECT"
    export MODEL IMGSZ PATIENCE EXP BBOX_SQUARE MARGIN MASKS PROJECT
    export RESUME=1

    echo "############################################################"
    echo "### margin=$MARGIN model=$MODEL imgsz=$IMGSZ -> $PROJECT"
    echo "############################################################"

    run_stage "$PROJECT" teacher_full env MODE=full bash run_teachers.sh
    run_stage "$PROJECT" teacher_oof  env MODE=oof  bash run_teachers.sh
    run_stage "$PROJECT" multitask_B  bash run_multitask.sh
    for COND in C D E F G H; do
      run_stage "$PROJECT" "distill_$COND" env COND="$COND" bash run_distill.sh
    done
  done
done

echo "=== 전체 스윕(모델 x margin) 완료 -> $ROOT/runs_sweep ($(date '+%F %T')) ==="
