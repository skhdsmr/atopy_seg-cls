#!/usr/bin/env bash
# =============================================================================
# run_distill.sh — 조건 C/D/E/F/G/H(Single->Multi 증류) 학습
#
# C/D/E 는 teacher_logits/<task>.pt(5-fold OOF, collect_oof.py 산출)를 증류한다.
# 먼저 run_teachers.sh MODE=oof (또는 MODE=both) 로 그 5개가 만들어져 있어야 한다.
#
# F/G/H 는 C/D/E 와 각각 같은 anneal/coherence 구성이지만, teacher 로짓 출처가
# teacher_logits_a/<task>.pt(조건 A 의 best.pt, train 전체 학습 — **in-sample**,
# collect_a.py 산출)로 바뀐 것이다. 먼저 run_teachers.sh MODE=full 로 조건 A 5개
# 모델(A_single_<task>_...)이 만들어져 있어야 한다. in-sample 로짓은 teacher 가
# 이미 본 표본에 대한 예측이라 OOF 보다 낙관적으로 편향될 수 있다 — F/G/H 는
# "OOF 로 누수를 막은 증류"(C/D/E)와 "조건 A 를 그대로 증류"(F/G/H)를 대조하기
# 위한 조건이지, F/G/H 가 방법론적으로 더 낫다는 뜻이 아니다.
#
# 조건 B는 run_multitask.sh 로 만든다.
#
# 사용법:
#   bash run_distill.sh                                       # DATA 기본값(아토피 dataset_all_final)
#   DATA=/path/to/dataset bash run_distill.sh                # C,D,E,F,G,H 순서로 전부
#   DATA=... COND=D bash run_distill.sh                       # 하나만(C|D|E|F|G|H)
#   DATA=... COND=cde bash run_distill.sh                      # OOF 트리오만(C,D,E)
#   DATA=... COND=fgh bash run_distill.sh                      # 조건 A 트리오만(F,G,H)
#   DATA=... MODEL=pvtv2b0 IMGSZ=224 bash run_distill.sh
#   DATA=... LAMBDA_CON=0.3 bash run_distill.sh                # 조건 E/H 의 일관성 가중
#   DATA=... EXP=bbox bash run_distill.sh                       # 마스크 union bbox 크롭
#     (teacher_logits/<task>_bbox[_sq].pt, teacher_logits_a/<task>_bbox[_sq].pt 가
#      있어야 한다 — run_teachers.sh 도 같은 EXP/BBOX_SQUARE 로 먼저 돌릴 것)
#   DATA=... TEACHER_DIR=/path bash run_distill.sh              # C/D/E teacher 로짓 위치 지정
#   DATA=... TEACHER_DIR_A=/path bash run_distill.sh            # F/G/H teacher 로짓 위치 지정
#   DATA=... MTL=fixed IGA_WEIGHT=0.5 bash run_distill.sh       # Kendall 대신 고정 배분
#     (조건 B와 대조하려면 run_multitask.sh 도 같은 MTL/IGA_WEIGHT 로 돌릴 것)
#   DATA=... RESUME=1 bash run_distill.sh                       # 끊긴 학습 이어서
# =============================================================================
set -uo pipefail

_abs() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
for _v in DATA PROJECT TEACHER_DIR TEACHER_DIR_A; do
  eval "_cur=\${$_v:-}"
  [ -n "$_cur" ] && eval "$_v=\$(_abs \"\$_cur\")"
done
unset _v _cur

cd "$(dirname "$0")"

export PYTHONPATH=""
export PYTHONUNBUFFERED=1   # 로그 파일로 리다이렉트해도 줄단위로 바로바로 flush
VENV="${VENV:-$(cd .. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ -x "$PY" ]; then
  SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  shopt -s nullglob
  NV_LIBS="$(printf '%s:' "$SP"/nvidia/*/lib)"
  shopt -u nullglob
  export LD_LIBRARY_PATH="$SP/torch/lib:${NV_LIBS}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  PY=python3
  echo "경고: 학습 venv($VENV) 없음 -> 시스템 python3." >&2
fi

MODEL="${MODEL:-pvtv2b0}"
IMGSZ="${IMGSZ:-224}"
DATA="${DATA:-$(cd .. && pwd)/dataset_all_final/images}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
PROJECT="${PROJECT:-$(pwd)/runs}"
TEACHER_DIR="${TEACHER_DIR:-$PROJECT/teacher_logits}"        # C/D/E (OOF)
TEACHER_DIR_A="${TEACHER_DIR_A:-$PROJECT/teacher_logits_a}"  # F/G/H (조건 A, in-sample)
LAMBDA_CON="${LAMBDA_CON:-0.3}"
COND="${COND:-all}"                 # C | D | E | F | G | H | cde | fgh | all
EXP="${EXP:-full}"                  # full | bbox — teacher_logits/<task>_bbox[_sq].pt 와 맞출 것
MASKS="${MASKS:-}"
MARGIN="${MARGIN:-0.15}"
BBOX_SQUARE="${BBOX_SQUARE:-0}"
MTL="${MTL:-uncertainty}"            # uncertainty | fixed
IGA_WEIGHT="${IGA_WEIGHT:-0.5}"
RESUME="${RESUME:-0}"
EXTRA="${EXTRA:-}"

CROP_FLAGS=(--exp "$EXP" --margin "$MARGIN")
[ -n "$MASKS" ] && CROP_FLAGS+=(--masks "$MASKS")
[ "$BBOX_SQUARE" = "1" ] && CROP_FLAGS+=(--bbox_square)
RESUME_FLAG=()
[ "$RESUME" = "1" ] && RESUME_FLAG=(--resume)

run_one() {
  local cond="$1" tdir="$2" tsrc="$3"; shift 3
  echo "=== 조건 $cond (teacher_source=$tsrc) ==="
  "$PY" train_distill.py --data "$DATA" --model "$MODEL" --imgsz "$IMGSZ" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --batch "$BATCH" --workers "$WORKERS" \
    --device "$DEVICE" --project "$PROJECT" --teacher_dir "$tdir" --teacher_source "$tsrc" \
    --mtl "$MTL" --iga_weight "$IGA_WEIGHT" "${CROP_FLAGS[@]}" "${RESUME_FLAG[@]}" "$@" $EXTRA
}

run_cde() {
  run_one C "$TEACHER_DIR" oof --anneal none
  run_one D "$TEACHER_DIR" oof --anneal linear
  run_one E "$TEACHER_DIR" oof --anneal linear --coherence --lambda_con "$LAMBDA_CON"
}

run_fgh() {
  run_one F "$TEACHER_DIR_A" single --anneal none
  run_one G "$TEACHER_DIR_A" single --anneal linear
  run_one H "$TEACHER_DIR_A" single --anneal linear --coherence --lambda_con "$LAMBDA_CON"
}

case "$COND" in
  C)   run_one C "$TEACHER_DIR" oof --anneal none ;;
  D)   run_one D "$TEACHER_DIR" oof --anneal linear ;;
  E)   run_one E "$TEACHER_DIR" oof --anneal linear --coherence --lambda_con "$LAMBDA_CON" ;;
  F)   run_one F "$TEACHER_DIR_A" single --anneal none ;;
  G)   run_one G "$TEACHER_DIR_A" single --anneal linear ;;
  H)   run_one H "$TEACHER_DIR_A" single --anneal linear --coherence --lambda_con "$LAMBDA_CON" ;;
  cde) run_cde ;;
  fgh) run_fgh ;;
  all) run_cde; run_fgh ;;
  *) echo "COND 는 C/D/E/F/G/H/cde/fgh/all 중 하나" >&2; exit 1 ;;
esac

echo "완료 -> $PROJECT"
