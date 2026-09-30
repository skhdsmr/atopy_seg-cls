#!/usr/bin/env bash
# =============================================================================
# run_teachers.sh — 조건 A(single-task teacher) + teacher OOF(5-fold) 학습
#
# 사용법:
#   bash run_teachers.sh                                       # DATA 기본값(아토피 dataset_all_final)
#   DATA=/path/to/dataset bash run_teachers.sh                # MODE=both (기본)
#   DATA=/path/to/dataset MODE=full bash run_teachers.sh       # 조건 A 만(5개 축)
#     -> 태스크마다 collect_a.py 를 자동 호출해 teacher_logits_a/<task>.pt 생성
#        (조건 A의 best.pt에서 뽑은 in-sample 로짓 — run_distill.sh COND=F/G/H 용.
#         OOF 아님: teacher 가 이미 본 train 표본에 대한 예측이다)
#   DATA=/path/to/dataset MODE=oof  bash run_teachers.sh       # teacher OOF 만(5축x5fold=25개)
#     -> oof 완료 후 태스크마다 collect_oof.py 를 자동 호출해 teacher_logits/<task>.pt 생성
#   DATA=... MODEL=pvtv2b0 IMGSZ=224 bash run_teachers.sh
#   DATA=... TASKS="erythema iga_grade" bash run_teachers.sh   # 축 일부만
#   DATA=... N_FOLDS=5 FOLDS_JSON=runs/folds.json bash run_teachers.sh
#   DATA=... EPOCHS=100 PATIENCE=5 BATCH=32 bash run_teachers.sh
#   DATA=... RESUME=1 bash run_teachers.sh                     # 끊긴 학습 이어서
#     (스윕 안의 모든 태스크/fold 호출에 --resume 를 붙인다 — last.pt 없는 것은
#      그냥 처음부터 시작하므로 아직 안 돌린 축까지 같이 줘도 안전하다)
#   DATA=... EXP=bbox bash run_teachers.sh                     # 마스크 union bbox 크롭
#   DATA=... EXP=bbox MASKS=/path bash run_teachers.sh         # 마스크가 데이터셋 밖에 있을 때만
#   DATA=... EXP=bbox MARGIN=0.2 BBOX_SQUARE=1 bash run_teachers.sh
#     (EXP=bbox 로 만든 teacher_logits/<task>_bbox[_sq].pt 는 student 쪽
#      train_distill.py 도 반드시 같은 --exp/--bbox_square 로 돌려야 찾는다)
# =============================================================================
set -uo pipefail

_abs() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
for _v in DATA PROJECT FOLDS_JSON; do
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
MODE="${MODE:-both}"                 # full | oof | both
N_FOLDS="${N_FOLDS:-5}"
FOLDS_JSON="${FOLDS_JSON:-$PROJECT/folds.json}"
TASKS="${TASKS:-erythema papulation excoriation lichenification iga_grade}"
EXP="${EXP:-full}"                   # full | bbox
MASKS="${MASKS:-}"
MARGIN="${MARGIN:-0.15}"
BBOX_SQUARE="${BBOX_SQUARE:-0}"
RESUME="${RESUME:-0}"
EXTRA="${EXTRA:-}"

CROP_FLAGS=(--exp "$EXP" --margin "$MARGIN")
[ -n "$MASKS" ] && CROP_FLAGS+=(--masks "$MASKS")
[ "$BBOX_SQUARE" = "1" ] && CROP_FLAGS+=(--bbox_square)
COLLECT_CROP_FLAGS=(--exp "$EXP")
[ "$BBOX_SQUARE" = "1" ] && COLLECT_CROP_FLAGS+=(--bbox_square)
RESUME_FLAG=()
[ "$RESUME" = "1" ] && RESUME_FLAG=(--resume)

echo "[run_teachers] MODE=$MODE MODEL=$MODEL IMGSZ=$IMGSZ DATA=$DATA EXP=$EXP"

run_full() {
  for t in $TASKS; do
    echo "=== 조건 A: $t ==="
    "$PY" train_single.py --task "$t" --data "$DATA" --model "$MODEL" --imgsz "$IMGSZ" \
      --epochs "$EPOCHS" --patience "$PATIENCE" --batch "$BATCH" --workers "$WORKERS" \
      --device "$DEVICE" --project "$PROJECT" "${CROP_FLAGS[@]}" "${RESUME_FLAG[@]}" $EXTRA
    echo "=== collect_a (조건 A in-sample 로짓, run_distill.sh F/G/H 용): $t ==="
    "$PY" collect_a.py --task "$t" --data "$DATA" --model "$MODEL" --imgsz "$IMGSZ" \
      --workers "$WORKERS" --device "$DEVICE" --project "$PROJECT" "${CROP_FLAGS[@]}"
  done
}

run_oof() {
  if [ ! -f "$FOLDS_JSON" ]; then
    echo "=== fold 분할 생성 -> $FOLDS_JSON ==="
    "$PY" make_folds.py --data "$DATA" --out "$FOLDS_JSON" --n_folds "$N_FOLDS"
  else
    echo "[run_teachers] 기존 fold 분할 재사용: $FOLDS_JSON"
  fi
  for t in $TASKS; do
    for k in $(seq 0 $((N_FOLDS - 1))); do
      echo "=== teacher OOF: $t fold $k/$N_FOLDS ==="
      "$PY" train_single.py --task "$t" --data "$DATA" --model "$MODEL" --imgsz "$IMGSZ" \
        --epochs "$EPOCHS" --patience "$PATIENCE" --batch "$BATCH" --workers "$WORKERS" \
        --device "$DEVICE" --project "$PROJECT" --fold "$k" --folds_json "$FOLDS_JSON" \
        "${CROP_FLAGS[@]}" "${RESUME_FLAG[@]}" $EXTRA
    done
    echo "=== collect_oof: $t ==="
    "$PY" collect_oof.py --task "$t" --model "$MODEL" --imgsz "$IMGSZ" --n_folds "$N_FOLDS" \
      --project "$PROJECT" --folds_json "$FOLDS_JSON" "${COLLECT_CROP_FLAGS[@]}"
  done
}

case "$MODE" in
  full) run_full ;;
  oof)  run_oof ;;
  both) run_full; run_oof ;;
  *) echo "MODE 는 full/oof/both 중 하나" >&2; exit 1 ;;
esac

echo "완료 -> $PROJECT"
