#!/usr/bin/env bash
# =============================================================================
# run.sh - 아토피 5축 MaMNet(MbN+SFEN+CFEN+ASPP): 단일 학습
#
# 기본: groups=sep(IGA/급성3/태선화 branch5) · sfen=share(CA 태스크별+SA 그룹공유) ·
#       cfen=all(그룹쌍 6개) · ASPP=1(IGA 분기)
#
# 사용법:
#   bash run.sh                        # 기본 설정
#   ASPP=0 bash run.sh                 # IGA 분기 ASPP 끄기 (sweep 축)
#   GROUP=merged bash run.sh          # branch5 없는 논문 2분기 대조군
#   SFEN=task bash run.sh              # SFEN 통째로 태스크별(논문 그대로)
#   IMGSZ=448 bash run.sh              # 해상도(16의 배수만). A100 batch16 에서 11.3GiB
#   CFEN=none bash run.sh              # cross-feature 전부 끄기
#   CFEN='lich<-acute,acute<-lich' bash run.sh   # 특정 쌍만
#   DATA=/path/to/images bash run.sh   # labels.csv + train/val/test 가 있는 폴더
#   NAME=my_exp FORCE=1 bash run.sh
# =============================================================================
set -uo pipefail

# DATA 등 경로는 cd 전에 절대경로로 굳힌다(상대경로가 조용히 바뀌는 사고 방지).
[ -n "${DATA:-}" ] && DATA="$(cd "$DATA" 2>/dev/null && pwd || echo "$DATA")"
[ -n "${PROJECT:-}" ] && PROJECT="$(cd "$PROJECT" 2>/dev/null && pwd || echo "$PROJECT")"
cd "$(dirname "$0")"

# --- 학습 venv(CUDA torch) 격리 ---
export PYTHONPATH=""
VENV="${VENV:-$(cd .. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ -x "$PY" ]; then
  SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  NV_LIBS="$(printf '%s:' "$SP"/nvidia/*/lib)"
  export LD_LIBRARY_PATH="$SP/torch/lib:${NV_LIBS}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  PY=python3
  echo "경고: 학습 venv($VENV) 없음 -> 시스템 python3." >&2
fi

DATA="${DATA:-$(cd .. && pwd)/dataset_all_final/images}"
PROJECT="${PROJECT:-$(pwd)/runs}"
IMGSZ="${IMGSZ:-224}"                 # 16의 배수만 (side branch 의 pool2->upsample2 정합)
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
WARMUP_LR="${WARMUP_LR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
PATIENCE="${PATIENCE:-10}"
GROUP="${GROUP:-sep}"               # sep | merged | flat
SFEN="${SFEN:-share}"                 # share | task | group
K="${K:-9}"                           # SA 비대칭 conv 커널(논문 최적)
CFEN="${CFEN:-all}"                   # all | none | 'lich<-acute,...'
ASPP="${ASPP:-1}"                     # 1 | 0   <- sweep 축
IGA_SKIP="${IGA_SKIP:-auto}"          # auto | f | none
LOSS="${LOSS:-corn}"                  # corn | ce
MTL="${MTL:-uncertainty}"             # uncertainty | fixed
IGA_W="${IGA_W:-0.5}"
INIT="${INIT:-vgg16}"                 # vgg16 | scratch
NO_SFEN="${NO_SFEN:-0}"
NO_AUG="${NO_AUG:-0}"
CW="${CW:-0}"
DROPOUT="${DROPOUT:-0.5}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

FLAGS=""
[ "$ASPP"    = "0" ] && FLAGS="$FLAGS --no_aspp"
[ "$NO_SFEN" = "1" ] && FLAGS="$FLAGS --no_sfen"
[ "$NO_AUG"  = "1" ] && FLAGS="$FLAGS --no_aug"
[ "$CW"      = "1" ] && FLAGS="$FLAGS --class_weight"

TAG="${GROUP}_${SFEN}"
[ "$ASPP" = "0" ] && TAG="${TAG}_noaspp" || TAG="${TAG}_aspp"
[ "$CFEN" != "all" ] && TAG="${TAG}_cfen$(echo "$CFEN" | tr -cd '[:alnum:]' | cut -c1-12)"
[ "$IGA_SKIP" != "auto" ] && TAG="${TAG}_skip${IGA_SKIP}"
[ "$NO_SFEN" = "1" ] && TAG="${TAG}_nosfen"
[ "$IMGSZ" != "224" ] && TAG="${TAG}_r${IMGSZ}"
[ "$K" != "9" ] && TAG="${TAG}_k${K}"
[ "$LOSS" != "corn" ] && TAG="${TAG}_${LOSS}"
[ "$MTL" != "uncertainty" ] && TAG="${TAG}_a${IGA_W}"
[ "$INIT" != "vgg16" ] && TAG="${TAG}_${INIT}"
NAME="${NAME:-amb_${TAG}}"

echo "==================== 아토피 5축 MaMNet: groups=$GROUP sfen=$SFEN aspp=$ASPP ===================="
echo " 데이터 -> $DATA"
echo " CFEN   -> $CFEN   IGA skip -> $IGA_SKIP   손실 -> $LOSS/$MTL"
echo " 결과   -> $PROJECT/$NAME"
echo "=========================================================================================="

if [ "${FORCE:-0}" != "1" ] && [ -f "$PROJECT/$NAME/done.txt" ]; then
  echo "skip (완료됨): $NAME  -> 재학습하려면 FORCE=1 bash run.sh"
  exit 0
fi

rm -f "$PROJECT/$NAME/done.txt"
if "$PY" train.py \
    --data "$DATA" --project "$PROJECT" --imgsz "$IMGSZ" --batch "$BATCH" --epochs "$EPOCHS" \
    --lr "$LR" --warmup_lr "$WARMUP_LR" --warmup_epochs "$WARMUP_EPOCHS" \
    --patience "$PATIENCE" --groups "$GROUP" --sfen "$SFEN" --sfen_k "$K" \
    --cfen "$CFEN" --iga_skip "$IGA_SKIP" --loss "$LOSS" --mtl "$MTL" \
    --iga_weight "$IGA_W" --init "$INIT" --dropout "$DROPOUT" \
    --workers "$WORKERS" --device "$DEVICE" --seed "$SEED" --name "$NAME" $FLAGS; then
  echo ""
  echo "###### 완료: $NAME ######"
  echo "체크포인트: $PROJECT/$NAME/best.pt  (test_metrics, gate_stats 포함)"
else
  echo "!! 실패: $NAME" >&2
  exit 1
fi
