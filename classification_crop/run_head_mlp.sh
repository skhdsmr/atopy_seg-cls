#!/usr/bin/env bash
# =============================================================================
# run_head_mlp.sh - pvtv2b0 bbox r512 b32: head 를 Linear 1층 vs MLP(FCNN) 로 비교
#
# 기준선 crop_bbox_pvtv2b0_r512_b32 (seed 42, head Linear) 와 같은 조건에서
# --head_hidden 만 바꾼다. 시드 3개로 비교(test 200장 SE ≈ ±0.05 라 단일 시드는 부족).
#
# 사용법:
#   bash run_head_mlp.sh                          # MLP(256) seed 42 43 44
#   HH=0 SEEDS="43 44" bash run_head_mlp.sh       # 대조군(Linear) 추가 시드
#   HH=512 SEEDS=42 bash run_head_mlp.sh          # 은닉 차원 변경
#   FORCE=1 bash run_head_mlp.sh                  # 완료된 런도 재학습
# 로그: runs/<name>/train.log
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=""
PY="${PY:-$(cd .. && pwd)/.venv-train/bin/python}"
SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SP/torch/lib:$(printf '%s:' "$SP"/nvidia/*/lib)/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

HH="${HH:-256}"
SEEDS="${SEEDS:-42 43 44}"
for s in $SEEDS; do
  if [ "$HH" = "0" ]; then NAME="crop_bbox_pvtv2b0_r512_b32_lin_s${s}"; else NAME="crop_bbox_pvtv2b0_r512_b32_mlp${HH}_s${s}"; fi
  if [ -f "runs/$NAME/done.txt" ] && [ "${FORCE:-0}" != "1" ]; then echo "[skip] $NAME"; continue; fi
  mkdir -p "runs/$NAME"
  echo "==== $NAME (head_hidden=$HH seed=$s) ===="
  "$PY" train.py --exp bbox --model pvtv2b0 --imgsz 512 --batch 32 \
    --data ../dataset_all_final/images --masks ../atopy_crop_masks \
    --loss corn --mtl uncertainty --epochs 100 --patience 20 --margin 0.15 \
    --dropout 0.5 --workers 6 --head_hidden "$HH" --seed "$s" --name "$NAME" \
    2>&1 | tee "runs/$NAME/train.log" | grep --line-buffered -E "^\[TEST\]|early stop|done\]|Traceback|Error"
done
