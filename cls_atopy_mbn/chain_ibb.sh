#!/usr/bin/env bash
# =============================================================================
# chain_ibb.sh - 앞선 스윕이 끝나면 상위 K개 설정에 IBB 샘플러를 붙여 이어서 학습
#
# 왜: 이 저장소에서 태스크 의존성을 특징/라벨 층에서 다룬 실험(CANet, CCNN pairwise,
#     cls_mbn 의 CFEN)은 전부 baseline 을 못 넘겼고, **데이터 층**의 IBB 샘플러만
#     4개 백본 전부에서 넘겼다(classification_crop --ibb, 최고 .5630).
#     그리드(특징 층)로 번 것과 샘플러(데이터 층)로 번 것이 더해지는지 확인한다.
#
# 짝 비교가 되도록, 그리드에서 나온 설정을 **그대로** 쓰고 --ibb 만 켠다.
# 선정 기준은 val_best_score 다 — test_score 로 고르면 test 가 샌다.
#
# 사용법:
#   WAIT_PID=7523 setsid bash chain_ibb.sh > logs/ibb.log 2>&1 < /dev/null &
#   TOPK=3 WAIT_PID=7523 bash chain_ibb.sh     # 상위 3개
#   TOPK=2 bash chain_ibb.sh                   # 대기 없이 바로(이미 끝난 경우)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

WAIT_PID="${WAIT_PID:-}"
TOPK="${TOPK:-2}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
DATA="${DATA:-$(cd .. && pwd)/dataset_all_final/images}"
PROJECT="${PROJECT:-$(pwd)/runs}"
WORKERS="${WORKERS:-8}"
IBB_ALPHA="${IBB_ALPHA:-0.5}"
IBB_CAP="${IBB_CAP:-4.0}"
IBB_EDGE_V="${IBB_EDGE_V:-0.25}"
IBB_WARMUP="${IBB_WARMUP:-3}"

# --- 학습 venv(CUDA torch) 격리 — run.sh 와 동일 ---
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

# ----------------------------------------------------------------- 1) 앞 스윕 대기
if [ -n "$WAIT_PID" ]; then
  echo "[chain] $(date '+%F %T')  앞 스윕(PID $WAIT_PID) 종료 대기..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    # PID 재사용 방지: 그 PID 가 아직 run_sweep 인지 확인한다
    if ! tr '\0' ' ' < "/proc/$WAIT_PID/cmdline" 2>/dev/null | grep -q "run_sweep"; then
      echo "[chain] PID $WAIT_PID 가 더 이상 run_sweep 이 아니다 -> 대기 종료"
      break
    fi
    sleep 60
  done
  echo "[chain] $(date '+%F %T')  앞 스윕 종료 확인"
fi

# ----------------------------------------------------------------- 2) 상위 K개 선정
mapfile -t PICKED < <("$PY" - "$TOPK" "$PROJECT" <<'PYEOF'
import sys, re
from pathlib import Path

topk, project = int(sys.argv[1]), Path(sys.argv[2])
rows = []
for d in sorted(project.glob("*/done.txt")):
    txt = d.read_text()
    kv = dict(re.findall(r"(\w+)=([^\s]+)", txt))
    if kv.get("ibb", "0") != "0":            # 이미 IBB 런이면 제외
        continue
    try:
        rows.append((float(kv["val_best_score"]), d.parent.name, kv))
    except (KeyError, ValueError):
        continue
rows.sort(key=lambda r: -r[0])
for score, name, kv in rows[:topk]:
    args = [f"--groups {kv['groups']}", f"--sfen {kv['sfen']}", f"--cfen {kv['cfen']}",
            f"--iga_skip {kv['iga_skip']}", f"--imgsz {kv['imgsz']}",
            f"--batch {kv['batch']}", f"--init {kv['init']}",
            f"--loss {kv['loss']}", f"--mtl {kv['mtl']}"]
    if kv.get("aspp") == "False":
        args.append("--no_aspp")
    print(f"{name}\t{score:.4f}\t{' '.join(args)}")
PYEOF
)

if [ "${#PICKED[@]}" -eq 0 ]; then
  echo "[chain] runs/ 에서 완료된(done.txt) 비-IBB 런을 찾지 못했다 -> 중단" >&2
  exit 1
fi

echo ""
echo "==================== IBB 이어달리기: ${#PICKED[@]}런 ===================="
printf '%s\n' "${PICKED[@]}" | awk -F'\t' '{printf "  %-34s val=%s\n      %s\n", $1, $2, $3}'
echo " alpha=$IBB_ALPHA cap=$IBB_CAP edge_V>=$IBB_EDGE_V warmup=$IBB_WARMUP"
echo "========================================================================"

# ----------------------------------------------------------------- 3) 샘플러 진단표
echo ""
echo "######## IBB 샘플러 진단(학습 없음) ########"
"$PY" sampler.py --data "$DATA" --split train --alpha "$IBB_ALPHA" --cap "$IBB_CAP" \
      --edge_v "$IBB_EDGE_V" 2>&1

# ----------------------------------------------------------------- 4) 학습
ok=0; fail=0
i=0
for line in "${PICKED[@]}"; do
  i=$((i+1))
  base="$(echo "$line" | cut -f1)"
  args="$(echo "$line" | cut -f3)"
  name="${base}_ibb"
  echo ""
  echo "######## [$i/${#PICKED[@]}] $name  (짝: $base) ########"
  if [ -f "$PROJECT/$name/done.txt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "skip (완료됨): $name"
    ok=$((ok+1)); continue
  fi
  rm -f "$PROJECT/$name/done.txt"
  # shellcheck disable=SC2086
  if "$PY" train.py --data "$DATA" --project "$PROJECT" --name "$name" \
        --epochs "$EPOCHS" --patience "$PATIENCE" --workers "$WORKERS" \
        --ibb --ibb_alpha "$IBB_ALPHA" --ibb_cap "$IBB_CAP" \
        --ibb_edge_v "$IBB_EDGE_V" --ibb_warmup "$IBB_WARMUP" $args 2>&1; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "!! 실패: $name" >&2
  fi
done

echo ""
echo "###### IBB 이어달리기 완료: 성공 $ok / 실패 $fail ######"
echo ""
echo "== IBB 런과 그 짝 비교 =="
for line in "${PICKED[@]}"; do
  base="$(echo "$line" | cut -f1)"
  for n in "$base" "${base}_ibb"; do
    [ -f "$PROJECT/$n/done.txt" ] || continue
    echo "--- $n"
    cat "$PROJECT/$n/done.txt"
  done
done
