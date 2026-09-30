#!/usr/bin/env bash
# =============================================================================
# chain_extra.sh - 끝난 baseline 런들의 설정을 그대로 가져와, 인자만 덧붙여 다시 돌린다
#
# chain_ibb.sh 의 일반화판. 무엇을 덧붙일지(EXTRA)와 런 이름 접미사(SUFFIX)만 바꾸면
# 새 축을 짝 비교로 얹을 수 있다. 설정은 runs/*/done.txt 에서 읽으므로 그리드와
# **완전히 같은 조건**이 되고, 차이는 EXTRA 하나뿐이다.
#
# 선정 기준은 val_best_score (test_score 로 고르면 test 가 샌다).
#
# 사용법:
#   # IBB + IGA 고정가중 0.5 를 12개 설정 전부에
#   SUFFIX=_ibb_a05 EXTRA="--ibb --mtl fixed --iga_weight 0.5" TOPK=12 \
#     WAIT_PID=42228 setsid bash chain_extra.sh > logs/x.log 2>&1 < /dev/null &
#
#   TOPK=3 ... bash chain_extra.sh      # 상위 3개만
#   DRY=1 ... bash chain_extra.sh       # 무엇이 돌지만 확인
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

SUFFIX="${SUFFIX:-_ibb_a05}"
EXTRA="${EXTRA:---ibb --mtl fixed --iga_weight 0.5}"
WAIT_PID="${WAIT_PID:-}"
WAIT_NAME="${WAIT_NAME:-}"          # 비면 PID 존재만 확인(재사용 검사 생략)
TOPK="${TOPK:-12}"
DRY="${DRY:-0}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
DATA="${DATA:-$(cd .. && pwd)/dataset_all_final/images}"
PROJECT="${PROJECT:-$(pwd)/runs}"
WORKERS="${WORKERS:-8}"

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

# ----------------------------------------------------------------- 1) 앞 작업 대기
if [ -n "$WAIT_PID" ]; then
  echo "[chain] $(date '+%F %T')  앞 작업(PID $WAIT_PID) 종료 대기..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    if [ -n "$WAIT_NAME" ] && \
       ! tr '\0' ' ' < "/proc/$WAIT_PID/cmdline" 2>/dev/null | grep -q "$WAIT_NAME"; then
      echo "[chain] PID $WAIT_PID 가 더 이상 '$WAIT_NAME' 이 아니다 -> 대기 종료"
      break
    fi
    sleep 60
  done
  echo "[chain] $(date '+%F %T')  앞 작업 종료 확인"
fi

# ----------------------------------------------------------------- 2) 설정 선정
# 접미사가 없는(= 그리드 baseline) 런만 후보. --mtl 은 EXTRA 가 정하므로 빼고 넘긴다.
mapfile -t PICKED < <("$PY" - "$TOPK" "$PROJECT" <<'PYEOF'
import sys, re
from pathlib import Path

topk, project = int(sys.argv[1]), Path(sys.argv[2])
rows = []
for d in sorted(project.glob("*/done.txt")):
    kv = dict(re.findall(r"(\w+)=([^\s]+)", d.read_text()))
    if kv.get("ibb", "0") != "0":                 # 이미 무언가 얹힌 런은 제외
        continue
    try:
        rows.append((float(kv["val_best_score"]), d.parent.name, kv))
    except (KeyError, ValueError):
        continue
rows.sort(key=lambda r: -r[0])
for score, name, kv in rows[:topk]:
    args = [f"--groups {kv['groups']}", f"--sfen {kv['sfen']}", f"--cfen {kv['cfen']}",
            f"--iga_skip {kv['iga_skip']}", f"--imgsz {kv['imgsz']}",
            f"--batch {kv['batch']}", f"--init {kv['init']}", f"--loss {kv['loss']}"]
    if kv.get("aspp") == "False":
        args.append("--no_aspp")
    print(f"{name}\t{score:.4f}\t{' '.join(args)}")
PYEOF
)

if [ "${#PICKED[@]}" -eq 0 ]; then
  echo "[chain] runs/ 에서 완료된 baseline 런(done.txt, ibb=0)을 찾지 못했다 -> 중단" >&2
  exit 1
fi

echo ""
echo "==================== 이어달리기: ${#PICKED[@]}런  EXTRA='$EXTRA' ===================="
printf '%s\n' "${PICKED[@]}" | awk -F'\t' -v s="$SUFFIX" \
  '{printf "  %-30s -> %s%s   (val=%s)\n", $1, $1, s, $2}'
echo "======================================================================================"
[ "$DRY" = "1" ] && { echo "(DRY=1 이라 여기서 종료)"; exit 0; }

# ----------------------------------------------------------------- 3) 학습
ok=0; fail=0; i=0
for line in "${PICKED[@]}"; do
  i=$((i+1))
  base="$(echo "$line" | cut -f1)"
  args="$(echo "$line" | cut -f3)"
  name="${base}${SUFFIX}"
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
        $args $EXTRA 2>&1; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "!! 실패: $name" >&2
  fi
done

echo ""
echo "###### 이어달리기 완료: 성공 $ok / 실패 $fail  (EXTRA='$EXTRA') ######"
echo ""
echo "== 짝 비교 =="
for line in "${PICKED[@]}"; do
  base="$(echo "$line" | cut -f1)"
  for n in "$base" "${base}_ibb" "${base}${SUFFIX}"; do
    [ -f "$PROJECT/$n/done.txt" ] || continue
    echo "--- $n"
    cat "$PROJECT/$n/done.txt"
  done
done
