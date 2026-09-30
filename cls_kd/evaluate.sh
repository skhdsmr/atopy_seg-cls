#!/usr/bin/env bash
# =============================================================================
# evaluate.sh — PROJECT 아래 A_single_*/B_*/C_*/D_*/E_*/F_*/G_*/H_* 런을 전부
# eval.py 로 평가
#
# 조건 A는 축마다 별도 런(A_single_<task>_...)이라 자연히 축별로 갈린다. 각 런의
# --data 는 학습 당시 저장된 값을 그대로 쓴다(eval.py 의 기본 동작) — 다른
# 데이터로 재평가하려면 DATA 를 명시할 것(전부 같은 경로로 강제된다).
#
# 사용법:
#   bash evaluate.sh                              # runs/ 안의 A~H 런 전부, test
#   PROJECT=/other/runs bash evaluate.sh
#   SPLIT=val bash evaluate.sh
#   DATA=/other/dataset bash evaluate.sh          # 전체 런에 다른 데이터 강제
#   LAST=1 bash evaluate.sh                       # best.pt 대신 last.pt
#   RUN=D_pvtv2b0_r224 bash evaluate.sh           # 런 하나만
#   NO_CONFUSION=1 bash evaluate.sh               # 혼동행렬 출력 생략(로그만 줄임)
# =============================================================================
set -uo pipefail

_abs() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
for _v in DATA PROJECT; do
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

PROJECT="${PROJECT:-$(pwd)/runs}"
SPLIT="${SPLIT:-test}"
DATA="${DATA:-}"
LAST="${LAST:-0}"
RUN="${RUN:-}"
BATCH="${BATCH:-}"
NO_CONFUSION="${NO_CONFUSION:-0}"
EXTRA="${EXTRA:-}"

FLAGS=(--split "$SPLIT")
[ -n "$DATA" ] && FLAGS+=(--data "$DATA")
[ "$LAST" = "1" ] && FLAGS+=(--last)
[ -n "$BATCH" ] && FLAGS+=(--batch "$BATCH")
[ "$NO_CONFUSION" = "1" ] && FLAGS+=(--no_confusion)

if [ -n "$RUN" ]; then
  dirs=("$PROJECT/$RUN")
else
  dirs=()
  while IFS= read -r -d '' d; do dirs+=("$d"); done < <(
    find "$PROJECT" -maxdepth 1 -mindepth 1 -type d \
      \( -name 'A_single_*' -o -name 'B_*' -o -name 'C_*' -o -name 'D_*' -o -name 'E_*' \
         -o -name 'F_*' -o -name 'G_*' -o -name 'H_*' \) \
      -print0 | sort -z)
fi

if [ "${#dirs[@]}" -eq 0 ]; then
  echo "[evaluate] $PROJECT 아래에서 A_single_*/B_*/C_*/D_*/E_*/F_*/G_*/H_* 런을 못 찾았다." >&2
  exit 1
fi

for d in "${dirs[@]}"; do
  ckpt="$d/best.pt"
  [ "$LAST" = "1" ] && ckpt="$d/last.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[evaluate] 건너뜀(없음): $ckpt" >&2
    continue
  fi
  echo "############################################################"
  echo "### $(basename "$d")"
  echo "############################################################"
  "$PY" eval.py --run "$d" "${FLAGS[@]}" $EXTRA
  echo
done

echo "완료 -> $PROJECT"
