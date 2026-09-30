#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh - crop 중증도 분류 스윕: 방법(exp) × 모델 × 해상도
#
# 방법1(마스킹 전체이미지=atopy_seg) 대안 3종을 여러 모델/해상도로 한 번에 비교.
# 사전 단계로 분할 ckpt 추론 마스크(atopy_crop_masks/)를 1회 생성한다(없을 때만).
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ 방법(EXP)별로 넘겨야 하는 인자 (스윕은 아래를 EXP_LIST 로 자동 순회)          │
# ├───────────────────────────────────────────────────────────────────────────┤
# │ ① bbox       : --exp bbox                                                   │
# │      마스크 union bbox + margin 크롭 1장(마스킹 X). 추가 인자 없음.          │
# │      관련: --margin 0.15 (병변 주변 정상피부 여유; 대비 문맥)                │
# │                                                                             │
# │ ② mil        : --exp mil --mil_pool {attention|max}                         │
# │      연결요소별 crop bag -> pooling. attention=여러 병변 종합,              │
# │      max=가장 심한 병변이 등급 결정. 관련: --min_area_frac 0.003            │
# │      (작은 노이즈 요소 제외), --max_instances 8 (bag 상한).                 │
# │                                                                             │
# │ ③ twostream  : --exp twostream [--twostream_share]                          │
# │      전체(global)+bbox 크롭(local) 두 스트림 concat.                        │
# │      --twostream_share = 백본 공유(소규모 데이터 과적합↓; 기본은 독립).      │
# └───────────────────────────────────────────────────────────────────────────┘
# 공통(세 방법 모두): --data(원본) --masks(사전계산) --model --imgsz --loss corn
#                     --mtl uncertainty --margin --epochs --patience
#
# 사용법:
#   bash run_sweep.sh                                  # 전체: 3방법 × 5모델 × 2해상도 = 30런
#   EXP_LIST="bbox mil" bash run_sweep.sh              # 특정 방법만
#   MODEL="effb0 mnv4s" bash run_sweep.sh              # 특정 모델만
#   IMGSZ_LIST="224" bash run_sweep.sh                 # 해상도 축소
#   MIL_POOL_LIST="attention max" bash run_sweep.sh    # (mil) 풀링 두 방식 대조 -> mil 런 2배
#   TWOSTREAM_SHARE_LIST="0 1" bash run_sweep.sh       # (twostream) 독립 vs 공유 대조 -> 2배
#   IBB_LIST="0 1" bash run_sweep.sh                   # 2-step IBB on/off 대조(모든 exp 에 곱해짐) -> 2배
#   IBB_LIST=1 IBB_ARGS="--ibb_alpha 0.7 --ibb_alpha_end 0.0" bash run_sweep.sh
#   SUFFIX=_ibbcmp bash run_sweep.sh                   # 런 이름 접미사 변경(기존 _final 결과 보존)
#   PW_LIST=1 IBB_LIST="0 1" bash run_sweep.sh         # A안: CCNN + CCNN-IT (논문 Table1 3,4열)
#   PW_LIST="0 1" IBB_LIST="0 1" bash run_sweep.sh     # OCNN/OCNN-IT/CCNN/CCNN-IT 4열 전부
#   LOGDIR=/path/to/logs bash run_sweep.sh             # 로그 위치 변경(기본 runs/logs)
#
# 로그: 스윕 전체 -> runs/logs/sweep<SUFFIX>_<날짜>_<시각>.log,  런별 -> runs/<NAME>/train.log
#   MARGIN=0.2 MAX_INSTANCES=6 bash run_sweep.sh
#   EPOCHS=100 BATCH=16 PATIENCE=20 bash run_sweep.sh
#   FORCE=1 bash run_sweep.sh                          # 완료된 런도 재학습(기본 FORCE=0: 건너뜀)
#   SKIP_MASKS=1 bash run_sweep.sh                     # 마스크 재생성 생략
#
# 주의: MIL_POOL_LIST 는 exp=mil 에만, TWOSTREAM_SHARE_LIST 는 exp=twostream 에만 축으로 붙는다
#       (bbox 는 부가 축이 없어 모델×해상도만큼만 돈다).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# --- 로그 기록: 스윕 전체 로그 1개 + 런별 train.log ---
# LOGDIR 로 위치 변경 가능. 스윕 로그는 실행 시각으로 구분해 덮어쓰지 않는다.
LOGDIR="${LOGDIR:-runs/logs}"
mkdir -p "$LOGDIR"
SWEEP_LOG="$LOGDIR/sweep${SUFFIX:-_final}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[log] 스윕 로그 -> $SWEEP_LOG"

# --- 학습 venv(CUDA torch) 격리 (다른 run.sh 와 동일 패턴) ---
export PYTHONPATH=""
VENV="${VENV:-$(cd .. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ -x "$PY" ]; then
  SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  NV_LIBS="$(printf '%s:' "$SP"/nvidia/*/lib)"
  export LD_LIBRARY_PATH="$SP/torch/lib:${NV_LIBS}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  PY=python3
  echo "경고: 학습 venv($VENV) 없음 -> 시스템 python3 (torch 충돌 가능)." >&2
fi

# --- 스윕 축 ---
EXP_LIST="${EXP_LIST:-bbox}"          # 방법 3종
MODELS_TO_RUN="${MODEL:-efflite0 effb0 mnv3s mnv4s mnv4m}"  # 온디바이스 후보 5종
IMGSZ_LIST="${IMGSZ_LIST:-224 512}"                 # 해상도 2종
# 방법별 부가 축(해당 exp 일 때만 순회)
MIL_POOL_LIST="${MIL_POOL_LIST:-attention}"         # (mil) attention | max | "attention max"
TWOSTREAM_SHARE_LIST="${TWOSTREAM_SHARE_LIST:-0}"   # (twostream) 0=독립 1=공유 | "0 1"
SUFFIX="${SUFFIX:-_final}"                          # 런 이름 접미사(다른 값이면 기존 결과와 분리 저장)
IBB_LIST="${IBB_LIST:-0}"                           # 2-step IBB 0=끔 1=켬 | "0 1"(대조군 동시 생성)
IBB_ARGS="${IBB_ARGS:---ibb_alpha 0.5 --ibb_cap 4.0 --ibb_edge_v 0.25 --ibb_step2_ratio 0.5 --ibb_warmup 3}"
PW_LIST="${PW_LIST:-0}"                             # A안 코퓰러 pairwise 0=끔 1=켬 | "0 1"
PW_ARGS="${PW_ARGS:-}"                              # 비우면 train.py 기본값 사용

# --- 공통 고정 ---
DATA="${DATA:-../dataset_all_final/images}"
MASKS="${MASKS:-../atopy_crop_masks}"
CKPT="${CKPT:-../segmentation/encoder_decoder/runs/effb0_unetpp_512/checkpoint_best.pth}"
LOSS="${LOSS:-corn}"                                # head loss: corn(순서형, QWK 정렬) | ce
MTL="${MTL:-uncertainty}"                           # 태스크 결합: uncertainty(Kendall σ) | fixed
MARGIN="${MARGIN:-0.15}"
MIN_AREA_FRAC="${MIN_AREA_FRAC:-0.003}"
MAX_INSTANCES="${MAX_INSTANCES:-8}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-3e-4}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
DROPOUT="${DROPOUT:-0.5}"
EMBED_DIM="${EMBED_DIM:-512}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"

# 해상도별 기본 배치(512 OOM 방지). BATCH env 를 주면 그 값으로 고정.
# (mil 은 bag 이라 이미지당 instance 수만큼 forward -> 여유 위해 한 단계 작게)
pick_batch() {
  if [ -n "${BATCH:-}" ]; then echo "$BATCH"; return; fi
  local sz="$1" exp="$2" b
  case "$sz" in 512) b=8 ;; 384) b=16 ;; 256) b=24 ;; *) b=32 ;; esac
  [ "$exp" = "mil" ] && b=$(( b / 2 > 0 ? b / 2 : 1 ))
  echo "$b"
}

# --- 1) 마스크 사전계산 (없을 때만) ---
if [ "${SKIP_MASKS:-0}" != "1" ] && [ ! -d "$MASKS/train" ]; then
  echo "==================== 마스크 사전계산 -> $MASKS ===================="
  "$PY" precompute_masks.py --ckpt "$CKPT" --src "$DATA" --out "$MASKS" || {
    echo "[에러] 마스크 생성 실패. seg ckpt/env 확인." >&2; exit 1; }
else
  echo "[skip] 마스크 존재($MASKS) 또는 SKIP_MASKS=1 -> 재생성 생략"
fi

# --- 부가 축을 exp 별로 펼쳐 '런 목록(NAME|args)' 생성 ---
# 각 항목: NAME<TAB>추가인자   (bbox 는 부가축 없음)
base_variants() {   # $1=exp
  local exp="$1"
  case "$exp" in
    mil)
      for pool in $MIL_POOL_LIST; do
        printf '%s\t%s\n' "_${pool}" "--mil_pool $pool"
      done ;;
    twostream)
      for sh in $TWOSTREAM_SHARE_LIST; do
        if [ "$sh" = "1" ]; then printf '%s\t%s\n' "_share" "--twostream_share"
        else                     printf '%s\t%s\n' "" ""; fi
      done ;;
    *)  printf '%s\t%s\n' "" "" ;;   # bbox: 변형 1개(부가인자 없음)
  esac
}

# IBB(B안) x PAIRWISE(A안) 는 exp 와 무관한 직교 축 -> 모든 변형에 곱한다.
# 두 축의 조합이 곧 논문 Table 1 의 4열이다:
#   ib=0 pw=0 -> OCNN      (CORN head 독립)          런이름 접미사 없음
#   ib=1 pw=0 -> OCNN-IT   (B안: 샘플러만 2-step)    _ibb
#   ib=0 pw=1 -> CCNN      (코퓰러 joint 학습)       _ccnn
#   ib=1 pw=1 -> CCNN-IT   (A안 3-step, theta 전용)  _ccnnit
build_variants() {   # $1=exp
  base_variants "$1" | while IFS="$(printf '\t')" read -r tag args; do
    for ib in $IBB_LIST; do
      for pw in $PW_LIST; do
        t="$tag"; a="$args"
        [ "$ib" = "1" ] && a="$a --ibb $IBB_ARGS"
        [ "$pw" = "1" ] && a="$a --pairwise $PW_ARGS"
        case "$ib$pw" in
          10) t="${t}_ibb" ;;
          01) t="${t}_ccnn" ;;
          11) t="${t}_ccnnit" ;;
        esac
        printf '%s\t%s\n' "$t" "$a"
      done
    done
  done
}

# --- 총 런 수 집계 ---
TOTAL=0
for EXP in $EXP_LIST; do
  NV=$(build_variants "$EXP" | wc -l)
  for M in $MODELS_TO_RUN; do for SZ in $IMGSZ_LIST; do TOTAL=$((TOTAL + NV)); done; done
done
echo "###### 스윕 시작: 총 $TOTAL 런  (방법 [$EXP_LIST] × 모델 [$MODELS_TO_RUN] × 해상도 [$IMGSZ_LIST], 각 ${EPOCHS}ep) ######"
echo "###### 공통: loss=$LOSS mtl=$MTL margin=$MARGIN  |  mil:pool[$MIL_POOL_LIST] twostream:share[$TWOSTREAM_SHARE_LIST] ibb[$IBB_LIST] pairwise[$PW_LIST] ######"
[ "${FORCE:-0}" = "1" ] && echo "(FORCE=1: 완료된 런도 재학습)"

# --- 2) 실행 ---
IDX=0; RAN=0; SKIPPED=0; FAILED=""
for EXP in $EXP_LIST; do
  while IFS="$(printf '\t')" read -r VTAG VARGS; do
    for SZ in $IMGSZ_LIST; do
      B=$(pick_batch "$SZ" "$EXP")
      for M in $MODELS_TO_RUN; do
        NAME="crop_${EXP}_${M}_r${SZ}${VTAG}${SUFFIX}"    # 설정마다 다른 폴더 -> 안 덮어씀
        IDX=$((IDX+1))
        if [ "${FORCE:-0}" != "1" ] && [ -f "runs/$NAME/done.txt" ]; then
          echo "[$IDX/$TOTAL] skip (완료됨): $NAME"; SKIPPED=$((SKIPPED+1)); continue
        fi
        echo ""
        echo "==================== [$IDX/$TOTAL] exp=$EXP  $M  r$SZ  (batch $B) $VARGS ===================="
        echo " 결과 -> runs/$NAME"
        echo "======================================================================"
        mkdir -p "runs/$NAME"
        rm -f "runs/$NAME/done.txt"                 # 재학습 시 낡은 마커 정리
        if "$PY" train.py \
            --exp "$EXP" --model "$M" --imgsz "$SZ" --batch "$B" \
            --data "$DATA" --masks "$MASKS" \
            --loss "$LOSS" --mtl "$MTL" --epochs "$EPOCHS" --patience "$PATIENCE" \
            --margin "$MARGIN" --min_area_frac "$MIN_AREA_FRAC" --max_instances "$MAX_INSTANCES" \
            --lr "$LR" --backbone_lr_scale "$BACKBONE_LR_SCALE" \
            --weight_decay "$WEIGHT_DECAY" --dropout "$DROPOUT" \
            --embed_dim "$EMBED_DIM" --label_smoothing "$LABEL_SMOOTHING" \
            --workers "$WORKERS" --device "$DEVICE" --name "$NAME" $VARGS \
            2>&1 | tee "runs/$NAME/train.log"; then
          RAN=$((RAN+1))
        else
          echo "!! 실패(계속 진행): $NAME" >&2; FAILED="$FAILED $NAME"
        fi
      done
    done
  done < <(build_variants "$EXP")
done

echo ""
echo "###### 스윕 완료: 학습 $RAN / 건너뜀 $SKIPPED / 전체 $TOTAL ######"
[ -n "$FAILED" ] && echo "실패한 런:$FAILED" || echo "실패 없음"
echo "체크포인트: runs/<NAME>/best.pt  |  런별 로그: runs/<NAME>/train.log"
echo "스윕 로그: $SWEEP_LOG"
echo "비교: for d in runs/crop_*/; do echo \"\$d: \$(tail -1 \$d/done.txt 2>/dev/null)\"; done"
