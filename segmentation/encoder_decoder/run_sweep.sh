#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh - U-Net++/smp 세그 스윕: 인코더 × 디코더 × 이미지크기
#
# ENCODERS × DECODERS × IMGSZS 격자의 모든 조합을 순차 학습한다. 각 조합은 run.sh 를
# 호출하므로 venv 격리·하이퍼파라미터 관례(LOSS/ALPHA/PATIENCE 등)를 그대로 재사용한다.
# 실험 이름은 <인코더태그>_<디코더>_<imgsz> (예: effb0_unetpp_512).
#
# ── 해상도 안내 (imgsz 는 32의 배수) ─────────────────────────────────────────
#     경량/배포:   128 192 224 256
#     표준:        384 512
#     고해상도:    768 1024   (작은 병변에 유리, 메모리↑ / HRNet 권장)
#   기존 effb0_unetpp_deeplab_128..512 스윕과 동일한 축이다.
# ─────────────────────────────────────────────────────────────────────────────
#
# 사용법:
#   bash run_sweep.sh                                   # effb0 × unetpp × {128..512}
#   IMGSZS="256 512" bash run_sweep.sh                  # 해상도만 좁혀서
#   ENCODERS="efficientnet-b0 efficientnet-b3" bash run_sweep.sh   # 인코더 2종
#   DECODERS="unetpp unet manet" bash run_sweep.sh      # 디코더 대조
#   ENCODERS="tu-hrnet_w18" DECODERS="unet" bash run_sweep.sh      # HRNet(고해상도 유지)
#   DATA=../../dataset_all_final bash run_sweep.sh      # 학습 데이터셋 지정
#   LOSS=tversky ALPHA=0.5 EPOCHS=100 PATIENCE=20 bash run_sweep.sh # 공통 하이퍼 오버라이드
#   PREFIX=sweep1 bash run_sweep.sh                     # 이름 접두사
#   FORCE=1 bash run_sweep.sh                           # 완료된 런도 재학습(기본은 건너뜀)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# --- 학습 venv(CUDA torch) 격리 (run.sh 와 동일 패턴, 요약 집계에만 사용) ---
export PYTHONPATH=""
VENV="${VENV:-$(cd ../.. && pwd)/.venv-train}"
PY="${PY:-$VENV/bin/python}"
if [ -x "$PY" ]; then
  SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
  NV_LIBS="$(printf '%s:' "$SP"/nvidia/*/lib)"
  export LD_LIBRARY_PATH="$SP/torch/lib:${NV_LIBS}/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
else
  PY=python3
  echo "경고: 학습 venv($VENV) 없음 -> 시스템 python3 (torch 충돌 가능)." >&2
fi

# --- 스윕 격자 ---------------------------------------------------------------
ENCODERS="${ENCODERS:-efficientnet-b0}"            # efficientnet-b0/b3, resnet34/50, tu-hrnet_w18 ...
DECODERS="${DECODERS:-unetpp}"                     # unetpp / unet / manet
IMGSZS="${IMGSZS:-128 192 224 256 320 384 512}"    # 32의 배수

# --- 모든 조합에 공통 적용될 하이퍼파라미터 (run.sh 로 env 전달) --------------
export DATA="${DATA:-../../dataset_all_final}"
export EPOCHS="${EPOCHS:-100}"
export PATIENCE="${PATIENCE:-20}"
export LR="${LR:-0.0001}"
export ENC_LR_SCALE="${ENC_LR_SCALE:-1.0}"
export LOSS="${LOSS:-tversky}"
export ALPHA="${ALPHA:-0.5}"
export GAMMA="${GAMMA:-1}"
export ATT="${ATT:-none}"
export AUG="${AUG:-on}"
export DEVICE="${DEVICE:-cuda}"
PREFIX="${PREFIX:-}"

# 인코더 이름 -> 짧은 태그 (실험 폴더명용). run.sh 의 NAME 규칙과 맞춘다.
enc_tag() {
  case "$1" in
    efficientnet-b0) echo "effb0" ;;
    efficientnet-b3) echo "effb3" ;;
    tu-hrnet_w18)    echo "hrnet18" ;;
    tu-hrnet_w32)    echo "hrnet32" ;;
    tu-hrnet_w48)    echo "hrnet48" ;;
    *)               echo "${1//[^a-zA-Z0-9]/}" ;;   # 그 외: 특수문자 제거
  esac
}

# 해상도별 기본 배치(고해상도 OOM 방지). BATCH env 를 주면 그 값으로 고정.
pick_batch() {
  if [ -n "${BATCH:-}" ]; then echo "$BATCH"; return; fi
  case "$1" in 1024) echo 4 ;; 768|512) echo 8 ;; 384) echo 12 ;; 256|320) echo 16 ;; *) echo 24 ;; esac
}

# --- 총 조합 수 집계 ---
n=0; for e in $ENCODERS; do for d in $DECODERS; do for s in $IMGSZS; do n=$((n+1)); done; done; done
echo "==================== smp 세그 스윕 ===================="
echo " ENCODERS = $ENCODERS"
echo " DECODERS = $DECODERS"
echo " IMGSZS   = $IMGSZS"
echo " 총 $n 개 조합  |  DATA=$DATA EPOCHS=$EPOCHS PATIENCE=$PATIENCE LOSS=$LOSS ALPHA=$ALPHA ATT=$ATT"
[ "${FORCE:-0}" = "1" ] && echo " (FORCE=1: 완료된 런도 재학습)"
echo "======================================================"

NAMES=""; i=0; RAN=0; SKIPPED=0; FAILED=""
for enc in $ENCODERS; do
  tag="$(enc_tag "$enc")"
  for dec in $DECODERS; do
    for imgsz in $IMGSZS; do
      i=$((i+1))
      name="${PREFIX:+${PREFIX}_}${tag}_${dec}_${imgsz}"   # 조합마다 다른 폴더 -> 안 덮어씀
      NAMES="$NAMES $name"
      B="$(pick_batch "$imgsz")"
      if [ "${FORCE:-0}" != "1" ] && [ -f "runs/$name/checkpoint_best.pth" ]; then
        echo "[$i/$n] skip (완료됨): $name"; SKIPPED=$((SKIPPED+1)); continue
      fi
      echo ""
      echo ">>> [$i/$n] ENC=$enc  DEC=$dec  IMGSZ=$imgsz  (batch $B)  ->  runs/$name"
      if BATCH="$B" ENCODER="$enc" DECODER="$dec" IMGSZ="$imgsz" NAME="$name" bash run.sh; then
        RAN=$((RAN+1))
      else
        echo "!! 실패(계속 진행): $name" >&2; FAILED="$FAILED $name"
      fi
    done
  done
done

# --- 요약: 각 run 의 checkpoint_best.pth 에 저장된 best val 지표 집계 ---------
echo ""
echo "==================== 스윕 요약 (val best, checkpoint_best.pth) ===================="
printf "%-30s %6s %8s %8s %8s %8s\n" "RUN" "ep" "Dice" "IoU" "Recall" "Prec"
for name in $NAMES; do
  ck="runs/$name/checkpoint_best.pth"
  if [ -f "$ck" ]; then
    "$PY" - "$name" "$ck" <<'PY'
import sys, torch
name, ck = sys.argv[1], sys.argv[2]
try:
    d = torch.load(ck, map_location="cpu", weights_only=False)
    v = d.get("val", {}); ep = d.get("epoch", "-")
    print(f"{name:<30} {str(ep):>6} {v.get('dice',float('nan')):>8.4f} "
          f"{v.get('iou',float('nan')):>8.4f} {v.get('recall',float('nan')):>8.4f} "
          f"{v.get('precision',float('nan')):>8.4f}")
except Exception as e:
    print(f"{name:<30} (읽기 실패: {e})")
PY
  else
    printf "%-30s %6s\n" "$name" "(no ckpt)"
  fi
done
echo "=================================================================================="
echo "###### 스윕 완료: 학습 $RAN / 건너뜀 $SKIPPED / 전체 $n ######"
[ -n "$FAILED" ] && echo "실패한 런:$FAILED" || echo "실패 없음"
echo "체크포인트: runs/<RUN>/checkpoint_best.pth  (RUN=<인코더태그>_<디코더>_<imgsz>)"
