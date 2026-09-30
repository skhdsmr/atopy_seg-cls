#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh - 아토피 5축 MaMNet 스윕
#
#   MODE=aspp    ASPP on/off                      2런  <- IGA 분기 전역문맥이 실제로 버는가
#   MODE=groups  sep / merged / flat              3런  <- branch5(태선화 분리)가 버는가
#   MODE=main    (groups x aspp)                  6런  <- 위 두 축의 교호작용까지
#   MODE=sfen    share / task / group             3런  <- CA·SA 배치
#   MODE=cfen    all / none / 태선화쌍만          3런  <- cross-feature 가 버는가
#   MODE=igaskip auto / f / none                  3런  <- ASPP 축의 F skip 교란 분리
#   MODE=imgsz   224 / 448 / 512                  3런  <- 병변 질감이 해상도를 버는가
#   MODE=grid    groups x imgsz(224,512) x aspp  12런  <- 세 축 전체 그리드(교호작용까지)
#   MODE=one     기본 설정 1런
#
# 사용법:
#   bash run_sweep.sh                 # MODE=main
#   MODE=aspp bash run_sweep.sh
#   MODE=aspp GROUP=merged bash run_sweep.sh     # 다른 축 고정한 채 ASPP 만
#   MODE=imgsz BATCH=8 bash run_sweep.sh          # VRAM 모자라면 배치를 낮춰서
#   DRY=1 MODE=main bash run_sweep.sh             # 무엇이 돌지만 확인
#   FORCE=1 ... 는 run.sh 로 그대로 전달된다
# =============================================================================
set -uo pipefail
[ -n "${DATA:-}" ] && DATA="$(cd "$DATA" 2>/dev/null && pwd || echo "$DATA")"
cd "$(dirname "$0")"

MODE="${MODE:-main}"
DRY="${DRY:-0}"
LOG="sweep_$(date +%Y%m%d_%H%M%S).txt"

# 축으로 쓰지 않는 값은 여기서 고정되고, 환경변수로 덮어쓸 수 있다.
GROUP="${GROUP:-sep}"
SFEN="${SFEN:-share}"
CFEN="${CFEN:-all}"
ASPP="${ASPP:-1}"
IGA_SKIP="${IGA_SKIP:-auto}"
IMGSZ="${IMGSZ:-224}"        # 16의 배수여야 한다(side branch 의 pool2->upsample2 정합)

# 태선화 가설 검증용 쌍: 라벨 상관상 lich<-acute(0.38)는 살고 acute<-lich(0.21)는 죽어야 한다.
LICH_PAIRS="lich<-acute,acute<-lich"

runs=()          # "설명|VAR=값 VAR=값 ..."
case "$MODE" in
  aspp)
    runs+=("ASPP on  (IGA 분기 전역문맥)|ASPP=1")
    runs+=("ASPP off (IGA 도 F skip)   |ASPP=0")
    ;;
  groups)
    runs+=("sep    IGA/급성3/태선화(branch5)|GROUP=sep")
    runs+=("merged 논문 2분기(branch5 없음) |GROUP=merged")
    runs+=("flat   태스크당 1분기(5분기)    |GROUP=flat")
    ;;
  main)
    for g in sep merged flat; do
      for a in 1 0; do
        runs+=("groups=$g aspp=$a|GROUP=$g ASPP=$a")
      done
    done
    ;;
  sfen)
    runs+=("share CA태스크별+SA그룹공유|SFEN=share")
    runs+=("task  논문 그대로 태스크별  |SFEN=task")
    runs+=("group 그룹별(가장 쌈)       |SFEN=group")
    ;;
  cfen)
    runs+=("cfen all  (그룹쌍 전부)|CFEN=all")
    runs+=("cfen none (cross 없음) |CFEN=none")
    runs+=("cfen 태선화쌍만        |CFEN=$LICH_PAIRS")
    ;;
  grid)
    # 세 축 전체(3 x 2 x 2 = 12). 싼 것부터 돌도록 224 를 먼저 깐다.
    # A100 batch16 peak: sep 3.6/14.8 · merged 3.2/13.6 · flat 4.6/17.7 GiB (224/512px)
    for sz in 224 512; do
      for g in sep merged flat; do
        for a in 1 0; do
          runs+=("groups=$g imgsz=$sz aspp=$a|GROUP=$g IMGSZ=$sz ASPP=$a")
        done
      done
    done
    ;;
  imgsz)
    # A100 batch16 실측: 224px 3.4GiB/84ms, 448px 11.3GiB/264ms, 512px 14.6GiB/338ms.
    # cls_mbn 스윕에서는 r512 가 r224 보다 나빴다(effb0 .5449->.5205, pvtv2b0 .5810->.5747).
    # 1,800장이라 과적합이 빨라진 것인지 해상도가 원래 안 버는 것인지를 여기서 가른다.
    runs+=("224px 논문 Table 2 기본     |IMGSZ=224")
    runs+=("448px 구진·찰상 질감 보존   |IMGSZ=448")
    runs+=("512px cls_mbn 과 같은 비교점|IMGSZ=512")
    ;;
  igaskip)
    runs+=("iga_skip auto (ASPP 있으면 F 제외)|IGA_SKIP=auto")
    runs+=("iga_skip f    (ASPP+F 둘 다)      |IGA_SKIP=f")
    runs+=("iga_skip none (항상 F 제외)       |IGA_SKIP=none")
    ;;
  one)
    runs+=("기본 설정|")
    ;;
  *)
    echo "알 수 없는 MODE: $MODE  (grid|aspp|groups|main|sfen|cfen|igaskip|imgsz|one)" >&2
    exit 2
    ;;
esac

echo "==================== 스윕 MODE=$MODE : ${#runs[@]}런 ===================="
printf '%s\n' "${runs[@]}" | sed 's/|/   ->   /'
echo " 고정값: GROUP=$GROUP SFEN=$SFEN CFEN=$CFEN ASPP=$ASPP IGA_SKIP=$IGA_SKIP IMGSZ=$IMGSZ (축으로 쓰는 것은 덮어씀)"
echo " 로그  : $LOG"
echo "========================================================================"
[ "$DRY" = "1" ] && { echo "(DRY=1 이라 여기서 종료)"; exit 0; }

ok=0; fail=0
for item in "${runs[@]}"; do
  desc="${item%%|*}"
  vars="${item#*|}"
  echo "" | tee -a "$LOG"
  echo "######## [$((ok+fail+1))/${#runs[@]}] $desc ########" | tee -a "$LOG"
  if env GROUP="$GROUP" SFEN="$SFEN" CFEN="$CFEN" ASPP="$ASPP" IGA_SKIP="$IGA_SKIP" \
         IMGSZ="$IMGSZ" \
         $vars bash run.sh 2>&1 | tee -a "$LOG"; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    echo "!! 실패: $desc" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "###### 스윕 완료: 성공 $ok / 실패 $fail ######" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "== 런별 결과(done.txt) ==" | tee -a "$LOG"
for d in runs/*/done.txt; do
  [ -f "$d" ] || continue
  echo "--- $(dirname "$d" | xargs basename)" | tee -a "$LOG"
  cat "$d" | tee -a "$LOG"
done
