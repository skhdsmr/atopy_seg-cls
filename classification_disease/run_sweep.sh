#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh - 질환 6-way 분류 스윕: 각도 × 모델 × 해상도
#
# classification_topk/run_sweep.sh 와 동일 패턴. 사전 단계로 dataset_disease/ 를
# 1회 생성한다(없을 때만). 조합마다 train.py 를 직접 부른다(run.sh 의 skip 로직과
# 이름 규칙은 여기서 그대로 재현 — 중간 단계를 하나 줄여 실패 지점을 좁힌다).
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ 스윕 축                                                                    │
# ├───────────────────────────────────────────────────────────────────────────┤
# │ 각도 3종 : both = 정면+측면 전부                                           │
# │            front = 1024px 실사 얼굴만                                      │
# │            side  = 512px 피부 접사(Z4)만                                   │
# │            둘은 해상도·구도·출처가 전부 달라 사실상 다른 도메인이다. 섞은  │
# │            both 의 accuracy 는 쉬운 side 가 끌어올린 평균일 수 있다 ->     │
# │            단독 성능을 따로 봐야 어느 쪽이 실제로 어려운지 드러난다.        │
# │ 모델     : 기본 mnv3l mnv3s. MODEL 로 확장 — 온디바이스 후보 전체는        │
# │            mnv3s mnv4s efflite0 effb0 efflite1 mnv3l efflite2 efflite3     │
# │            mnv4m efflite4 (10종, 파라미터 오름차순) — ALL=1 로 전부.        │
# │            2.0M(mnv3s) ~ 12.4M(efflite4) 구간을 훑는다.                    │
# │            BIG=1 은 예산 밖 참고 백본 7종(pvt_b0 mnv4m mnv4mh pvt_b1       │
# │            pvt_b2 cnxt_t mnv4l, 3.6M~32M). 배포 후보가 아니라 대조군이다 — │
# │            지금 test macroF1 이 0.92 근처에서 멈춘 게 용량 부족 때문인지    │
# │            데이터/라벨의 천장 때문인지 가른다. 여기서도 안 오르면 백본을    │
# │            키우는 방향은 접고 crop/출처 쪽을 손대야 한다.                   │
# │ 해상도   : 224 512                                                         │
# │            224 = topk/crop 실험과 동일 조건(비교용)                        │
# │            512 = 1024->224 축소로 뭉개지는 텍스처를 살리는 처방            │
# └───────────────────────────────────────────────────────────────────────────┘
# 공통 고정: --select macro_f1 --label_smoothing 0.1 --epochs/--patience
#            클래스가 균형(각 900장)이라 --class_weight 는 기본 off.
#            배치도 해상도만 보고 정한다(모델별로 바꾸지 않는다) — 배치가 다르면
#            모델 비교가 오염된다. r512/batch8 기준 실측 VRAM 은 BIG 그룹 최대가
#            pvt_b2 의 3.9GiB 라 8GB GPU 면 그대로 돌아간다. 모자라면 BATCH=4.
#
# 사용법:
#   bash run_sweep.sh                          # 기본: 3각도 × 2모델 × 2해상도 = 12런
#   ANGLE_LIST="both" bash run_sweep.sh        # 섞은 것만 (4런)
#   ANGLE_LIST="front side" bash run_sweep.sh  # 도메인 분리 대조 (8런)
#   ALL=1 bash run_sweep.sh                    # 온디바이스 후보 10종 전부 (10런)
#   BIG=1 bash run_sweep.sh                    # 참고 백본 7종 전부 (7런)
#   ALL=1 BIG=1 bash run_sweep.sh              # 둘 다 (16런, mnv4m 중복 제거됨)
#   MODEL="pvt_b2 cnxt_t" bash run_sweep.sh                      # 임의 조합
#   IMGSZ_LIST="224" bash run_sweep.sh         # 해상도 축소
#   CLASS_WEIGHT=1 bash run_sweep.sh           # 역빈도 클래스 가중
#   SELECT=balanced_accuracy bash run_sweep.sh # 선택 기준 변경
#   EPOCHS=100 PATIENCE=20 BATCH=16 bash run_sweep.sh
#   LINK=1 bash run_sweep.sh                   # dataset 생성 시 복사 대신 심볼릭 링크
#   FORCE=1 bash run_sweep.sh                  # 완료된 런(done.txt)도 재학습
#   SKIP_DATA=1 bash run_sweep.sh              # dataset 재생성 생략
#
# 주의: test 는 제공된 Validation(새 케이스 + 새 출처)이다. 스윕으로 30런을 돌린 뒤
#       test 최고점을 골라 보고하면 그 순간 test 가 val 이 된다. 모델 선택은 val 로만
#       하고, test 표는 '고른 뒤 한 번 보는 값'으로 취급할 것.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

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
ANGLE_LIST="${ANGLE_LIST:-both}"                  # both | front | side
# 온디바이스 후보 전체(파라미터 오름차순). ALL=1 이면 이걸 전부 돈다.
ONDEVICE="mnv3s mnv4s efflite0 effb0 efflite1 mnv3l efflite2 efflite3 mnv4m efflite4"
# 예산 밖 참고 백본(파라미터 오름차순). BIG=1. mnv4m 은 ONDEVICE 와 겹치지만
# 'MobileNetV4 medium 의 conv/hybrid 대조'가 이 그룹의 질문이라 여기에도 둔다
# (ALL=1 BIG=1 이면 아래에서 중복 제거).
# pvt_b0 mnv4m 
BIG_MODELS="mnv4mh pvt_b1 pvt_b2 cnxt_t mnv4l"
DEFAULT_MODELS="mnv3l mnv3s"
[ "${ALL:-0}" = "1" ] && DEFAULT_MODELS="$ONDEVICE"
[ "${BIG:-0}" = "1" ] && DEFAULT_MODELS="$([ "${ALL:-0}" = "1" ] && echo "$ONDEVICE") $BIG_MODELS"
MODELS_TO_RUN="${MODEL:-$DEFAULT_MODELS}"   # MODEL 을 주면 ALL/BIG 보다 우선
# 순서를 지킨 채 중복만 제거(ALL=1 BIG=1 의 mnv4m).
MODELS_TO_RUN="$(echo $MODELS_TO_RUN | tr ' ' '\n' | awk '!seen[$0]++' | tr '\n' ' ')"
IMGSZ_LIST="${IMGSZ_LIST:-512}"                          # 해상도 2종

# --- 데이터 ---
DATA="${DATA:-../dataset_disease}"
SRC="${SRC:-../skin_dataset}"
VAL_FRAC="${VAL_FRAC:-0.125}"
LINK_FLAG=""; [ "${LINK:-0}" = "1" ] && LINK_FLAG="--link"

# --- 공통 고정 ---
SELECT="${SELECT:-macro_f1}"            # macro_f1 | accuracy | balanced_accuracy | macro_auroc
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
CW_FLAG=""; [ "${CLASS_WEIGHT:-0}" = "1" ] && CW_FLAG="--class_weight"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
LR="${LR:-3e-4}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
DROPOUT="${DROPOUT:-0.5}"
EMBED_DIM="${EMBED_DIM:-512}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-8}"

# 해상도별 기본 배치(512 OOM 방지). BATCH env 를 주면 그 값으로 고정.
pick_batch() {
  if [ -n "${BATCH:-}" ]; then echo "$BATCH"; return; fi
  case "$1" in 512) echo 8 ;; 384) echo 16 ;; 256) echo 24 ;; *) echo 32 ;; esac
}

# --- 1) dataset_disease 생성 (없을 때만) ---
if [ "${SKIP_DATA:-0}" != "1" ] && [ ! -f "$DATA/labels.csv" ]; then
  echo "==================== dataset_disease 생성 -> $DATA ===================="
  "$PY" make_dataset_disease.py --src "$SRC" --out "$DATA" \
    --val_frac "$VAL_FRAC" $LINK_FLAG || {
    echo "[에러] dataset 생성 실패. SRC 경로($SRC)와 다운로드 완료 여부 확인." >&2; exit 1; }
else
  echo "[skip] dataset 존재($DATA) 또는 SKIP_DATA=1 -> 재생성 생략"
fi

# --- 총 런 수 집계 ---
TOTAL=0
for A in $ANGLE_LIST; do for M in $MODELS_TO_RUN; do for SZ in $IMGSZ_LIST; do
  TOTAL=$((TOTAL+1)); done; done; done
echo "###### 스윕 시작: 총 $TOTAL 런  (각도 [$ANGLE_LIST] × 모델 [$MODELS_TO_RUN] × 해상도 [$IMGSZ_LIST], 각 ${EPOCHS}ep) ######"
echo "###### 공통: select=$SELECT label_smoothing=$LABEL_SMOOTHING class_weight=${CLASS_WEIGHT:-0} patience=$PATIENCE ######"
[ "${FORCE:-0}" = "1" ] && echo "(FORCE=1: 완료된 런도 재학습)"

# --- 2) 실행 ---
IDX=0; RAN=0; SKIPPED=0; FAILED=""
for A in $ANGLE_LIST; do
  for SZ in $IMGSZ_LIST; do
    B=$(pick_batch "$SZ")
    for M in $MODELS_TO_RUN; do
      # 각도가 이름에 들어가야 both/front/side 가 서로 덮어쓰지 않는다(run.sh 와 같은 규칙).
      NAME="dis_${M}_r${SZ}_${A}${CLASS_WEIGHT:+_cw}"
      IDX=$((IDX+1))
      if [ "${FORCE:-0}" != "1" ] && [ -f "runs/$NAME/done.txt" ]; then
        echo "[$IDX/$TOTAL] skip (완료됨): $NAME"; SKIPPED=$((SKIPPED+1)); continue
      fi
      echo ""
      echo "==================== [$IDX/$TOTAL] angle=$A  $M  r$SZ  (batch $B) ===================="
      echo " 결과 -> runs/$NAME"
      echo "======================================================================"
      rm -f "runs/$NAME/done.txt"                 # 재학습 시 낡은 마커 정리
      if "$PY" train.py \
          --model "$M" --imgsz "$SZ" --batch "$B" --data "$DATA" --angle "$A" \
          --select "$SELECT" --label_smoothing "$LABEL_SMOOTHING" $CW_FLAG \
          --epochs "$EPOCHS" --patience "$PATIENCE" \
          --lr "$LR" --backbone_lr_scale "$BACKBONE_LR_SCALE" \
          --weight_decay "$WEIGHT_DECAY" --dropout "$DROPOUT" \
          --embed_dim "$EMBED_DIM" \
          --workers "$WORKERS" --device "$DEVICE" --name "$NAME"; then
        RAN=$((RAN+1))
      else
        echo "!! 실패(계속 진행): $NAME" >&2; FAILED="$FAILED $NAME"
      fi
    done
  done
done

echo ""
echo "###### 스윕 완료: 학습 $RAN / 건너뜀 $SKIPPED / 전체 $TOTAL ######"
[ -n "$FAILED" ] && echo "실패한 런:$FAILED" || echo "실패 없음"

# --- 3) 요약표 (test_report.json 파싱) ---
# accuracy 만 보면 안 된다. 클래스는 균형이지만 출처(Z4/H*)는 전혀 균형이 아니라,
# 촬영 출처를 외운 모델도 높은 accuracy 를 낸다. src_spread(출처별 정확도 폭)를 같이 본다.
echo ""
echo "###### 요약 ######"
"$PY" - <<'PYEOF'
import json, glob, os, re
rows, legacy = [], []
for f in sorted(glob.glob("runs/dis_*/test_report.json")):
    name = os.path.basename(os.path.dirname(f))
    m = re.match(r"dis_(\w+?)_r(\d+)_(both|front|side)", name)
    if not m:
        legacy.append(name)          # 이름 규약 밖 -> 축을 못 읽으니 표에 섞지 않는다
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    ts, vs = d.get("test_summary", {}), d.get("val_summary", {})
    # 출처별 정확도 폭: n>=20 인 출처만(소표본 출처는 0/1 로 튄다)
    accs = [v["acc"] for v in (d.get("by_source") or {}).values() if v.get("n", 0) >= 20]
    spread = (max(accs) - min(accs)) if len(accs) >= 2 else float("nan")
    rows.append({
        "name": name, "model": m.group(1), "sz": int(m.group(2)), "angle": m.group(3),
        "acc": ts.get("accuracy", float("nan")),
        "bal": ts.get("balanced_accuracy", float("nan")),
        "f1": ts.get("macro_f1", float("nan")),
        "auroc": ts.get("macro_auroc", float("nan")),
        "vf1": vs.get("macro_f1", float("nan")),
        "spread": spread, "ep": d.get("epoch", 0),
    })
if legacy:
    print(f"  [제외] 이름 규약 밖 런 {len(legacy)}개: {', '.join(legacy)}\n")
if not rows:
    print("  (아직 완료된 런 없음)")
    raise SystemExit

print(f"{'run':34s} {'acc':>6s} {'balAcc':>7s} {'macroF1':>8s} {'mAUROC':>7s} "
      f"{'valF1':>6s} {'v-t':>6s} {'srcSpread':>10s} {'ep':>4s}")
for r in sorted(rows, key=lambda r: (r["angle"], r["sz"], r["model"])):
    print(f"{r['name']:34s} {r['acc']:6.3f} {r['bal']:7.3f} {r['f1']:8.3f} {r['auroc']:7.3f} "
          f"{r['vf1']:6.3f} {r['vf1']-r['f1']:+6.3f} {r['spread']:10.3f} {r['ep']:4d}")
print("\n  v-t = val macroF1 - test macroF1. 양수로 크면 출처 과적합(val 은 같은 출처의")
print("  새 케이스, test 는 다른 출처의 새 케이스). srcSpread 는 출처별 정확도 최대-최소 폭.")

# 각도 짝지어 비교 — 이게 이 스윕의 질문이다.
pair = {}
for r in rows:
    pair.setdefault((r["model"], r["sz"]), {})[r["angle"]] = r
both = {k: v for k, v in pair.items() if "front" in v and "side" in v}
if both:
    print(f"\n--- front vs side (같은 모델·해상도 짝) : 단독 도메인 macroF1 ---")
    print(f"{'model':12s} {'sz':>5s} {'front':>7s} {'side':>7s} {'both':>7s} {'side-front':>11s}")
    for (mo, sz), v in sorted(both.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        bo = v["both"]["f1"] if "both" in v else float("nan")
        print(f"{mo:12s} {sz:5d} {v['front']['f1']:7.3f} {v['side']['f1']:7.3f} "
              f"{bo:7.3f} {v['side']['f1']-v['front']['f1']:+11.3f}")
    print("\n  side-front 가 크게 양수면 측면(512 피부 접사)이 훨씬 쉽다는 뜻이고, 그러면")
    print("  both 의 점수는 '정면을 잘 풀어서'가 아니라 '측면이 절반을 채워서' 나온 값이다.")
    print("  배포 대상이 정면 얼굴이면 front 단독 점수만 믿을 것.")

# 해상도 짝지어 비교 — 224 -> 512 가 실제로 텍스처를 살리는가.
res = {}
for r in rows:
    res.setdefault((r["model"], r["angle"]), {})[r["sz"]] = r
rboth = {k: v for k, v in res.items() if 224 in v and 512 in v}
if rboth:
    print(f"\n--- 224 vs 512 (같은 모델·각도 짝) : 512 - 224 ---")
    print(f"{'model':12s} {'angle':>6s} {'ΔmacroF1':>9s} {'ΔmAUROC':>9s}")
    for (mo, an), v in sorted(rboth.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"{mo:12s} {an:>6s} {v[512]['f1']-v[224]['f1']:+9.3f} "
              f"{v[512]['auroc']-v[224]['auroc']:+9.3f}")
    n = len(rboth)
    print(f"{'평균':12s} {'':6s} "
          f"{sum(v[512]['f1']-v[224]['f1'] for v in rboth.values())/n:+9.3f} "
          f"{sum(v[512]['auroc']-v[224]['auroc'] for v in rboth.values())/n:+9.3f}")
PYEOF
echo ""
echo "체크포인트: runs/<NAME>/best.pt   클래스별 표·혼동행렬: runs/<NAME>/test_report.json"
echo "주의: 모델 선택은 val 로만. test(제공 Validation)는 고른 뒤 한 번 보는 값이다."
