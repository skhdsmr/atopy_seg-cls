# classification_crop — 분할 결과 기반 crop 중증도 분류 (세 실험)

방법1(마스킹 전체이미지 = `atopy_seg`, 배경 0)이 성능이 나빴던 것에 대한 대안 3종.
공통 원칙: **배경을 지우지 않는다.** 마스크는 "어디를 볼지"만 정하고, crop 은 원본 픽셀
그대로 가져와 병변–정상피부 대비/텍스처를 보존한다.

| exp | 입력 | 모델 | 언제 유리 |
|-----|------|------|-----------|
| `full` | 원본 이미지 전체(마스크 미사용) | `MultiTaskNet` | crop 대조군. 병변 분포/범위 보존 |
| `bbox` | 마스크 union bbox + margin 크롭 1장 | `MultiTaskNet` | 기본. 유효해상도↑, 최소 변경 |
| `mil` | 연결요소별 crop 여러 장(bag) | `MILNet`(attention/max pool) | polygon 다수·병변 분산 |
| `twostream` | 전체(global) + bbox 크롭(local) | `TwoStreamNet` | 분포+텍스처 둘 다, 데이터 충분 시 |

라벨은 이미지 단위 등급(`atopy_face/labels.csv`). 학습 로직(CORN 순서형 loss, Kendall
uncertainty MTL, QWK 선택기준)은 `classification/mobile/train.py` 와 동일 — dataset/model 만 교체.

## 파일

- `precompute_masks.py` — 분할 ckpt로 `atopy_face` 전 이미지 마스크(PNG) 1회 생성 → `atopy_crop_masks/`
- `canet.py` — CANet(TMI'20) cross-disease attention 이식(`--canet`)
- `run_canet_sweep.sh` — CANet 4런 스윕(모델 2 × 입력 2)
- `crop_datasets.py` — 세 데이터셋 + collate
- `model.py` — `MILNet` / `TwoStreamNet` (+ `MultiTaskNet` 재사용)
- `train.py` — `--exp {full,bbox,mil,twostream}` 통합 트레이너
- `dataset.py`, `metrics.py`, `model_base.py` — `classification/` 에서 복사(라벨 파싱·QWK·백본)
- `run.sh` — env 격리 + 마스크 자동 생성 + 세 실험 1회 실행
- `run_sweep.sh` — 방법(exp) × 모델 × 해상도 스윕(+ mil 풀링 / twostream 공유 부가축)

## 실행

```bash
# 전부(마스크 없으면 자동 생성 후 bbox/mil/twostream 순차)
bash run.sh

# 특정 실험만 / 옵션
EXP=mil MIL_POOL=max bash run.sh
EXP=twostream TWOSTREAM_SHARE=1 bash run.sh
MODEL=mnv4s IMGSZ=256 bash run.sh

# 수동
python3 precompute_masks.py            # 1회
python3 train.py --exp bbox --model effb0
python3 train.py --exp mil  --model effb0 --mil_pool attention
python3 train.py --exp twostream --model effb0
```

### 여러 모델 스윕 (run_sweep.sh)

방법(exp) × 모델 × 해상도를 한 번에. mil 은 `--mil_pool`, twostream 은 `--twostream_share`
가 부가 축으로 붙는다(각 방법에 넘길 인자는 `run_sweep.sh` 상단 표 참고).

```bash
bash run_sweep.sh                                # 3방법 × 5모델 × 2해상도 = 30런
EXP_LIST="bbox mil" MODEL="effb0 mnv4s" bash run_sweep.sh
MIL_POOL_LIST="attention max" bash run_sweep.sh  # (mil) 풀링 두 방식 대조
TWOSTREAM_SHARE_LIST="0 1" bash run_sweep.sh     # (twostream) 독립 vs 공유
IMGSZ_LIST=224 FORCE=1 bash run_sweep.sh

# 결과 비교
for d in runs/crop_*/; do echo "$d: $(tail -1 $d/done.txt 2>/dev/null)"; done
```

결과: `runs/crop_{exp}_{model}_r{imgsz}/` (best.pt, done.txt). `done.txt` 의 `test_mean_qwk` 로 비교.

## 주요 하이퍼파라미터

- `--margin 0.15` : bbox/crop 여유(병변 주변 정상피부 = 대비 문맥)
- `--min_area_frac 0.003` / `--max_instances 8` : (mil) 노이즈 요소 제외 / bag 상한
- `--mil_pool attention|max` : 여러 병변 종합(attention) vs 최악 병변 우선(max)
- `--twostream_share` : 두 스트림 백본 공유(소규모 데이터 과적합↓)

## 2-step IBB 샘플링 (`sampler.py`, `--ibb`)

Walecki+ 2017 "Deep Structured Learning for Facial AU Intensity Estimation" 의 Iterative
Balanced Batch(Alg.1)를 이 데이터에 맞춰 축소 적용한 것. **B안 = 모델 파라미터를 늘리지
않고 샘플러만 교체**하는 버전.

논문은 3-step(subject / AU-level / AU-cooccurrence)이지만 **subject 축은 제거**했다.
`dataset_all_final` 은 환자 1명당 이미지가 1.06장(고유 ID 1691 / 이미지 1800)이고 split 이
이미 ID 단위로 분리돼 있어(train∩val=0, train∩test=0) subject 균등 배치가 uniform 과
사실상 같기 때문. 대신 앞 `--ibb_warmup` 에폭을 자연분포로 돌려 백본 W 를 워밍업한다.

| step | 배치 구성 | 역전파 대상 | 논문 대응 |
|------|-----------|-------------|-----------|
| Step 1 `lv:<task>` | 그 태스크의 **등급**이 평탄해지도록 샘플링 | 해당 태스크 head 만 | `∀q: φ^q` |
| Step 2 `co:<a>-<b>` | 태스크 쌍의 **등급조합 셀**이 평탄해지도록 샘플링 | 그 쌍의 두 태스크만 | `∀(rs): θ^rs` |

두 스텝은 배치 단위로 인터리브된다(블록 분리 시 BN/optimizer 가 흔들리는 것 방지).

### 왜 '완전균등'이 아닌가
train 의 `severity=Clear` 는 n=2 라 완전균등이면 에폭당 140배 반복 → 2장 암기가 된다.
그래서 빈도의 `--ibb_alpha` 승 역가중(기본 0.5=sqrt) + 반복배수 상한 `--ibb_cap`(기본 4x)
을 쓴다. Step 2 도 최소 셀이 n=1 인 쌍이 있어(최대 100배 반복) `--ibb_min_cell` 미만 셀은
제외한다 — 그 샘플들은 Step 1 과 워밍업 에폭에서만 본다.

### 엣지 선별
증상 4종 등급의 **전체 조인트는 쓸 수 없다** — 129개 조합 / 이론상 256개, 싱글톤 30개,
5회 이하 75개. 논문도 전체 조인트가 아니라 pairwise edge 단위이므로 쌍으로 간다.
쌍은 Cramér's V(논문의 θ 가지치기 대응)로 고르고, `--ibb_edge_v` 기본 0.25 에서 4개가 남는다:

```
채택: seve-eryt .47 | seve-papu .33 | eryt-papu .32 | seve-exco .26
제외: eryt-exco .19 | papu-lich .18 | papu-exco .16 | eryt-lich .09 | exco-lich .09 | seve-lich .07
```

분포/엣지/반복배수 진단표는 학습 없이 확인할 수 있다:
```bash
python3 sampler.py --data ../dataset_all_final/images/labels.csv
```

### 실행
```bash
IBB=1 bash run.sh                       # runs/crop_<exp>_<model>_r<sz>_all_ibb
IBB_LIST="0 1" bash run_sweep.sh        # IBB on/off 대조군까지 자동 생성(런 2배)
python3 train.py --exp bbox --model effb0 --ibb --ibb_alpha_end 0.0
```

### 주요 노브
- `--ibb_alpha 0.5` / `--ibb_alpha_end` : 균등화 강도. `alpha_end 0.0` 이면 학습 말미에
  자연분포로 선형 복귀 — **선택지표 QWK 는 val/test 자연분포에서 재므로, 균등 리샘플링이
  예측 주변분포를 밀어 QWK 를 떨어뜨릴 수 있다. 그 완화용.**
- `--ibb_cap 4.0` : 샘플별 기대 반복배수 상한(희소 등급 암기 방지).
- `--ibb_edge_v 0.25` / `--ibb_min_cell 5` : Step 2 엣지·셀 선별.
- `--ibb_step2_ratio 0.5` : Step2/자연분포에폭 배치 비율. 기본 설정에서 에폭당 60배치(자연 44).
- `--ibb_step1_mult 1.0` : Step1 을 태스크 5개가 나눠 쓰므로 기본값에선 **head 하나당
  업데이트가 자연분포 에폭의 1/5**(공유 trunk 는 매 배치 갱신되므로 영향은 제한적).
  head 수렴이 느리면 5.0(비용도 5배).
- `--ibb_warmup 3` : 앞 N 에폭 자연분포(논문 Step1 제거 보완).

### 읽을 때 주의
- IBB 에폭의 `tr=` 는 배치마다 태스크 1~2개 loss 만 합산하므로 **워밍업/베이스라인의 tr 과
  직접 비교하면 안 된다.** 비교는 `va=` 와 QWK 로.
- 로그의 `S1:` / `S2:` 줄이 스텝별 loss 평균이다.
- **B안의 한계**: 이 모델에는 논문의 코퓰러 pairwise 파라미터 θ 가 없다. Step 2 는 θ 전용
  갱신이 아니라 조인트 층화 샘플링으로 동작한다 — 주변분포 지름길 학습을 억제하는 정규화
  효과를 노리는 것이지 논문 Step 3 그 자체는 아니다. 논문에 충실하려면 CORN 임계값 위에
  pairwise 항을 얹는 A안이 필요하다.
- 참고: `--class_weight` 는 기본값 `--loss corn` 경로에서 무효다(CE weight 를 CORN 이 쓰지
  않음). 즉 IBB 가 이 파이프라인의 **첫 불균형 대책**이고, 베이스라인은 `--ibb` 없이 그대로 두면 된다.

## A안: 코퓰러 pairwise 항 (`copula.py`, `--pairwise`)

논문 저자 구현 **RWalecki/copula_ordinal_regression** 을 대조해 포팅했다.
대조 파일: `copulas.py`(frank) / `statistics.py`(log_prob·node_potn·edge_potn) /
`BASE.py`(_cdf·_pdf) / `COR.py`(_loss·predict·_init_para).

```bash
python3 train.py --exp bbox --model effb0 --pairwise                     # CCNN
python3 train.py --exp bbox --model effb0 --pairwise --ibb \
        --ibb_step2_ratio 1.0                                            # CCNN-IT (권장)
```

| 모드 | 구성 | 논문 |
|------|------|------|
| (없음) | CORN head 5개 독립 | OCNN |
| `--ibb` | B안: 샘플러만 2-step | OCNN-IT |
| `--pairwise` | 코퓰러 + composite likelihood joint 학습 | CCNN |
| `--pairwise --ibb` | Step2 가 주변분포 detach 후 theta 만 갱신 | CCNN-IT |

### 저자 구현에서 그대로 가져온 것
- **theta 파라미터화**: `theta = (sigmoid(raw) - 0.5) * cut`, `cut=25` -> `theta in (-12.5, +12.5)`,
  raw 초기값 `0.01`. (theta=0 은 독립 특이점이라 저자도 0 을 피한다)
- **노드 포텐셜** `-log P(y^q=l)`, **엣지 포텐셜** `-log[C(u1,v1)+C(u0,v0)-C(u0,v1)-C(u1,v0)]`
- **손실 결합** `w_nodes * mean(노드) + (1-w_nodes) * mean(엣지)`.
  저자 기본 `w_nodes=0.1` — 즉 **pairwise 에 0.9** 를 준다(`--pairwise_w_nodes`).
- **추론 가중**: 노드/엣지에 같은 `w_nodes` 를 적용해 MAP (저자 `predict` 의 AD3 호출과 동일)
- **log_prob 안정화** `realmin=1e-20` 클리핑 + NaN/inf 처리
- **shared_copula** (`--pairwise_shared`): 1=엣지당 theta 스칼라, 0=엣지당 (Kr,Ks) 행렬
- **indep 코퓰러** (`--pairwise_copula indep`) — 위생검사용

### 저자와 의도적으로 다르게 한 것
1. **`frank()` 를 수치 안정형으로 재작성.** 저자 원식은 float32 에서 |theta| 가 cut/2 에
   가까우면 `1+U*V/D` 가 뭉개진다(경계조건 오차 4.9e-4; 재작성형 6.0e-8). 원식은
   `frank_ref()` 로 남겨 두었고, 테스트에서 float64 기준 **3.6e-12 까지 동치**를 확인한다.
2. **주변 CDF 를 CORN head 에서 생성**(저자는 선형 threshold 모델 `BASE._cdf`).
   CORN 은 sigmoid 누적곱이라 F 단조증가가 구조적으로 보장돼 순서형 제약을 그대로 만족한다.
   -> 논문 3.1절 ordinal unary 를 CORN 이 담당. **CNN 연결부는 이 저장소 고유.**
3. **결합 추론을 AD3 근사 대신 전수 열거로.** 저자/논문은 AU 10+ 라 `pystruct.inference_ad3`
   를 쓰지만 여기는 `5*4*4*4*4=1280` 조합뿐이라 **정확한 MAP** 을 낸다.
4. **IBB 는 이 저장소 구현**(`sampler.py`). 저자 저장소에는 IBB 가 없다(논문에만 있음).
   Step2 가 주변분포를 detach 하고 그 엣지의 theta 만 갱신한다 = 논문 Alg.1 Step3.

### 함정 (전부 '돌아가지만 결과가 무의미해지는' 종류)
1. **theta_raw=0 초기화 시 파라미터가 죽음** — theta=0 은 독립 특이점이라 gradient 가 0.
   저자도 `0.01` 로 피한다.
2. **주 lr 로는 theta 가 안 움직여 A안이 B안으로 붕괴** — `--pairwise_theta_lr`(기본 0.02)로
   분리. 라벨의 경험적 Kendall tau 역산 상한은 seve-eryt +8.26 / seve-papu +5.11 /
   eryt-papu +5.08 / seve-exco +3.68.
3. **theta 추정 분산** — 엣지당 Step2 배치가 적으면 theta 가 요동친다.
   `--ibb_step2_ratio 1.0` 권장(0.5 면 경고를 띄운다).

### theta 읽는 법
theta 는 **이미지 특징으로 조건화한 뒤 남는 잔여 의존성**이다(논문 식(6) 아래). 위 역산값은
상한이지 목표가 아니다 — CNN 이 공존 패턴을 설명할수록 theta 는 낮게 수렴하는 게 정상.


## CANet 이식 (`canet.py`, `--canet`)

**xmengli/CANet**(Li et al., *CANet: Cross-disease Attention Network for Joint DR and DME
Grading*, TMI 2020) 의 `crossCBAM` 분기를 이식했다. 대조 파일: `models/cbam.py`(CBAM),
`models/resnet50.py:339-392`(forward), `baseline.py:377-382,98`(손실·λ).

태스크 의존성을 **특징 층**에서 다룬다 — 데이터 층(IBB)·라벨 층(코퓰러 θ)과 축이 다르다.

```bash
python3 train.py --exp full --model effb0   --canet     # 원본 이미지
python3 train.py --exp bbox --model pvtv2b0 --canet     # bbox 크롭
bash run_canet_sweep.sh                                 # 4런 (모델 2 × 입력 2)
```

### 구조 (저자 forward 와 1:1)

```
F  = backbone(x)                        # (B,C,H,W) — global_pool="" 로 풀링 전 특징맵
Aq = CBAM(C)(F)                         # ① 태스크별 어텐션(채널+공간)  branch_bam1/2
zq = neck_q(GAP(Aq))                    # ② 태스크별 임베딩            classifier_dep1/2
     head_spec_q(zq)                    #    specific 헤드(보조, λ=0.25) classifier_specific_*
aq = CBAM(D, no_spatial)(zq)            # ③ 교차 게이팅(채널전용)       branch_bam3/4
zq'= zq + mean_{r∈nbr(q)} a_r           #    교차 주입                  choice="both"
     head_joint_q(zq')                  #    joint 헤드(주 예측)        classifier1/2
loss_q = CORN(joint) + λ·CORN(specific) #    저자 loss1+loss2+λ(loss3+loss4)
```

### 저자와 다르게 한 것
1. **백본** ResNet-50 → timm(effb0/pvtv2b0). CBAM 채널수를 백본 출력에서 받는다.
2. **헤드** CE softmax → CORN 순서형. 선택지표가 QWK 라 순서형이어야 하고, CANet 의 기여는
   헤드가 아니라 특징 융합이므로 교체해도 잃는 게 없다.
3. **neck** 저자 `classifier_dep` 은 활성 없는 Linear 1개. 여기선 대조군 `MultiTaskNet` 과
   같은 Linear+BN+GELU+Dropout — 대조군과의 차이를 어텐션으로만 국한시키기 위함.
4. **2 → 5 태스크.** 교차 엣지가 방향 포함 20개로 늘어 1400장에 과하다. IBB 와 같은
   Cramér's V 기준(`--canet_edge_v 0.25`)으로 4엣지만 연결한다:
   `seve<-eryt+papu+exco, eryt<-seve+papu, papu<-seve+eryt, exco<-seve`.
   lichenification 은 모든 쌍이 V≤.18 이라 고립 — 두 헤드가 같은 z 를 보고, λ 보조헤드의
   정규화 효과만 남는다.
5. **교차 CBAM 은 source 태스크당 1개**(엣지쌍당이 아님) — 저자가 `branch_bam3(x1)` 의
   출력을 x2 로 보내는 구조 그대로.
6. **다중 이웃 합성은 `mean`**(`--canet_fuse`). 저자는 이웃이 항상 1개라 단순 덧셈이고,
   이웃이 1개면 mean 과 sum 이 동일하다. severity 는 이웃 3개라 sum 이면 스케일이 커진다.

### 주요 노브
- `--canet_lambda 0.25` : specific 보조헤드 가중(저자 `lambda_value` 기본값)
- `--canet_edge_v 0.25` : 교차 엣지 선별 하한. 0 이면 전체 10쌍(과적합 주의)
- `--canet_fuse mean|sum` / `--canet_reduction 16`(저자 `reduction_ratio`)

### 파라미터 비용 (r512, embed_dim 512)
| 모델 | baseline | +CANet | 증가 |
|------|----------|--------|------|
| effb0 (feat 1280) | 4.67M | 8.47M | +3.80M |
| pvtv2b0 (feat 256) | 3.55M | 4.27M | +0.71M |

증가분의 대부분은 **태스크별 neck 5개**(저자 `classifier_dep` 가 태스크당 1개인 구조를 따름)
라 백본 특징차원에 비례한다. effb0 는 백본(4.67M)만큼 파라미터가 붙는 셈이라 1400장에서
과적합 위험이 pvtv2b0 보다 크다 — 결과 해석 시 감안할 것.

### 함정
- `--canet` 은 단일 이미지 입력 전용(`--exp full|bbox`). mil/twostream 은 막아 둔다.
- `--canet` + `--pairwise` 는 **금지**(에러). CCNN 경로는 unary 를 `node_loss(cdfs)` 로
  계산해 specific 헤드를 통째로 무시하므로, 조합하면 λ 항이 조용히 사라진다.
- 배치 크기는 r512 에서 8(peak 2.0GiB/1.2GiB) — 대조군과 같은 값이라 비교 가능하다.

### 대조군
`run_canet_sweep.sh` 는 `CANET=0 SUFFIX=_base` 로 어텐션 없는 같은 조건 4런을 만든다.
bbox 쪽은 기존 결과가 이미 있다 — `crop_bbox_effb0_r512_b32`(0.4995),
`crop_bbox_pvtv2b0_r512_b32`(0.5297). full 쪽은 대조군이 없으므로 새로 만들어야 한다.
