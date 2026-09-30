# classification_disease — 질환 6-way 분류

안면부 피부질환 데이터로 **어떤 질환인가**만 맞히는 분류기.
건선 / 아토피 / 여드름 / 정상 / 주사 / 지루 6클래스.

중증도(IGA/EASI)나 징후 태그는 다루지 않는다 — 그건 `classification/`,
`classification_crop/`, `classification_topk/` 쪽 문제다.

## 빠른 시작

```bash
bash run.sh                    # dataset 생성 + 학습 (effb0, r224, 정면+측면)
ANGLE=front bash run.sh        # 정면만
ANGLE=side  bash run.sh        # 측면만
LINK=1 bash run.sh             # 이미지 복사 대신 심볼릭 링크(2GB+ 절약)
```

## 평가 — 학습 끝난 체크포인트 다시 재기

```bash
bash evaluate.sh --list                            # 런 목록 + 저장된 test 성적
bash evaluate.sh --run dis_effb0_r512_both         # test / best.pt 상세 리포트
bash evaluate.sh --run dis_effb0_r512_both --png   # 혼동행렬 PNG 도 저장
bash evaluate.sh --run dis_effb0_r512_both --split val --ckpt last
bash evaluate.sh --run dis_mnv3l_r512_both dis_effb0_r512_both   # 나란히 비교
bash evaluate.sh --run all                         # 전 런 비교표(macro-F1 순)
```

해상도·각도·embed_dim 은 체크포인트에 박힌 학습 당시 값을 그대로 재현하므로 지정할
필요가 없다. `--angle front|side` 로 덮어쓰면 한 모델을 각도별로 쪼개 잴 수 있다.
결과는 `runs/<런>/eval_<split>_<ckpt>.json` (+`--png` 시 `confusion_*.png`)에 저장된다.

## 구조

| 파일 | 역할 |
|---|---|
| `make_dataset_disease.py` | `skin_dataset` JSON -> `dataset_disease/{train,val,test}/*.png` + `labels.csv` |
| `dataset.py` | split 폴더 glob + stem 으로 CSV 조회 -> `(image, disease_idx)` |
| `model.py` | timm 백본 + Neck + 6-way 헤드 (`DiseaseNet`) |
| `metrics.py` | 혼동행렬 / 클래스별 P·R·F1·AUROC / **출처별 정확도 분해** |
| `train.py` | CE 학습, val 로 best 선택, test 1회 보고 |
| `evaluate.py` | 저장된 `runs/*/best.pt` 재평가 — acc/F1/혼동행렬, 런 선택·비교 |
| `run.sh` / `evaluate.sh` | 학습 / 평가 실행 래퍼 (`.venv-train` 격리) |

`labels.csv` 헤더: `split, stem, disease, angle, source, subject`
(`atopy_crop_masks`, `dataset_topk` 와 동일한 자립형 규약)

## split 설계 — 왜 이렇게 나눴나

데이터를 조사해서 나온 세 가지 사실이 설계를 결정했다.

**1. 제공된 Validation 이 유일한 진짜 held-out 이다.**
Training 과 피험자 겹침이 0이고 출처(prefix) 분포도 다르다 — Training 정면의
H0(1,817장)는 Validation 에 한 장도 없고, Validation 의 H10/H11 은 Training 에 없다.
'새 환자 + 새 출처' 조건을 만족하므로 **test 로 봉인**하고 최종 1회만 본다.
튜닝에 쓰면 이 성질이 사라진다.

**2. 한 케이스 키가 여러 장을 갖고 있다.**
`H1_500531_P14_L0` 에서 앞 두 필드(`H1_500531`)가 케이스 키다. 건선 정면은 800장에
498키뿐이다. 파일 단위로 랜덤 분할하면 같은 케이스가 train/val 양쪽에 들어가 val 이
부풀려진다. 그래서 val 은 **키 단위**로, (질환, 각도, 출처)별 층화해서 뗀다.

주의 — 이 키는 **'같은 얼굴'이 아니다.** 합성 데이터라 키를 공유하는 파일도 서로 다른
얼굴로 생성돼 있다(`H2_21172_P5` 와 `P2`, `H2_18405_P2_L0` 와 `L1` 을 열어보면 머리색·
얼굴형·배경이 전부 다르다. Validation 1,200장은 전부 픽셀 고유라 완전중복도 없다).
공유하는 건 병변 케이스 쪽이다. 그래도 묶어서 나누는 이유는 얼굴 기억 방지가 아니라
같은 원본 케이스가 갈라지면 val 이 부풀려지기 때문이다.

결과적으로 val 은 *같은 출처의 새 케이스*를, test 는 *다른 출처의 새 케이스*를 잰다.
**둘의 격차가 곧 출처 과적합의 크기**이고, train.py 가 이 격차를 매번 찍는다.

**3. 질환을 가로지르는 키 겹침이 하나 있다 — 무해함(확인 완료).**
`H2_2820` 이 Training 에선 아토피, Validation 에선 건선이다. 두 PNG 를 직접 열어
확인한 결과 **서로 다른 사람**이다(헤어스타일·배경·옷·얼굴형 전부 다름). 라벨 오류도
아니고 한 사람의 두 질환도 아니며, 키가 우연히 겹친 별개 생성 이미지다. 즉 이 겹침은
누수가 아니다.

빌더는 그래도 기본값으로 train 쪽 1건을 버린다 — 10,800건 중 1건이라 버리는 비용이
0에 가깝고, 겹침을 0으로 유지하면 stats.txt 의 누수 점검이 잡음 없이 읽히기 때문이다.
되살리려면 `--keep_test_leaks` 를 쓰면 된다(무해함이 확인됐으므로 안전하다).

## 읽을 때 조심할 것 — accuracy 를 믿지 말 것

클래스가 균형(각 900장)이라 accuracy 가 그럴듯해 보이기 쉽지만, 이 데이터엔 질환과
무관한 지름길이 두 개 있다.

**출처(prefix)가 질환을 거의 알려준다.** test 기준 주사는 100% H1, 여드름은 99% H2,
정상은 전부 H9/H10/H11 이다. 모델이 촬영 조건만 보고도 점수를 올릴 수 있다.
그래서 `metrics.group_accuracy` 로 출처별 정확도를 쪼개 찍는다 — 편차가 크면 의심할 것.

**정면과 측면은 사실상 다른 도메인이다.**

| | 정면 | 측면 |
|---|---|---|
| ID prefix | H0~H9 | Z4 단독 (100%) |
| 해상도 | 1024×1024 | 512×512 |
| 메타데이터 | gender/age 실제값 | 전부 N/A |
| 화면 | 배경·옷 포함 얼굴 전체 | 얼굴 없는 피부 접사 |

정면/측면의 피험자 겹침도 0 — 같은 사람의 두 각도가 아니라 아예 다른 사람들이다.
Z4 는 ID 가 클래스별 연속 블록(건선 69001~72561, 아토피 74007~78541 …)이라 클래스별
일괄 생성된 것으로 보이며, 생성 배치 아티팩트가 클래스와 상관될 수 있다.
`--angle both` 로 합치기 전에 `front`/`side` 단독 성능을 따로 재고, **측면만 유독
높으면** 질환이 아니라 도메인 단서를 본 것으로 의심할 것.

## 다음 단계 후보

- **정면 얼굴 crop**: 배경·옷·헤어가 출처 단서로 샌다.
  `classification_crop/precompute_masks.py` / `atopy_crop_masks` 자산을 재활용 가능.
- **출처 단위(leave-source-out) 평가**: 실배포는 새 병원/기기다. 다만 주사는 H0/H1
  뿐이라 완전한 leave-source-out 은 불가능 — 가능한 클래스에서만 절충.
