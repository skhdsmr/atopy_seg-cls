# cls_atopy_mbn — 아토피 5축 MaMNet (그룹 분기 + CA/SA 분리 + ASPP)

MaMNet(Xing et al., *A Multi-Task Learning and Multi-Branch Network for DR and DME Joint
Grading*, Appl. Sci. 2024)을 **IGA + 증상 4종(홍반·구진·찰상·태선화)** 5축 아토피 등급
분류로 이식한 실험. [`cls_idrid`](../cls_idrid)(논문 그대로의 IDRiD 2분기 재현)가 출발점이고,
[`cls_mbn`](../cls_mbn)(5축을 평평하게 편 일반화)의 실패를 고치는 게 목적이다.

## cls_mbn 과 무엇이 다른가

| | cls_mbn | cls_atopy_mbn |
|---|---|---|
| 백본/분기 | 사전학습 backbone + dwsep 분기 | **VGG16-bn MbN**(논문 Table 1 1:1, cls_idrid 와 동일) |
| 분기 단위 | 태스크 5개 대칭 | **그룹 3개**(IGA / 급성3 / 태선화) — `--groups` |
| SFEN | 태스크별 통째로 | **CA 태스크별 + SA 그룹공유** — `--sfen` |
| CFEN | 5태스크 공유 softmax attention | **그룹 쌍마다 독립 CA 게이트**(논문 원형, 6쌍) |
| ASPP | 미포함 | **IGA 분기에 적용, on/off 가 스윕 축** |
| 결과 | attention 이 균등분포로 붕괴(0.20~0.31, 균등 0.25) | softmax 가 없어 붕괴 모드가 구조적으로 없음 |

## 설계 근거 — 실데이터 라벨 상관

`/home/work/Data/atopy.csv` 9,150행의 스피어만 상관:

| | iga | ery | pap | exc | lic |
|---|---|---|---|---|---|
| iga | — | 0.72 | 0.67 | 0.59 | **0.36** |
| ery | 0.72 | — | 0.57 | 0.44 | **0.21** |
| pap | 0.67 | 0.57 | — | 0.48 | **0.29** |
| exc | 0.59 | 0.44 | 0.48 | — | **0.38** |

태선화만 모든 축에서 떨어져 있고(IGA 와도 0.36, 나머지 셋은 0.59~0.72), 가장 가까운 짝이
찰상(0.38)이다 — 긁어서 만성화되는 임상 경로와 일치한다. **홍반·구진·찰상 = 급성 염증 /
태선화 = 만성 재형성**이라는 그룹이 라벨에 이미 있으므로, 학습으로 찾게 두지 말고
구조에 박아 넣는다(cls_mbn 은 찾게 뒀다가 못 찾았다).

> 기대효과는 "태선화가 좋아진다"가 아니다. cls_mbn 스윕의 축별 QWK 는
> `seve=.663 eryt=.555 papu=.327 exco=.546 lich=.518` 로 제일 망가진 축은 구진이다.
> 가설은 **상관 없는 축이 공유 트렁크에서 빠져 나머지 셋의 그래디언트 충돌이 줄어든다**
> 쪽이고, 그래서 봐야 할 숫자는 구진·홍반 QWK 다.

## 구조

```
                    ┌─ branch_iga   ─ S + C<-acute + C<-lich + G(ASPP) ─ head(IGA)
x ─ Branch1/2(VGG) ─┼─ branch_acute ─ F + S + C<-iga + C<-lich ─┬─ head(홍반)
                    │                                          ├─ head(구진)
                    │                                          └─ head(찰상)
                    └─ branch_lich  ─ F + S + C<-acute + C<-iga ─ head(태선화)
```

- **F** = MbN 원특징(게이트 안 거친 skip). CA/SA 는 전부 sigmoid 라 **억제만** 할 수 있어서,
  한 번 눌린 채널·위치를 되살릴 통로가 필요하다. IGA 분기는 논문 `D_DR` 처럼 F 자리를
  ASPP 전역특징으로 갈아끼운다(ASPP 의 rate=1 분기가 F 를 거의 그대로 통과시킨다).
- **CA 태스크별 / SA 그룹공유**: SA 는 출력이 1채널 H×W 라 "어디를"만 정한다. 홍반·구진·
  찰상은 같은 병변 영역에 같이 나타나므로 공유해도 되고, "그 영역에서 무슨 특징을"(색/
  융기/선상결손)은 채널 선택이라 CA 가 맡는다. 비용도 이쪽이 유리하다 — SFEN 파라미터의
  95%(2.36M/2.50M)가 SA 의 비대칭 conv 이고 CA 는 131K 다.
- **CFEN 쌍별 독립**: 논문 CFEN 은 softmax 없이 `C_i = CA(F_i) ⊗ F_j` 뿐이다. 그룹이 3개면
  방향 포함 6쌍이라 이 원형을 그대로 쓸 수 있다(5태스크 20쌍은 무리였다). 쌍끼리 경쟁하지
  않으므로 정보 없는 방향은 그 쌍만 죽는다.

### 논문에서 의도적으로 벗어난 곳

1. `--sfen share` 의 **SA 입력이 F_g(원본)**다. 논문 Eq(4)는 CA 통과 후 F' 를 받는데,
   CA 가 태스크별이면 F' 도 태스크별이 되어 SA 를 그룹에서 공유할 수 없다. 논문 그대로를
   보려면 `--sfen task`.
2. **선택기준/early stop = `0.5*IGA QWK + 0.5*증상4 QWK 평균`** (논문은 Joint Ac).
   5축 동시정답률은 값이 너무 낮아 선택 신호가 노이즈다(`joint_ac` 는 참고로만 출력).
   논문 4.1.3 은 *test* loss 로 조기종료하는데 그러면 test 가 샌다 — 여기선 val 만 쓴다.
3. **warmup 3ep**(논문 10ep). 논문 학습셋은 300장, 여기는 1,400~9,150장이다.
4. **손실**: CORN 순서형 + Kendall uncertainty 가 기본. 논문의 α·β 그리드는 2태스크라
   가능했던 것이고 5축이면 차원이 5개다. `--mtl fixed --iga_weight 0.5` 로 고정가중 가능.

## 데이터

```
<data>/labels.csv
<data>/train/*.png|jpg   <data>/val/*   <data>/test/*
```

`labels.csv` 컬럼: `stem`, `severity`(또는 `iga`/`iga_grade`), `erythema`, `papulation`,
`excoriation`, `lichenification`. 기본 경로는 [`../dataset_all_final/images`](../dataset_all_final)(1400/200/200).

**라벨 공간은 CSV 에서 유도한다.** 두 출처의 표기가 다르기 때문이다:

- 합성(dataset_all_final): `Clear`/`Almost Clear`/`Mild`/`Moderate`/`Severe` 문자열 → 표준 등급표(IGA 5 / 증상 4)
- 실데이터(`/home/work/Data/atopy.csv`): 정수. **IGA 가 1~4 이고 0 이 한 건도 없다**(9,150행 확인)
  → 관측된 값만 오름차순으로 0..n-1 에 매핑 = **IGA 4등급**

영원히 0장인 로짓이 생기거나 QWK 가중행렬이 왜곡되는 걸 막기 위한 처리다. 어느 쪽으로
해석했는지는 학습 시작 시 매핑표와 등급별 건수로 찍힌다.

## 실행

```bash
bash run.sh                          # 기본: groups=sep sfen=share cfen=all ASPP=1
ASPP=0 bash run.sh                   # IGA 분기 ASPP 끄기
GROUP=merged bash run.sh             # branch5 없는 논문 2분기 대조군
SFEN=task bash run.sh                # SFEN 통째로 태스크별(논문 그대로)
CFEN=none bash run.sh                # cross-feature 전부 끄기
CFEN='lich<-acute,acute<-lich' bash run.sh    # 특정 쌍만
DATA=/path/to/images bash run.sh     # 다른 데이터셋
```

### 스윕

```bash
bash run_sweep.sh                    # MODE=main: groups x aspp = 6런
MODE=aspp    bash run_sweep.sh       # ASPP on/off             2런
MODE=groups  bash run_sweep.sh       # sep/merged/flat         3런
MODE=sfen    bash run_sweep.sh       # share/task/group        3런
MODE=cfen    bash run_sweep.sh       # all/none/태선화쌍만     3런
MODE=igaskip bash run_sweep.sh       # auto/f/none             3런
MODE=imgsz   bash run_sweep.sh       # 224/448/512             3런
DRY=1 MODE=main bash run_sweep.sh    # 무엇이 돌지만 확인
MODE=aspp GROUP=merged bash run_sweep.sh   # 다른 축 고정한 채 ASPP 만
```

### 해상도 축

`--imgsz` 는 **16의 배수만** 된다. MbN side branch 가 stride8 특징을 `pool(2,2)` 로 절반
냈다가 `Upsample x2` 로 되돌리므로 H/8 이 홀수면 Branch1 출력과 1픽셀 어긋난다
(300px -> 37 vs 36). train.py 가 시작 전에 막는다.

A100 batch16 실측: `224px 3.4GiB/84ms · 384px 8.5GiB/196ms · 448px 11.3GiB/264ms ·
512px 14.6GiB/338ms` (fwd+bwd 1스텝). VRAM 이 모자라면 `BATCH=8`.

기본값 224 는 논문 Table 2 값이자 **이 저장소의 실측**이기도 하다 — `cls_mbn` 스윕에서
r512 가 두 백본 모두 r224 보다 나빴다(effb0 .5449->.5205, pvtv2b0 .5810->.5747, best epoch
은 오히려 뒤로 밀림). 1,800장이라 해상도를 올린 만큼 과적합이 빨라진 것으로 보이는데,
구진·찰상 같은 국소 질감이 224 에서 살아남는지는 별개 문제다. 실데이터(9,150장, 원본
1024x1024)에서는 전제가 둘 다 바뀌므로 **다시 재야 하는 축**이고, 그래서 스윕에 넣었다.

`MODE=aspp` 를 볼 때 주의: `--iga_skip auto`(기본)는 ASPP 를 끄면 IGA 분기에 F 를 돌려준다.
게이트 안 거친 자기 경로를 통째로 잃는 걸 막기 위해서인데, 그만큼 "ASPP 효과"와 "F 유무"가
섞인다. 순수하게 ASPP 만 재려면 `MODE=igaskip` 으로 `f`(항상 F 포함) 또는 `none`(항상 제외)
한쪽에 고정한 뒤 ASPP 를 흔들 것.

## 학습 후 반드시 볼 것 — CFEN 쌍별 게이트

```
[CFEN 진단] 쌍별 채널게이트(test 평균)
  iga<-acute       mean=0.549 std=0.1567
  lich<-acute      mean=0.495 std=0.0930
  ...
```

std 가 0 에 가까운 방향은 게이트가 채널을 못 가려 cross-feature 가 상대 특징맵의
**상수배**일 뿐이라는 뜻이다(cls_mbn 의 균등 붕괴와 같은 실패, 여기선 쌍마다 독립이라
쌍별로 갈린다). 라벨 상관대로라면 `lich<-acute`(0.38)는 살고 `acute<-lich`(0.21)는 죽어야
한다 — **그렇게 나오면 그룹 가정이 데이터와 맞는다는 증거**이고, 아니면 가정을 재검토할
신호다. `best.pt["gate_stats"]`, `test_report.json` 에도 남는다.

## 파일 · 파라미터 규모

- `model.py` — `AtopyMaMNet`(`MbN` + `ChannelAttention`/`SpatialAttention` + `CFEN` + `ASPP` + `GradingHead`)
- `dataset.py` — labels.csv 라벨공간 유도(문자열/정수), split 폴더 데이터셋, 그룹 프리셋
- `metrics.py` — 5축 QWK/acc/±1/MAE/혼동행렬 + joint_ac(Eq 13 확장) + 선택점수
- `train.py` — CORN/CE, Kendall uncertainty, warmup 동결→해동, 선택점수 early stop, 게이트 진단
- `run.sh` / `run_sweep.sh`

| `--groups` | 분기 | ASPP on | ASPP off |
|---|---|---|---|
| `sep`(기본) | 3 | 65.6M | 62.8M |
| `merged` | 2 | 48.4M | 45.6M |
| `flat` | 5 | 100.7M | 97.9M |

`merged`+ASPP off = 45.6M 으로 `cls_idrid`(45.2M)와 사실상 같다 — 같은 구조를 5축으로만
편 것이기 때문이고, 그래서 이 조합이 자연스러운 하한 대조군이다.

## 실데이터로 옮기기 전에 남은 것

지금 이 폴더는 **합성 데이터(dataset_all_final 1,800장)로만** 돌려봤다. 실데이터
(`/home/work/Data/atopy.csv`, 9,150행)에는 **이미지 키가 없다** — 컬럼이 5축 등급뿐이고
`stem`/`subject`/`split` 이 없어서 이미지와 매칭할 방법이 없다. 필요한 것:

- `stem` (이미지 파일명)
- `subject` (환자 키 — 한 환자 여러 장이면 split 누수 방지에 필수)
- `source` (출처/기관 — 지름길 학습 진단축)

키가 붙으면 `DATA=` 만 바꿔서 그대로 돌아간다(라벨 정수 표기는 dataset.py 가 이미 처리).
9,150장은 합성 1,400장의 6.5배라 `--warmup_epochs`, `--patience`, `--batch` 는 그때 다시
잡는 게 맞다.
