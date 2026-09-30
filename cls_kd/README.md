# cls_kd/ — BAM(Born-Again Multi-Task) 지식증류로 아토피 중증도 분류

Clark et al., *"BAM! Born-Again Multi-Task Networks for Natural Language
Understanding"* (ACL 2019, [arXiv:1907.04829](https://arxiv.org/abs/1907.04829))의
Single→Multi 지식증류 + teacher annealing을 EASI 5축 순서형(CORN) 아토피
중증도 분류에 옮긴 것. `/home/work/Code/cls_kd`(오프라인 학습 서버용 사본)를
이 저장소로 들여오면서, 백본 로딩만 이 저장소의 관례에 맞췄다: 여기는 인터넷이
되므로 [`classification_crop`](../classification_crop)/[`cls_atopy_mbn`](../cls_atopy_mbn)과
같이 timm 을 바로 쓴다(`model.py::build_backbone` = `timm.create_model(...,
pretrained=True)`) — 로컬 `pretrained/<name>.pth` 반입이나 timm 소스 벤더링
(`backbones/`)이 필요 없다. `dataset.py`/`labels.py`/`corn.py`/`metrics.py`/
`kd_common.py`는 데이터·손실·지표 쪽 로직이라 그대로 옮겼다.

## 0. 실험 설계 — 조건 A~H

| 조건 | 구성 | 확인하려는 것 | 만드는 스크립트 |
|---|---|---|---|
| A | single-task 모델 5개 | teacher 성능 = 상한 기준 | `train_single.py` (fold 없이) |
| B | 일반 multi-task 모델 1개 | multi-task 성능 저하가 실제로 있는지 | `train_multitask.py` |
| C | Single→Multi 증류 (annealing 없음), teacher=OOF | 증류만의 효과 | `train_distill.py --anneal none` |
| D | Single→Multi 증류 + teacher annealing, teacher=OOF | BAM 방식의 최종 효과 | `train_distill.py --anneal linear` |
| E | D + 소견-IGA 일관성 손실(coherence), teacher=OOF | 소견 간 일관성 제약의 추가 효과 | `train_distill.py --anneal linear --coherence` |
| F | C와 동일, teacher=조건 A 직접(in-sample) | OOF 누수 방지가 실제로 중요한지 | `train_distill.py --anneal none --teacher_source single` |
| G | D와 동일, teacher=조건 A 직접(in-sample) | 〃 | `train_distill.py --anneal linear --teacher_source single` |
| H | E와 동일, teacher=조건 A 직접(in-sample) | 〃 | `train_distill.py --anneal linear --coherence --teacher_source single` |

여덟 조건 전부 같은 아키텍처(`MultiTaskNet`: 공유 trunk + 5개 독립 CORN 헤드)와
같은 5축 결합 방식(`--mtl`, 기본 `uncertainty`, §7)을 쓴다. B는 손실이 hard
label CORN 손실 그대로이고, C~H는 그 자리를 지식증류 손실로 바꾼 것뿐이다 —
그래야 조건 사이 성능차가 "무엇으로 학습했는가"로만 좁혀진다.

**F/G/H는 C/D/E와 anneal/coherence 구성이 완전히 같고, teacher 로짓의 출처만
다르다.** C/D/E는 `collect_oof.py`가 만든 5-fold OOF 로짓(student가 한 번도
보지 못한 teacher 예측, 누수 없음)을 쓰고, F/G/H는 `collect_a.py`가 만든 조건 A
`best.pt`의 in-sample 로짓(teacher가 train 전체로 학습됐으므로 이 표본들을 이미
본 상태의 예측)을 쓴다. F/G/H는 "OOF로 누수를 막은 증류가 실제로 더 나은가"를
C/D/E와 짝지어 보기 위한 대조 조건이지, F/G/H 쪽이 방법론적으로 더 우수해서
있는 게 아니다 — in-sample teacher는 암기한 표본에 낙관적인 soft label을 줄
수 있다.

## 1. 데이터

기본 `DATA`는 [`dataset_all_final/images`](../dataset_all_final/images)다(이
저장소의 아토피 5축 데이터 — `labels.csv` + `train/val/test/` 폴더, `cls_atopy_mbn`이
쓰는 것과 같은 규약). 다른 경로를 주려면 `DATA=/path/to/dataset`.

```
<data>/labels.csv
<data>/train/*.png|jpg   <data>/val/*   <data>/test/*
```

CSV 헤더는 `split,identifier,severity,erythema,papulation,excoriation,
lichenification[,diagnosis]`(`identifier`는 과거 `stem`/`filename`도 계속
받는다, 값은 이미지 파일명에서 확장자를 뺀 것과 같아야 한다; 등급 컬럼명
흔들림은 `labels.py::_COLUMN_ALIASES`가 흡수한다). `dataset.py`는 원본 아토피
JSON([`atopy/`](../atopy))도 바로 읽을 수 있다(`--json_roots`) — CSV/JSON 어느
쪽이든 산출되는 라벨 인덱스 형식은 같다.

CSV 의 split 컬럼은 **읽지 않는다**. split 은 `images/{train,val,test}/` 폴더
구조가 정한다 — 두 곳이 어긋나면 조용히 누수가 나므로 진실의 출처를 하나로
못박는다.

- **5-fold는 train 안에서만** 나눈다. val/test는 split 폴더 구조가 이미
  분리해 두므로 fold 분할이 손댈 일이 없다.
- **teacher의 early stopping은 고정 val로 한다** — fold에서 빠진 부분(held-out
  fold)으로 모델을 고르면 그 fold의 OOF soft label이 이미 그 데이터로 한 번 고른
  모델의 예측이라 낙관적으로 편향된다.
- **5개 teacher(홍반/구진/찰상/태선화/IGA)는 전부 같은 fold 분할을 쓴다**
  (`make_folds.py`가 한 번 만든 JSON을 재사용). 사진 한 장이 홍반 teacher에서는
  fold 1, IGA teacher에서는 fold 3이면 소견별 soft label의 "무엇을 못 봤는지"가
  서로 어긋난다.
- **student는 k-fold 없이 train 전체로 한 번만 학습**한다. student의
  예측을 다른 모델 학습에 쓰지 않고(OOF 불필요), 고정 val/test가 이미 있어
  성능 추정을 위한 k-fold도 불필요하다.

> **주의 — 5-fold teacher OOF는 BAM 논문에 없는 절차다.** 원 논문(및 공식 코드
> `bam/run_classifier.py`)은 teacher를 train 전체로 한 번 학습하고, 그 teacher가
> **자신이 학습한 바로 그 데이터**에 대해 예측한 값을 그대로 distillation
> target으로 쓴다 — teacher가 train에 다소 과적합돼 있어도 teacher annealing이
> 뒤로 갈수록 정답 라벨 비중을 높이므로 문제 삼지 않는다. 여기서 5-fold OOF를
> 쓰는 것은 그 낙관적 편향을 미리 없애려는 별도의 설계 선택이며, teacher
> annealing(§4)이나 CORN 증류 손실 자체와는 독립적인 부분이다 — B~E 비교표의
> 결론(annealing/증류/coherence 효과)에는 영향을 주지 않는다.

## 2. 실행 순서

```bash
cd cls_kd

# --- teacher: 5-fold 분할 + OOF soft label 생성 (25번 학습: 5축 x 5fold) ---
MODE=oof bash run_teachers.sh
#   -> runs/folds.json                                (5개 teacher 공유 fold)
#   -> runs/oof_<task>_<model>_r<imgsz>_fold<k>of5/    (fold별 teacher ckpt+로짓)
#   -> runs/teacher_logits/<task>.pt                   (collect_oof.py 가 자동 병합)

# --- 조건 A: 단일축 teacher(전체 데이터, 보고용 상한 기준) ---
MODE=full bash run_teachers.sh
#   -> runs/A_single_<task>_<model>_r<imgsz>/test_report.json
#   -> runs/teacher_logits_a/<task>.pt                 (collect_a.py 가 자동 생성 —
#                                                        in-sample, F/G/H 용)

# --- 조건 B: 표준 multi-task ---
bash run_multitask.sh
#   -> runs/B_<model>_r<imgsz>/test_report.json

# --- 조건 C/D/E/F/G/H: 증류 ---
bash run_distill.sh                            # C,D,E,F,G,H 순서로 전부
#   -> runs/{C,D,E,F,G,H}_<model>_r<imgsz>/test_report.json
#   (C/D/E 는 teacher_logits/ OOF 로짓, F/G/H 는 teacher_logits_a/ in-sample 로짓 —
#    COND=cde 또는 COND=fgh 로 한쪽만 돌릴 수도 있다)

# --- 비교표(요약 숫자만, 이미 있는 test_report.json 재사용) ---
python3 compare.py --model pvtv2b0 --imgsz 224

# --- 재평가(혼동행렬 포함, test 를 다시 돌려 확인) ---
bash evaluate.sh
```

`DATA`를 안 주면 위 명령 모두 `../dataset_all_final/images`를 기본값으로 쓴다.
다른 데이터로 돌리려면 `DATA=/path/to/dataset ...` 를 앞에 붙인다. 세 `run_*.sh`는
같은 venv 관례(`../.venv-train` 탐색, 없으면 시스템 `python3`로 폴백, 상대경로를
cd 전에 절대경로로 고정)를 따른다. 하나씩 직접 돌리려면 `train_single.py --help` /
`train_multitask.py --help` / `train_distill.py --help` 로 개별 CLI를 볼 것.

## 재평가 — `eval.py` / `evaluate.sh` (혼동행렬 포함)

`compare.py`(§0 위 실행 순서)는 각 학습 스크립트가 끝날 때 이미 만들어 둔
`test_report.json`을 그러모아 표로 보여줄 뿐, 다시 추론하지 않고 혼동행렬도
없다. **`eval.py`는 ckpt 하나를 받아 test(또는 다른 split)를 처음부터 다시
돌려 축별 지표 + 혼동행렬을 보여준다**. 조건 A(단일축, `train_single.py` 산출)와
B~H(다축, `train_multitask.py`/`train_distill.py` 산출) ckpt를 ckpt 안의 `"task"`/
`"tasks"` 키로 자동 구분해서 받는다.

```bash
python3 eval.py --run runs/A_single_erythema_pvtv2b0_r224      # --data 는 학습 때 값 재사용
python3 eval.py --run runs/D_pvtv2b0_r224 --split val
python3 eval.py --run runs/D_pvtv2b0_r224 --last                # best.pt 대신 last.pt
python3 eval.py --run runs/G_pvtv2b0_r224                       # F/G/H도 동일하게

bash evaluate.sh                          # runs/ 안의 A_single_*/B_*/C_*/D_*/E_*/F_*/G_*/H_* 전부
SPLIT=val bash evaluate.sh
RUN=D_pvtv2b0_r224 bash evaluate.sh       # 하나만
```

평가 설정(imgsz/`--exp`/`--mtl` 등 결과를 바꾸는 값)은 ckpt에 저장된 학습 당시
`args`를 그대로 따른다 — 학습 때와 다른 해상도·crop으로 재평가하면 숫자가
비교 불가능해지기 때문이다. `pretrained=False`로 백본을 만들고 ckpt가 전부
덮어쓰므로 평가에는 ImageNet 사전학습 가중치도 필요 없다. 결과는 화면 출력과
함께 `<run 폴더>/eval_<split>.json`(축별 지표 + 혼동행렬 raw 배열)에 남는다.

### teacher OOF 평가 — `eval_oof.py` (개별 fold + 전체, soft 확률까지)

`eval.py`는 A~H 처럼 **test set으로 최종 평가**하는 ckpt만 다룬다.
`train_single.py --fold`가 만드는 teacher OOF 조각(`oof_logits.pt`)은 애초에
test를 보지 않으므로(held-out **train** fold에 대한 예측일 뿐) 별도
스크립트인 `eval_oof.py`로 다룬다. 이 스크립트는:

1. **fold 하나하나 개별 평가** — 그 fold의 held-out 표본만 놓고 QWK/혼동행렬을
   본다. 유난히 나쁜 fold가 있으면 여기서 드러난다.
2. **전체(합산) 평가** — N개 fold를 전부 이어붙여 train 전체(각 표본이 자신을
   학습하지 않은 teacher의 예측)에 대해 본다. `collect_oof.py`가 만드는
   `teacher_logits/<task>.pt`와 커버리지가 같다 — **즉 증류에 실제로 쓰이는
   soft label을 hard 라벨로 디코딩했을 때의 품질**이다.
3. **`--show_probs N`** — CORN 로짓을 `corn_class_probs`로 카테고리 확률
   분포(합이 1)로 디코딩해서 표본 N개의 실제 숫자를 보여주고, fold별/전체로
   "진짜 등급에 실린 확률질량"(교정 proxy)과 "분포 엔트로피"(확신도)를
   요약한다 — **teacher의 soft 확률이 어떻게 흘러가는지**를 직접 보는 부분.

```bash
python3 eval_oof.py --task erythema --model pvtv2b0 --imgsz 224
python3 eval_oof.py --task iga_grade --model pvtv2b0 --imgsz 224 \
    --show_probs 5                     # fold별/전체 표본 5개씩 실제 분포 출력
python3 eval_oof.py --task erythema --model pvtv2b0 --imgsz 224 \
    --exp bbox --bbox_square           # bbox 로 만든 teacher 평가(태그를 맞출 것)
```

`--model`/`--imgsz`/`--exp`/`--bbox_square`/`--n_folds`는 `train_single.py`
(OOF 학습 때)와 `collect_oof.py`에 준 값과 같아야 한다 — 런 폴더 이름
(`oof_<task>_<model>_r<imgsz>{_bbox[_sq]}_fold<k>of<n>`)을 그대로 재구성해서
찾기 때문이다. 다른 값을 주면 "없음, 건너뜀" 경고 뒤에 fold를 하나도 못
찾았다는 에러로 죽는다.

## 병변 마스크 crop — `--exp bbox`

기본은 `--exp full`(원본 이미지 전체, 마스크 불필요)이다. 분할 마스크가 있으면
`--exp bbox`로 마스크의 union bbox(+margin)를 크롭해서 볼 수 있다 —
`train_single.py`/`train_multitask.py`/`train_distill.py` 세 스크립트, 그리고
`run_teachers.sh`/`run_multitask.sh`/`run_distill.sh` 세 래퍼 전부 아래 옵션을
동일하게 받는다(`kd_common.py::add_crop_args`에 한 곳에만 정의돼 있다):

| CLI (env) | 기본값 | 의미 |
|---|---|---|
| `--exp` (`EXP`) | `full` | `full`\|`bbox` |
| `--masks` (`MASKS`) | (자동탐색) | 마스크 루트를 직접 지정. 비우면 `<data>/masks/<split>` → `<data>/images/masks/<split>` → `<data>/../masks/<split>` 순으로 찾는다 |
| `--margin` (`MARGIN`) | `0.15` | union bbox 각 변을 이 비율만큼 바깥으로 확장 — 병변 경계뿐 아니라 주변 정상 피부까지 같이 보려는 것 |
| `--bbox_square` (`BBOX_SQUARE=1`) | 끔 | 크롭을 긴 변 기준 정사각으로 넓힌다. 안 켜면 `Resize((imgsz,imgsz))`가 비등방 변환이라 구진이 타원이 되는 등 모양이 찌그러진다 |

**배경을 지우지 않는다** — 마스크는 "어디를 크롭할지"만 정하고, 크롭된 영역의
픽셀은 원본 그대로 가져온다(자세한 원리는 `dataset.py`의 `_bbox_from_mask`/
`_square_crop` 참고). 마스크가 비었거나 없는 표본은 자동으로 전체 이미지로
폴백한다.

```bash
EXP=bbox bash run_teachers.sh              # A + teacher OOF, bbox 크롭
                                            # (A 쪽에서 teacher_logits_a/ 도 같이 만들어진다)
EXP=bbox BBOX_SQUARE=1 bash run_multitask.sh  # B
EXP=bbox BBOX_SQUARE=1 bash run_distill.sh    # C/D/E/F/G/H
```

**주의 — `--exp`/`--bbox_square`는 런 이름과 teacher 로짓 파일명에 그대로 남는다**
(`crop_tag`): `--exp bbox --bbox_square`로 만들면 `A_single_<task>_..._bbox_sq/`,
`teacher_logits/<task>_bbox_sq.pt`/`teacher_logits_a/<task>_bbox_sq.pt`처럼
`_bbox`/`_bbox_sq` 접미사가 붙는다 — `--exp full`(기본)로 만든 결과와 같은
폴더/파일을 덮어쓰지 않게 하려는 것이다. 그래서 **student(`train_distill.py`)는
teacher를 만들 때와 반드시 같은 `--exp`/`--bbox_square`를 줘야 한다**(C/D/E는
`run_teachers.sh MODE=oof`, F/G/H는 `run_teachers.sh MODE=full` 기준) — 다르면
`teacher_logits[_a]/<task>{태그}.pt`를 못 찾고 바로 에러로 죽는다(어느 축이
문제인지, 어떤 조합으로 다시 만들어야 하는지까지 에러 메시지에 나온다).

## 3. 백본 — timm

`--model`은 짧은 이름 -> timm 모델명 매핑이다(`model.py::MODELS`):
`efflite0`/`effb0`/`mnv3s`/`mnv4s`/`mnv4m`/`mnv4l`/`mnv4hm`/`pvtv2b0`/`pvtv2b1`/
`pvtv2b2`. `build_backbone`이 `timm.create_model(name, pretrained=True,
num_classes=0, global_pool=...)`을 그대로 부르므로, timm 이 지원하는 다른
모델명도 `MODELS`에 한 줄만 추가하면 바로 쓸 수 있다(ImageNet 가중치는 처음
실행 시 자동으로 받아진다). `--no_pretrained`로 from-scratch 대조군을 만들 수
있고, 평가(`eval.py`)는 ckpt가 백본까지 전부 덮어쓰므로 `pretrained=False`로
돈다.

## 4. CORN 지식증류 — `corn_distill_loss` (kd_common.py)

BAM 논문의 KD 손실은 원래 softmax 분류를 전제로 한다:

```
ℓ(λ·y + (1-λ)·f_teacher(x), f_student(x))     (paper §3.2)
```

여기서는 softmax가 아니라 **CORN**(K등급을 K-1개의 조건부 이진 문제로 분해,
`corn.py::corn_loss`)을 쓴다. 그래서 BAM의 블렌드를 CORN의 조건부 이진 구조
그대로 확장한다: 레벨 `i`의 이진 타깃(원래 "등급>i" 여부의 0/1)을
`λ·hard + (1-λ)·sigmoid(teacher_logit_i)`로 블렌드하고, 나머지(조건부 마스킹,
BCE)는 `corn_loss`와 완전히 같다.

- `λ=1` → `corn_loss`와 완전히 동일(순수 지도학습)
- `λ=0` → 순수 증류 — teacher의 레벨별 조건부확률 `P(등급>i | 등급>i-1)`을 그대로 모사
- teacher 로짓은 항상 `.detach()`(증류 방향은 student → teacher 한쪽뿐)

`--anneal none`(조건 C)은 λ=0 고정 — 공식 구현(`bam/task_specific/classification/
classification_tasks.py`)의 `config.teacher_annealing=False` 분기(`labels =
true*(1-distill_weight) + teacher*distill_weight`, `distill_weight=1`)와 같다.

`--anneal linear`(조건 D/E)는 BAM 공식 코드(`bam/run_classifier.py`)의 스케줄을
그대로 옮긴 것이다:

```python
percent_done = global_step / num_train_steps      # 매 학습 스텝마다 갱신
labels = true_labels * percent_done + teacher_labels * (1 - percent_done)
```

즉 λ(=percent_done)는 **에폭이 아니라 스텝 단위**로 선형 증가한다 — `anneal_lambda`는
`(step, total_steps)`를 받고, `train_distill.py::train_one_epoch`가 배치마다 λ를
새로 계산한다(에폭당 한 번만 갱신하면 한 에폭 안의 모든 배치가 같은 λ를 쓰는
계단식 근사가 되어 원 스케줄과 어긋난다).

## 5. 일관성 손실(coherence loss) — 조건 E

실데이터 분석(`atopy.csv`, 9,150개 표본)에 따르면 `IGA(총평) -
max(홍반,구진,찰상,태선화)`의 분포는:

| 차이 | -1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| 개수 | 96 | 3,211 | 5,275 | 544 | 24 |

98.69%(9,030/9,150)가 `{0,1,2}`에 몰려 있다 — 즉 IGA는 구조적으로 가장 심한
소견 등급과 거의 같거나 그보다 최대 2등급 더 심하게 매겨진다. 이걸 힌지 손실로
반영한다(`kd_common.coherence_loss`):

```
g = E[IGA] - max(E[홍반], E[구진], E[찰상], E[태선화])
L_con = mean( max(0, -g) + max(0, g-2) )
```

`E[·]`는 CORN 로짓의 기대 등급(`corn_expected_value` — `corn_predict`와 같은
누적곱 규칙 `P(등급>i)=∏sigmoid(logit_j)`를 쓰되 0.5 이산화 대신 기대값을 써서
미분 가능하게 만든 것, `E[Y]=Σ P(Y>i)`). **정답 라벨이 아니라 student 자신의
현재 예측**에 거는 제약이라 라벨 결측과 무관하게 매 배치 걸 수 있다.

`--lambda_con`(기본 0.3)으로 가중을 조절하고, `--con_warmup_epochs`로 초반
N epoch은 끌 수 있다(모델이 아직 무작위에 가까울 때 큰 그래디언트를 주는 것을
피하고 싶으면 사용).

## 6. 파일

| 파일 | 역할 |
|---|---|
| `labels.py` | 태스크 정의, 등급 문자열 파싱, CSV 컬럼 별칭 |
| `dataset.py` | `SevDataset` — 이미지 폴더(+labels.csv 또는 원본 JSON) → (텐서, 등급) |
| `model.py` | `MultiTaskNet`, timm 백본 로딩(`build_backbone`) |
| `metrics.py` | QWK/±1/MAE, `selection_score`(0.5·IGA + 0.5·증상4종) |
| `corn.py` | CORN 손실/예측/`UncertaintyWeighter`/transform 등 학습 루프 유틸 |
| `kd_common.py` | 위 전부 + KD 손실/유틸(이 폴더의 "라이브러리" 역할) |
| `make_folds.py` | train split 5-fold 층화 분할(IGA 기준) → `folds.json` |
| `train_single.py` | 조건 A(전체 데이터) + teacher OOF(fold별) — `--fold` 유무로 모드 전환 |
| `collect_oof.py` | 5-fold OOF 로짓 병합 → `teacher_logits/<task>.pt`(C/D/E용) |
| `collect_a.py` | 조건 A의 `best.pt`에서 in-sample 로짓 추출 → `teacher_logits_a/<task>.pt`(F/G/H용) |
| `train_multitask.py` | 조건 B |
| `train_distill.py` | 조건 C/D/E/F/G/H — `--anneal {none,linear}` `--coherence` `--teacher_source {oof,single}` |
| `compare.py` | A~H 결과를 한 표로(재추론 없이 기존 `test_report.json` 재사용) |
| `eval.py` | ckpt 하나를 다시 추론해 축별 지표 + 혼동행렬(`eval_<split>.json`) |
| `eval_oof.py` | teacher OOF fold별/전체 평가 + `--show_probs`로 soft 확률 분포 확인 |
| `run_teachers.sh` / `run_multitask.sh` / `run_distill.sh` / `evaluate.sh` | venv 관례를 따르는 실행 래퍼 |

## 7. 5축 결합 방식 — `--mtl {uncertainty,fixed}`

`train_multitask.py`(B)와 `train_distill.py`(C/D/E/F/G/H)는 5축 손실 결합 방식을
**똑같은 `--mtl`/`--iga_weight`**로 고른다(`kd_common.add_mtl_args`, 세부 구현은
`corn.combine_losses`) — 그래야 B~H 사이의 성능차가 손실 결합 방식 차이가
아니라 **증류/annealing/coherence/teacher 출처 그 자체**에서만 나온다. 조건을
대조할 때는 B와 C~H에 항상 같은 `--mtl`(과, fixed라면 같은 `--iga_weight`)을 줄 것.

| `--mtl` | 방식 | 비고 |
|---|---|---|
| `uncertainty`(기본) | `corn.UncertaintyWeighter` — 태스크마다 학습되는 log-분산 `σ`로 자동가중(Kendall et al.) | `cls_atopy_mbn` 등 이 저장소 계열에서도 기본값 |
| `fixed` | `--iga_weight`\*IGA손실 + (1-`--iga_weight`)\*mean(증상4종손실), 상수 배분 | 손실결합 자체를 대조군으로 켜보고 싶을 때 |

모델 선택 기준(`selection_score = 0.5·IGA QWK + 0.5·증상4종 QWK 평균`)과 평가
방식(`run_epoch_eval`)은 `--mtl`과 무관하게 `metrics.py`/`corn.head_predict`를
그대로 불러 쓰므로 다섯 조건의 지표가 전부 같은 정의로 계산된다.

`--exp bbox`와 마찬가지로 **`--mtl fixed`는 런 이름에 `_fixed<iga_weight>`
태그가 붙는다**(`mtl_tag`) — `uncertainty`(기본)로 만든 결과를 덮어쓰지 않는다.

```bash
MTL=fixed IGA_WEIGHT=0.5 bash run_multitask.sh   # B, 고정 배분
MTL=fixed IGA_WEIGHT=0.5 bash run_distill.sh      # C/D/E/F/G/H, 고정 배분
```

## 학습 재개 — `--resume`

`train_single.py`/`train_multitask.py`/`train_distill.py` 세 스크립트 전부
중단된 학습을 이어갈 수 있다(`kd_common.try_resume`/`rng_state`). 매 epoch 끝에
`<out_dir>/last.pt`에 다음을 전부 저장한다:

- 모델 가중치, `UncertaintyWeighter`(쓰는 경우) 상태
- optimizer/scheduler/AMP scaler 상태
- random/numpy/torch(+CUDA) RNG 상태
- `best`/`best_ep`/`no_improve`(조기종료 카운터)

`--resume`를 주면 이 파일에서 위 상태를 전부 복원하고 `epoch+1`부터 이어서
돈다 — 옵티마이저 모멘텀이나 조기종료 카운터가 리셋되지 않는다. `best.pt`는
지금까지와 동일하게 가볍게 유지한다(모델+지표만, 평가에는 그거면 충분하다).

```bash
RESUME=1 bash run_teachers.sh     # A/OOF — 스윕 전체에 --resume 적용
RESUME=1 bash run_multitask.sh    # B
RESUME=1 bash run_distill.sh      # C/D/E/F/G/H
```

`--model`/`--imgsz`/`--batch`/`--exp`/`--mtl` 등 결과를 바꾸는 인자가 ckpt와
다르면(`--task`/`--fold`는 `train_single.py`, `--anneal`/`--coherence`/
`--teacher_source`는 `train_distill.py`에서 추가로 체크) 조용히 이어받지 않고
바로 에러로 죽는다.
`run_teachers.sh RESUME=1`처럼 스윕 전체에 걸어도 안전하다 — `last.pt`가 없는
축/fold는 그냥 처음부터 시작한다.
