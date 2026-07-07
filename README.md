# atopy_seg-cls

아토피 피부 병변 **분할(Segmentation)** + 중증도/증상 **분류(Classification)** 파이프라인과 Streamlit 데모.

- **분할**: 두 가지 결과(데모 상단 단추로 전환)
  - `face` — HRNet-w18 + U-Net (`tu-hrnet_w18`), `dataset_face_strat_merged`
  - `lesion` — U-Net++ (EfficientNet-b3), `dataset_lesion_merged`
- **분류**: MobileNetV4-conv-medium 멀티태스크 — IGA 중증도(5단계) + 증상 4종(홍반·구진·찰상·태선화) 순서형 등급 (두 분할 결과가 공유)
- **데모**: `app/app.py` (Streamlit, 3분할 대시보드)
  - 왼쪽: 테스트셋 목록 / 가운데: 분할 오버레이·중증도·증상(정답 GT 포함) / 오른쪽: 누적 성능(IoU·Dice·IGA QWK)
  - 목록 아래 **📷 촬영 / 🖼 불러오기** 로 test 셋 외 이미지를 올리거나 카메라로 찍어 현재 모델로 즉시 추론(정답 없음 → 지표 미반영)

> ⚠️ 연구/교육용입니다. 의료 진단 도구가 아닙니다.
> ⚠️ 환자 원본 이미지(`atopy/`)와 데이터셋(`dataset_*`)은 **개인정보 보호를 위해 저장소에 포함하지 않습니다.**

## 구조

```
segmentation/              # 분할 파이프라인
  dataset.py augment.py losses.py metrics.py tiling.py   # 모델 공유 모듈
  visualize.py boundary_dice.py threshold_sweep.py       # 모델 무관 평가·진단
  postprocess.py viz_failures.py viz_tolerance.py        #   (예측 시각화·경계허용 지표·오탐 후처리)
  merge_face_fragments.py  #   폴리곤 국소 병합 도구(데이터셋 전처리)
  encoder_decoder/         #   Encoder-Decoder 분할 (U-Net++/EfficientNet-b3, HRNet+U-Net)  (train.py, model.py)
  unext/                   #   UNeXt 대조군 (archs.py, train_unext.py, run.sh)
classification/            # 분류 파이프라인
  dataset.py metrics.py    #   공유 모듈
  mobilenet/               #   MobileNetV4 멀티태스크 (train.py, model.py, run.sh)
app/                       # Streamlit 데모
  app.py                   #   테스트셋 대시보드 + 업로드/카메라 실시간 추론
  eval_precompute.py       #   사전 평가(무거운 추론 1회) → app/outputs/<variant>/ 생성
  start_app.sh             #   eval_precompute + streamlit 원샷 실행
requirements.txt
```

## 데모 실행

```bash
# 1) 의존성 설치 (torch 는 환경에 맞게 별도 설치 권장)
pip install -r requirements.txt

# 2) 모델 가중치 받기 (Git LFS)
git lfs install
git lfs pull

# 3) 테스트셋 대시보드
#    데이터셋(dataset_face_strat_merged, dataset_lesion_merged)이 있는 환경에서 실행.
bash app/start_app.sh          # 사전 평가 → 대시보드 실행을 한 번에

# (수동으로 나눠서 실행하려면)
python3 app/eval_precompute.py                 # face·lesion 모두 → app/outputs/<variant>/ 생성
# python3 app/eval_precompute.py --variant face   # 하나만 다시 만들 때
streamlit run app/app.py
```

무거운 계산(모델 추론)은 `eval_precompute.py` 가 한 번만 수행해 `app/outputs/` 에
예측/정답 마스크와 지표(`eval_meta.json`)를 저장하고, `app/app.py` 는 이를 읽어 **표시만** 합니다.
대시보드는 상단 단추로 **face/lesion** 을 전환하며 3분할로 보여주고, 패널 경계를 드래그해
너비를 조절할 수 있습니다. 업로드/카메라 이미지는 현재 모델로 그 자리에서 추론합니다.

## 모델 가중치 (Git LFS)

데모용 best 체크포인트만 Git LFS 로 관리합니다.

| 용도 | 경로 |
|---|---|
| 분할 face (HRNet-w18 + U-Net) | `segmentation/encoder_decoder/runs/hrnet18_unet/checkpoint_best.pth` |
| 분할 lesion (U-Net++ EfficientNet-b3) | `segmentation/encoder_decoder/runs/unetpp_effb3_lesion/checkpoint_best.pth` |
| 분류 (MobileNetV4, atopy) | `classification/mobilenet/runs/cls_mobilenetv4_conv_medium_atopy/best.pt` |

## 학습 재현

데이터셋(`dataset_lesion_merged/` 등, YOLO 폴리곤 라벨)이 있는 환경에서:

```bash
# 분할 (Encoder-Decoder)
python3 segmentation/encoder_decoder/train.py --data dataset_lesion_merged --epochs 100 --batch 16
# UNeXt 대조군
bash segmentation/unext/run.sh

# 분류 (MobileNetV4 멀티태스크)
bash classification/mobilenet/run.sh                 # 기본: source=face, conv-medium
# 또는 직접:
python3 classification/mobilenet/train.py --source atopy --version conv --size medium --epochs 100
```

학습에는 `albumentations`, `opencv-python-headless`, `timm` 이 추가로 필요합니다.
