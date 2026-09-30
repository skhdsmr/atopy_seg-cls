"""질환 6-way 단일라벨 데이터셋 — dataset_disease/{train,val,test}/*.png + labels.csv 소비.

classification_topk/dataset.py 와의 차이:
    라벨축 : 태그(공존 가능, sigmoid) + IGA(순서형) -> 질환 1개(배타적, softmax).
             질환은 서로 배타적이고 등급처럼 순서가 없다. 아토피와 건선 사이엔 '거리'가
             없으므로 CORN/QWK 같은 순서형 장치를 쓰면 안 된다.
    부가축 : angle/source/subject 를 샘플마다 들고 다닌다. 라벨이 아니라 진단용이다 —
             출처별로 정확도를 쪼개 봐야 모델이 질환을 배웠는지 촬영 출처를 배웠는지
             구분할 수 있다(train↔test 출처 분포가 다르다).
그대로 유지: split 폴더 glob + stem 으로 CSV 조회하는 자립형 구조(atopy_crop_masks 규약).

CSV 헤더: split, stem, disease, angle, source, subject
"""
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# make_dataset_disease.py 와 같은 순서. 인덱스가 체크포인트에 박히므로 바꾸지 말 것.
DISEASES = ["건선", "아토피", "여드름", "정상", "주사", "지루"]
NUM_DISEASES = len(DISEASES)
_DIS2IDX = {d: i for i, d in enumerate(DISEASES)}

ANGLES = ["정면", "측면"]
_ANGLE_FILTER = {"front": {"정면"}, "side": {"측면"}, "both": set(ANGLES)}


def label_index_from_csv(csv_path):
    """dataset_disease/labels.csv -> {stem: {"disease": idx, "angle", "source", "subject"}}.

    정의 밖 질환이나 stem 이 빈 행은 조용히 버린다(make_dataset_disease.py 가 이미
    걸러 두므로 정상 경로에선 발생하지 않는다).
    """
    index = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem, disease = row.get("stem"), row.get("disease")
            if not stem or disease not in _DIS2IDX:
                continue
            index[stem] = {"disease": _DIS2IDX[disease], "angle": row.get("angle", ""),
                           "source": row.get("source", ""), "subject": row.get("subject", "")}
    return index


class DiseaseDataset(Dataset):
    """이미지 폴더 + label_index -> (image, disease_idx) 쌍.

    angle: both | front(정면만) | side(측면만).
    정면과 측면은 사실상 다른 도메인이라(정면=1024 얼굴 전체, 측면=512 피부 접사,
    케이스 키도 겹치지 않음) 섞을지 여부를 실험 변수로 둔다.
    """

    def __init__(self, img_dir, label_index, transform=None, angle="both"):
        self.transform = transform
        self.samples = []
        want = _ANGLE_FILTER[angle]
        skipped = 0
        for img_path in sorted(Path(img_dir).glob("*.png")):
            rec = label_index.get(img_path.stem)
            if rec is None:
                skipped += 1
                continue
            if rec["angle"] not in want:
                continue
            self.samples.append((img_path, rec["disease"], rec["angle"],
                                 rec["source"], rec["subject"]))
        if skipped:
            print(f"[dataset] 건너뜀(라벨없음): {skipped}")
        n_sub = len({s[4] for s in self.samples})
        print(f"[dataset] {Path(img_dir).name:5s}: 샘플 {len(self.samples):5d}  "
              f"키 {n_sub:5d}  각도={angle}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, y, *_ = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)

    def class_counts(self):
        """질환별 샘플 수(인덱스 순). 클래스 가중/불균형 진단용."""
        c = Counter(y for _, y, *_ in self.samples)
        return [c[i] for i in range(NUM_DISEASES)]

    def meta(self, key):
        """샘플 순서대로 angle/source/subject 배열. 지표를 이 축으로 쪼갤 때 쓴다."""
        idx = {"angle": 2, "source": 3, "subject": 4}[key]
        return [s[idx] for s in self.samples]
