"""아토피 5축 등급 데이터셋 — labels.csv + split 폴더.

    <data>/labels.csv
    <data>/train/*.png|jpg   <data>/val/*   <data>/test/*

labels.csv 컬럼(`/home/work/Code/README.md` 의 데이터 규약과 동일):

    stem                                       확장자 뺀 파일명(이미지 매칭 키)
    severity | iga | iga_grade                 IGA 중증도
    erythema papulation excoriation lichenification   증상 4축

**라벨 공간을 CSV 에서 유도한다.** 두 데이터 출처의 표기가 다르기 때문이다:

    합성(dataset_all_final) : 'Clear'/'Almost Clear'/'Mild'/'Moderate'/'Severe' 문자열
    실데이터(Data/atopy.csv): 정수. IGA 가 **1~4 이고 0 이 한 건도 없다**(9,150행 확인)

문자열이면 표준 등급표를 그대로 쓰고(IGA 5등급 / 증상 4등급), 정수면 **관측된 값만**
오름차순으로 0..n-1 에 매핑한다. 실데이터 IGA 는 이 규칙으로 4등급이 되어, 영원히
0장인 Clear 로짓이 생기거나 QWK 가중행렬이 왜곡되는 일이 없다. 어느 쪽으로 해석했는지는
build_label_space 가 매핑표와 등급별 건수를 찍는다 — 조용히 바뀌면 안 되는 값이라서다.
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

TASK_NAMES = ["severity", "erythema", "papulation", "excoriation", "lichenification"]
SYMPTOMS = [t for t in TASK_NAMES if t != "severity"]

# CSV 에서 각 축을 찾을 때 볼 컬럼 이름(앞에서부터)
CSV_COLS = {
    "severity": ["severity", "iga", "iga_grade"],
    "erythema": ["erythema"],
    "papulation": ["papulation"],
    "excoriation": ["excoriation"],
    "lichenification": ["lichenification"],
}
# 문자열 표기일 때의 표준 등급표(순서 = 등급 오름차순)
STR_LEVELS = {
    "severity": ["Clear", "Almost Clear", "Mild", "Moderate", "Severe"],
    **{t: ["None", "Mild", "Moderate", "Severe"] for t in SYMPTOMS},
}

# 그룹 프리셋 — 곧 branch5(태선화 분리)의 ablation 축이다. model.py 상단 docstring 참고.
GROUP_PRESETS = {
    # 태선화 분리(기본): 라벨 상관에서 태선화만 나머지와 떨어져 있다(스피어만 0.21~0.38)
    "sep": {"iga": ["severity"],
            "acute": ["erythema", "papulation", "excoriation"],
            "lich": ["lichenification"]},
    # 논문과 동일한 2분기(DR/DME 자리) — branch5 를 뺀 대조군
    "merged": {"iga": ["severity"], "sym": SYMPTOMS},
    # 태스크당 1분기(5분기) — cls_mbn 의 평평한 구조에 대응, side branch 5개라 가장 무겁다
    "flat": {t: [t] for t in TASK_NAMES},
}
IGA_GROUP = {"sep": "iga", "merged": "iga", "flat": "severity"}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _pick(row, names):
    for n in names:
        if n in row and str(row[n]).strip() != "":
            return str(row[n]).strip()
    return None


def build_label_space(csv_path):
    """labels.csv -> (label_index, num_classes, levels).

    label_index: {stem: {task: idx}}   — 5축이 전부 있는 행만 담는다
    num_classes: {task: n}
    levels:      {task: [원표기, ...]}  (인덱스 = 등급)
    """
    csv_path = Path(csv_path)
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = _pick(row, ["stem", "filename", "file", "image"])
            if not stem:
                continue
            vals = {t: _pick(row, CSV_COLS[t]) for t in TASK_NAMES}
            if any(v is None for v in vals.values()):
                continue
            rows.append((Path(stem).stem, vals))
    if not rows:
        raise SystemExit(f"[dataset] {csv_path} 에서 5축이 모두 있는 행을 찾지 못했다.")

    levels, mapping = {}, {}
    for t in TASK_NAMES:
        seen = {v for _, vals in rows for v in [vals[t]]}
        canon = {s.lower(): i for i, s in enumerate(STR_LEVELS[t])}
        if all(v.lower() in canon for v in seen):            # 문자열 표기 -> 표준 등급표
            levels[t] = list(STR_LEVELS[t])
            mapping[t] = canon
            kind = "문자열"
        else:                                                # 정수 표기 -> 관측값만 사용
            try:
                nums = sorted({int(float(v)) for v in seen})
            except ValueError as e:
                raise SystemExit(f"[dataset] '{t}' 축의 값을 해석할 수 없다: {sorted(seen)[:8]}") from e
            levels[t] = [str(n) for n in nums]
            mapping[t] = {str(n): i for i, n in enumerate(nums)}
            kind = "정수"
        levels[t] = [f"{v}" for v in levels[t]]
        levels[t + "__kind"] = kind

    index = {}
    for stem, vals in rows:
        try:
            index[stem] = {t: mapping[t][vals[t].lower() if levels[t + "__kind"] == "문자열"
                                        else str(int(float(vals[t])))] for t in TASK_NAMES}
        except (KeyError, ValueError):
            continue
    num_classes = {t: len(levels[t]) for t in TASK_NAMES}

    print(f"[labels] {csv_path}  행 {len(index)}개")
    for t in TASK_NAMES:
        cnt = [0] * num_classes[t]
        for lb in index.values():
            cnt[lb[t]] += 1
        pairs = ", ".join(f"{levels[t][i]}->{i}({cnt[i]})" for i in range(num_classes[t]))
        print(f"  {t:<16} {num_classes[t]}등급 [{levels[t + '__kind']}]  {pairs}")
    return index, num_classes, {t: levels[t] for t in TASK_NAMES}


class AtopyDataset(Dataset):
    """split 폴더의 이미지 + stem 으로 조회한 라벨."""

    def __init__(self, image_dir, label_index, num_classes, transform=None, split=""):
        self.transform = transform
        self.num_classes = num_classes
        self.samples = []
        missing = 0
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise SystemExit(f"[dataset] 이미지 폴더가 없다: {image_dir}")
        for p in sorted(image_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            lb = label_index.get(p.stem)
            if lb is None:
                missing += 1
                continue
            self.samples.append((p, lb))
        if not self.samples:
            raise SystemExit(f"[dataset] {image_dir} 에서 라벨이 붙은 이미지를 찾지 못했다 "
                             f"(파일 {missing}개가 labels.csv 에 없음).")
        note = f"  (라벨없어 건너뜀 {missing})" if missing else ""
        print(f"[dataset] {split or image_dir.name}: {len(self.samples)}장{note}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, labels = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, labels

    def label_arrays(self):
        """{task: np.ndarray(int64)} — IBB 샘플러(sampler.py)가 쓰는 형식.
        라벨 없는 축은 -1(UNKNOWN). 이 폴더는 5축이 다 있는 행만 담으므로 지금은 안 생기지만,
        실데이터에서 일부 축이 비는 경우를 샘플러가 그대로 처리할 수 있게 규약을 맞춰둔다."""
        return {t: np.array([lb.get(t, -1) for _, lb in self.samples], dtype=np.int64)
                for t in TASK_NAMES}

    def class_counts(self):
        counts = {t: [0] * self.num_classes[t] for t in TASK_NAMES}
        for _, lb in self.samples:
            for t, idx in lb.items():
                counts[t][idx] += 1
        return counts
