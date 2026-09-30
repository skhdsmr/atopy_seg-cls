"""
아토피 멀티태스크 분류 데이터셋 (원본 이미지 입력).

- 입력: 원본 얼굴 이미지 전체 (seg crop 아님 — 중증도의 '범위' 정보 보존 + seg 오류 전파 회피).
- 라벨: atopy/ JSON 의 easi_score → IGA 중증도 + 증상 4종 (모두 순서형 등급).
- 뷰: 정면(front)/측면(side)/둘다(both). 정면·측면을 각각 독립 샘플로 취급.
- split: atopy 폴더 규약상 T*=train, V*=val (test 분리는 추후).
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# 순서형 등급 (인덱스 = 등급 오름차순). JSON easi_score 키와 매핑.
TASKS = {
    "severity":        ["Clear", "Almost Clear", "Mild", "Moderate", "Severe"],  # iga_grade
    "erythema":        ["None", "Mild", "Moderate", "Severe"],
    "papulation":      ["None", "Mild", "Moderate", "Severe"],
    "excoriation":     ["None", "Mild", "Moderate", "Severe"],
    "lichenification": ["None", "Mild", "Moderate", "Severe"],
}
JSON_KEY = {"severity": "iga_grade", "erythema": "erythema", "papulation": "papulation",
            "excoriation": "excoriation", "lichenification": "lichenification"}
TASK_NAMES = list(TASKS.keys())
NUM_CLASSES = {t: len(v) for t, v in TASKS.items()}
_LABEL2IDX = {t: {n: i for i, n in enumerate(v)} for t, v in TASKS.items()}


class AtopyClsDataset(Dataset):
    def __init__(self, pairs, transform=None):
        """pairs: [(image_dir, label_dir), ...]."""
        self.transform = transform
        self.samples = []
        skipped = 0
        for img_dir, lbl_dir in pairs:
            img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)
            for img_path in sorted(img_dir.glob("*.png")):
                jp = lbl_dir / f"{img_path.stem}.json"
                if not jp.exists():
                    skipped += 1
                    continue
                labels = self._parse(jp)
                if labels is None:
                    skipped += 1
                    continue
                self.samples.append((img_path, labels))
        if skipped:
            print(f"[dataset] 건너뜀(라벨없음/이상): {skipped}")
        print(f"[dataset] 샘플 {len(self.samples)}개")

    @staticmethod
    def _parse(jp):
        d = json.load(open(jp, encoding="utf-8"))
        try:
            es = d["annotations"][0]["diagnosis_info"]["easi_score"]
        except (KeyError, IndexError):
            return None
        out = {}
        for t in TASK_NAMES:
            v = es.get(JSON_KEY[t])
            if v not in _LABEL2IDX[t]:
                return None
            out[t] = _LABEL2IDX[t][v]
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, labels = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, labels

    def class_counts(self):
        counts = {t: [0] * NUM_CLASSES[t] for t in TASK_NAMES}
        for _, lb in self.samples:
            for t, idx in lb.items():
                if t in counts:                 # '_src'(다중출처 인덱스)는 등급이 아니다
                    counts[t][idx] += 1
        return counts


def label_index_from_csv(csv_path):
    """atopy_face/labels.csv (등급 '문자열') -> {stem: {task: idx}}.

    atopy JSON 없이도 동작하는 자립형 라벨 인덱스(make_atopy_face.py 산출물 소비).
    CSV 헤더: split, stem, severity, erythema, papulation, excoriation, lichenification.
    """
    import csv as _csv
    from pathlib import Path as _P
    index = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            stem = row.get("stem") or row.get("filename")
            if not stem:
                continue
            stem = _P(stem).stem
            try:
                index[stem] = {t: _LABEL2IDX[t][row[t]] for t in TASK_NAMES}
            except KeyError:
                continue
    return index


SOURCE_SPECS = {
    # 논문 Eq(12)의 '데이터셋'에 해당하는 맥락 축. 합성 AI-Hub 데이터에는 촬영 각도가 있고,
    # 실제로 의존구조가 다르다(train 1400장 Cramer's V):
    #   구진-태선화  정면 0.120 / 측면 0.274  -> 합치면 0.180 으로 희석
    #   중증도-태선화 정면 0.145 / 측면 0.142 -> 합치면 0.071 로 사라짐
    # 즉 태선화의 엣지는 두 맥락을 뭉개는 순간 없어진다. theta 를 출처별로 두면 살아난다.
    "angle": [("front", ["TL_아토피_정면", "VL_아토피_정면"]),
              ("side",  ["TL_아토피_측면", "VL_아토피_측면"])],
}


def build_source_index(atopy_root, spec="angle"):
    """{stem: 출처인덱스}, [출처이름] 반환. 원본 atopy/ 폴더 이름으로 출처를 정한다.

    라벨 CSV 에 source 컬럼이 생기면 그걸 우선 쓰는 게 맞지만, 지금 dataset_all_final 에는
    없고 stem 은 원본 폴더와 1:1 이라 폴더로 역추적한다."""
    root = Path(atopy_root)
    groups = SOURCE_SPECS[spec]
    index, names = {}, [g[0] for g in groups]
    for si, (_, dirs) in enumerate(groups):
        for d in dirs:
            p = root / d
            if not p.is_dir():
                continue
            for jp in p.glob("*.json"):
                index[jp.stem] = si
    return index, names


def build_label_index(atopy_root):
    """atopy 폴더 전체를 훑어 {파일명stem: labels} 인덱스 구성.

    dataset_face / dataset_lesion 처럼 이미지만 있는 폴더의 easi_score 라벨을
    파일명(stem) 매칭으로 조회하기 위한 용도.
    """
    index = {}
    for jp in Path(atopy_root).rglob("*.json"):
        labels = AtopyClsDataset._parse(jp)
        if labels is not None:
            index[jp.stem] = labels
    return index


class ImageDirClsDataset(AtopyClsDataset):
    """이미지 폴더(dataset_face/dataset_lesion 의 images/{train,val})만 주어질 때 사용.

    라벨은 세그멘테이션용 txt 가 아니라 build_label_index 로 만든 atopy JSON
    인덱스(stem->labels)에서 조회한다. 이미지는 atopy 원본과 픽셀 동일한 서브셋.
    """
    def __init__(self, image_dirs, label_index, transform=None):
        self.transform = transform
        self.samples = []
        skipped = 0
        for img_dir in image_dirs:
            for img_path in sorted(Path(img_dir).glob("*.png")):
                lb = label_index.get(img_path.stem)
                if lb is None:
                    skipped += 1
                    continue
                self.samples.append((img_path, lb))
        if skipped:
            print(f"[dataset] 건너뜀(라벨없음): {skipped}")
        print(f"[dataset] 샘플 {len(self.samples)}개")
