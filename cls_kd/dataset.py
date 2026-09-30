"""라벨 인덱스 + 데이터셋 — labels.csv 또는 원본 아토피 JSON 둘 다 소비한다.

CSV 경로(권장):
    split,identifier,severity,erythema,papulation,excoriation,lichenification
    train,H0_107994_P1_L0,Severe,Moderate,Mild,None,None

(과거 실데이터는 이 키 컬럼을 `stem`으로 썼다 — 지금은 `identifier`로 바뀌었지만
`stem`/`filename`도 계속 받는다. 어떤 이름이든 값 자체는 이미지 파일명에서
확장자를 뺀 것과 같아야 한다: `images/train/H0_107994_P1_L0.png` 라면 이 컬럼
값도 `H0_107994_P1_L0`.)

JSON 경로(원본 /home/work/ogw/atopy/*/*.json):
    annotations[0].diagnosis_info.easi_score.{iga_grade,erythema,...}

CSV 의 split 컬럼은 **읽지 않는다**. split 은 폴더 구조(images/{train,val,test})가
정한다 — 두 곳이 어긋나면 조용히 누수가 나므로 진실의 출처를 하나로 못박는다.

폴더 배치는 둘 다 받는다. 아래가 표준형이고, 평평한 <root>/<split>/ 도 허용한다:

    dataset/
      images/{train,valid,test}/    이미지
      masks/{train,valid,test}/     마스크 (--exp bbox)
      labels.csv

split 폴더 이름은 별칭을 받는다(SPLIT_ALIASES) — val/valid/validation 은 같은 것이다.

입력은 두 가지다(`--exp`):
    full : 원본 이미지 전체(마스크 불필요, 기저선)
    bbox : 마스크 union bbox + margin 크롭 1장

bbox 도 **마스킹(배경 0)을 하지 않는다.** 배경을 지우면 병변-정상피부 대비와
질감이 사라져 중증도 판단이 오히려 나빠진다. 마스크는 '어디를 볼지'만 정하고
픽셀은 원본 그대로 가져온다. 마스크가 비면(전경 0px) 전체 이미지로 폴백한다.

빈 칸/모르는 등급은 UNKNOWN(-1)이 되어 그 축만 손실·지표·샘플러에서 빠진다.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from labels import (NUM_CLASSES, TASK_NAMES, UNKNOWN, parse_easi, parse_grade,
                    resolve_columns, resolve_diagnosis_column)


# ----------------------------------------------------------------------------- 경로
# split 폴더 이름의 흔들림 흡수. 코드 안에서 split 은 항상 정규명(train/val/test)이고,
# 디스크에서는 아래 별칭 중 **실제로 있는 폴더**를 쓴다. 'valid' 로 내보내는 도구가
# 흔한데, 그것 때문에 데이터를 통째로 복사하거나 이름을 바꾸게 하고 싶지 않다.
#
# 순서가 곧 우선순위다. 정규명이 먼저라, 같은 루트에 val/ 과 valid/ 가 둘 다 있으면
# val/ 이 이긴다(그런 데이터는 애초에 의심스럽지만, 조용히 고르지는 않게 경고한다).
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val":   ("val", "valid", "validation"),
    "test":  ("test", "testing"),
}


def split_names(split):
    """정규 split 이름 -> 디스크에서 찾아볼 폴더 이름들(우선순위 순)."""
    return SPLIT_ALIASES.get(split, (split,))


def canonical_split(name):
    """디스크 폴더 이름 -> 정규 split 이름. 모르면 그대로 돌려준다."""
    low = str(name).strip().lower()
    for canon, aliases in SPLIT_ALIASES.items():
        if low in aliases:
            return canon
    return low


def resolve_split_dir(root, split):
    """dataset/<split>/ 과 dataset/images/<split>/ 둘 다 허용하고, split 이름은
    별칭(val/valid/validation 등)까지 받는다. 반환한 Path 의 .name 이 실제 폴더명이다.

    마스크도 같은 이름을 써야 하므로(resolve_mask_dir) 호출한 쪽이 .name 을 넘겨 준다.
    """
    root = Path(root)
    cands = [base / nm for nm in split_names(split) for base in (root, root / "images")]
    found = [c for c in cands if c.is_dir()]
    if found:
        if len({c.name for c in found}) > 1:
            print(f"[경고] '{split}' 후보 폴더가 여러 개다: "
                  f"{[str(c) for c in found]} -> {found[0]} 를 쓴다. 같은 split 이 두 이름"
                  f"으로 나뉘어 있으면 일부 이미지가 조용히 빠진다.")
        return found[0]
    tried = "\n         ".join(str(c.resolve()) for c in cands)
    raise FileNotFoundError(
        f"'{split}' 이미지 폴더를 찾지 못함. 탐색한 경로:\n         {tried}\n"
        f"       현재 작업 폴더: {Path.cwd()}\n"
        f"       {_REL_HINT}")


_REL_HINT = ("상대경로를 줬다면 run.sh/run_sweep.sh/evaluate.sh 가 스크립트 폴더로 cd 하므로\n"
             "       기준 위치가 달라진다 — 절대경로로 주는 편이 안전하다.")


def find_labels_csv(root, explicit=None):
    """labels.csv 위치. 못 찾으면 '무엇을 어디서 찾았는지'를 전부 찍고 죽는다.

    이 예외는 --data 가 틀렸을 때 **가장 먼저** 터지는 지점이다(라벨을 split 폴더보다
    먼저 읽으므로). 그래서 메시지가 'labels.csv 가 없다'로만 끝나면, 정작 원인인
    '--data 폴더가 딴 데를 가리킨다'를 놓치게 된다 — 그 둘을 갈라서 알려준다.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"labels.csv 없음: {p} (절대경로 {p.resolve()})")
        return p
    root = Path(root)
    cands = (root / "labels.csv", root / "images" / "labels.csv",
             root / "labels" / "labels.csv", root.parent / "labels.csv")
    for cand in cands:
        if cand.exists():
            return cand
    if not root.is_dir():
        raise FileNotFoundError(
            f"--data 폴더 자체가 없다: {root}\n"
            f"       절대경로로 풀면: {root.resolve()}\n"
            f"       현재 작업 폴더: {Path.cwd()}\n"
            f"       {_REL_HINT}")
    try:
        names = sorted(x.name + ("/" if x.is_dir() else "") for x in root.iterdir())
    except OSError as e:
        names = [f"(읽을 수 없음: {e})"]
    listing = ", ".join(names[:15]) + (" ..." if len(names) > 15 else "")
    tried = "\n         ".join(str(c.resolve()) for c in cands)
    raise FileNotFoundError(
        f"labels.csv 를 찾지 못함. 탐색한 경로:\n         {tried}\n"
        f"       {root.resolve()} 안에 있는 것: {listing}\n"
        f"       파일명이 정확히 'labels.csv' 인지(대소문자 구분) 확인하거나, "
        f"--labels_csv 로 직접 지정할 것.")


def resolve_mask_dir(data_root, masks_arg, split, dir_name=None):
    """(bbox) 마스크 폴더 찾기. --masks 를 주면 그 아래 <split>/ 만 본다.

    비워 두면 자동 탐색한다. 마스크가 데이터셋 **안에** 있는 배치가 흔하기 때문이다:
        dataset/images/{train,val,test}/ 이미지
        dataset/masks/{train,val,test}/  마스크     <- 이 형태를 먼저 찾는다
        dataset/{train,val,test}/        이미지(평평한 배치)
        dataset/../masks/{train,val,test}/          <- seg/precompute_masks.py 기본 산출

    dir_name: 이미지 쪽이 실제로 쓴 폴더 이름(resolve_split_dir(...).name). 이미지가
    valid/ 에 있으면 마스크도 valid/ 를 **먼저** 본다 — 두 쪽을 따로 풀면 이미지는
    valid/, 마스크는 val/ 처럼 엇갈려서 조용히 다른 split 의 마스크를 씌울 수 있다.

    반환: (Path 또는 None, 탐색한 후보 목록).
    """
    root = Path(data_root)
    # 이미지가 쓴 이름을 맨 앞에, 그 뒤에 나머지 별칭을 중복 없이.
    names = ([dir_name] if dir_name else []) + [n for n in split_names(split)
                                                if n != dir_name]
    if masks_arg:
        cands = [Path(masks_arg) / nm for nm in names]
    else:
        cands = [base / nm for nm in names
                 for base in (root / "masks", root / "images" / "masks",
                              root.parent / "masks")]
    for c in cands:
        if c.is_dir():
            return c, cands
    return None, cands


# ----------------------------------------------------------------------------- 라벨
def label_index_from_csv(csv_path, diagnosis_col=None):
    """-> ({stem: {task: idx, 'diagnosis':..., 'source':..., 'subject':...}}, info dict).

    split 컬럼은 읽지 않는다(폴더가 진실의 출처). CSV 한 장에 train/val/test 가 모두
    들어 있어도 그대로 쓰면 된다 — 각 split 폴더의 파일명으로 조회할 뿐이다.
    """
    index = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = resolve_columns(reader.fieldnames)
        dx_col = resolve_diagnosis_column(reader.fieldnames, diagnosis_col)
        missing = [t for t, c in cols.items() if c is None]
        for row in reader:
            stem = (row.get("identifier") or row.get("stem") or row.get("filename") or "").strip()
            if not stem:
                continue
            rec = {t: (parse_grade(t, row.get(cols[t])) if cols[t] else UNKNOWN)
                   for t in TASK_NAMES}
            rec["diagnosis"] = ((row.get(dx_col) or "").strip() if dx_col else "")
            rec["source"] = (row.get("source") or "unknown").strip()
            rec["subject"] = (row.get("subject") or "").strip()
            index[Path(stem).stem] = rec
    info = {"columns": cols, "missing_columns": missing, "diagnosis_column": dx_col,
            "n_rows": len(index), "src": str(csv_path)}
    return index, info


# ----------------------------------------------------------------------------- 진단명 집계
def diagnosis_counts(records):
    """{진단명: 건수}. 값이 비어 있으면 '(빈칸)' 으로 묶는다."""
    out = {}
    for r in records:
        k = (r.get("diagnosis") or "").strip() or "(빈칸)"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def format_diagnosis_counts(records, col):
    """진단명 분포 한 블록. 필터는 없고 **무엇이 얼마나 들어갔는지**만 보여준다.

    행을 고르는 로직이 사라진 뒤에도 이 표는 남긴다 — labels.csv 에 아토피 외 질환이나
    비병변이 섞여 있으면 등급 분포가 통째로 달라지는데, 로그에 안 남으면 나중에 점수
    차이가 방법 때문인지 데이터 구성 때문인지 구분할 수가 없다.
    """
    counts = diagnosis_counts(records)
    if not col:
        return "[dx] 진단명 컬럼 없음 -> 진단명 분포를 찍지 않는다(학습에는 영향 없음)."
    lines = [f"[dx] 진단명 컬럼 '{col}'  (필터 없음 — 전부 학습)"]
    w = max([len(v) for v in counts] + [8])
    for value, cnt in counts.items():
        lines.append(f"[dx]   {value:{w}s} {cnt:6d}")
    lines.append(f"[dx]   -> 합계 {sum(counts.values())} 행")
    return "\n".join(lines)


def labeled_fraction(records):
    """5축 중 하나라도 라벨이 있는 행의 비율. EASI 칸이 빈 행(비병변 등)을 잡는다."""
    if not records:
        return 0.0, 0
    n = sum(1 for r in records if any(int(r[t]) >= 0 for t in TASK_NAMES))
    return n / len(records), n


def label_index_from_json(json_roots):
    """원본 아토피 JSON 폴더들(여러 개) -> 같은 형식의 인덱스.

    stem 은 파일명에서 딴다. 파싱 실패한 축은 UNKNOWN 이 되고, easi_score 자체가
    없으면 그 파일을 통째로 건너뛴다.
    """
    index, skipped = {}, 0
    for root in json_roots:
        for jp in sorted(Path(root).rglob("*.json")):
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
                ann = d["annotations"][0]
                easi = ann["diagnosis_info"]["easi_score"]
            except (KeyError, IndexError, ValueError, OSError):
                skipped += 1
                continue
            rec = parse_easi(easi)
            gp = ann.get("generated_parameters", {}) or {}
            # '출처'는 라벨이 아니라 진단축이다. 촬영 조건별로 지표를 쪼개 봐야
            # 병변 형태를 배웠는지 촬영 조건을 외웠는지 구분할 수 있다.
            rec["diagnosis"] = ""                    # JSON 경로에는 진단명 축이 없다
            rec["source"] = Path(root).name
            rec["subject"] = str(gp.get("age_range") or "")
            index[jp.stem] = rec
    info = {"columns": {t: t for t in TASK_NAMES}, "missing_columns": [],
            "diagnosis_column": None, "n_rows": len(index), "skipped": skipped,
            "src": ", ".join(str(r) for r in json_roots)}
    return index, info


def build_label_index(data_root, labels_csv=None, json_roots=None, diagnosis_col=None):
    """json_roots 가 있으면 JSON 을, 없으면 CSV 를 쓴다."""
    if json_roots:
        return label_index_from_json(json_roots)
    return label_index_from_csv(find_labels_csv(data_root, labels_csv), diagnosis_col)


def describe_labels(index, title=""):
    """축별 라벨 가용성 + 등급 분포. 학습 전에 무엇을 학습할 수 있는지 눈으로 확인."""
    lines = [f"[labels] {title} n={len(index)}"]
    for t in TASK_NAMES:
        col = np.array([r[t] for r in index.values()]) if index else np.zeros(0, int)
        known = int((col >= 0).sum())
        dist = [int((col == k).sum()) for k in range(NUM_CLASSES[t])]
        flag = "  -> 라벨 없음(이 축은 비활성)" if known == 0 else ""
        lines.append(f"  {t:16s}: 라벨있음 {known:5d}  분포 {dist}{flag}")
    return "\n".join(lines)


def class_counts(records):
    """{task: [class0 수, class1 수, ...]} — 클래스 가중 산출용. UNKNOWN 제외."""
    counts = {t: [0] * NUM_CLASSES[t] for t in TASK_NAMES}
    for rec in records:
        for t in TASK_NAMES:
            k = int(rec[t])
            if k >= 0:
                counts[t][k] += 1
    return counts


def class_weights(counts, power=0.5, clip_max=3.0):
    """역빈도^power 가중, 평균 1 로 정규화 후 clip.

    power=1(순수 역빈도)은 이 데이터에서 너무 공격적이다 — IGA 의 'Clear' 가
    train 1400여 장 중 2장이라 가중이 수백 배가 되고 그 2장에 모델이 끌려간다.
    제곱근(power=0.5)으로 눌러 두고 clip_max 로 상한을 건다. power=0 이면 가중 없음.

    주의: --loss corn 경로에서도 이 가중은 살아 있다(corn_loss 가 등급별 weight 를
    각 이진 문제에 실어 준다). 즉 IBB 와 클래스 가중이 **둘 다** 불균형을 건드리므로,
    IBB 효과만 보려면 --class_weight_power 0 으로 두고 대조하는 편이 깨끗하다.
    """
    out = {}
    for t, c in counts.items():
        c = np.asarray(c, dtype=np.float64)
        present = c > 0
        w = np.ones_like(c)
        if present.any() and power > 0:
            inv = np.where(present, c.sum() / np.maximum(c, 1.0), 0.0) ** power
            # 없는 클래스는 가중 1 로 둔다. 역빈도를 그대로 두면 1/0 이 폭주해서
            # 나머지 클래스 가중을 전부 0 으로 만들어 버린다.
            mean = inv[present].mean() if present.any() else 1.0
            w = np.where(present, inv / max(mean, 1e-9), 1.0)
            w = np.clip(w, 1.0 / clip_max, clip_max)
        out[t] = w.astype(np.float32)
    return out


# ----------------------------------------------------------------------------- 데이터셋
def _bbox_from_mask(mask, margin):
    """mask (H,W) bool -> (x0,y0,x1,y1) union bbox, 각 변을 margin(비율)만큼 확장.
    빈 마스크면 None(=전체 이미지)."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))


def _square_crop(im, box):
    """box 를 긴 변 기준 정사각으로 넓혀 크롭한다(--bbox_square).

    왜 필요한가: 뒤의 T.Resize((s,s)) 는 비등방(anisotropic)이라 종횡비가 1 이 아닌
    크롭을 찌그러뜨린다. 구진이 타원이 되고 찰상의 방향·길이가 바뀐다. 크롭을 미리
    정사각으로 만들어 두면 같은 Resize 가 등방 변환이 되어 모양이 보존된다.

    이미지 안에서 평행이동으로 채울 수 있으면 실제 픽셀만 쓴다(패딩 0). 원본이
    정사각이 아니어서 창이 넘칠 때만 모자란 축을 패딩한다 — 레터박스와 달리
    '있는 픽셀은 쓰고 없는 만큼만' 채우므로 단순 레터박스보다 나쁠 수 없다.

    패딩은 edge 복제다. 검은 띠보다 경계 인공물이 적고, reflect 와 달리 병변을
    복제하지 않는다 — 구진/찰상 등급이 개수·범위 기준이라 복제는 등급을 부풀린다.
    """
    W, H = im.size
    x0, y0, x1, y1 = box
    s = int(round(max(x1 - x0, y1 - y0)))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ix0 = int(round(min(max(cx - s / 2.0, 0), max(W - s, 0))))
    iy0 = int(round(min(max(cy - s / 2.0, 0), max(H - s, 0))))
    ix1, iy1 = ix0 + s, iy0 + s
    if ix0 >= 0 and iy0 >= 0 and ix1 <= W and iy1 <= H:
        return im.crop((ix0, iy0, ix1, iy1))          # 패딩 불필요 — 실제 픽셀만
    cx0, cy0 = max(ix0, 0), max(iy0, 0)
    cx1, cy1 = min(ix1, W), min(iy1, H)
    arr = np.asarray(im.crop((cx0, cy0, cx1, cy1)))
    pad = ((max(0, cy0 - iy0), max(0, iy1 - cy1)),
           (max(0, cx0 - ix0), max(0, ix1 - cx1)), (0, 0))
    return Image.fromarray(np.pad(arr, pad, mode="edge"))


def _load_mask(mask_path, size):
    """마스크 PNG -> bool ndarray (H,W). 없으면 전부 False(전체 폴백).
    size 는 PIL 규약의 (w, h)."""
    if mask_path is None or not Path(mask_path).exists():
        return np.zeros((size[1], size[0]), dtype=bool)
    m = Image.open(mask_path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return np.asarray(m) > 127


class SevDataset(Dataset):
    """이미지 폴더(+선택적 마스크 폴더) -> (tensor, {task: 등급}).

    mask_dir=None 이면 항상 전체 이미지("full"). 주어지면 union bbox 크롭("bbox").
    """

    def __init__(self, image_dir, mask_dir, label_index, transform=None, margin=0.15,
                 square=False):
        self.transform = transform
        self.margin = margin
        self.square = bool(square)
        self.samples = []
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir) if mask_dir else None
        skipped, no_mask = 0, 0
        exts = ("*.png", "*.jpg", "*.jpeg")
        for img_path in sorted(p for e in exts for p in image_dir.glob(e)):
            rec = label_index.get(img_path.stem)
            if rec is None:
                skipped += 1
                continue
            mp = None
            if mask_dir:
                # 마스크는 .png 로 저장되므로 확장자가 다를 수 있다 -> stem 으로도 찾는다.
                for cand in (mask_dir / img_path.name, mask_dir / (img_path.stem + ".png")):
                    if cand.exists():
                        mp = cand
                        break
                if mp is None:
                    no_mask += 1
            self.samples.append((img_path, mp, rec))
        if skipped:
            print(f"[dataset] 건너뜀(labels.csv 에 없거나 진단 필터로 제외됨): {skipped}")
        if no_mask:
            print(f"[dataset] 마스크 없음 {no_mask}장 -> 전체 이미지로 폴백")
        print(f"[dataset] {image_dir.name:5s}: 샘플 {len(self.samples):5d}"
              f"  ({'bbox 크롭' if mask_dir else '전체 이미지'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, mask_path, rec = self.samples[i]
        im = Image.open(img_path).convert("RGB")
        src_side = (im.size[0] * im.size[1]) ** 0.5
        if mask_path is not None:                  # 없으면(=full, 또는 마스크 결측) 전체 이미지
            box = _bbox_from_mask(_load_mask(mask_path, im.size), self.margin)
            if box is not None:                    # 전경 0px -> 전체 폴백
                im = _square_crop(im, box) if self.square else im.crop(box)
        # 절대 배율: 크롭이 원본의 몇 분의 몇인가. log2 라 0=전체프레임, -1=선형 1/2.
        # Resize 가 모든 크롭을 같은 imgsz 로 늘리므로 이 값이 사라진다 — 등급이
        # 병변의 '크기'에 달린 축(구진/태선화)에서는 이게 곧 정보 손실이다.
        crop_side = (im.size[0] * im.size[1]) ** 0.5
        scale = math.log2(max(crop_side, 1.0) / max(src_side, 1.0))
        if self.transform:
            im = self.transform(im)
        return im, {t: int(rec[t]) for t in TASK_NAMES}, {"scale": scale}

    def records(self):
        return [r for _, _, r in self.samples]

    def label_arrays(self):
        """{task: np.ndarray(int64)} — IBB 샘플러 입력. UNKNOWN(-1) 이 그대로 들어간다."""
        return {t: np.array([r[t] for _, _, r in self.samples], dtype=np.int64)
                for t in TASK_NAMES}

    def meta(self, key):
        return [(r.get(key) or "unknown") for _, _, r in self.samples]


def collate(batch):
    """-> (imgs, labels, meta). meta 는 모델 입력이 아닌 부가 정보다(현재 'scale' 하나).

    3-튜플인 이유: 크롭 배율은 라벨도 이미지도 아니라서 둘 중 어디에 얹어도
    의미가 어긋난다. 쓰지 않는 경로는 그냥 meta 를 무시하면 된다.
    """
    imgs = torch.stack([b[0] for b in batch])
    labels = {t: torch.tensor([b[1][t] for b in batch], dtype=torch.long)
              for t in TASK_NAMES}
    meta = {"scale": torch.tensor([b[2]["scale"] for b in batch], dtype=torch.float32)}
    return imgs, labels, meta
