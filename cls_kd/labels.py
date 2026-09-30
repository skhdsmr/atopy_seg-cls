"""중증도 라벨 정의 — EASI 계열 5개 축, 전부 순서형(ordinal).

OCNN/OCNN-IT 가 성립하려면 축이 (a) 순서형이고 (b) 서로 상관돼 있어야 한다.
이 5축은 둘 다 만족한다 — IGA 는 나머지 4개 증상 등급을 종합한 총평이라
구조적으로 상관이 강하고, 증상끼리도 같은 병기에서 함께 나타난다. 그 상관을
샘플러(Step 2)가 쓰는 것이 IT(Iterative Training)의 핵심이다.

  iga_grade       : 5등급 (Clear < Almost Clear < Mild < Moderate < Severe)
  erythema        : 4등급 (None < Mild < Moderate < Severe)  홍반
  papulation      : 4등급                                     구진
  excoriation     : 4등급                                     찰상
  lichenification : 4등급                                     태선화

증상을 '있다/없다' 이진으로 뭉개면 안 된다. EASI 는 증상마다 0~3 을 매기므로
이진화하면 '약한 홍반'과 '심한 홍반'이 같은 라벨이 되어 정보가 통째로 날아가고,
그와 동시에 Step 2 가 쓰는 등급조합 셀도 4x4 에서 2x2 로 붕괴한다.

태스크 이름은 원본 JSON 의 easi_score 키와 **일부러 똑같이** 맞췄다(도메인 ckpt 의
헤드 이름과도 같다 — 이름을 맞춰 두면 헤드까지 그대로 이식된다).

인덱스는 등급 오름차순이고 ckpt 에 박히므로 **순서를 바꾸지 말 것**.
"""

# IGA 등급 이름. 기본은 5등급이고, SPEC 모드에서는 4등급으로 병합된다(set_iga_merge).
IGA_LEVELS_5 = ["Clear", "Almost Clear", "Mild", "Moderate", "Severe"]
IGA_LEVELS_MERGED = ["Clear/AlmostClear", "Mild", "Moderate", "Severe"]

# 태스크 -> 등급 이름(오름차순). 인덱스가 곧 등급 값이다.
TASKS = {
    "iga_grade":       list(IGA_LEVELS_5),
    "erythema":        ["None", "Mild", "Moderate", "Severe"],
    "papulation":      ["None", "Mild", "Moderate", "Severe"],
    "excoriation":     ["None", "Mild", "Moderate", "Severe"],
    "lichenification": ["None", "Mild", "Moderate", "Severe"],
}
TASK_NAMES = list(TASKS)
NUM_CLASSES = {t: len(v) for t, v in TASKS.items()}

# 등급축(IGA)과 증상축을 분리해서 읽어야 한다. 전체 평균 하나로 뭉치면 'IGA 는
# 잘 맞추는데 증상은 전부 최빈값' 같은 붕괴가 안 보인다.
IGA_TASK = "iga_grade"
SIGN_TASKS = [t for t in TASK_NAMES if t != IGA_TASK]

UNKNOWN = -1

# --- IGA 4등급 병합 (SPEC.md §1.2, 기본 꺼짐) ---------------------------------
# SPEC 은 5축 전부 4등급 {0,1,2,3} 을 요구한다. IGA 만 원래 5등급(IGA 0-4)이므로
# 'Clear' 와 'Almost Clear' 를 한 칸으로 합친다: 새 인덱스 = max(0, 원 인덱스 - 1).
#
# 왜 병합인가 — 레포 데이터의 'Clear' 는 train 2장 / val 0장이다. val 에 support 가
# 아예 없는 클래스는 QWK 를 흔들기만 하고 배울 수도 없다. 병합이 그 문제까지 같이
# 없앤다(SPEC.md §1.2 표).
#
# 대가: 병합 후에는 'IGA >= max(증상)' 제약이 79% 로 깨진다. 의미 축의 원점이 1만큼
# 어긋나기 때문이다 — 그래서 SPEC 은 제약에 오프셋 +1 을 넣는다(§1.3, spec.py 참조).
#
# 이 함수는 TASKS/NUM_CLASSES 를 **제자리에서** 고친다. 두 dict 는 다른 모듈이
# `from labels import NUM_CLASSES` 로 객체째 붙들고 있으므로 재할당이 아니라 제자리
# 수정이라야 전파된다. 반드시 데이터셋/모델을 만들기 **전에** 한 번만 부를 것.
_STATE = {"iga_merge": False}


def set_iga_merge(on):
    """IGA 를 4등급으로 병합할지 정한다. 데이터셋 생성 전에 호출."""
    _STATE["iga_merge"] = bool(on)
    TASKS[IGA_TASK] = list(IGA_LEVELS_MERGED if on else IGA_LEVELS_5)
    NUM_CLASSES[IGA_TASK] = len(TASKS[IGA_TASK])


def iga_merge_enabled():
    return _STATE["iga_merge"]


# 진단명 컬럼. 학습 대상을 고르는 데는 **쓰지 않는다** — labels.csv 의 모든 행을
# 그대로 학습한다. 이 컬럼은 로그/리포트에 "무엇이 얼마나 들어갔는지"를 남기는
# 용도로만 읽는다(train.py 의 [dx] 줄, test_report.json 의 train_diagnosis).
DIAGNOSIS_ALIASES = ["diagnosis", "disease", "dx", "진단", "진단명", "질환"]

# CSV 컬럼명 흔들림 흡수. dataset_all_final/images/labels.csv 는 IGA 를 'severity'
# 로 쓰는데 JSON 은 'iga_grade' 다 — 둘 다 받는다.
_COLUMN_ALIASES = {
    "iga_grade": ["iga_grade", "severity", "iga", "igagrade"],
    "erythema": ["erythema", "홍반"],
    "papulation": ["papulation", "구진"],
    "excoriation": ["excoriation", "찰상"],
    "lichenification": ["lichenification", "태선화"],
}

# 등급 문자열 -> 인덱스. 표기 흔들림(대소문자/공백/언더스코어/숫자/한글)을 흡수한다.
_GRADE_ALIASES = {
    "iga_grade": {
        "clear": 0, "0": 0, "없음": 0,
        "almostclear": 1, "almost clear": 1, "almost_clear": 1, "1": 1, "거의없음": 1,
        "mild": 2, "2": 2, "경증": 2,
        "moderate": 3, "3": 3, "중등증": 3,
        "severe": 4, "4": 4, "중증": 4,
    },
    "_sign": {
        "none": 0, "absent": 0, "0": 0, "없음": 0,
        "mild": 1, "1": 1, "경증": 1,
        "moderate": 2, "2": 2, "중등증": 2,
        "severe": 3, "3": 3, "중증": 3,
    },
}


def _norm(s):
    return str(s).strip().lower().replace("-", " ").replace("_", " ").replace("  ", " ")


def parse_grade(task, value):
    """등급 문자열/숫자 -> 인덱스. 비었거나 모르는 값이면 UNKNOWN(-1).

    UNKNOWN 을 예외로 던지지 않고 값으로 흘리는 이유: 일부 축만 라벨된 데이터를
    정상 경로로 받기 위해서다. 손실·지표·IBB 샘플러 셋 다에서 그 축만 빠진다.
    """
    if value is None:
        return UNKNOWN
    v = _norm(value)
    if v in ("", "nan", "none_", "unknown", "-1", "na"):
        return UNKNOWN
    table = _GRADE_ALIASES["iga_grade" if task == IGA_TASK else "_sign"]
    # 'none' 은 증상축에서는 등급 0 이지만 IGA 에는 그런 등급이 없다 -> UNKNOWN 유지.
    key = v.replace(" ", "") if v.replace(" ", "") in table else v
    idx = table.get(key, UNKNOWN)
    # 병합은 5등급 표로 읽은 **뒤에** 적용한다. 표 자체를 갈아끼우면 'Clear' 와
    # 'Almost Clear' 가 같은 키로 충돌해 원 등급을 되짚을 수 없게 된다.
    if idx >= 0 and task == IGA_TASK and _STATE["iga_merge"]:
        idx = max(0, idx - 1)
    return idx


def resolve_columns(fieldnames):
    """CSV 헤더 -> {task: 실제 컬럼명 또는 None}."""
    have = {_norm(c): c for c in (fieldnames or [])}
    out = {}
    for task, aliases in _COLUMN_ALIASES.items():
        out[task] = next((have[_norm(a)] for a in aliases if _norm(a) in have), None)
    return out


def resolve_diagnosis_column(fieldnames, explicit=None):
    """CSV 헤더 -> 진단명 컬럼 이름(없으면 None)."""
    have = {_norm(c): c for c in (fieldnames or [])}
    for a in ([explicit] if explicit else []) + DIAGNOSIS_ALIASES:
        if a and _norm(a) in have:
            return have[_norm(a)]
    return None


def parse_easi(easi):
    """JSON 의 easi_score dict -> {task: idx}. 키 이름이 태스크명과 동일하다."""
    return {t: parse_grade(t, (easi or {}).get(t)) for t in TASK_NAMES}
