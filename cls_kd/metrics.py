"""순서형 등급 지표 — QWK 가 주 지표다.

정확도만 보면 안 되는 이유: 등급은 순서가 있어서 'Severe 를 Moderate 로' 틀린 것과
'Severe 를 Clear 로' 틀린 것의 무게가 전혀 다르다. QWK 는 오차의 제곱으로 벌점을
주므로 이 차이를 반영하고, 우연 일치도 보정한다. acc1(±1 이내)은 임상에서 흔히
쓰는 '한 등급 오차는 허용' 관점이라 같이 본다.

UNKNOWN(-1) 라벨은 모든 지표에서 제외된다.
"""
import numpy as np

from labels import IGA_TASK, NUM_CLASSES, SIGN_TASKS, TASK_NAMES


def quadratic_weighted_kappa(y_true, y_pred, num_classes):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if len(yt) == 0:
        return 0.0
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(yt, yp):
        O[t, p] += 1
    if O.sum() == 0:
        return 0.0
    i, j = np.indices((num_classes, num_classes))
    w = ((i - j) ** 2) / ((num_classes - 1) ** 2 + 1e-9)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (w * E).sum()
    if den < 1e-9:            # 전부 한 클래스면 정의 불가 -> NaN 대신 0
        return 0.0
    return float(1 - (w * O).sum() / den)


def macro_f1(y_true, y_pred, num_classes):
    """클래스별 F1 의 단순 평균. 희소 등급이 통째로 버려졌는지 드러난다."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    f1s = []
    for k in range(num_classes):
        tp = int(((yt == k) & (yp == k)).sum())
        fp = int(((yt != k) & (yp == k)).sum())
        fn = int(((yt == k) & (yp != k)).sum())
        if tp + fn == 0:      # 정답에 없는 클래스는 평균에서 제외
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def per_task_metrics(y_true, y_pred, num_classes):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if len(yt) == 0:
        return {"n": 0, "qwk": 0.0, "acc": 0.0, "acc1": 0.0, "mae": 0.0, "f1": 0.0}
    return {
        "n": int(len(yt)),
        "qwk": quadratic_weighted_kappa(yt, yp, num_classes),
        "acc": float((yt == yp).mean()),
        "acc1": float((np.abs(yt - yp) <= 1).mean()),   # 한 등급 이내
        "mae": float(np.abs(yt - yp).mean()),
        "f1": macro_f1(yt, yp, num_classes),
    }


def evaluate_all(y_true, y_pred):
    """y_true/y_pred: {task: array}. UNKNOWN(-1) 제외 후 태스크별 지표."""
    out = {}
    for t in TASK_NAMES:
        yt = np.asarray(y_true[t], dtype=int)
        yp = np.asarray(y_pred[t], dtype=int)
        m = yt >= 0
        out[t] = per_task_metrics(yt[m], yp[m], NUM_CLASSES[t])
    return out


def selection_score(metrics):
    """모델 선택 점수 = 0.5*IGA QWK + 0.5*(증상 4종 QWK 평균).

    단순 5개 평균이 아닌 이유: 증상이 4개고 IGA 가 1개라 평균을 내면 IGA 비중이
    1/5 로 눌린다. IGA 는 최종 임상 판단에 해당하는 축이라 절반을 준다.
    라벨이 없어 n=0 인 축은 평균에서 뺀다 — 0.0 으로 세면 그 축이 점수를 끌어내려
    멀쩡한 모델이 버려진다.
    """
    def q(t):
        m = metrics.get(t, {})
        return m["qwk"] if m.get("n", 0) > 0 else None

    iga = q(IGA_TASK)
    signs = [v for v in (q(t) for t in SIGN_TASKS) if v is not None]
    sign_mean = float(np.mean(signs)) if signs else None
    if iga is None and sign_mean is None:
        return 0.0
    if iga is None:
        return sign_mean
    if sign_mean is None:
        return iga
    return 0.5 * iga + 0.5 * sign_mean


def mean_qwk(metrics):
    vals = [m["qwk"] for m in metrics.values() if m.get("n", 0) > 0]
    return float(np.mean(vals)) if vals else 0.0


def format_metrics(metrics, prefix=""):
    lines = []
    for t in TASK_NAMES:
        m = metrics[t]
        lines.append(f"{prefix}{t:16s} n={m['n']:4d}  QWK {m['qwk']:.4f}  "
                     f"acc {m['acc']:.3f}  ±1 {m['acc1']:.3f}  "
                     f"MAE {m['mae']:.3f}  F1 {m['f1']:.3f}")
    lines.append(f"{prefix}{'선택점수':16s} {selection_score(metrics):.4f}  "
                 f"(mean QWK {mean_qwk(metrics):.4f})")
    return "\n".join(lines)


def confusion(y_true, y_pred, num_classes):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    m = yt >= 0
    C = np.zeros((num_classes, num_classes), dtype=int)
    for a, b in zip(yt[m], yp[m]):
        C[a, b] += 1
    return C


def format_confusion(cm, class_names, title=""):
    """행=정답, 열=예측. 오른쪽에 등급별 recall 을 붙인다.

    순서형 혼동행렬은 **대각선만 보면 안 된다.** 오답이 대각선 바로 옆(±1)에 몰려
    있는지 멀리 흩어져 있는지가 QWK 를 가르는 지점이고, 그건 행렬 모양으로만 보인다.
    한 열에 전부 쏠려 있으면 그 축은 최빈등급만 찍고 있는 것이다(acc 는 높게 나온다).
    """
    w = max(max(len(c) for c in class_names), 6)
    cw = max(9, w + 1)
    lines = []
    if title:
        lines.append(title)
    lines.append(" " * (w + 2) + "".join(f"{c:>{cw}s}" for c in class_names)
                 + f"{'n':>7s}{'recall':>9s}")
    for i, c in enumerate(class_names):
        row = cm[i]
        tot = int(row.sum())
        rec = row[i] / tot if tot else 0.0
        lines.append(f"{c:>{w}s} |" + "".join(f"{int(v):{cw}d}" for v in row)
                     + f"{tot:7d}{rec:9.3f}")
    lines.append(" " * (w + 2) + "".join(f"{int(v):{cw}d}" for v in cm.sum(0))
                 + f"{int(cm.sum()):7d}   <- 예측 합")
    return "\n".join(lines)


def mean_acc(metrics):
    """축별 정확도의 단순 평균. 라벨이 없어 n=0 인 축은 뺀다."""
    vals = [m["acc"] for m in metrics.values() if m.get("n", 0) > 0]
    return float(np.mean(vals)) if vals else 0.0
