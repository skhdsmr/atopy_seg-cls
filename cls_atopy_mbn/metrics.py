"""순서형 5축 등급 지표.

- 주 지표는 QWK. 등급이 순서형이고 클래스 불균형이 커서 단순 정확도는 off-by-one 을
  완전 오답과 같이 취급하고 다수 클래스에 좌우된다.
- **선택점수 = 0.5*IGA QWK + 0.5*(증상 4축 QWK 평균)** — 5축 단순평균이면 증상이 4개라
  IGA 비중이 1/5 로 눌린다(`cls_atopy_sev` 와 동일 규약).
- joint_ac 는 논문 Eq(13)의 5축 확장(전 축 동시 정답률). 5축이면 값이 매우 낮아
  모델 선택 신호로는 노이즈라 **참고지표로만** 찍는다.
"""
import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, num_classes):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(yt, yp):
        O[t, p] += 1
    if O.sum() == 0:
        return 0.0
    w = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2 + 1e-9)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (w * E).sum()
    if den < 1e-9:                       # 전부 한 클래스면 정의 불가
        return 0.0
    return float(1 - (w * O).sum() / den)


def confusion(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_task_metrics(y_true, y_pred, num_classes):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    return {
        "qwk":  quadratic_weighted_kappa(yt, yp, num_classes),
        "acc":  float((yt == yp).mean()),
        "acc1": float((np.abs(yt - yp) <= 1).mean()),       # 한 등급 이내
        "mae":  float(np.abs(yt - yp).mean()),
        "cm":   confusion(yt, yp, num_classes).tolist(),
    }


def all_metrics(trues, preds, num_classes, task_names):
    m = {t: per_task_metrics(trues[t], preds[t], num_classes[t]) for t in task_names}
    arr = np.stack([np.asarray(trues[t]) == np.asarray(preds[t]) for t in task_names])
    m["joint_ac"] = float(arr.all(axis=0).mean())            # 논문 Eq(13)의 5축 확장
    m["score"] = sel_score(m, task_names)
    m["mean_qwk"] = float(np.mean([m[t]["qwk"] for t in task_names]))
    return m


def sel_score(m, task_names):
    sym = [t for t in task_names if t != "severity"]
    return 0.5 * m["severity"]["qwk"] + 0.5 * float(np.mean([m[t]["qwk"] for t in sym]))


def format_confusion(cm, name, levels=None):
    cm = np.asarray(cm)
    n = cm.shape[0]
    head = levels or [str(i) for i in range(n)]
    head = [h[:6] for h in head]
    lines = [f"  [{name}] 행=정답 열=예측",
             "        " + " ".join(f"{h:>6}" for h in head)]
    for i in range(n):
        lines.append(f"  {head[i]:>6}" + " ".join(f"{cm[i, j]:6d}" for j in range(n)))
    return "\n".join(lines)
