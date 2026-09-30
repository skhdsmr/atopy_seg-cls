"""
순서형(ordinal) 등급 분류 지표.

- 이 태스크는 등급이 순서(Clear<Mild<Moderate<Severe)이고 클래스 불균형이 크다.
  단순 정확도는 (a) off-by-one(거의 맞음)을 완전 오답과 동일 취급, (b) 다수 클래스에
  좌우돼 오해를 준다. 그래서 QWK 를 주 지표로 쓴다.
- QWK(Quadratic Weighted Kappa): 의료 중증도 등급의 표준. 예측이 정답에서 멀수록
  (제곱) 벌점, 우연 일치를 보정. 1=완벽, 0=우연 수준, 음수=우연보다 나쁨.
- 함께: exact(정확), acc±1(한 등급 이내), MAE(평균 등급 오차).
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
    act = O.sum(1)
    pred = O.sum(0)
    E = np.outer(act, pred) / O.sum()
    den = (w * E).sum()
    if den < 1e-9:                       # 모두 한 클래스면 정의 불가
        return 0.0
    return float(1 - (w * O).sum() / den)


def per_task_metrics(y_true, y_pred, num_classes):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    return {
        "qwk":  quadratic_weighted_kappa(yt, yp, num_classes),
        "acc":  float((yt == yp).mean()),
        "acc1": float((np.abs(yt - yp) <= 1).mean()),   # 한 등급 이내
        "mae":  float(np.abs(yt - yp).mean()),
    }
