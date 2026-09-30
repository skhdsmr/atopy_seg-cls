"""질환 6-way 단일라벨 지표 — 혼동행렬 / 클래스별 P·R·F1 / macro-AUROC / 출처별 분해.

classification_topk/metrics.py 에서 가져오지 않고 다시 쓴 이유:
    거기 지표는 멀티라벨(threshold 탐색, BP-MLL, 이미지별 MAP)과 순서형(QWK)에 맞춰져
    있다. 질환은 배타적이고 순서가 없어서 threshold 도 등급거리도 의미가 없다.
    argmax 하나로 예측이 확정되므로 필요한 건 혼동행렬 계열이다.

주의: 정확도(accuracy)만 보지 말 것. 클래스가 균형(각 900장)이라 accuracy 가 그럴듯해
보이기 쉽지만, 이 데이터는 출처(prefix)가 질환과 강하게 상관돼 있어서 모델이 촬영
출처만 보고도 점수를 올릴 수 있다. group_accuracy 로 출처별로 쪼개 보는 게 핵심이다.
"""
import numpy as np

NAN = float("nan")


def confusion(y_true, y_pred, k):
    """(k,k) 행렬. 행=정답, 열=예측."""
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        cm[t, p] += 1
    return cm


def prf(y_true, y_pred):
    """이진 (precision, recall, f1). 양성 예측/정답이 없으면 해당 항을 0 으로 둔다.
    classification_topk/metrics.py 와 동일 정의."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def auroc(y_true, scores):
    """Mann-Whitney U 기반 ROC-AUC(동점은 평균 순위).
    classification_topk/metrics.py 와 동일 정의."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return NAN
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=float)
    r = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(sorted_s):                       # 동점 구간은 평균 순위로
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = r
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_class_metrics(y_true, y_pred, scores, classes):
    """{클래스: {precision, recall, f1, auroc, support}} — one-vs-rest."""
    out = {}
    for i, c in enumerate(classes):
        yt = (np.asarray(y_true) == i).astype(int)
        yp = (np.asarray(y_pred) == i).astype(int)
        p, r, f = prf(yt, yp)
        out[c] = {"precision": p, "recall": r, "f1": f,
                  "auroc": auroc(yt, scores[:, i]) if scores is not None else NAN,
                  "support": int(yt.sum())}
    return out


def summarize(y_true, y_pred, scores, classes):
    """accuracy / balanced_accuracy / macro-F1 / macro-AUROC.

    balanced_accuracy = 클래스별 recall 의 평균. 클래스가 균형이면 accuracy 와 거의
    같지만, 이미지 다운로드가 덜 끝나 클래스 수가 어긋난 동안에도 정직한 값을 준다.
    무작위 기저선은 accuracy/balanced_accuracy 둘 다 1/K, macro-AUROC 는 0.5 다.

    macro 계열은 모두 '그 split 에 실제로 존재하는 클래스'(support>0)만 평균한다.
    DISEASES 는 6개로 고정인데 다운로드가 덜 끝나 4개만 있는 동안 없는 클래스의 f1=0
    까지 평균에 넣으면 macro_f1 이 4/6 배로 눌려서 절대값을 못 읽는다. 정답이 하나도
    없는 클래스는 '못 맞힌 것'이 아니라 '물어보지 않은 것'이다.
    주의: test 에는 있는데 train 에 없는 클래스(현재 주사/지루)는 support>0 이라 f1=0
    이 그대로 잡힌다 — 이건 의도한 것이다. 실제로 못 맞히는 게 맞다.
    """
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    pc = per_class_metrics(y_true, y_pred, scores, classes)
    present = [c for c in classes if pc[c]["support"] > 0]
    aurocs = [pc[c]["auroc"] for c in present if not np.isnan(pc[c]["auroc"])]
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else NAN,
        "balanced_accuracy": (float(np.mean([pc[c]["recall"] for c in present]))
                              if present else NAN),
        "macro_f1": float(np.mean([pc[c]["f1"] for c in present])) if present else NAN,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else NAN,
        "n": int(len(y_true)),
        "n_classes_present": len(present),
    }


def group_accuracy(y_true, y_pred, groups):
    """{그룹: {acc, n}} — 출처(prefix)나 각도별로 정확도를 쪼갠다.

    이 데이터에서 가장 중요한 진단이다. 예를 들어 test 주사는 전부 H1, 여드름은 99%가
    H2 라서, 출처별 정확도가 고르지 않고 특정 출처에서만 높다면 모델이 질환이 아니라
    촬영 조건을 외웠다는 신호다.
    """
    out = {}
    for g in sorted(set(groups)):
        m = np.array([x == g for x in groups])
        out[g] = {"acc": float((np.asarray(y_true)[m] == np.asarray(y_pred)[m]).mean()),
                  "n": int(m.sum())}
    return out


def format_confusion(cm, classes):
    w = max(len(c) for c in classes) + 1
    lines = [" " * (w + 2) + "".join(f"{c:>7s}" for c in classes) + f"{'recall':>9s}"]
    for i, c in enumerate(classes):
        row = cm[i]
        rec = row[i] / row.sum() if row.sum() else 0.0
        lines.append(f"{c:>{w}s} |" + "".join(f"{v:7d}" for v in row) + f"{rec:9.3f}")
    lines.append(" " * (w + 2) + "".join(f"{v:7d}" for v in cm.sum(0)) + "   <- 예측 합")
    return "\n".join(lines)


def format_report(per_class, summary, classes, title=""):
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'질환':8s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AUROC':>7s} {'n':>6s}")
    for c in classes:
        m = per_class[c]
        lines.append(f"{c:8s} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
                     f"{m['auroc']:7.3f} {m['support']:6d}")
    lines.append(f"{'-' * 46}")
    n_present = summary.get("n_classes_present", len(classes))
    lines.append(f"acc={summary['accuracy']:.3f}  bal_acc={summary['balanced_accuracy']:.3f}  "
                 f"macroF1={summary['macro_f1']:.3f}  macroAUROC={summary['macro_auroc']:.3f}  "
                 f"n={summary['n']}")
    if n_present < len(classes):
        # macro 는 이 n_present 개만 평균한 값이다. 6클래스 런과 직접 비교하지 말 것.
        lines.append(f"  주의: 정답이 존재하는 클래스 {n_present}/{len(classes)}개 "
                     f"-> macro 지표는 그 {n_present}개만 평균한 값")
    return "\n".join(lines)


def format_groups(g, title):
    """출처/각도별 정확도 한 줄 표. 편차가 크면 지름길 학습을 의심할 것."""
    lines = [title]
    for k, v in sorted(g.items(), key=lambda x: -x[1]["n"]):
        lines.append(f"  {k:8s} acc={v['acc']:.3f}  n={v['n']:5d}")
    accs = [v["acc"] for v in g.values() if v["n"] >= 20]
    if len(accs) >= 2:
        lines.append(f"  -> n>=20 그룹 간 정확도 편차: {min(accs):.3f} ~ {max(accs):.3f} "
                     f"(폭 {max(accs) - min(accs):.3f})")
    return "\n".join(lines)
