"""TSTR 결과표 — 합성만으로 학습한 모델이 실제 학습 모델의 몇 %를 따라오는가.

읽는 것
    runs/dis_effb0_r512_front/test_report.json        TRTR 기준선 (실제 4,200장 학습)
    runs/tstr_effb0_r512_<arm>_<A|B>/test_report.json TSTR 각 런
    runs/tstr_effb0_r512_<arm>_B/eval_test_best_front.json
                                                      B 런을 원본 test 600장만으로 재평가

비율은 프로토콜 A 에서만 계산한다
    A 의 test 는 TRTR 기준선이 쓴 test 600장과 **완전히 같은 집합**이라 나눗셈이
    성립한다. B 의 test(2,698장)에는 원래 train 이던 실제 이미지가 섞여 있어서,
    그 이미지로 학습한 TRTR 모델과는 애초에 같은 잣대로 잴 수 없다(누출).
    그래서 B 런도 원본 test 600장으로 한 번 더 재서 그 값으로 비율을 낸다.
    B 자체의 2,698장 숫자는 '더 넓고 분산이 작은 TSTR 절대 성능'으로만 읽는다.

B 의 test 를 쪼개 봐야 하는 이유
    옮겨온 실제 train 은 학습 출처(H0/H1 위주)와 같은 분포고, 원래 test 는 새 출처다.
    합쳐 평균 내면 쉬운 쪽이 4.5배 많아 절대 수치가 올라간다. source 끝에 '*' 가
    붙은 것이 옮겨온 것이라 두 덩어리로 갈라 같이 적는다.

사용법:
    python3 tstr_report.py
    python3 tstr_report.py --baseline dis_effb0_r512_front
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
ARMS = [("gan", "GAN"), ("sdft", "SD-전체FT")]
METRICS = [("accuracy", "acc"), ("macro_f1", "macro-F1"), ("macro_auroc", "macro-AUROC")]


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def split_by_moved(by_source):
    """{source: {acc,n}} -> (원본 test, 옮긴 train) 각각 (가중평균 acc, n)."""
    out = {}
    for moved in (False, True):
        sel = [(v["acc"], v["n"]) for k, v in by_source.items()
               if k.endswith("*") == moved]
        n = sum(c for _, c in sel)
        out[moved] = ((sum(a * c for a, c in sel) / n if n else float("nan")), n)
    return out[False], out[True]


def verdict(ratio):
    if ratio != ratio:
        return "-"
    return "매우 높음" if ratio >= 0.90 else ("높음" if ratio >= 0.85 else "미달")


def main():
    ap = argparse.ArgumentParser(description="TSTR 결과표",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--baseline", default="dis_effb0_r512_front")
    ap.add_argument("--prefix", default="tstr_effb0_r512")
    args = ap.parse_args()

    base = load(RUNS / args.baseline / "test_report.json")
    if base is None:
        raise SystemExit(f"TRTR 기준선이 없다: runs/{args.baseline}/test_report.json")
    bs = base["test_summary"]
    print(f"TRTR 기준선  runs/{args.baseline}  (실제 train 4,200장 학습, test {bs['n']}장)")
    print("  " + "  ".join(f"{lab}={bs[k]:.4f}" for k, lab in METRICS))

    # ---- 표 1: 프로토콜 A — 같은 test 600장, 비율이 성립하는 유일한 자리
    print("\n" + "=" * 86)
    print("1. 프로토콜 A — val/test 원래대로 (test 600장, TRTR 과 동일 집합)")
    print(f"{'학습 데이터':<16}{'n':>6}" + "".join(f"{lab:>12}" for _, lab in METRICS)
          + f"{'F1 비율':>9}{'판정':>10}")
    print("-" * 86)
    print(f"{'실제 (TRTR)':<16}{bs['n']:>6}"
          + "".join(f"{bs[k]:>12.4f}" for k, _ in METRICS) + f"{'100.0%':>9}{'기준':>10}")
    rows = {}
    for arm, lab in ARMS:
        r = load(RUNS / f"{args.prefix}_{arm}_A" / "test_report.json")
        if r is None:
            print(f"{lab + ' (TSTR-A)':<16}{'-':>6}  (아직 없음)"); continue
        s = r["test_summary"]
        ratio = s["macro_f1"] / bs["macro_f1"]
        rows[(arm, "A")] = (s, ratio)
        print(f"{lab + ' (TSTR)':<16}{s['n']:>6}"
              + "".join(f"{s[k]:>12.4f}" for k, _ in METRICS)
              + f"{ratio * 100:>8.1f}%{verdict(ratio):>10}")
    print("  * 판정: macro-F1 기준 90% 이상 '매우 높음', 85% 이상 '높음'.")

    # ---- 표 2: 프로토콜 B — 넓힌 test
    print("\n" + "=" * 86)
    print("2. 프로토콜 B — 실제 train 4,200장을 val/test 로 이동 (test 2,698장)")
    print(f"{'학습 데이터':<16}{'n':>6}" + "".join(f"{lab:>12}" for _, lab in METRICS)
          + f"{'원본test acc':>13}{'이동분 acc':>12}")
    print("-" * 86)
    for arm, lab in ARMS:
        # 재평가본(eval_test_best.json)이 있으면 그쪽을 쓴다 — 학습 당시 test 폴더에
        # 옛 링크 4장이 남아 있었고(고침), 지운 뒤 다시 잰 값이 이 파일이다.
        d = RUNS / f"{args.prefix}_{arm}_B"
        r = load(d / "eval_test_best.json")
        key = "summary"
        if r is None:
            r, key = load(d / "test_report.json"), "test_summary"
        if r is None:
            print(f"{lab + ' (TSTR-B)':<16}{'-':>6}  (아직 없음)"); continue
        s = r[key]
        (a_orig, n_orig), (a_moved, n_moved) = split_by_moved(r.get("by_source", {}))
        rows[(arm, "B")] = (s, None)
        print(f"{lab + ' (TSTR)':<16}{s['n']:>6}"
              + "".join(f"{s[k]:>12.4f}" for k, _ in METRICS)
              + f"{a_orig:>9.4f}({n_orig}){a_moved:>8.4f}({n_moved})")
    print("  * 원본test = 새 출처·새 케이스, 이동분 = 학습 데이터와 같은 출처 분포.")
    print("    두 값 차이가 크면 합성 모델도 출처 편향을 그대로 물려받은 것이다.")

    # ---- 표 3: B 런을 원본 test 600장으로 재평가 -> 비율
    print("\n" + "=" * 86)
    print("3. B 런을 원본 test 600장만으로 재평가 (TRTR 과 같은 잣대)")
    print(f"{'학습 데이터':<16}{'n':>6}" + "".join(f"{lab:>12}" for _, lab in METRICS)
          + f"{'F1 비율':>9}{'판정':>10}")
    print("-" * 86)
    for arm, lab in ARMS:
        r = load(RUNS / f"{args.prefix}_{arm}_B" / "eval_test_best_front.json")
        if r is None:
            print(f"{lab + ' (TSTR-B)':<16}{'-':>6}  (아직 없음)"); continue
        s = r["summary"]
        ratio = s["macro_f1"] / bs["macro_f1"]
        print(f"{lab + ' (TSTR)':<16}{s['n']:>6}"
              + "".join(f"{s[k]:>12.4f}" for k, _ in METRICS)
              + f"{ratio * 100:>8.1f}%{verdict(ratio):>10}")
    print("  * A 와 여기의 차이는 '학습이 같은데 val(선택 집합)만 달랐을 때'의 차이다.")

    # ---- 표 4: 클래스별 F1
    classes = base["classes"]
    print("\n" + "=" * 86)
    print("4. 클래스별 F1 (프로토콜 A)")
    print(f"{'학습 데이터':<16}" + "".join(f"{c:>10}" for c in classes))
    print("-" * 86)
    print(f"{'실제 (TRTR)':<16}"
          + "".join(f"{base['per_class'][c]['f1']:>10.3f}" for c in classes))
    for arm, lab in ARMS:
        r = load(RUNS / f"{args.prefix}_{arm}_A" / "test_report.json")
        if r is None:
            continue
        print(f"{lab:<16}" + "".join(f"{r['per_class'][c]['f1']:>10.3f}" for c in classes))
    print("  * TRTR 대비 특정 질환만 무너지면 그 질환의 합성이 라벨과 어긋난 것이다."
          " label_agreement / FID 클래스별 값과 대조해 볼 것.")


if __name__ == "__main__":
    main()
