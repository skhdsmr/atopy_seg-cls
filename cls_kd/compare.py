"""A~H 결과 비교표 — done.txt / test_report.json 을 모아 BAM Table 1 형태로 찍는다.

조건 A 는 5개 축마다 별도 run(A_single_<task>_...)이라 표에서 한 행으로 합친다.
B~H 는 전부 이 폴더(cls_kd/runs)에서 나온다(다른 cls_* 폴더를 보지 않는다).
C/D/E 는 teacher=OOF(collect_oof.py), F/G/H 는 teacher=조건 A 직접(collect_a.py,
in-sample) — 나머지(anneal/coherence 구성)는 C/D/E 와 F/G/H 가 각각 대응한다.

사용법:
    python3 compare.py --project runs --model pvtv2b0 --imgsz 224
"""
import argparse
import json
from pathlib import Path


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).is_file() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--model", default="pvtv2b0")
    ap.add_argument("--imgsz", type=int, default=224)
    args = ap.parse_args()

    project = Path(args.project)
    tasks = ["iga_grade", "erythema", "papulation", "excoriation", "lichenification"]

    def print_task_metrics(rep, indent="  "):
        for t in tasks:
            m = rep["test_metrics"][t]
            print(f"{indent}{t:16s} QWK {m['qwk']:.4f}  acc {m['acc']:.3f}  "
                 f"±1 {m['acc1']:.3f}  MAE {m['mae']:.3f}")
        print(f"{indent}{'mean QWK':16s} {rep['test_score']:.4f}")

    print("=== 조건 A: single-task teacher (상한 기준) ===")
    for t in tasks:
        rep = load_json(project / f"A_single_{t}_{args.model}_r{args.imgsz}" / "test_report.json")
        if rep:
            m = rep["test_metrics"]
            print(f"  {t:16s} QWK {m['qwk']:.4f}  acc {m['acc']:.3f}  "
                 f"±1 {m['acc1']:.3f}  MAE {m['mae']:.3f}")
        else:
            print(f"  {t:16s} (없음 — train_single.py 로 먼저 학습할 것)")

    print("\n=== 조건 B/C/D/E (teacher=OOF) ===")
    for cond, name, script in (
        ("B", f"B_{args.model}_r{args.imgsz}", "train_multitask.py"),
        ("C", f"C_{args.model}_r{args.imgsz}", "train_distill.py --anneal none"),
        ("D", f"D_{args.model}_r{args.imgsz}", "train_distill.py --anneal linear"),
        ("E", f"E_{args.model}_r{args.imgsz}",
         "train_distill.py --anneal linear --coherence"),
    ):
        rep = load_json(project / name / "test_report.json")
        print(f"  [{cond}] {name}")
        if not rep:
            print(f"      (없음 — {script} 로 먼저 학습할 것)")
            continue
        print_task_metrics(rep, indent="      ")

    print("\n=== 조건 F/G/H (teacher=조건 A 직접, in-sample — C/D/E 와 대응) ===")
    for cond, name, script in (
        ("F", f"F_{args.model}_r{args.imgsz}",
         "train_distill.py --anneal none --teacher_source single"),
        ("G", f"G_{args.model}_r{args.imgsz}",
         "train_distill.py --anneal linear --teacher_source single"),
        ("H", f"H_{args.model}_r{args.imgsz}",
         "train_distill.py --anneal linear --coherence --teacher_source single"),
    ):
        rep = load_json(project / name / "test_report.json")
        print(f"  [{cond}] {name}")
        if not rep:
            print(f"      (없음 — {script} 로 먼저 학습할 것)")
            continue
        print_task_metrics(rep, indent="      ")

    print("\n비고: A 는 각 축의 상한(teacher) 기준, B~H 는 5축을 동시에 내는 "
         "하나의 모델이다. B vs C 차이 = 증류 자체의 효과, C vs D 차이 = teacher "
         "annealing 효과, D vs E 차이 = 일관성 손실 효과, B vs D 차이 = Single->Multi "
         "증류 전체 효과(BAM 핵심 주장). C vs F(D vs G, E vs H) 차이 = OOF로 누수를 "
         "막은 teacher 와 조건 A를 그대로 쓴 teacher(in-sample) 의 차이.")


if __name__ == "__main__":
    main()
