"""조건 B~H 하이퍼파라미터 탐색 — pvtv2b0 고정, optuna TPE, 조건별 독립 study.

hpo_a.py(조건 A)와 구조는 같지만 trial 하나 = 모델 1개 학습(조건 A처럼 5축을 따로
학습하는 게 아니라, B~H는 원래 5헤드 공유 trunk 모델 하나라 학습도 한 번뿐이다).
목표(최대화) = val mean QWK(train_multitask.py/train_distill.py가 이미 계산해서
test_report.json에 val_score로 저장한 값 그대로). test는 기록만 하고 보고에만 쓴다.

조건 B~H는 서로 다른 study.db를 쓰는 완전히 독립된 탐색이다(A와 달리 조건마다
아키텍처는 같지만 손실이 달라서 최적 하이퍼파라미터가 다를 수 있다는 전제).
distill 조건(C~H)의 teacher_logits/teacher_logits_a는 조건 A HPO에서 최종
선택된 trial 24 설정(runs_bh_sweep/trial_24/)의 것을 고정으로 재사용한다 —
탐색 중인 매 trial마다 margin/imgsz가 바뀌는데, teacher_logits 파일은 학습
표본의 stem을 key로 쓰고 학생의 그 시점 crop과 무관하므로(파일명 태그는
exp/bbox_square에만 좌우) 문제 없다.

탐색 공간은 조건 A HPO(hpo_a.py)의 10개(lr/backbone_lr_scale/weight_decay/
dropout/warmup_epochs/scheduler(+cosine_epochs|plateau_patience)/
class_weight_power/label_smoothing/margin/imgsz)에 조건별로:
  - B 이외 전부: anneal_epochs(5~40) 추가
  - E/H: lambda_con(일관성 손실 가중) 추가

단계(조건마다 각각 실행):
  search  : TPE 탐색. 완료 trial 이 --n_trials 개가 될 때까지, 중단 뒤 재실행하면
            study.db 에서 이어서 돈다.
  report  : final_report.md 작성.

산출물(runs_hpo_bh/<COND>/): study.db, results.csv, leaderboard.txt, trial_XXX/,
                            final_report.md
"""
import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "runs_hpo_bh"
DATA = (ROOT.parent / "dataset_all_final" / "images").resolve()
MASKS = (ROOT.parent / "atopy_crop_masks").resolve()
SEARCH_SEED = 42

# 조건 A HPO 최종 선택(trial 24)의 teacher 산출물 재사용 — C~H 전 trial 공통.
TEACHER_BASE = ROOT / "runs_bh_sweep" / "trial_24"
TEACHER_DIR = TEACHER_BASE / "teacher_logits"
TEACHER_DIR_A = TEACHER_BASE / "teacher_logits_a"

COND_ARGS = {
    "B": dict(script="train_multitask.py", fixed=[]),
    "C": dict(script="train_distill.py", fixed=["--teacher_source", "oof", "--anneal", "none"]),
    "D": dict(script="train_distill.py", fixed=["--teacher_source", "oof", "--anneal", "linear"]),
    "E": dict(script="train_distill.py",
             fixed=["--teacher_source", "oof", "--anneal", "linear", "--coherence"]),
    "F": dict(script="train_distill.py", fixed=["--teacher_source", "single", "--anneal", "none"]),
    "G": dict(script="train_distill.py", fixed=["--teacher_source", "single", "--anneal", "linear"]),
    "H": dict(script="train_distill.py",
             fixed=["--teacher_source", "single", "--anneal", "linear", "--coherence"]),
}
CONDS = list(COND_ARGS)

# trial 24(조건 A 최종 선택) 하이퍼파라미터 — B~H 탐색의 기준점(1번 trial)으로 고정 투입.
BASELINE_BASE = {
    "lr": 2.70e-4, "backbone_lr_scale": 0.198, "weight_decay": 2.84e-3, "dropout_x10": 4,
    "warmup_epochs": 3, "scheduler": "cosine", "cosine_epochs": 30,
    "class_weight_power": 0.5, "label_smoothing": 0.0, "margin": 0.3, "imgsz": 224,
}

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def out_dir(cond):
    return OUT_BASE / cond


def baseline(cond):
    b = dict(BASELINE_BASE)
    if cond != "B":
        b["anneal_epochs"] = 20
    if cond in ("E", "H"):
        b["lambda_con"] = 0.3
    return b


# ----------------------------------------------------------------------------- 탐색 공간
def suggest(trial, cond):
    p = {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "backbone_lr_scale": trial.suggest_float("backbone_lr_scale", 0.03, 1.0, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 0.1, log=True),
        "dropout": trial.suggest_int("dropout_x10", 0, 5) / 10,
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 5),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "plateau", "constant"]),
        "class_weight_power": trial.suggest_categorical("class_weight_power", [0.0, 0.5, 1.0]),
        "label_smoothing": trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1]),
        "margin": trial.suggest_categorical("margin", [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]),
        "imgsz": trial.suggest_categorical("imgsz", [224, 320, 384, 448, 512]),
    }
    if p["scheduler"] == "cosine":
        p["cosine_epochs"] = trial.suggest_categorical("cosine_epochs", [15, 30, 100])
    elif p["scheduler"] == "plateau":
        p["plateau_patience"] = trial.suggest_int("plateau_patience", 1, 3)
    if cond != "B":
        p["anneal_epochs"] = trial.suggest_int("anneal_epochs", 5, 40)
    if cond in ("E", "H"):
        p["lambda_con"] = trial.suggest_float("lambda_con", 0.05, 0.6, log=True)
    return p


def fmt_params(p, cond):
    s = (f"lr={p['lr']:.2e} bb×{p['backbone_lr_scale']:.3f} wd={p['weight_decay']:.2e} "
         f"drop={p['dropout']:.1f} warm={p['warmup_epochs']} sched={p['scheduler']}")
    if p["scheduler"] == "cosine":
        s += f"({p['cosine_epochs']})"
    elif p["scheduler"] == "plateau":
        s += f"(pat{p['plateau_patience']})"
    s += (f" cw={p['class_weight_power']} ls={p['label_smoothing']} "
          f"margin={p['margin']} imgsz={p['imgsz']}")
    if cond != "B":
        s += f" ae={p['anneal_epochs']}"
    if cond in ("E", "H"):
        s += f" λcon={p['lambda_con']:.2f}"
    return s


# ----------------------------------------------------------------------------- 학습 1회
def run_task(cond, p, run_dir, seed, workers):
    """조건 하나(B~H) 1회 학습. 이미 test_report.json 있으면 재사용(재개용)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    name = cond
    rep_path = run_dir / name / "test_report.json"
    if not rep_path.is_file():
        spec = COND_ARGS[cond]
        cmd = [sys.executable, str(ROOT / spec["script"]),
               "--data", str(DATA), "--model", "pvtv2b0", "--imgsz", str(p["imgsz"]),
               "--epochs", "100", "--patience", "5", "--batch", "32",
               "--lr", str(p["lr"]), "--backbone_lr_scale", str(p["backbone_lr_scale"]),
               "--weight_decay", str(p["weight_decay"]), "--dropout", str(p["dropout"]),
               "--warmup_epochs", str(p["warmup_epochs"]), "--scheduler", p["scheduler"],
               "--class_weight_power", str(p["class_weight_power"]),
               "--label_smoothing", str(p["label_smoothing"]),
               "--exp", "bbox", "--bbox_square", "--margin", str(p["margin"]),
               "--masks", str(MASKS), "--workers", str(workers), "--seed", str(seed),
               "--project", str(run_dir), "--name", name]
        if p["scheduler"] == "cosine":
            cmd += ["--cosine_epochs", str(p["cosine_epochs"])]
        elif p["scheduler"] == "plateau":
            cmd += ["--plateau_patience", str(p["plateau_patience"])]
        cmd += spec["fixed"]
        if cond != "B":
            cmd += ["--anneal_epochs", str(p["anneal_epochs"])]
            tdir = TEACHER_DIR if cond in ("C", "D", "E") else TEACHER_DIR_A
            cmd += ["--teacher_dir", str(tdir)]
        if cond in ("E", "H"):
            cmd += ["--lambda_con", str(p["lambda_con"])]
        task_log = run_dir / f"{name}.log"
        with open(task_log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not rep_path.is_file():
            raise RuntimeError(f"{cond} 학습 실패(returncode={r.returncode}) — {task_log}")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    (run_dir / name / "last.pt").unlink(missing_ok=True)
    vm = rep.get("val_metrics", {})
    per_task_val = {t: float(vm[t]["qwk"]) for t in vm} if vm else {}
    return {"val": float(rep["val_score"]), "test": float(rep["test_score"]),
            "epoch": int(rep["epoch"]), "per_task_val": per_task_val}


# ----------------------------------------------------------------------------- study
def make_study(cond):
    d = out_dir(cond)
    d.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{d / 'study.db'}",
                                         heartbeat_interval=60, grace_period=240)
    sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=12, multivariate=True,
                                         group=True, constant_liar=True)
    return optuna.create_study(study_name=f"hpo_bh_{cond}_pvtv2b0", storage=storage,
                               direction="maximize", sampler=sampler, load_if_exists=True)


def completed(study):
    return study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))


def write_tables(study, cond):
    d = out_dir(cond)
    trials = sorted(completed(study), key=lambda t: t.value, reverse=True)
    if not trials:
        return
    cols = ["trial", "val", "test", "epoch", "params"]
    with open(d / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in trials:
            ua = t.user_attrs
            w.writerow([t.number, round(ua["val"], 4), round(ua["test"], 4), ua["epoch"],
                       fmt_params(ua["params_full"], cond)])
    lines = [f"조건 {cond} — val mean QWK 기준 (완료 {len(trials)}개, test 는 results.csv)",
             f"{'#':>4} {'val':>7} {'test':>7} {'ep':>4}  params"]
    for t in trials[:15]:
        ua = t.user_attrs
        lines.append(f"{t.number:4d} {ua['val']:7.4f} {ua['test']:7.4f} {ua['epoch']:4d}  "
                     + fmt_params(ua["params_full"], cond))
    (d / "leaderboard.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def search(cond, n_trials, n_jobs, workers):
    study = make_study(cond)
    optuna.storages.fail_stale_trials(study)
    if not any(t.user_attrs.get("baseline") for t in
               study.get_trials(deepcopy=False,
                                states=(TrialState.COMPLETE, TrialState.WAITING))):
        study.enqueue_trial(baseline(cond), user_attrs={"baseline": True})
        log(f"[{cond}] 기준점(trial 24 조건 A 설정)을 첫 trial 로 넣었다.")
    n_done = len(completed(study))
    if n_done >= n_trials:
        log(f"[{cond}] 이미 완료 trial {n_done}개 >= {n_trials} — search 건너뜀")
        return
    log(f"[{cond}] search 시작: 완료 {n_done}/{n_trials}, 동시 {n_jobs}개, workers {workers}")

    fails = {"n": 0}

    def on_done(st, tr):
        if tr.state == TrialState.FAIL:
            fails["n"] += 1
            log(f"[{cond}] trial {tr.number} 실패 ({fails['n']}번째)")
            if fails["n"] >= 8:
                log(f"[{cond}] 실패가 8번 쌓여 search 를 멈춘다 — 로그를 확인할 것.")
                st.stop()
        with _lock:
            write_tables(st, cond)

    def objective(trial):
        p = suggest(trial, cond)
        run_dir = out_dir(cond) / f"trial_{trial.number:03d}"
        tag = " [기준점]" if trial.user_attrs.get("baseline") else ""
        log(f"[{cond}] trial {trial.number} 시작{tag}  {fmt_params(p, cond)}")
        t0 = time.time()
        r = run_task(cond, p, run_dir, SEARCH_SEED, workers)
        trial.set_user_attr("val", r["val"])
        trial.set_user_attr("test", r["test"])
        trial.set_user_attr("epoch", r["epoch"])
        trial.set_user_attr("per_task_val", r["per_task_val"])
        trial.set_user_attr("params_full", p)
        log(f"[{cond}] trial {trial.number} 완료  val {r['val']:.4f}  test {r['test']:.4f}  "
            f"best_ep {r['epoch']:2d}  ({time.time() - t0:.0f}s)")
        return r["val"]

    study.optimize(objective, n_jobs=n_jobs, catch=(RuntimeError,),
                   callbacks=[optuna.study.MaxTrialsCallback(n_trials,
                                                             states=(TrialState.COMPLETE,)),
                              on_done])
    write_tables(study, cond)
    log(f"[{cond}] search 종료: 완료 {len(completed(study))}개")


# ----------------------------------------------------------------------------- report
def report(cond):
    study = make_study(cond)
    trials = sorted(completed(study), key=lambda t: t.value, reverse=True)
    states = [t.state for t in study.get_trials(deepcopy=False)]
    L = [f"# 조건 {cond} 하이퍼파라미터 탐색 (pvtv2b0)", "",
         f"- 완료 trial {len(trials)}개 / 실패 {states.count(TrialState.FAIL)}개",
         "- 목표: val mean QWK (선택은 val, test 는 보고용)",
         f"- teacher: runs_bh_sweep/trial_24/teacher_logits{'_a' if cond in ('F','G','H') else ''}"
         if cond != "B" else "- teacher 없음(hard label 만)", ""]

    if trials:
        best = trials[0]
        ua = best.user_attrs
        L += ["## 최고 trial", "",
              f"trial {best.number} — val {ua['val']:.4f}  test {ua['test']:.4f}  "
              f"best_ep {ua['epoch']}", "",
              "```", fmt_params(ua["params_full"], cond), "```", ""]
        if ua.get("per_task_val"):
            L += ["| 축 | val QWK |", "|---|---|"]
            for t, v in ua["per_task_val"].items():
                L.append(f"| {t} | {v:.4f} |")
            L.append("")

    L += ["## 탐색 상위 10 (val)", "",
          "| trial | val | test | best_ep | 설정 |", "|---|---|---|---|---|"]
    for t in trials[:10]:
        ua = t.user_attrs
        L.append(f"| {t.number} | {ua['val']:.4f} | {ua['test']:.4f} | {ua['epoch']} | "
                 f"{fmt_params(ua['params_full'], cond)} |")
    L.append("")

    try:
        imp = optuna.importance.get_param_importances(study)
        L += ["## 파라미터 중요도 (fANOVA, val 기준)", "", "| 파라미터 | 중요도 |", "|---|---|"]
        L += [f"| {k} | {v:.3f} |" for k, v in imp.items()]
        L.append("")
    except Exception as e:
        L += [f"(파라미터 중요도 계산 실패: {type(e).__name__}: {e})", ""]

    text = "\n".join(L)
    (out_dir(cond) / "final_report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", required=True, choices=CONDS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--n_trials", type=int, default=60)
    s.add_argument("--n_jobs", type=int, default=2)
    s.add_argument("--workers", type=int, default=5)
    sub.add_parser("report")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.cmd == "search":
        search(args.cond, args.n_trials, args.n_jobs, args.workers)
    else:
        report(args.cond)


if __name__ == "__main__":
    main()
