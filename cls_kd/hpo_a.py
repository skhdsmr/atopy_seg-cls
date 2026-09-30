"""조건 A(단일축 teacher) 하이퍼파라미터 탐색 — pvtv2b0 고정, optuna TPE.

trial 하나 = 같은 하이퍼파라미터로 5개 축을 train_single.py 로 전부 학습한 것.
목표(최대화) = mean(5축 val QWK) - BALANCE * std(5축 val QWK)
  한 축만 튀고 나머지가 무너진 설정보다 5축이 고르게 높은 설정을 고르려는 것.
  선택은 val 로만 한다. test 는 기록만 하고 최종 설정의 보고에만 쓴다 — test 로
  고르면 200장짜리 test 에 과적합된 숫자를 보고하게 된다.

단계:
  search  : TPE 탐색. 완료 trial 이 --n_trials 개가 될 때까지 돌고, 중단 뒤 다시 실행하면
            study.db 에서 이어서 돈다(끊긴 trial 은 heartbeat 로 FAIL 처리되고 버려진다).
  confirm : 상위 --top_k 설정을 --seeds 로 재학습 — val 200장 + 시드 1개짜리 점수에는
            운이 섞이므로, 최종 선택은 시드 평균으로 한다.
  report  : final_report.md 작성.

산출물(runs_hpo_a/): study.db, results.csv, leaderboard.txt, trial_XXX/, confirm/,
                     confirm.json, final_report.md
"""
import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState

from labels import TASK_NAMES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs_hpo_a"
DATA = (ROOT.parent / "dataset_all_final" / "images").resolve()
MASKS = (ROOT.parent / "atopy_crop_masks").resolve()
STUDY = "hpo_a_pvtv2b0"
BALANCE = 0.5
SEARCH_SEED = 42

# 지난 스윕(runs_sweep/m15/pvtv2b0)의 조건 A 설정 — 개선폭을 잴 기준점으로 1번 trial 에 넣는다.
BASELINE = {
    "lr": 3e-4, "backbone_lr_scale": 0.1, "weight_decay": 0.05, "dropout_x10": 3,
    "warmup_epochs": 3, "scheduler": "cosine", "cosine_epochs": 100,
    "class_weight_power": 0.0, "label_smoothing": 0.0, "margin": 0.15, "imgsz": 512,
}

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- 탐색 공간
def suggest(trial):
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
    return p


def fmt_params(p):
    s = (f"lr={p['lr']:.2e} bb×{p['backbone_lr_scale']:.3f} wd={p['weight_decay']:.2e} "
         f"drop={p['dropout']:.1f} warm={p['warmup_epochs']} sched={p['scheduler']}")
    if p["scheduler"] == "cosine":
        s += f"({p['cosine_epochs']})"
    elif p["scheduler"] == "plateau":
        s += f"(pat{p['plateau_patience']})"
    return (s + f" cw={p['class_weight_power']} ls={p['label_smoothing']} "
                f"margin={p['margin']} imgsz={p['imgsz']}")


# ----------------------------------------------------------------------------- 학습 1회
def run_task(task, p, run_dir, seed, workers):
    """train_single.py 1회. 이미 test_report.json 이 있으면 다시 돌리지 않는다(confirm 재개용)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"A_{task}"
    rep_path = run_dir / name / "test_report.json"
    if not rep_path.is_file():
        cmd = [sys.executable, str(ROOT / "train_single.py"), "--task", task,
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
        task_log = run_dir / f"{name}.log"
        with open(task_log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not rep_path.is_file():
            raise RuntimeError(f"{task} 학습 실패(returncode={r.returncode}) — {task_log}")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    (run_dir / name / "last.pt").unlink(missing_ok=True)       # 재개용 상태 — 끝난 뒤엔 불필요
    return {"val": float(rep["val_score"]), "test": float(rep["test_score"]),
            "epoch": int(rep["epoch"])}


def summarize(per):
    vals = np.array([per[t]["val"] for t in TASK_NAMES])
    tests = np.array([per[t]["test"] for t in TASK_NAMES])
    return {"score": float(vals.mean() - BALANCE * vals.std()),
            "val_mean": float(vals.mean()), "val_std": float(vals.std()),
            "val_min": float(vals.min()), "test_mean": float(tests.mean())}


# ----------------------------------------------------------------------------- study
def make_study():
    OUT.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{OUT / 'study.db'}",
                                         heartbeat_interval=60, grace_period=240)
    sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=12, multivariate=True,
                                         group=True, constant_liar=True)
    return optuna.create_study(study_name=STUDY, storage=storage, direction="maximize",
                               sampler=sampler, load_if_exists=True)


def completed(study):
    return study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))


def write_tables(study):
    trials = sorted(completed(study), key=lambda t: t.value, reverse=True)
    if not trials:
        return
    cols = (["trial", "score", "val_mean", "val_std", "val_min", "test_mean"]
            + [f"val_{t}" for t in TASK_NAMES] + [f"test_{t}" for t in TASK_NAMES]
            + [f"ep_{t}" for t in TASK_NAMES] + ["params"])
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in trials:
            ua = t.user_attrs
            w.writerow([t.number] + [round(ua.get(c, float("nan")), 4) for c in cols[1:-1]]
                       + [fmt_params(ua["params_full"])])
    lines = [f"균형점수 = val mean - {BALANCE}*std  (완료 {len(trials)}개, val 기준 — test 는 results.csv)",
             f"{'#':>4} {'score':>7} {'mean':>7} {'std':>6} {'min':>6}  "
             + " ".join(f"{t[:6]:>7}" for t in TASK_NAMES) + "  params"]
    for t in trials[:15]:
        ua = t.user_attrs
        lines.append(f"{t.number:4d} {t.value:7.4f} {ua['val_mean']:7.4f} {ua['val_std']:6.4f} "
                     f"{ua['val_min']:6.4f}  "
                     + " ".join(f"{ua[f'val_{x}']:7.4f}" for x in TASK_NAMES)
                     + "  " + fmt_params(ua["params_full"]))
    (OUT / "leaderboard.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def search(n_trials, n_jobs, workers):
    study = make_study()
    optuna.storages.fail_stale_trials(study)
    if not any(t.user_attrs.get("baseline") for t in
               study.get_trials(deepcopy=False,
                                states=(TrialState.COMPLETE, TrialState.WAITING))):
        study.enqueue_trial(BASELINE, user_attrs={"baseline": True})
        log("기준점(지난 스윕 margin 0.15/512 설정)을 첫 trial 로 넣었다.")
    n_done = len(completed(study))
    if n_done >= n_trials:
        log(f"이미 완료 trial {n_done}개 >= {n_trials} — search 건너뜀")
        return
    log(f"search 시작: 완료 {n_done}/{n_trials}, 동시 {n_jobs}개, workers {workers}")

    fails = {"n": 0}

    def on_done(st, tr):
        if tr.state == TrialState.FAIL:
            fails["n"] += 1
            log(f"trial {tr.number} 실패 ({fails['n']}번째)")
            if fails["n"] >= 5:
                log("실패가 5번 쌓여 search 를 멈춘다 — 로그를 확인할 것.")
                st.stop()
        with _lock:
            write_tables(st)

    def objective(trial):
        p = suggest(trial)
        run_dir = OUT / f"trial_{trial.number:03d}"
        tag = " [기준점]" if trial.user_attrs.get("baseline") else ""
        log(f"trial {trial.number} 시작{tag}  {fmt_params(p)}")
        per = {}
        for task in TASK_NAMES:
            t0 = time.time()
            per[task] = run_task(task, p, run_dir, SEARCH_SEED, workers)
            for k in ("val", "test"):
                trial.set_user_attr(f"{k}_{task}", per[task][k])
            trial.set_user_attr(f"ep_{task}", per[task]["epoch"])
            log(f"trial {trial.number}  {task:16s} val {per[task]['val']:.4f}  "
                f"test {per[task]['test']:.4f}  best_ep {per[task]['epoch']:2d}  "
                f"({time.time() - t0:.0f}s)")
        s = summarize(per)
        for k, v in s.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("params_full", p)
        log(f"trial {trial.number} 완료  균형점수 {s['score']:.4f}  "
            f"(val mean {s['val_mean']:.4f} ± {s['val_std']:.4f}, min {s['val_min']:.4f})")
        return s["score"]

    study.optimize(objective, n_jobs=n_jobs, catch=(RuntimeError,),
                   callbacks=[optuna.study.MaxTrialsCallback(n_trials,
                                                             states=(TrialState.COMPLETE,)),
                              on_done])
    write_tables(study)
    log(f"search 종료: 완료 {len(completed(study))}개")


# ----------------------------------------------------------------------------- confirm
def confirm(top_k, seeds, n_jobs, workers):
    study = make_study()
    top = sorted(completed(study), key=lambda t: t.value, reverse=True)[:top_k]
    if not top:
        sys.exit("[에러] 완료된 trial 이 없다 — search 를 먼저 돌릴 것.")
    log(f"confirm 시작: 상위 {len(top)}개 trial {[t.number for t in top]} x 추가 시드 {seeds}")

    def work(job):
        t, seed = job
        run_dir = OUT / "confirm" / f"trial_{t.number:03d}_seed{seed}"
        per = {task: run_task(task, t.user_attrs["params_full"], run_dir, seed, workers)
               for task in TASK_NAMES}
        s = summarize(per)
        log(f"confirm trial {t.number} seed {seed}: 균형점수 {s['score']:.4f} "
            f"(val mean {s['val_mean']:.4f}, test mean {s['test_mean']:.4f})")
        return t.number, seed, per

    jobs = [(t, s) for t in top for s in seeds]
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        extra = list(ex.map(work, jobs))

    out = []
    for t in top:
        runs = {SEARCH_SEED: {task: {"val": t.user_attrs[f"val_{task}"],
                                     "test": t.user_attrs[f"test_{task}"]}
                              for task in TASK_NAMES}}
        runs.update({seed: per for num, seed, per in extra if num == t.number})
        scores = [summarize(per)["score"] for per in runs.values()]
        agg = {task: {k: [runs[s][task][k] for s in runs] for k in ("val", "test")}
               for task in TASK_NAMES}
        out.append({
            "trial": t.number, "params": t.user_attrs["params_full"],
            "baseline": bool(t.user_attrs.get("baseline")), "seeds": sorted(runs),
            "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
            "per_task": {task: {f"{k}_{st}": float(fn(agg[task][k]))
                                for k in ("val", "test")
                                for st, fn in (("mean", np.mean), ("std", np.std))}
                         for task in TASK_NAMES},
        })
    out.sort(key=lambda r: r["score_mean"], reverse=True)
    (OUT / "confirm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    log(f"confirm 종료: 최종 선택 trial {out[0]['trial']} (시드 평균 균형점수 "
        f"{out[0]['score_mean']:.4f} ± {out[0]['score_std']:.4f})")


# ----------------------------------------------------------------------------- report
def report():
    study = make_study()
    trials = sorted(completed(study), key=lambda t: t.value, reverse=True)
    states = [t.state for t in study.get_trials(deepcopy=False)]
    conf_path = OUT / "confirm.json"
    conf = json.loads(conf_path.read_text(encoding="utf-8")) if conf_path.is_file() else []
    L = ["# 조건 A 하이퍼파라미터 탐색 (pvtv2b0)", "",
         f"- 완료 trial {len(trials)}개 / 실패 {states.count(TrialState.FAIL)}개",
         f"- 목표: 5축 val QWK 의 mean − {BALANCE}·std (선택은 val, test 는 보고용)", ""]

    if conf:
        best = conf[0]
        L += ["## 최종 선택 설정 (상위 후보를 시드 여러 개로 재학습한 평균 기준)", "",
              f"trial {best['trial']} — 시드 {best['seeds']} 평균 균형점수 "
              f"**{best['score_mean']:.4f} ± {best['score_std']:.4f}**", "",
              "```", fmt_params(best["params"]), "```", "",
              "| 축 | val QWK (mean±std) | test QWK (mean±std) |", "|---|---|---|"]
        for task in TASK_NAMES:
            r = best["per_task"][task]
            L.append(f"| {task} | {r['val_mean']:.4f} ± {r['val_std']:.4f} | "
                     f"{r['test_mean']:.4f} ± {r['test_std']:.4f} |")
        vm = np.mean([best["per_task"][t]["val_mean"] for t in TASK_NAMES])
        tm = np.mean([best["per_task"][t]["test_mean"] for t in TASK_NAMES])
        L += [f"| **평균** | **{vm:.4f}** | **{tm:.4f}** |", "",
              "### 재학습 후보 비교", "",
              "| trial | 시드 평균 균형점수 | val mean | test mean | 설정 |", "|---|---|---|---|---|"]
        for r in conf:
            vmean = np.mean([r["per_task"][t]["val_mean"] for t in TASK_NAMES])
            tmean = np.mean([r["per_task"][t]["test_mean"] for t in TASK_NAMES])
            L.append(f"| {r['trial']} | {r['score_mean']:.4f} ± {r['score_std']:.4f} | "
                     f"{vmean:.4f} | {tmean:.4f} | {fmt_params(r['params'])} |")
        L.append("")

    base = next((t for t in trials if t.user_attrs.get("baseline")), None)
    if base is not None:
        ua = base.user_attrs
        L += ["## 기준점 (지난 스윕 설정, 시드 42 한 번)", "",
              f"균형점수 {base.value:.4f}, val mean {ua['val_mean']:.4f}, "
              f"test mean {ua['test_mean']:.4f}  —  " + " / ".join(
                  f"{t} {ua[f'val_{t}']:.3f}" for t in TASK_NAMES), ""]

    L += ["## 탐색 상위 10 (시드 42, val)", "",
          "| trial | 균형점수 | val mean | std | min | " + " | ".join(TASK_NAMES) + " | 설정 |",
          "|" + "---|" * (6 + len(TASK_NAMES))]
    for t in trials[:10]:
        ua = t.user_attrs
        L.append(f"| {t.number} | {t.value:.4f} | {ua['val_mean']:.4f} | {ua['val_std']:.4f} | "
                 f"{ua['val_min']:.4f} | " + " | ".join(f"{ua[f'val_{x}']:.4f}" for x in TASK_NAMES)
                 + f" | {fmt_params(ua['params_full'])} |")
    L.append("")

    L += ["## 축별 최고 설정 (각 축만 보면, val)", "",
          "| 축 | trial | val | 그 trial 의 5축 평균 | 설정 |", "|---|---|---|---|---|"]
    for task in TASK_NAMES:
        t = max(trials, key=lambda x: x.user_attrs[f"val_{task}"])
        ua = t.user_attrs
        L.append(f"| {task} | {t.number} | {ua[f'val_{task}']:.4f} | {ua['val_mean']:.4f} | "
                 f"{fmt_params(ua['params_full'])} |")
    L.append("")

    try:
        imp = optuna.importance.get_param_importances(study)
        L += ["## 파라미터 중요도 (fANOVA, 균형점수 기준)", "",
              "| 파라미터 | 중요도 |", "|---|---|"]
        L += [f"| {k} | {v:.3f} |" for k, v in imp.items()]
        L.append("")
    except Exception as e:                                    # 조건부 파라미터/표본 부족 등
        L += [f"(파라미터 중요도 계산 실패: {type(e).__name__}: {e})", ""]

    text = "\n".join(L)
    (OUT / "final_report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--n_trials", type=int, default=60)
    s.add_argument("--n_jobs", type=int, default=2)
    s.add_argument("--workers", type=int, default=5)
    c = sub.add_parser("confirm")
    c.add_argument("--top_k", type=int, default=3)
    c.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    c.add_argument("--n_jobs", type=int, default=2)
    c.add_argument("--workers", type=int, default=5)
    sub.add_parser("report")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.cmd == "search":
        search(args.n_trials, args.n_jobs, args.workers)
    elif args.cmd == "confirm":
        confirm(args.top_k, args.seeds, args.n_jobs, args.workers)
    else:
        report()


if __name__ == "__main__":
    main()
