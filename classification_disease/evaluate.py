"""저장된 체크포인트(runs/*/best.pt) 평가 — accuracy / F1 / 혼동행렬.

train.py 는 학습이 끝날 때 test 를 딱 1회 보고 끝난다. 이 스크립트는 그 뒤에
    - 어떤 런(모델)을 볼지 고르고
    - 어떤 split(test/val/train)에 대해
    - best/last 중 어느 체크포인트로
평가를 다시 돌리기 위한 것이다. 가중치만 복원해 추론하므로 학습에 영향이 없다.

체크포인트에 학습 당시 args 가 통째로 박혀 있어서(imgsz/angle/embed_dim/dropout)
전처리를 그대로 재현한다 — 런마다 해상도가 달라도 손댈 필요 없다.

사용법:
    python3 evaluate.py --list                          # 런 목록 + 저장된 test 성적
    python3 evaluate.py --run dis_effb0_r512_both       # 기본: test, best.pt
    python3 evaluate.py --run dis_effb0_r512_both --split val --ckpt last
    python3 evaluate.py --run dis_effb0_r512_both --png # 혼동행렬 PNG 도 저장
    python3 evaluate.py --run all                       # 전 런 비교표
    python3 evaluate.py --run dis_mnv3s_r512_both dis_effb0_r512_both

읽을 때: accuracy 만 보지 말 것. 출처(prefix)가 질환을 거의 알려주는 데이터라
출처별 정확도 편차가 크면 질환이 아니라 촬영 조건을 외운 것이다(README 참고).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent            # classification_disease
OGW = ROOT.parent
sys.path.insert(0, str(ROOT))

from dataset import (DISEASES, DiseaseDataset, NUM_DISEASES,             # noqa: E402
                     label_index_from_csv)
from metrics import (confusion, format_confusion, format_groups,          # noqa: E402
                     format_report, group_accuracy, per_class_metrics,
                     summarize)
from model import DiseaseNet                                              # noqa: E402
from train import build_transforms                                        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="질환 6-way 분류 체크포인트 평가",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--run", nargs="+", default=None,
                   help="runs/ 아래 런 이름(또는 경로). 여러 개 주면 비교표. 'all' 이면 전부")
    p.add_argument("--ckpt", default="best", choices=["best", "last"],
                   help="best=val 최고점 시점, last=마지막 epoch")
    p.add_argument("--split", default="test", choices=["test", "val", "train"],
                   help="test=제공 Validation(새 케이스+새 출처), val=같은 출처의 새 케이스")
    p.add_argument("--data", default=None,
                   help="dataset_disease 루트. 기본은 체크포인트에 박힌 학습 당시 경로")
    p.add_argument("--angle", choices=["both", "front", "side"], default=None,
                   help="기본은 학습 당시 값. front/side 로 쪼개 재평가할 때만 지정")
    p.add_argument("--imgsz", type=int, default=None, help="기본은 학습 당시 해상도")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--png", action="store_true", help="혼동행렬 PNG 저장(matplotlib 필요)")
    p.add_argument("--no_save", action="store_true", help="JSON 리포트 저장 안 함")
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--list", action="store_true", help="런 목록만 출력하고 종료")
    return p.parse_args()


def find_runs(project):
    """runs/ 아래 best.pt 를 가진 디렉터리 목록(이름순)."""
    proj = Path(project)
    if not proj.exists():
        return []
    return sorted((d for d in proj.iterdir() if d.is_dir() and (d / "best.pt").exists()),
                  key=lambda d: d.name)


def resolve_run(spec, project):
    """런 이름 또는 경로 -> Path. 없으면 후보를 보여주고 종료."""
    cand = Path(spec)
    if not cand.is_dir():
        cand = Path(project) / spec
    if not cand.is_dir():
        names = [d.name for d in find_runs(project)]
        print(f"[에러] 런을 찾을 수 없음: {spec}\n  가능한 런: " +
              (", ".join(names) if names else "(없음)"), file=sys.stderr)
        sys.exit(1)
    return cand


def list_runs(project):
    """런 목록 + 학습 때 이미 저장해 둔 test 성적(done.txt/test_report.json)."""
    runs = find_runs(project)
    if not runs:
        print(f"[list] {project} 에 런이 없다. 먼저 bash run.sh 로 학습할 것.")
        return
    print(f"[list] {project}  ({len(runs)}개)")
    print(f"  {'런':28s} {'ckpt':11s} {'ep':>4s} {'저장된 test acc':>15s} {'macroF1':>8s}")
    print("  " + "-" * 72)
    for d in runs:
        has = "+".join(n for n in ("best", "last") if (d / f"{n}.pt").exists())
        ep, acc, f1 = "-", "-", "-"
        rp = d / "test_report.json"
        if rp.exists():
            try:
                r = json.loads(rp.read_text(encoding="utf-8"))
                ep = str(r.get("epoch", "-"))
                ts = r.get("test_summary", {})
                acc = f"{ts.get('accuracy', float('nan')):.4f}"
                f1 = f"{ts.get('macro_f1', float('nan')):.4f}"
            except (json.JSONDecodeError, OSError):
                pass
        done = "" if (d / "done.txt").exists() else "  (학습 미완료)"
        print(f"  {d.name:28s} {has:11s} {ep:>4s} {acc:>15s} {f1:>8s}{done}")
    print("\n  위 숫자는 '학습 때 저장된' 값이다. 다시 재기려면 --run <이름>.")


@torch.no_grad()
def predict(model, loader, device, use_amp):
    """반환: (y_true(N,), y_pred(N,), scores(N,K)). scores 는 softmax 확률."""
    model.eval()
    ys, ps, ss = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
        lg = logits.float()
        ys.append(y.numpy())
        ps.append(lg.argmax(1).cpu().numpy())
        ss.append(torch.softmax(lg, dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)


def load_run(run_dir, a):
    """체크포인트 + 데이터로더 준비. 반환: (model, dataset, loader, ckpt, cfg)."""
    ckpt_path = run_dir / f"{a.ckpt}.pt"
    if not ckpt_path.exists():
        print(f"[에러] 체크포인트 없음: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})

    # 학습 당시 설정을 기본값으로, CLI 로 준 것만 덮어쓴다.
    cfg = {
        "imgsz": a.imgsz or targs.get("imgsz", 224),
        "angle": a.angle or targs.get("angle", "both"),
        "embed_dim": targs.get("embed_dim", 512),
        "dropout": targs.get("dropout", 0.5),
        "model_name": ckpt.get("model_name", targs.get("model", "?")),
    }
    data = Path(a.data) if a.data else Path(targs.get("data", OGW / "dataset_disease"))
    if not data.is_absolute():                       # ckpt 의 '../dataset_disease' 같은 상대경로
        data = (ROOT / data).resolve()
    csv_path = data / "labels.csv"
    if not csv_path.exists():
        print(f"[에러] labels.csv 없음: {csv_path}\n"
              f"       --data 로 dataset_disease 루트를 지정하거나 "
              f"make_dataset_disease.py 를 먼저 실행할 것.", file=sys.stderr)
        sys.exit(1)
    cfg["data"] = str(data)

    _, eval_tf = build_transforms(cfg["imgsz"])
    ds = DiseaseDataset(data / a.split, label_index_from_csv(csv_path), eval_tf,
                        angle=cfg["angle"])
    if len(ds) == 0:
        print(f"[에러] {a.split} 샘플 0개 (angle={cfg['angle']}).", file=sys.stderr)
        sys.exit(1)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                        pin_memory=True)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = DiseaseNet(cfg["model_name"], NUM_DISEASES, embed_dim=cfg["embed_dim"],
                       dropout=cfg["dropout"], pretrained=False).to(device)  # 가중치는 ckpt 로
    model.load_state_dict(ckpt["model"])
    return model, ds, loader, ckpt, cfg, device


def save_confusion_png(cm, classes, path, title):
    """행 정규화 혼동행렬 히트맵. 셀에는 (개수/행비율) 둘 다 찍는다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        print("[png] matplotlib 없음 -> 건너뜀 (pip install matplotlib)")
        return None

    # 한글 폰트가 없으면 클래스명이 두부(□)로 나오므로 있으면 쓰고 없으면 로마자로.
    korean = next((f.name for f in font_manager.fontManager.ttflist
                   if f.name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
                                 "Malgun Gothic", "AppleGothic", "UnDotum")), None)
    if korean:
        plt.rcParams["font.family"] = korean
        labels = list(classes)
    else:
        labels = [_ROMAN.get(c, c) for c in classes]
    plt.rcParams["axes.unicode_minus"] = False

    row = cm.sum(1, keepdims=True)
    norm = cm / np.maximum(row, 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.1 * n + 2.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("예측" if korean else "predicted")
    # 한글은 세로로 눕히면 읽기 나쁘다 -> y축 라벨만 수평 유지
    ax.set_ylabel("정답" if korean else "true", rotation=0, labelpad=18, va="center")
    ax.set_title(title, fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="행 비율" if korean else "row ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


_ROMAN = {"건선": "psoriasis", "아토피": "atopy", "여드름": "acne",
          "정상": "normal", "주사": "rosacea", "지루": "seborrheic"}


def evaluate_one(run_dir, a, verbose=True):
    """한 런 평가. 반환: 리포트 dict."""
    model, ds, loader, ckpt, cfg, device = load_run(run_dir, a)
    classes = ckpt.get("diseases", DISEASES)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    if verbose:
        print("=" * 78)
        print(f"[run] {run_dir.name}  ckpt={a.ckpt}.pt (epoch {ckpt.get('epoch', '?')})")
        print(f"  model={cfg['model_name']}  params={n_params:.1f}M  imgsz={cfg['imgsz']}  "
              f"angle={cfg['angle']}  split={a.split}  n={len(ds)}")
        print(f"  data={cfg['data']}  device={device.type}")
        print("=" * 78)

    y, p, s = predict(model, loader, device, use_amp=device.type == "cuda")

    per_class = per_class_metrics(y, p, s, classes)
    summary = summarize(y, p, s, classes)
    cm = confusion(y, p, len(classes))
    by_source = group_accuracy(y, p, ds.meta("source"))
    by_angle = group_accuracy(y, p, ds.meta("angle"))

    if verbose:
        print("\n" + format_report(per_class, summary, classes,
                                   f"[{a.split.upper()}] 클래스별 지표"))
        print(f"\n[{a.split.upper()}] 혼동행렬 (행=정답, 열=예측)\n"
              f"{format_confusion(cm, classes)}")

        # 어디서 무너지는지 — 오분류가 몰린 쌍 상위 5개
        off = [(classes[i], classes[j], int(cm[i, j]))
               for i in range(len(classes)) for j in range(len(classes))
               if i != j and cm[i, j] > 0]
        off.sort(key=lambda x: -x[2])
        if off:
            print(f"\n[{a.split.upper()}] 최다 오분류 쌍")
            for t, q, c in off[:5]:
                print(f"  {t:>6s} -> {q:<6s} {c:4d}장 ({c / len(y) * 100:4.1f}%)")

        print("\n" + format_groups(by_source, f"[{a.split.upper()}] 출처(prefix)별 정확도"))
        if cfg["angle"] == "both" and len(by_angle) > 1:
            print("\n" + format_groups(by_angle, f"[{a.split.upper()}] 각도별 정확도"))
            print("  측면=512px 피부 접사(Z4), 정면=1024px 실사 얼굴. 측면만 유독 높으면 "
                  "질환이 아니라 도메인 단서를 본 것일 수 있다.")

        # 학습 때 val 과 비교 — 격차가 곧 출처 과적합의 크기
        vsum = ckpt.get("val_summary")
        if vsum and a.split == "test":
            print(f"\n[val -> test 격차] acc: {vsum['accuracy']:.3f} -> "
                  f"{summary['accuracy']:.3f} ({summary['accuracy'] - vsum['accuracy']:+.3f})  |  "
                  f"macroF1: {vsum['macro_f1']:.3f} -> {summary['macro_f1']:.3f} "
                  f"({summary['macro_f1'] - vsum['macro_f1']:+.3f})")
            print("  val 은 '같은 출처의 새 케이스', test 는 '다른 출처의 새 케이스'다.")

    report = {
        "run": run_dir.name, "ckpt": a.ckpt, "epoch": ckpt.get("epoch"),
        "split": a.split, "model_name": cfg["model_name"], "params_m": round(n_params, 3),
        "imgsz": cfg["imgsz"], "angle": cfg["angle"], "data": cfg["data"],
        "n": int(len(y)), "classes": classes,
        "summary": summary, "per_class": per_class, "confusion": cm.tolist(),
        "by_source": by_source, "by_angle": by_angle,
    }

    tag = f"{a.split}_{a.ckpt}" + ("" if a.angle is None else f"_{cfg['angle']}")
    if not a.no_save:
        out = run_dir / f"eval_{tag}.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if verbose:
            print(f"\n[저장] {out}")
    if a.png:
        png = save_confusion_png(
            cm, classes, run_dir / f"confusion_{tag}.png",
            f"{run_dir.name} / {a.split} / {a.ckpt}  "
            f"acc={summary['accuracy']:.3f} macroF1={summary['macro_f1']:.3f}")
        if png and verbose:
            print(f"[저장] {png}")
    return report


def print_comparison(reports, split):
    """여러 런 한 줄씩 — macro-F1 내림차순."""
    print("\n" + "=" * 90)
    print(f"[비교] split={split}  (macro-F1 내림차순)")
    print("=" * 90)
    print(f"  {'런':28s} {'params':>7s} {'imgsz':>6s} {'acc':>7s} {'bal_acc':>8s} "
          f"{'macroF1':>8s} {'AUROC':>7s} {'출처편차':>9s}")
    print("  " + "-" * 88)
    for r in sorted(reports, key=lambda r: -r["summary"]["macro_f1"]):
        accs = [v["acc"] for v in r["by_source"].values() if v["n"] >= 20]
        spread = f"{max(accs) - min(accs):.3f}" if len(accs) >= 2 else "-"
        s = r["summary"]
        print(f"  {r['run']:28s} {r['params_m']:6.1f}M {r['imgsz']:6d} "
              f"{s['accuracy']:7.3f} {s['balanced_accuracy']:8.3f} {s['macro_f1']:8.3f} "
              f"{s['macro_auroc']:7.3f} {spread:>9s}")
    print("\n  출처편차 = n>=20 출처 그룹 간 정확도 최대-최소. 크면 촬영 조건을 외운 것.")


def main():
    a = parse_args()
    if a.list or not a.run:
        list_runs(a.project)
        if not a.run:
            print("\n평가하려면: python3 evaluate.py --run <런 이름>")
        return

    specs = a.run
    if len(specs) == 1 and specs[0] == "all":
        run_dirs = find_runs(a.project)
        if not run_dirs:
            print(f"[에러] {a.project} 에 런이 없다.", file=sys.stderr)
            sys.exit(1)
    else:
        run_dirs = [resolve_run(s, a.project) for s in specs]

    # 런 하나면 전체 리포트, 여러 개면 개별 출력은 접고 비교표만 낸다.
    multi = len(run_dirs) > 1
    reports = []
    for i, d in enumerate(run_dirs, 1):
        if multi:
            print(f"[{i}/{len(run_dirs)}] {d.name} ...", flush=True)
        reports.append(evaluate_one(d, a, verbose=not multi))
    if multi:
        print_comparison(reports, a.split)
        print("\n  런 하나를 자세히 보려면: python3 evaluate.py --run <런 이름>")


if __name__ == "__main__":
    main()
