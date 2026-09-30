"""ckpt 하나(조건 A~H 어느 것이든) -> split(기본 test) 축별 지표 + 혼동행렬.

`train_single.py`(조건 A/OOF)와 `train_multitask.py`/`train_distill.py`(조건
B/C/D/E/F/G/H)가 저장한 `best.pt`(또는 `--last` 로 `last.pt`)를 전부 받는다 — ckpt
안에 `"task"`(단일축) 키가 있으면 그 축만, `"tasks"`(다축) 키가 있으면 5축
전부 평가한다. 사전학습 가중치는 필요 없다(ckpt 가 backbone 까지 전부
덮어쓴다 — `cls_sev/evaluate.py` 와 같은 이유).

평가 설정(imgsz/exp/masks/margin/bbox_square/model)은 ckpt 안의 학습 당시
`args` 를 그대로 따른다 — 학습 때와 다른 해상도/crop 방식으로 재평가하면
숫자가 비교 불가능해지기 때문이다. `--data` 도 비우면 학습 당시 경로를
그대로 재사용한다(같은 머신에서 재평가하는 게 보통이라서).

사용법:
    python3 eval.py --run runs/A_single_erythema_pvtv2b0_r224          # --data 자동
    python3 eval.py --ckpt runs/D_pvtv2b0_r224/best.pt --data /path
    python3 eval.py --run runs/D_pvtv2b0_r224 --split val
    python3 eval.py --run runs/D_pvtv2b0_r224 --last                    # last.pt 평가

여러 런을 한 번에 훑으려면 evaluate.sh 참고(`PROJECT` 아래 A_single_*/B_*/
C_*/D_*/E_*/F_*/G_*/H_* 를 전부 찾아 이 스크립트를 돌린다).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kd_common import (
    MultiTaskNet, NUM_CLASSES, TASK_NAMES, TASKS,
    collate_kd, confusion, corn, format_confusion, format_metrics, make_full_dataset,
    per_task_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="", help="ckpt 경로 직접 지정(--run 대신)")
    p.add_argument("--run", default="", help="run 폴더 — 안의 best.pt(또는 --last 면 last.pt)를 본다")
    p.add_argument("--last", action="store_true", help="best.pt 대신 last.pt 평가")
    p.add_argument("--data", default="", help="비우면 ckpt 저장 당시 --data 를 재사용")
    p.add_argument("--labels_csv", default="")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="", help="결과 JSON 저장 경로. 비우면 ckpt 옆 eval_<split>.json")
    p.add_argument("--no_confusion", dest="confusion", action="store_false", default=True,
                   help="혼동행렬 출력을 끈다(로그를 줄이고 싶으면)")
    p.add_argument("--no_save", dest="save", action="store_false", default=True,
                   help="eval_<split>.json 저장을 끈다")
    return p.parse_args()


def resolve_ckpt_path(args):
    if args.ckpt:
        return Path(args.ckpt)
    if args.run:
        return Path(args.run) / ("last.pt" if args.last else "best.pt")
    sys.exit("[에러] --ckpt 또는 --run 중 하나는 줘야 한다.")


@torch.no_grad()
def run_eval(model, loader, tasks, device):
    model.eval()
    ys = {t: [] for t in tasks}
    ps = {t: [] for t in tasks}
    for inputs, labels, meta, _stems in loader:
        inputs = inputs.to(device, non_blocking=True)
        scale = meta["scale"].to(device, non_blocking=True)
        out = model(inputs, scale=scale)
        for t in tasks:
            y = labels[t]
            m = y >= 0
            if not bool(m.any()):
                continue
            pred = corn.corn_predict(out[t][m].float()).cpu()
            ys[t].append(y[m].numpy())
            ps[t].append(pred.numpy())
    y = {t: (np.concatenate(v) if v else np.zeros(0, dtype=int)) for t, v in ys.items()}
    p = {t: (np.concatenate(v) if v else np.zeros(0, dtype=int)) for t, v in ps.items()}
    return y, p


def main():
    args = parse_args()
    ckpt_path = resolve_ckpt_path(args)
    if not ckpt_path.is_file():
        sys.exit(f"[에러] ckpt 없음: {ckpt_path}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    prev = ckpt.get("args", {}) or {}

    tasks = [ckpt["task"]] if "task" in ckpt else list(ckpt.get("tasks", TASK_NAMES))
    condition = ckpt.get("condition", "A" if "task" in ckpt else "?")
    model_name = ckpt.get("model_name")
    if model_name is None:
        sys.exit(f"[에러] ckpt 에 model_name 이 없다 — {ckpt_path} 가 cls_kd 산출물이 맞는지 확인할 것.")

    data = args.data or prev.get("data")
    if not data:
        sys.exit("[에러] --data 를 못 정했다 — ckpt 의 args 에도 저장돼 있지 않다. 직접 줄 것.")

    imgsz = int(prev.get("imgsz", 224))
    embed_dim = int(prev.get("embed_dim", 512))
    exp = prev.get("exp", "full")
    masks = prev.get("masks") or None
    margin = float(prev.get("margin", 0.15))
    bbox_square = bool(prev.get("bbox_square", False))

    print(f"[eval] {ckpt_path}  조건={condition}  축={tasks}  epoch={ckpt.get('epoch')}")
    print(f"[eval] data={data}  split={args.split}  imgsz={imgsz}  exp={exp}"
         + (f"  margin={margin}  square={bbox_square}" if exp == "bbox" else ""))

    _, eval_tf = corn.build_transforms(imgsz)
    ds, index, info = make_full_dataset(data, args.split, eval_tf, args.labels_csv or None,
                                        exp=exp, masks=masks, margin=margin,
                                        bbox_square=bbox_square)
    print(f"[labels] {info['src']}  {args.split}={len(ds)}")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_kd, pin_memory=True)

    # pretrained=False: ckpt 가 backbone 까지 전부 덮어쓰므로 로컬 사전학습
    # 가중치 파일이 없어도 평가는 그대로 된다.
    model = MultiTaskNet(model_name, embed_dim=embed_dim, dropout=0.0, ordinal=True,
                         pretrained=False, use_neck=True).to(device)
    model.load_state_dict(ckpt["model"])

    y, p = run_eval(model, loader, tasks, device)

    metrics_by_task = {}
    cm_by_task = {}
    for t in tasks:
        m = per_task_metrics(y[t], p[t], NUM_CLASSES[t])
        metrics_by_task[t] = m
        print(f"\n[{t}] n={m['n']}  QWK {m['qwk']:.4f}  acc {m['acc']:.3f}  "
             f"±1 {m['acc1']:.3f}  MAE {m['mae']:.3f}  F1 {m['f1']:.3f}")
        if m["n"] == 0:
            continue
        cm = confusion(y[t], p[t], NUM_CLASSES[t])
        cm_by_task[t] = cm
        if args.confusion:
            print(format_confusion(cm, TASKS[t], title=f"  혼동행렬({t})"))

    if len(tasks) > 1:
        print(f"\n=== 전체({args.split}, 조건 {condition}) ===")
        print(format_metrics(metrics_by_task, prefix="  "))

    if args.save:
        out_path = Path(args.out) if args.out else ckpt_path.parent / f"eval_{args.split}.json"
        out_path.write_text(json.dumps({
            "ckpt": str(ckpt_path), "condition": condition, "split": args.split,
            "tasks": tasks, "epoch": ckpt.get("epoch"), "data": data,
            "metrics": metrics_by_task,
            "confusion": {t: cm.tolist() for t, cm in cm_by_task.items()},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n완료 -> {out_path}")


if __name__ == "__main__":
    main()
