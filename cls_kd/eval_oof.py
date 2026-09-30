"""teacher OOF(5-fold) 로짓 평가 — fold별 개별 평가 + 전체(합친 train) 평가,
그리고 그 soft 확률이 어떻게 생겼는지(교정/엔트로피/샘플별 분포) 확인.

train_single.py --fold 0..N-1 로 만든 oof_logits.pt 는 그 fold가 "한 번도 보지
못한" held-out 표본에 대한 teacher 예측이다(collect_oof.py 가 이걸 이어붙여
train_distill.py 의 soft label 로 쓴다). 이 스크립트는 그 원본 fold 조각들을
다시 읽어서:

  1) fold 하나하나 개별 평가 — 어떤 fold의 teacher가 유난히 나쁜지 본다.
  2) 5개 fold를 합친 전체 평가 — collect_oof.py 가 만드는 teacher_logits/<task>.pt
     와 커버리지가 같다. 즉 "증류에 실제로 쓰이는 soft label을 hard 라벨로
     디코딩했을 때의 품질"이다.
  3) --show_probs N — CORN 로짓을 클래스별 확률 분포로 디코딩해서(corn_class_probs,
     corn_distill_loss 가 blend 하는 레벨별 조건부확률과는 다른, 사람이 읽는
     카테고리 분포) 표본 N개의 실제 숫자를 보여주고, 진짜 라벨에 실린 확률질량
     (교정 proxy)과 분포 엔트로피(확신도)를 fold별/전체로 요약한다 — "soft
     확률이 어떻게 흘러가는지"를 직접 확인하는 부분.

사용법:
    python3 eval_oof.py --task erythema --data /path --model pvtv2b0 --imgsz 224
    python3 eval_oof.py --task iga_grade --data /path --model pvtv2b0 --imgsz 224 \
        --show_probs 5                      # fold별/전체 표본 5개씩 분포 출력
    python3 eval_oof.py --task erythema --data /path --model pvtv2b0 --imgsz 224 \
        --exp bbox --bbox_square            # bbox 로 만든 teacher 평가
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from kd_common import (
    NUM_CLASSES, ROOT, TASKS,
    build_label_index, confusion, corn, corn_class_probs, crop_tag, format_confusion,
    per_task_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(NUM_CLASSES))
    p.add_argument("--data", required=True)
    p.add_argument("--labels_csv", default="")
    p.add_argument("--model", required=True)
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--exp", choices=["full", "bbox"], default="full",
                   help="OOF 를 만들 때 준 --exp 와 같아야 한다(런 폴더 이름이 갈린다)")
    p.add_argument("--bbox_square", action="store_true",
                   help="OOF 를 만들 때 --bbox_square 를 줬으면 여기도 줄 것")
    p.add_argument("--no_confusion", dest="confusion", action="store_false", default=True,
                   help="혼동행렬 출력을 끈다")
    p.add_argument("--show_probs", type=int, default=0,
                   help="fold별/전체에서 표본 N개의 CORN 디코딩 확률분포를 출력한다(0=끔)")
    p.add_argument("--seed", type=int, default=0, help="--show_probs 표본 추출 시드")
    return p.parse_args()


def decode_fold(path, task, true_index):
    """oof_logits.pt 하나 -> (stems, true, pred, probs, num_classes). 진짜 라벨이
    없거나(UNKNOWN) labels.csv 에 그 stem 자체가 없는 표본은 뺀다(둘 다 흔치 않은
    상황이라 다르지 않게 다루지만, 있으면 [경고]로 몇 개인지는 알려준다)."""
    d = torch.load(path, map_location="cpu")
    if d["task"] != task:
        sys.exit(f"[에러] {path} 는 task={d['task']} 인데 --task {task} 와 다르다.")
    num_classes = d.get("num_classes", NUM_CLASSES[task])
    stems = list(d["logits"].keys())
    logits = torch.stack([d["logits"][s] for s in stems]).float()
    probs = corn_class_probs(logits)
    pred = corn.corn_predict(logits).numpy()

    true, keep = [], []
    dropped = 0
    for i, s in enumerate(stems):
        rec = true_index.get(s)
        if rec is None or rec[task] < 0:
            dropped += 1
            continue
        true.append(rec[task])
        keep.append(i)
    if dropped:
        print(f"[경고] {path.parent.name}: 진짜 라벨 없어 제외 {dropped}/{len(stems)}",
             file=sys.stderr)
    keep = np.array(keep, dtype=int)
    return ([stems[i] for i in keep], np.array(true, dtype=int), pred[keep],
           probs[keep].numpy(), num_classes)


def print_metrics(name, true, pred, num_classes, task, show_confusion):
    m = per_task_metrics(true, pred, num_classes)
    print(f"\n[{name}] n={m['n']}  QWK {m['qwk']:.4f}  acc {m['acc']:.3f}  "
         f"±1 {m['acc1']:.3f}  MAE {m['mae']:.3f}  F1 {m['f1']:.3f}")
    if show_confusion and m["n"] > 0:
        cm = confusion(true, pred, num_classes)
        print(format_confusion(cm, TASKS[task], title=f"  혼동행렬({name})"))
    return m


def prob_summary(name, true, probs, class_names, rng, show_n):
    """soft 확률이 어떻게 흘러가는지: 진짜 라벨에 실린 확률질량(교정 proxy)과
    분포 엔트로피(확신도)를 요약하고, 원하면 표본 몇 개의 실제 분포를 찍는다."""
    if len(true) == 0:
        return
    eps = 1e-12
    p_true = probs[np.arange(len(true)), true]
    entropy = -(probs * np.log(probs + eps)).sum(axis=1)
    max_entropy = np.log(probs.shape[1])
    print(f"  [soft 확률 요약: {name}] 진짜 등급 확률질량 평균 {p_true.mean():.3f}"
         f"(±{p_true.std():.3f})  엔트로피 평균 {entropy.mean():.3f}/{max_entropy:.3f}"
         f"(낮을수록 확신)")
    if show_n <= 0:
        return
    idx = rng.choice(len(true), size=min(show_n, len(true)), replace=False)
    header = "    " + "".join(f"{c[:8]:>9s}" for c in class_names) + "     true  argmax"
    print(header)
    for i in idx:
        row = "".join(f"{v:9.3f}" for v in probs[i])
        print(f"    {row}   {true[i]:5d}  {int(np.argmax(probs[i])):6d}")


def main():
    args = parse_args()
    index, info = build_label_index(args.data, args.labels_csv or None)
    print(f"[labels] {info['src']}")

    tag = crop_tag(args)
    project = Path(args.project)
    class_names = TASKS[args.task]
    rng = np.random.RandomState(args.seed)

    all_true, all_pred, all_probs = [], [], []
    for k in range(args.n_folds):
        run_dir = project / f"oof_{args.task}_{args.model}_r{args.imgsz}{tag}_fold{k}of{args.n_folds}"
        f = run_dir / "oof_logits.pt"
        if not f.is_file():
            print(f"[경고] 없음, 건너뜀: {f}", file=sys.stderr)
            continue
        stems, true, pred, probs, num_classes = decode_fold(f, args.task, index)
        print_metrics(f"fold {k}", true, pred, num_classes, args.task, args.confusion)
        prob_summary(f"fold {k}", true, probs, class_names, rng, args.show_probs)
        all_true.append(true)
        all_pred.append(pred)
        all_probs.append(probs)

    if not all_true:
        sys.exit("[에러] 평가할 fold 결과가 하나도 없다 — --project/--model/--imgsz/--exp 가 "
                 "train_single.py 에 준 값과 같은지 확인할 것.")

    true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    probs_all = np.concatenate(all_probs)
    print(f"\n=== 전체 OOF(train 전체, {len(all_true)}/{args.n_folds}-fold 합산) "
         f"— {args.task} ===")
    print_metrics("전체", true_all, pred_all, NUM_CLASSES[args.task], args.task, args.confusion)
    prob_summary("전체", true_all, probs_all, class_names, rng, args.show_probs)


if __name__ == "__main__":
    main()
