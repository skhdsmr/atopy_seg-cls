"""5-fold teacher OOF 로짓 -> teacher_logits/<task>.pt 로 병합.

train_single.py 를 --fold 0..N-1 로 전부 돌린 뒤 이걸로 합친다. N개 fold 산출물이
train 전체의 stem 을 정확히 한 번씩 덮어야 한다 — 겹치거나 빠지면 fold 분할이나
--data 경로가 어긋난 것이므로 조용히 넘어가지 않고 에러로 죽는다.
(참고: 이 스크립트는 --data 를 다시 읽지 않는다 — "전체 개수가 맞는지"는 train_ds
전체 크기를 알아야 확인 가능하지만, 겹침 검사만으로도 fold 분할 실수는 대부분 걸린다.
전체 커버리지까지 엄격히 확인하려면 make_folds.py 의 --out JSON 속 "fold" 키 개수와
비교할 것 — --folds_json 을 주면 자동으로 그렇게 한다.)

사용법:
    python3 collect_oof.py --task erythema --model pvtv2b0 --imgsz 224 \
        --n_folds 5 --folds_json runs/folds.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from kd_common import crop_tag


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--project", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--folds_json", default="", help="주면 전체 stem 커버리지까지 검사")
    ap.add_argument("--exp", choices=["full", "bbox"], default="full",
                    help="train_single.py 에 준 --exp 와 똑같이 줄 것(런 폴더 이름이 갈린다)")
    ap.add_argument("--bbox_square", action="store_true",
                    help="train_single.py 에 --bbox_square 를 줬으면 여기도 줄 것")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tag = crop_tag(args)
    project = Path(args.project)
    merged = {}
    num_classes = None
    for k in range(args.n_folds):
        run_dir = project / f"oof_{args.task}_{args.model}_r{args.imgsz}{tag}_fold{k}of{args.n_folds}"
        f = run_dir / "oof_logits.pt"
        if not f.is_file():
            sys.exit(f"[에러] 없음: {f}\n"
                     f"       먼저 돌릴 것: python3 train_single.py --task {args.task} "
                     f"--model {args.model} --imgsz {args.imgsz} --fold {k} "
                     f"--folds_json <folds.json>")
        d = torch.load(f, map_location="cpu")
        if d["task"] != args.task:
            sys.exit(f"[에러] {f} 는 task={d['task']} 인데 --task {args.task} 와 다르다.")
        if num_classes is None:
            num_classes = d["num_classes"]
        elif num_classes != d["num_classes"]:
            sys.exit(f"[에러] fold 간 num_classes 불일치: {num_classes} vs {d['num_classes']}")
        overlap = set(d["logits"]) & set(merged)
        if overlap:
            sys.exit(f"[에러] fold 겹침 {len(overlap)}개(fold 분할이 어긋났다): "
                     f"{list(overlap)[:5]}")
        merged.update(d["logits"])
        print(f"[collect_oof] fold {k}: +{len(d['logits'])} (누적 {len(merged)})")

    if args.folds_json:
        want = set(json.loads(Path(args.folds_json).read_text(encoding="utf-8"))["fold"])
        missing = want - set(merged)
        extra = set(merged) - want
        if missing or extra:
            sys.exit(f"[에러] 커버리지 불일치: 누락 {len(missing)}개, 여분 {len(extra)}개 "
                     f"(folds.json 의 stem 목록과 안 맞는다)."
                     + (f" 누락 예시: {list(missing)[:5]}" if missing else "")
                     + (f" 여분 예시: {list(extra)[:5]}" if extra else ""))

    out = Path(args.out) if args.out else project / "teacher_logits" / f"{args.task}{tag}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": args.task, "num_classes": num_classes, "logits": merged}, out)
    print(f"[collect_oof] {args.task}: 총 {len(merged)} stem, K={num_classes} -> {out}")


if __name__ == "__main__":
    main()
