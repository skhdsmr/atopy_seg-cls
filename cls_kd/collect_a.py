"""조건 A(BAM "Single" teacher, train 전체로 학습한 best.pt) -> teacher_logits_a/
<task>[_bbox[_sq]].pt — train_distill.py --teacher_source single(조건 F/G/H) 용.

collect_oof.py 가 만드는 teacher_logits/<task>.pt 는 5-fold OOF라 student가 한
번도 보지 못한 teacher 예측이다(누수 없음). 이 스크립트가 만드는 로짓은 그와
다르다 — A_single_<task>_.../best.pt 는 train 전체(7,116)로 학습된 모델이라,
여기서 뽑는 로짓은 teacher가 이미 본 표본에 대한 예측(in-sample)이다. teacher가
train 표본을 일부 암기했을 수 있어 OOF보다 낙관적으로 편향된 soft label을 줄 수
있다 — F/G/H는 "OOF로 누수를 막은 증류"(C/D/E)와 "조건 A를 그대로 증류"(F/G/H)를
대조하기 위한 조건이지, F/G/H가 방법론적으로 더 나아서 있는 게 아니다.

사용법:
    python3 train_single.py --task erythema --data /path/to/dataset --model pvtv2b0   # 조건 A
    ... (5개 축 반복, 이미 했다면 생략)

    python3 collect_a.py --task erythema        --model pvtv2b0 --imgsz 224
    python3 collect_a.py --task papulation      --model pvtv2b0 --imgsz 224
    python3 collect_a.py --task excoriation     --model pvtv2b0 --imgsz 224
    python3 collect_a.py --task lichenification --model pvtv2b0 --imgsz 224
    python3 collect_a.py --task iga_grade       --model pvtv2b0 --imgsz 224
    # (run_distill.sh COND=F/G/H/all 은 이 단계를 자동으로 먼저 돌린다)
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kd_common import (
    MODELS, MultiTaskNet, NUM_CLASSES, ROOT, add_crop_args, collate_kd, corn, crop_tag,
    make_full_dataset,
)
from train_single import dump_logits


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", choices=list(MODELS), default="pvtv2b0")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--data", required=True,
                    help="조건 A(train_single.py) 를 학습할 때 쓴 것과 같은 --data")
    ap.add_argument("--labels_csv", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--project", default=str(ROOT / "runs"),
                    help="A_single_<task>_... 런이 들어있는 project 폴더")
    add_crop_args(ap)
    ap.add_argument("--out", default="", help="비우면 <project>/teacher_logits_a/<task>{태그}.pt")
    return ap.parse_args()


def main():
    args = parse_args()
    tag = crop_tag(args)
    run_dir = Path(args.project) / f"A_single_{args.task}_{args.model}_r{args.imgsz}{tag}"
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        sys.exit(f"[에러] 없음: {ckpt_path}\n"
                 f"       먼저 돌릴 것: python3 train_single.py --task {args.task} "
                 f"--model {args.model} --imgsz {args.imgsz} --data {args.data}"
                 + (f" --exp {args.exp}" + (" --bbox_square" if args.bbox_square else "")
                    if args.exp == "bbox" else ""))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt["task"] != args.task:
        sys.exit(f"[에러] {ckpt_path} 는 task={ckpt['task']} 인데 --task {args.task} 와 다르다.")
    ck_args = ckpt["args"]

    model = MultiTaskNet(ckpt["model_name"], embed_dim=ck_args["embed_dim"],
                         dropout=ck_args["dropout"], ordinal=True, pretrained=False,
                         use_neck=True).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, eval_tf = corn.build_transforms(args.imgsz)
    tr_ds, _, info = make_full_dataset(args.data, "train", eval_tf, args.labels_csv or None,
                                       exp=args.exp, masks=args.masks or None,
                                       margin=args.margin, bbox_square=args.bbox_square)
    print(f"[labels] {info['src']}  train={len(tr_ds)}  (조건 A 가 이미 본 표본 — in-sample)")
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       collate_fn=collate_kd, pin_memory=True)

    logits = dump_logits(model, tr_ld, args.task, device)
    out = (Path(args.out) if args.out
          else Path(args.project) / "teacher_logits_a" / f"{args.task}{tag}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task": args.task, "num_classes": NUM_CLASSES[args.task], "logits": logits,
               "source": "A_single(in-sample, not OOF)"}, out)
    print(f"[collect_a] {args.task}: 총 {len(logits)} stem, K={NUM_CLASSES[args.task]} -> {out}"
         f"  (주의: OOF 아님 — teacher 가 이미 본 표본에 대한 로짓)")


if __name__ == "__main__":
    main()
