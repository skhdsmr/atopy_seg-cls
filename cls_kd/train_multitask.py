"""조건 B — 표준 multi-task(자연분포 CORN, 지식증류 없음).

공유 trunk + 5개 독립 CORN 헤드를 hard label 만으로 학습한다. C/D/E(`train_distill.py`)
가 쓰는 것과 완전히 같은 아키텍처(`MultiTaskNet`)·5축 결합·평가 방식을 그대로 쓰고,
유일한 차이는 손실이 지식증류가 아니라 그냥 CORN hard-label 손실이라는 것뿐이다 —
그래야 B와 C/D/E의 성능차가 오직 "무엇으로 학습했는가"에서만 나온다(BAM 논문의
"Multi" 조건에 해당).

5축 결합은 --mtl 로 고른다(기본 uncertainty):
    uncertainty(기본) : Kendall 불확실성 자동가중(UncertaintyWeighter)
    fixed             : --iga_weight*IGA손실 + (1-iga_weight)*mean(증상4종손실) 고정 배분

사용법:
    python3 train_multitask.py --data /path/to/dataset --model pvtv2b0
    python3 train_multitask.py --data /path/to/dataset --mtl fixed --iga_weight 0.5
    python3 train_multitask.py --data /path/to/dataset --resume   # 중단된 학습 이어서

학습이 중간에 끊겨도(SSH 끊김, OOM 등) --resume 로 이어갈 수 있다: 매 epoch 끝에
<out_dir>/last.pt 에 optimizer/scheduler/AMP scaler/RNG 상태까지 저장해 두고,
--resume 를 주면 그 epoch 바로 다음부터 이어서 학습한다(cls_sev/train.py 와 같은
방식). --model/--imgsz/--batch/--exp/--mtl 등이 ckpt 와 다르면 바로 에러로 죽는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import class_counts, class_weights
from kd_common import (
    MODELS, MultiTaskNet, ROOT, TASK_NAMES, EpochScheduler,
    add_crop_args, add_mtl_args, add_resume_arg, add_sched_args, build_optimizer,
    build_weighter, collate_kd, corn, crop_tag, evaluate_all, format_metrics,
    make_full_dataset, mean_qwk, mtl_tag, rng_state, seed_everything,
    try_resume,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS), default="pvtv2b0")
    p.add_argument("--data", required=True)
    p.add_argument("--labels_csv", default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5,
                   help="val loss 가 이 epoch 수만큼 갱신되지 않으면 조기종료(best 갱신 기준과 별개)")
    p.add_argument("--min_delta", type=float, default=0.003,
                   help="best.pt 갱신 기준(mean QWK)의 최소 증가폭 — early stop 과 무관")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--name", default="")
    p.add_argument("--no_amp", dest="amp", action="store_false", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false", default=True)
    p.add_argument("--class_weight_power", type=float, default=0.0,
                   help="축별 역빈도^power 클래스 가중(dataset.py::class_weights). 0=끔")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="CORN 이진 타깃 smoothing(0/1 -> ε/2, 1-ε/2). 0=끔")
    add_crop_args(p)
    add_mtl_args(p)
    add_sched_args(p)
    add_resume_arg(p)
    return p.parse_args()


def train_one_epoch(model, loader, weighter, iga_weight, device, opt, scaler, clip_grad,
                    cls_w=None, smoothing=0.0):
    model.train(True)
    tot, nb = 0.0, 0
    for inputs, labels, meta, _stems in loader:
        inputs = inputs.to(device, non_blocking=True)
        scale = meta["scale"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            out = model(inputs, scale=scale)
            losses = corn.head_losses(out, labels, device, cls_w=cls_w, smoothing=smoothing)
            loss = corn.combine_losses(losses, weighter, iga_weight)
            if loss is None:
                continue
        corn.backward_step(loss, opt, scaler, model, clip_grad)
        tot += float(loss.detach())
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def run_epoch_eval(model, loader, device):
    model.eval()
    ys = {t: [] for t in TASK_NAMES}
    ps = {t: [] for t in TASK_NAMES}
    tot, nb = 0.0, 0
    for inputs, labels, meta, _stems in loader:
        inputs = inputs.to(device, non_blocking=True)
        scale = meta["scale"].to(device, non_blocking=True)
        out = model(inputs, scale=scale)
        losses = corn.head_losses(out, labels, device)
        if losses:
            tot += float(sum(losses.values()) / len(losses))
            nb += 1
        pred = corn.head_predict(out)
        for t in TASK_NAMES:
            ys[t].append(labels[t].numpy())
            ps[t].append(pred[t].cpu().numpy())
    y = {t: np.concatenate(v) for t, v in ys.items()}
    p = {t: np.concatenate(v) for t, v in ps.items()}
    return tot / max(nb, 1), evaluate_all(y, p)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    model_name = MODELS[args.model]
    name = args.name or f"B_{args.model}_r{args.imgsz}{crop_tag(args)}{mtl_tag(args)}"
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] 조건 B -> {out_dir}")

    train_tf, eval_tf = corn.build_transforms(args.imgsz)

    def ds_for(split, transform):
        return make_full_dataset(args.data, split, transform, args.labels_csv or None,
                                 exp=args.exp, masks=args.masks or None,
                                 margin=args.margin, bbox_square=args.bbox_square)

    tr_ds, index, info = ds_for("train", train_tf)
    va_ds, _, _ = ds_for("val", eval_tf)
    te_ds, _, _ = ds_for("test", eval_tf)
    print(f"[labels] {info['src']}  train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}"
         + (f"  [exp=bbox margin={args.margin} square={args.bbox_square}]"
            if args.exp == "bbox" else ""))

    def dl(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                          collate_fn=collate_kd, pin_memory=True, drop_last=shuffle)

    tr_ld, va_ld, te_ld = dl(tr_ds, True), dl(va_ds, False), dl(te_ds, False)

    model = MultiTaskNet(model_name, embed_dim=args.embed_dim, dropout=args.dropout,
                         ordinal=True, pretrained=args.pretrained, use_neck=True).to(device)
    weighter = build_weighter(args.mtl, TASK_NAMES, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    mtl_desc = ("Kendall σ 자동가중" if args.mtl == "uncertainty"
               else f"고정(iga_weight={args.iga_weight})")
    print(f"[model] {args.model} ({model_name})  params={n_params:.2f}M  조건=B(표준 multi-task)  "
         f"mtl={mtl_desc}")

    cls_w = None
    if args.class_weight_power > 0:
        w_all = class_weights(class_counts(tr_ds.records()), power=args.class_weight_power)
        cls_w = {t: torch.tensor(w, dtype=torch.float32, device=device)
                 for t, w in w_all.items()}
        print(f"[loss] class weight(power={args.class_weight_power}) "
              + ", ".join(f"{t}={[round(float(v), 3) for v in w_all[t]]}" for t in TASK_NAMES))
    if args.label_smoothing > 0:
        print(f"[loss] CORN label smoothing ε={args.label_smoothing}")

    opt = build_optimizer(model, args.lr, args.backbone_lr_scale, args.weight_decay,
                          extra_params=weighter.parameters() if weighter is not None else None)
    sched = EpochScheduler(opt, args.scheduler, args.warmup_epochs, args.epochs,
                          cosine_epochs=args.cosine_epochs,
                          plateau_patience=args.plateau_patience)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

    def save(path, vm, ep, extra=None):
        ck = {"model": model.state_dict(), "args": vars(args), "model_name": model_name,
             "tasks": TASK_NAMES, "condition": "B", "val_metrics": vm, "epoch": ep,
             "weighter": weighter.state_dict() if weighter is not None else None}
        torch.save({**ck, **(extra or {})}, path)

    check_keys = ("model", "embed_dim", "imgsz", "batch", "exp", "mtl")
    start_ep, best, best_ep, no_improve, best_val_loss = try_resume(
        out_dir / "last.pt", args, model, opt, sched, scaler, weighter, check_keys)

    ep = start_ep - 1
    for ep in range(start_ep, args.epochs + 1):
        tr_loss = train_one_epoch(model, tr_ld, weighter, args.iga_weight, device, opt,
                                  scaler, args.clip_grad, cls_w=cls_w,
                                  smoothing=args.label_smoothing)
        va_loss, vm = run_epoch_eval(model, va_ld, device)
        sched.step(va_loss)
        score = mean_qwk(vm)
        print(f"[ep {ep:3d}/{args.epochs}] train {tr_loss:.4f}  val {va_loss:.4f}  "
             f"mean QWK {score:.4f}  lr {opt.param_groups[-1]['lr']:.2e}")
        print(format_metrics(vm, prefix="    "))
        # best.pt 갱신(mean QWK)과 early stop 카운터(val loss)는 서로 다른 축이다 — QWK가
        # 갱신되지 않아도 val loss 가 내려가면 조기종료 카운터는 리셋된다(그 반대도 마찬가지).
        score_improved = score > best + args.min_delta
        if score_improved:
            best, best_ep = score, ep
        val_loss_improved = va_loss < best_val_loss
        if val_loss_improved:
            best_val_loss, no_improve = va_loss, 0
        else:
            no_improve += 1
        save(out_dir / "last.pt", vm, ep, extra={
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": rng_state(), "best": best, "best_ep": best_ep,
            "no_improve": no_improve, "best_val_loss": best_val_loss})
        if score_improved:
            save(out_dir / "best.pt", vm, ep)
            print(f"    -> best 갱신 (ep {ep}, {best:.4f})")
        if val_loss_improved:
            print(f"    -> val loss 갱신 (ep {ep}, {best_val_loss:.4f}) -> early stop 카운터 리셋")
        elif args.patience > 0 and no_improve >= args.patience:
            print(f"[early stop] val loss 가 {args.patience} epoch 동안 갱신 없음 "
                 f"(best_val_loss={best_val_loss:.4f}, best mean QWK ep {best_ep})")
            break

    if not (out_dir / "best.pt").is_file():
        sys.exit(f"[에러] {out_dir/'best.pt'} 가 없다 — 한 에폭도 갱신하지 못했다.")
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _, tm = run_epoch_eval(model, te_ld, device)
    test_score = mean_qwk(tm)
    print(f"\n=== TEST (best ep{ckpt['epoch']}) 조건 B mean QWK {test_score:.4f} ===")
    print(format_metrics(tm, prefix="  "))

    torch.save({**ckpt, "test_metrics": tm}, out_dir / "best.pt")
    (out_dir / "test_report.json").write_text(json.dumps({
        "condition": "B", "model": args.model, "model_name": model_name,
        "epoch": ckpt["epoch"], "val_metrics": ckpt["val_metrics"], "test_metrics": tm,
        "val_score": best, "test_score": test_score, "args": vars(args),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "done.txt").write_text(
        f"condition=B model={args.model} imgsz={args.imgsz} batch={args.batch} "
        f"n_train={len(tr_ds)} stopped_epoch={ep} best_epoch={ckpt['epoch']} "
        f"val_score={best:.4f} test_score={test_score:.4f} "
        f"test_mean_qwk={mean_qwk(tm):.4f}\n", encoding="utf-8")
    print(f"\n완료 -> {out_dir}")


if __name__ == "__main__":
    main()
