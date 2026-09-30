"""조건 A(BAM "Single" teacher) + teacher OOF(5-fold) 로짓 생성 — 단일 스크립트, 두 모드.

MultiTaskNet(5헤드 전부 있는 구조)을 그대로 쓰되, 손실/평가/모델선택을
--task 하나로만 좁힌다. 헤드가 5개 다 있어도 상관없다 — 나머지 헤드는 그 태스크의
손실에 전혀 기여하지 않으므로 gradient 가 가지 않고, 공유 trunk 는 오직 --task 하나의
신호로만 학습된다 = 진짜 single-task 학습(BAM 논문의 "Single" 조건과 동일한 구조:
같은 아키텍처, 태스크별 classifier 만 다름).

두 모드:
  --fold 없음(기본)      : train 전체(7,116)로 학습 -> val(고정)로 조기종료 ->
                           test(고정)로 1회 평가. 이게 조건 A 의 보고 수치
                           (teacher 성능 = 상한 기준)다.
  --fold K --folds_json F : train 에서 fold(F)!=K 인 표본으로 학습(조기종료는 여전히
                           고정 val), fold==K 표본에 대해 이 태스크의 CORN 로짓을
                           뽑아 저장한다(OOF). teacher 의 early stopping을 fold로
                           뺀 부분이 아니라 고정 val 로 하는 이유: fold 로 빠진
                           부분으로 모델 선택까지 하면 그 부분의 soft label 이
                           낙관적으로 편향된다(그 데이터로 이미 한 번 골랐으므로).
                           5개 fold 를 전부 돌리면 train 7,116장 전체에 '한 번도
                           보지 못한 teacher' 의 로짓이 생긴다. collect_oof.py 로
                           5개를 병합해 distill 단계의 soft label 로 쓴다.

사용법:
    # 조건 A (5개 축 반복)
    python3 train_single.py --task erythema --data /path/to/dataset --model pvtv2b0
    python3 train_single.py --task iga_grade --data /path/to/dataset --model pvtv2b0

    # teacher OOF (한 축당 5번, 5개 축 x 5 fold = 25번)
    python3 make_folds.py --data /path/to/dataset --out runs/folds.json
    for k in 0 1 2 3 4; do
      python3 train_single.py --task erythema --data /path/to/dataset --model pvtv2b0 \
          --fold $k --folds_json runs/folds.json
    done
    python3 collect_oof.py --task erythema --model pvtv2b0 --imgsz 224

    # 중단된 학습 이어서(조건 A/OOF fold 학습 둘 다)
    python3 train_single.py --task erythema --data /path/to/dataset --model pvtv2b0 --resume

학습이 중간에 끊겨도 --resume 로 이어갈 수 있다: 매 epoch 끝에 <out_dir>/last.pt 에
optimizer/scheduler/AMP scaler/RNG 상태까지 저장해 두고, --resume 를 주면 그
epoch 바로 다음부터 이어서 학습한다(cls_sev/train.py 와 같은 방식). --task/
--model/--imgsz/--batch/--exp/--fold 등이 ckpt 와 다르면 바로 에러로 죽는다.
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
    MODELS, MultiTaskNet, NUM_CLASSES, TASK_NAMES, ROOT, EpochScheduler,
    add_crop_args, add_resume_arg, add_sched_args, build_optimizer, collate_kd, corn,
    crop_tag, keep_stems, make_full_dataset, per_task_metrics, rng_state, seed_everything,
    try_resume,
)


def parse_args():
    p = argparse.ArgumentParser(description="조건 A / teacher OOF — 단일축 CORN 순서형 분류")
    p.add_argument("--task", required=True, choices=TASK_NAMES)
    p.add_argument("--model", choices=list(MODELS), default="pvtv2b0")
    p.add_argument("--data", required=True)
    p.add_argument("--labels_csv", default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5,
                   help="val loss 가 이 epoch 수만큼 갱신되지 않으면 조기종료(best 갱신 기준과 별개)")
    p.add_argument("--min_delta", type=float, default=0.003,
                   help="best.pt 갱신 기준(QWK)의 최소 증가폭 — early stop 과 무관")
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
    add_crop_args(p)
    add_sched_args(p)
    add_resume_arg(p)
    # --- 손실 변형(CORN 출력 형식은 유지 — 증류 teacher 호환) ---
    p.add_argument("--class_weight_power", type=float, default=0.0,
                   help="등급별 역빈도^power 가중(0=끔). train 손실에만 적용 — val loss 는 raw")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="CORN 이진 타깃 smoothing ε(0=끔). train 손실에만 적용")
    # --- teacher OOF ---
    p.add_argument("--fold", type=int, default=None,
                   help="주면 OOF 모드(fold!=이 값으로 학습, fold==이 값에 로짓 저장)")
    p.add_argument("--folds_json", default="", help="make_folds.py 산출물(--fold 필수 동반)")
    return p.parse_args()


def run_epoch_single(model, loader, task, device, opt=None, scaler=None, clip_grad=1.0,
                     weight=None, smoothing=0.0):
    train = opt is not None
    if not train:
        weight, smoothing = None, 0.0
    model.train(train)
    tot, nb = 0.0, 0
    ys, ps = [], []
    for inputs, labels, meta, _stems in loader:
        inputs = inputs.to(device, non_blocking=True)
        scale = meta["scale"].to(device, non_blocking=True)
        y = labels[task].to(device, non_blocking=True)
        m = y >= 0
        if not bool(m.any()):
            continue
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                out = model(inputs, scale=scale)
            logit = out[task][m].float()
            loss = corn.corn_loss(logit, y[m], NUM_CLASSES[task], weight=weight,
                                  smoothing=smoothing)
        if train:
            corn.backward_step(loss, opt, scaler, model, clip_grad)
        tot += float(loss.detach())
        nb += 1
        with torch.no_grad():
            pred = corn.corn_predict(logit)
        ys.append(y[m].cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_all = np.concatenate(ys) if ys else np.zeros(0, dtype=int)
    p_all = np.concatenate(ps) if ps else np.zeros(0, dtype=int)
    return tot / max(nb, 1), per_task_metrics(y_all, p_all, NUM_CLASSES[task])


@torch.no_grad()
def dump_logits(model, loader, task, device):
    """held-out 표본에 대한 이 태스크의 원본 CORN 로짓(K-1,) -> {stem: tensor}."""
    model.eval()
    out_map = {}
    for inputs, _labels, meta, stems in loader:
        inputs = inputs.to(device, non_blocking=True)
        scale = meta["scale"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=False):
            out = model(inputs, scale=scale)
        logit = out[task].float().cpu()
        for s, row in zip(stems, logit):
            out_map[s] = row.clone()
    return out_map


def main():
    args = parse_args()
    if (args.fold is None) != (not args.folds_json):
        sys.exit("[에러] --fold 와 --folds_json 은 같이 주거나 둘 다 비워야 한다.")
    oof_mode = args.fold is not None

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    model_name = MODELS[args.model]

    if oof_mode:
        folds = json.loads(Path(args.folds_json).read_text(encoding="utf-8"))
        n_folds = folds["n_folds"]
        if not (0 <= args.fold < n_folds):
            sys.exit(f"[에러] --fold 는 0..{n_folds - 1} 범위")
        name = (args.name or
               f"oof_{args.task}_{args.model}_r{args.imgsz}{crop_tag(args)}_fold{args.fold}of{n_folds}")
    else:
        name = args.name or f"A_single_{args.task}_{args.model}_r{args.imgsz}{crop_tag(args)}"
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {name} -> {out_dir}")

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

    holdout_ds = None
    if oof_mode:
        fold_map = folds["fold"]
        train_stems = [s for s, f in fold_map.items() if f != args.fold]
        holdout_stems = [s for s, f in fold_map.items() if f == args.fold]
        before, after = keep_stems(tr_ds, train_stems)
        print(f"[oof] fold {args.fold}/{n_folds}: train {before} -> {after} "
             f"(held-out {len(holdout_stems)})")
        holdout_ds, _, _ = ds_for("train", eval_tf)
        h_before, h_after = keep_stems(holdout_ds, holdout_stems)
        if h_after == 0:
            sys.exit("[에러] held-out fold 표본이 0개다 — folds_json 이 이 --data 로 만든 "
                     "것이 맞는지 확인할 것.")
        print(f"[oof] held-out 표본 {h_after}/{h_before}에 대해 학습 후 로짓을 저장한다.")

    def dl(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                          collate_fn=collate_kd, pin_memory=True, drop_last=shuffle)

    tr_ld, va_ld, te_ld = dl(tr_ds, True), dl(va_ds, False), dl(te_ds, False)
    ho_ld = dl(holdout_ds, False) if holdout_ds is not None else None

    model = MultiTaskNet(model_name, embed_dim=args.embed_dim, dropout=args.dropout,
                         ordinal=True, pretrained=args.pretrained, use_neck=True).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {args.model} ({model_name})  params={n_params:.2f}M  "
         f"단일축={args.task} (K={NUM_CLASSES[args.task]})")

    opt = build_optimizer(model, args.lr, args.backbone_lr_scale, args.weight_decay)
    sched = EpochScheduler(opt, args.scheduler, args.warmup_epochs, args.epochs,
                           cosine_epochs=args.cosine_epochs,
                           plateau_patience=args.plateau_patience)
    cls_w = None
    if args.class_weight_power > 0:
        w = class_weights(class_counts(tr_ds.records()), power=args.class_weight_power)[args.task]
        cls_w = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[loss] class weight(power={args.class_weight_power}) {args.task}: "
              f"{[round(float(v), 3) for v in w]}")
    if args.label_smoothing > 0:
        print(f"[loss] CORN label smoothing ε={args.label_smoothing}")
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

    def save(path, vm, ep, extra=None):
        ck = {"model": model.state_dict(), "args": vars(args), "model_name": model_name,
             "task": args.task, "num_classes": NUM_CLASSES[args.task],
             "val_metrics": vm, "epoch": ep}
        torch.save({**ck, **(extra or {})}, path)

    check_keys = ("task", "model", "imgsz", "embed_dim", "batch", "exp", "fold")
    start_ep, best, best_ep, no_improve, best_val_loss = try_resume(
        out_dir / "last.pt", args, model, opt, sched, scaler, None, check_keys)

    ep = start_ep - 1
    for ep in range(start_ep, args.epochs + 1):
        tr_loss, _ = run_epoch_single(model, tr_ld, args.task, device, opt, scaler,
                                      args.clip_grad, weight=cls_w,
                                      smoothing=args.label_smoothing)
        va_loss, vm = run_epoch_single(model, va_ld, args.task, device)
        sched.step(va_loss)
        score = vm["qwk"]
        print(f"[ep {ep:3d}/{args.epochs}] train {tr_loss:.4f}  val {va_loss:.4f}  "
             f"QWK {score:.4f}  acc {vm['acc']:.3f}  MAE {vm['mae']:.3f}  "
             f"lr {opt.param_groups[-1]['lr']:.2e}")
        # best.pt 갱신(QWK)과 early stop 카운터(val loss)는 서로 다른 축이다 — QWK가
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
            print(f"    -> best 갱신 (ep {ep}, QWK {best:.4f})")
        if val_loss_improved:
            print(f"    -> val loss 갱신 (ep {ep}, {best_val_loss:.4f}) -> early stop 카운터 리셋")
        elif args.patience > 0 and no_improve >= args.patience:
            print(f"[early stop] val loss 가 {args.patience} epoch 동안 갱신 없음 "
                 f"(best_val_loss={best_val_loss:.4f}, best QWK ep {best_ep})")
            break

    if not (out_dir / "best.pt").is_file():
        sys.exit(f"[에러] {out_dir/'best.pt'} 가 없다 — 한 에폭도 갱신하지 못했다.")
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if oof_mode:
        logits = dump_logits(model, ho_ld, args.task, device)
        torch.save({"task": args.task, "fold": args.fold, "n_folds": n_folds,
                   "num_classes": NUM_CLASSES[args.task], "logits": logits},
                  out_dir / "oof_logits.pt")
        print(f"[oof] {len(logits)}개 stem 의 로짓 저장 -> {out_dir/'oof_logits.pt'}")
    else:
        _, tm = run_epoch_single(model, te_ld, args.task, device)
        print(f"\n=== TEST (best ep{ckpt['epoch']}) {args.task} ===")
        print(f"  n={tm['n']}  QWK {tm['qwk']:.4f}  acc {tm['acc']:.3f}  "
             f"±1 {tm['acc1']:.3f}  MAE {tm['mae']:.3f}  F1 {tm['f1']:.3f}")
        torch.save({**ckpt, "test_metrics": tm}, out_dir / "best.pt")
        (out_dir / "test_report.json").write_text(json.dumps({
            "condition": "A_single", "task": args.task, "model": args.model,
            "model_name": model_name, "epoch": ckpt["epoch"], "val_metrics": ckpt["val_metrics"],
            "test_metrics": tm, "val_score": best, "test_score": tm["qwk"], "args": vars(args),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "done.txt").write_text(
            f"condition=A task={args.task} model={args.model} imgsz={args.imgsz} "
            f"batch={args.batch} n_train={len(tr_ds)} stopped_epoch={ep} "
            f"best_epoch={ckpt['epoch']} val_qwk={best:.4f} test_qwk={tm['qwk']:.4f} "
            f"test_acc={tm['acc']:.4f} test_mae={tm['mae']:.4f}\n", encoding="utf-8")
    print(f"\n완료 -> {out_dir}")


if __name__ == "__main__":
    main()
