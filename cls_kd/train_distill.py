"""조건 C/D/E/F/G/H — Single→Multi 지식증류 (BAM, Clark et al. 2019).

--teacher_source 로 teacher 로짓의 출처를 고른다. anneal/coherence 조합은
그대로 두고 이 값만 바꾸면 조건 문자가 C/D/E <-> F/G/H 로 갈린다:

  --teacher_source oof(기본)  : collect_oof.py 가 만든 teacher_logits/<task>.pt
                                (5-fold OOF, student 가 한 번도 보지 못한 teacher 예측)
      --anneal none                              C: 증류만(annealing 없음)
      --anneal linear                            D: BAM 본체(teacher annealing, λ:0->1 선형)
      --anneal linear --coherence                E: D + IGA-증상 일관성 손실

  --teacher_source single     : collect_a.py 가 만든 teacher_logits_a/<task>.pt
                                (조건 A 의 best.pt, train 전체로 학습 — **in-sample**,
                                 teacher 가 이미 본 표본에 대한 예측이라 OOF 보다 낙관적으로
                                 편향될 수 있다. F/G/H 는 "OOF 로 누수를 막은 증류"(C/D/E)와
                                 "조건 A 를 그대로 증류"(F/G/H)를 대조하기 위한 조건이지,
                                 F/G/H 가 방법론적으로 더 낫다는 뜻이 아니다)
      --anneal none                              F: C 와 동일 구성, teacher만 A 직접
      --anneal linear                            G: D 와 동일 구성, teacher만 A 직접
      --anneal linear --coherence                H: E 와 동일 구성, teacher만 A 직접

student 는 fold 없이 train 전체(7,116)로 **한 번**만 학습한다 — teacher 처럼 OOF 로
평가할 이유가 없다(student 예측을 다른 모델 학습에 쓰지 않고, 고정 val/test 로
조기종료·평가가 이미 가능하다).

5축 결합은 조건 B(train_multitask.py)와 똑같이 --mtl 로 고른다(기본
uncertainty=Kendall 자동가중, fixed=--iga_weight 고정 배분) — 그래야 B/C/D/E/F/G/H
사이의 성능차가 손실 결합 방식이 아니라 '증류/annealing/coherence/teacher 출처'
그 자체에서만 나온다. B와 다르게 여기서 맞추고 싶으면 --mtl/--iga_weight 를 B 와
동일하게 줄 것.

사용법:
    python3 collect_oof.py --task erythema      --model pvtv2b0 --imgsz 224
    python3 collect_oof.py --task papulation    --model pvtv2b0 --imgsz 224
    python3 collect_oof.py --task excoriation   --model pvtv2b0 --imgsz 224
    python3 collect_oof.py --task lichenification --model pvtv2b0 --imgsz 224
    python3 collect_oof.py --task iga_grade     --model pvtv2b0 --imgsz 224

    python3 train_distill.py --data /path/to/dataset --model pvtv2b0 --anneal none      # C
    python3 train_distill.py --data /path/to/dataset --model pvtv2b0 --anneal linear    # D
    python3 train_distill.py --data /path/to/dataset --model pvtv2b0 --anneal linear \
        --coherence --lambda_con 0.3                                                    # E

    python3 collect_a.py --task erythema        --model pvtv2b0 --imgsz 224
    ... (5축 반복 — run_teachers.sh MODE=full 로 조건 A 5개 모델이 먼저 있어야 한다)
    python3 train_distill.py --data /path/to/dataset --model pvtv2b0 \
        --teacher_source single --anneal none                                           # F
    python3 train_distill.py --data /path/to/dataset --model pvtv2b0 \
        --teacher_source single --anneal linear                                         # G
    python3 train_distill.py --data /path/to/dataset --model pvtv2b0 \
        --teacher_source single --anneal linear --coherence --lambda_con 0.3            # H

    python3 train_distill.py --data /path/to/dataset --model pvtv2b0 --anneal linear \
        --resume                                                     # 중단된 학습 이어서

학습이 중간에 끊겨도 --resume 로 이어갈 수 있다: 매 epoch 끝에 <out_dir>/last.pt 에
optimizer/scheduler/AMP scaler/RNG 상태까지 저장해 두고, --resume 를 주면 그
epoch 바로 다음부터 이어서 학습한다(teacher annealing λ 스텝 카운터도 epoch 수만
알면 그대로 재계산되므로 따로 저장하지 않는다). --model/--imgsz/--batch/--exp/
--mtl/--anneal/--coherence 등이 ckpt 와 다르면 바로 에러로 죽는다.
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
    MODELS, MultiTaskNet, NUM_CLASSES, ROOT, TASK_NAMES, EpochScheduler,
    add_crop_args, add_mtl_args, add_resume_arg, add_sched_args, anneal_lambda,
    build_optimizer, build_weighter, coherence_loss, collate_kd, corn, corn_distill_loss,
    crop_tag, evaluate_all, format_metrics, make_full_dataset,
    mean_qwk, mtl_tag, rng_state, seed_everything, try_resume,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS), default="pvtv2b0")
    p.add_argument("--data", required=True)
    p.add_argument("--labels_csv", default="")
    p.add_argument("--teacher_source", choices=["oof", "single"], default="oof",
                   help="teacher 로짓 출처. oof(기본)=collect_oof.py 산출(C/D/E) | "
                        "single=collect_a.py 산출, 조건 A 의 in-sample 로짓(F/G/H)")
    p.add_argument("--teacher_dir", default="",
                   help="비우면 --teacher_source 에 따라 자동 선택: oof -> "
                        "runs/teacher_logits, single -> runs/teacher_logits_a")
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
                   help="CORN 이진 타깃(hard 성분) smoothing(0/1 -> ε/2, 1-ε/2). 0=끔")
    add_crop_args(p)
    add_mtl_args(p)
    add_sched_args(p)
    add_resume_arg(p)
    # --- BAM 증류 ---
    p.add_argument("--anneal", choices=["none", "linear"], default="linear",
                   help="none=조건 C(λ=0 고정) | linear=조건 D/E(BAM 본체, λ 0->1 선형)")
    p.add_argument("--anneal_epochs", type=int, default=0,
                   help="(anneal=linear) λ 가 0->1 로 선형 증가하는 지평(epoch). 0 이면 "
                        "--epochs 를 그대로 쓴다(기존 동작). patience 로 조기종료가 "
                        "--epochs 100 보다 훨씬 일찍(보통 10~30ep) 걸리면 이 기본값은 "
                        "λ 가 절대 0.3~0.4 를 못 넘고 훈련 내내 teacher 쪽에 강하게 "
                        "쏠린 채로 끝난다 — 실제 조기종료 시점에 맞춰 더 작은 값을 줄 것.")
    p.add_argument("--coherence", action="store_true",
                   help="조건 E: IGA-증상 일관성 손실 추가")
    p.add_argument("--lambda_con", type=float, default=0.3, help="(--coherence) 일관성 손실 가중")
    p.add_argument("--con_warmup_epochs", type=int, default=0,
                   help="(--coherence) 처음 N epoch은 일관성 손실을 걸지 않음")
    return p.parse_args()


def load_teacher_logits(teacher_dir, tasks, tag, source="oof"):
    out = {}
    for t in tasks:
        f = Path(teacher_dir) / f"{t}{tag}.pt"
        if not f.is_file():
            hint = (f"collect_oof.py --task {t} --exp ... 를 먼저 돌릴 것(그 전엔 "
                    f"train_single.py --task {t} --fold 0..N-1 로 OOF 를 만들어야 한다)."
                    if source == "oof" else
                    f"collect_a.py --task {t} --exp ... 를 먼저 돌릴 것(그 전엔 "
                    f"train_single.py --task {t} 를 --fold 없이(조건 A) 돌려야 한다).")
            sys.exit(f"[에러] teacher 로짓 없음: {f}\n"
                     f"       {hint} "
                     f"student 의 --exp/--bbox_square 와 teacher 를 만들 때 준 값이 "
                     f"같아야 파일명이 맞는다.")
        d = torch.load(f, map_location="cpu")
        if d["num_classes"] != NUM_CLASSES[t]:
            sys.exit(f"[에러] {t}: teacher K={d['num_classes']} != 현재 NUM_CLASSES "
                     f"{NUM_CLASSES[t]} (--iga_merge 등 라벨 정의가 teacher 학습 때와 다르다)")
        out[t] = d["logits"]
        print(f"[teacher] {t}: {len(d['logits'])} stem 로짓 로드 ({f})")
    return out


def kd_head_losses(out, labels, stems, teacher_logits, lam, device, tasks=None, cls_w=None,
                   smoothing=0.0):
    """corn.head_losses 와 구조가 같다 — corn_loss 대신 corn_distill_loss."""
    cls_w = cls_w or {}
    losses = {}
    for t in (tasks or TASK_NAMES):
        y = labels[t].to(device, non_blocking=True)
        m = y >= 0
        if not bool(m.any()):
            continue
        idx = m.nonzero(as_tuple=True)[0]
        tmap = teacher_logits[t]
        idx_list = idx.tolist()
        missing = [stems[i] for i in idx_list if stems[i] not in tmap]
        if missing:
            raise KeyError(f"{t}: teacher 로짓에 없는 stem {len(missing)}개(예: "
                           f"{missing[:3]}) — collect_oof.py 의 커버리지가 이 --data 의 "
                           f"train 폴더와 안 맞는다.")
        t_logit = torch.stack([tmap[stems[i]] for i in idx_list]).to(device)
        s_logit = out[t][idx].float()
        losses[t] = corn_distill_loss(s_logit, t_logit, y[idx], NUM_CLASSES[t], lam,
                                      weight=cls_w.get(t), smoothing=smoothing)
    return losses


def train_one_epoch(model, loader, teacher_logits, step0, total_steps, anneal_mode,
                    lambda_con_eff, weighter, iga_weight, device, opt, scaler, clip_grad,
                    cls_w=None, smoothing=0.0):
    """BAM 공식 구현과 같은 스텝 단위 λ 스케줄: 배치마다 anneal_lambda(step, ...)를
    새로 계산한다(에폭당 한 번이 아니라). 반환: (평균 loss, 다음 step, 이 에폭
    처음/마지막 배치의 λ — 로그용)."""
    model.train(True)
    tot, nb = 0.0, 0
    step = step0
    lam_first = lam_last = None
    for inputs, labels, meta, stems in loader:
        lam = anneal_lambda(step, total_steps, anneal_mode)
        lam_first = lam if lam_first is None else lam_first
        lam_last = lam
        step += 1
        inputs = inputs.to(device, non_blocking=True)
        scale = meta["scale"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            out = model(inputs, scale=scale)
            losses = kd_head_losses(out, labels, stems, teacher_logits, lam, device,
                                    cls_w=cls_w, smoothing=smoothing)
            loss = corn.combine_losses(losses, weighter, iga_weight)
            if loss is None:
                continue
            if lambda_con_eff > 0:
                loss = loss + lambda_con_eff * coherence_loss(out, low=0.0, high=2.0)
        corn.backward_step(loss, opt, scaler, model, clip_grad)
        tot += float(loss.detach())
        nb += 1
    return tot / max(nb, 1), step, lam_first, lam_last


@torch.no_grad()
def run_epoch_eval(model, loader, device):
    """조건 B(train_multitask.py)와 정확히 같은 방식으로 평가 — hard label vs CORN 디코딩."""
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
    if args.coherence and args.anneal != "linear":
        print("[경고] --coherence 는 --anneal linear(조건 D/G) 위에 얹는 것을 상정했다"
             "(조건 E/H). anneal=none 조합은 실험 설계표에 없다.", file=sys.stderr)
    base_cond = "E" if args.coherence else ("D" if args.anneal == "linear" else "C")
    cond = base_cond if args.teacher_source == "oof" else {"C": "F", "D": "G", "E": "H"}[base_cond]

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    model_name = MODELS[args.model]
    name = args.name or f"{cond}_{args.model}_r{args.imgsz}{crop_tag(args)}{mtl_tag(args)}"
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] 조건 {cond} -> {out_dir}")

    teacher_dir = args.teacher_dir or str(ROOT / "runs" /
        ("teacher_logits" if args.teacher_source == "oof" else "teacher_logits_a"))
    teacher_logits = load_teacher_logits(teacher_dir, TASK_NAMES, crop_tag(args),
                                         source=args.teacher_source)

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
    anneal_epochs = args.anneal_epochs or args.epochs
    total_steps = len(tr_ld) * anneal_epochs
    print(f"[anneal] {len(tr_ld)} step/epoch x {anneal_epochs} anneal_epoch(--epochs={args.epochs}) "
         f"= 총 {total_steps} step (percent_done=global_step/총_step, "
         f"BAM 공식 구현과 동일한 스텝 단위 스케줄; anneal_epochs < epochs 면 그 이후는 λ=1.0 고정)")

    model = MultiTaskNet(model_name, embed_dim=args.embed_dim, dropout=args.dropout,
                         ordinal=True, pretrained=args.pretrained, use_neck=True).to(device)
    weighter = build_weighter(args.mtl, TASK_NAMES, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    mtl_desc = ("Kendall σ 자동가중" if args.mtl == "uncertainty"
               else f"고정(iga_weight={args.iga_weight})")
    print(f"[model] {args.model} ({model_name})  params={n_params:.2f}M  조건={cond}  "
         f"anneal={args.anneal}  coherence={'on(λ=' + str(args.lambda_con) + ')' if args.coherence else 'off'}  "
         f"mtl={mtl_desc}")

    cls_w = None
    if args.class_weight_power > 0:
        w_all = class_weights(class_counts(tr_ds.records()), power=args.class_weight_power)
        cls_w = {t: torch.tensor(w, dtype=torch.float32, device=device)
                 for t, w in w_all.items()}
        print(f"[loss] class weight(power={args.class_weight_power}) "
              + ", ".join(f"{t}={[round(float(v), 3) for v in w_all[t]]}" for t in TASK_NAMES))
    if args.label_smoothing > 0:
        print(f"[loss] CORN label smoothing(hard 성분) ε={args.label_smoothing}")

    opt = build_optimizer(model, args.lr, args.backbone_lr_scale, args.weight_decay,
                          extra_params=weighter.parameters() if weighter is not None else None)
    sched = EpochScheduler(opt, args.scheduler, args.warmup_epochs, args.epochs,
                          cosine_epochs=args.cosine_epochs,
                          plateau_patience=args.plateau_patience)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

    def save(path, vm, ep, lam_last, extra=None):
        ck = {"model": model.state_dict(), "args": vars(args), "model_name": model_name,
             "tasks": TASK_NAMES, "condition": cond, "lambda_teacher": lam_last,
             "val_metrics": vm, "epoch": ep,
             "weighter": weighter.state_dict() if weighter is not None else None}
        torch.save({**ck, **(extra or {})}, path)

    check_keys = ("model", "embed_dim", "imgsz", "batch", "exp", "mtl", "anneal", "coherence",
                 "teacher_source", "anneal_epochs")
    start_ep, best, best_ep, no_improve, best_val_loss = try_resume(
        out_dir / "last.pt", args, model, opt, sched, scaler, weighter, check_keys)
    # step 은 checkpoint 에 따로 안 싣는다 — 매 epoch 이 항상 len(tr_ld) 스텝이므로
    # (drop_last=True 라 배치 수가 고정된다) start_ep 만 알면 그대로 계산된다.
    step = (start_ep - 1) * len(tr_ld)

    ep = start_ep - 1
    for ep in range(start_ep, args.epochs + 1):
        lambda_con_eff = (args.lambda_con if (args.coherence and ep > args.con_warmup_epochs)
                          else 0.0)
        tr_loss, step, lam_first, lam_last = train_one_epoch(
            model, tr_ld, teacher_logits, step, total_steps, args.anneal, lambda_con_eff,
            weighter, args.iga_weight, device, opt, scaler, args.clip_grad,
            cls_w=cls_w, smoothing=args.label_smoothing)
        va_loss, vm = run_epoch_eval(model, va_ld, device)
        sched.step(va_loss)
        score = mean_qwk(vm)
        print(f"[ep {ep:3d}/{args.epochs}] λ_teacher {lam_first:.3f}->{lam_last:.3f}"
             + (f" λ_con={lambda_con_eff:.2f}" if args.coherence else "")
             + f"  train {tr_loss:.4f}  val {va_loss:.4f}  mean QWK {score:.4f}  "
               f"lr {opt.param_groups[-1]['lr']:.2e}")
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
        save(out_dir / "last.pt", vm, ep, lam_last, extra={
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": rng_state(), "best": best, "best_ep": best_ep,
            "no_improve": no_improve, "best_val_loss": best_val_loss})
        if score_improved:
            save(out_dir / "best.pt", vm, ep, lam_last)
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
    print(f"\n=== TEST (best ep{ckpt['epoch']}) 조건 {cond} mean QWK {test_score:.4f} ===")
    print(format_metrics(tm, prefix="  "))

    torch.save({**ckpt, "test_metrics": tm}, out_dir / "best.pt")
    (out_dir / "test_report.json").write_text(json.dumps({
        "condition": cond, "anneal": args.anneal, "coherence": args.coherence,
        "lambda_con": args.lambda_con if args.coherence else 0.0,
        "model": args.model, "model_name": model_name, "epoch": ckpt["epoch"],
        "val_metrics": ckpt["val_metrics"], "test_metrics": tm,
        "val_score": best, "test_score": test_score, "args": vars(args),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "done.txt").write_text(
        f"condition={cond} teacher_source={args.teacher_source} anneal={args.anneal} "
        f"coherence={args.coherence} "
        f"lambda_con={args.lambda_con if args.coherence else 0.0} model={args.model} "
        f"imgsz={args.imgsz} batch={args.batch} n_train={len(tr_ds)} stopped_epoch={ep} "
        f"best_epoch={ckpt['epoch']} val_score={best:.4f} test_score={test_score:.4f} "
        f"test_mean_qwk={mean_qwk(tm):.4f}\n", encoding="utf-8")
    print(f"\n완료 -> {out_dir}")


if __name__ == "__main__":
    main()
