"""아토피 5축 MaMNet(MbN+SFEN+CFEN+ASPP) 학습.

구조 설명은 model.py 상단, 라벨 처리는 dataset.py 상단 docstring 참고.
여기 있는 학습 레시피는 논문 Table 2 를 기본으로 하되 세 가지를 바꿨다.

1. **선택기준 = 0.5*IGA QWK + 0.5*증상4 QWK 평균** (논문은 Joint Ac).
   5축 동시정답률은 값이 너무 낮아 선택 신호가 노이즈다. joint_ac 는 참고로만 찍는다.
2. **early stop 도 같은 선택점수 기준** (논문 4.1.3 은 *test* loss 기준이라 test 가 샌다).
3. **warmup 기본 3ep** (논문 10ep). 논문은 학습셋이 300장이라 사전학습 MbN 을 오래
   묶어둘 이유가 있었지만 여기는 1,400~9,150장이다.

손실은 CORN(순서형, 기본) 또는 CE, 5축 결합은 Kendall uncertainty(기본) 또는
고정가중(`--mtl fixed --iga_weight 0.5`). 논문의 alpha/beta 그리드는 2태스크라 가능했던
것이고 5축이면 차원이 5개라 못 한다.

직접 실행:
    python3 train.py --groups sep --sfen share --cfen all
    python3 train.py --no_aspp                      # ASPP 축 대조군
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
OGW = ROOT.parent
sys.path.insert(0, str(ROOT))

from dataset import (AtopyDataset, GROUP_PRESETS, IGA_GROUP, TASK_NAMES,  # noqa: E402
                     build_label_space)
from metrics import all_metrics, format_confusion, sel_score               # noqa: E402
from model import AtopyMaMNet                                              # noqa: E402
from sampler import build_ibb_sampler, labels_from_dataset, select_edges   # noqa: E402

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description="아토피 5축 MaMNet(MbN+SFEN+CFEN+ASPP)")
    p.add_argument("--data", default=str(OGW / "dataset_all_final" / "images"),
                   help="labels.csv + {train,val,test}/ 가 있는 폴더")
    p.add_argument("--labels_csv", default=None, help="기본값: <data>/labels.csv")
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_lr", type=float, default=1e-3)
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="MbN 동결 구간(논문 10ep 은 300장 기준이라 줄였다)")
    p.add_argument("--patience", type=int, default=10, help="선택점수 기준 early stop")
    p.add_argument("--rlrop_patience", type=int, default=3)
    p.add_argument("--decay_drop", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    # --- 구조 축 ---
    p.add_argument("--groups", choices=list(GROUP_PRESETS), default="sep",
                   help="sep=IGA/급성3/태선화(branch5) | merged=논문 2분기 | flat=태스크당 1분기")
    p.add_argument("--sfen", choices=["share", "task", "group"], default="share",
                   help="share=CA 태스크별+SA 그룹공유(기본) | task=논문 그대로 | group=그룹별")
    p.add_argument("--sfen_k", type=int, default=9, help="SA 비대칭 conv 커널(논문 최적)")
    p.add_argument("--reduction", type=int, default=4, help="CA 축소비 r")
    p.add_argument("--no_sfen", action="store_true")
    p.add_argument("--cfen", default="all",
                   help="all | none | 쌍 목록 (예: 'lich<-acute,iga<-acute')")
    p.add_argument("--no_aspp", action="store_true", help="IGA 분기 ASPP 끄기(sweep 축)")
    p.add_argument("--iga_skip", choices=["auto", "f", "none"], default="auto",
                   help="IGA 분기의 F skip: auto=ASPP 있으면 빼고 없으면 넣음(기본)")
    p.add_argument("--head_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--init", choices=["vgg16", "scratch"], default="vgg16")
    # --- 손실 ---
    p.add_argument("--loss", choices=["corn", "ce"], default="corn")
    p.add_argument("--mtl", choices=["uncertainty", "fixed"], default="uncertainty")
    p.add_argument("--iga_weight", type=float, default=0.5, help="(mtl=fixed 일 때만)")
    p.add_argument("--class_weight", action="store_true", help="(loss=ce 일 때만) 역빈도 가중")
    p.add_argument("--cw_power", type=float, default=0.5, help="역빈도의 거듭제곱(0.5=제곱근)")
    p.add_argument("--cw_clip", type=float, default=3.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--no_aug", action="store_true")
    # --- IBB 샘플러(OCNN-IT) — 기본 off. 켜면 학습 배치 구성만 바뀌고 모델은 그대로다 ---
    p.add_argument("--ibb", action="store_true",
                   help="2-step IBB 샘플러(Walecki+ 2017 Alg.1 축소판). sampler.py 참고")
    p.add_argument("--ibb_alpha", type=float, default=0.5, help="역빈도 지수(0=자연분포, 1=완전균등)")
    p.add_argument("--ibb_alpha_end", type=float, default=None, help="에폭에 걸쳐 alpha 어닐링")
    p.add_argument("--ibb_cap", type=float, default=4.0, help="샘플별 기대 반복배수 상한")
    p.add_argument("--ibb_min_cell", type=int, default=5, help="Step2 에서 제외할 희소 셀 기준")
    p.add_argument("--ibb_edge_v", type=float, default=0.25, help="Step2 엣지 채택 Cramer's V")
    p.add_argument("--ibb_step2_ratio", type=float, default=0.5)
    p.add_argument("--ibb_step1_mult", type=float, default=1.0)
    p.add_argument("--ibb_edge_loss", choices=["pair", "all"], default="pair")
    p.add_argument("--ibb_warmup", type=int, default=3,
                   help="앞 N 에폭은 자연분포로 학습(MbN 해동 시점과 맞춰둔다)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", default=None)
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ============================================================================= 데이터
def collate_batch(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = {t: torch.tensor([b[1][t] for b in batch]) for t in TASK_NAMES}
    return imgs, labels


def build_loaders(args, num_classes, index):
    aug = [] if args.no_aug else [
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    ]
    train_tf = T.Compose([T.Resize((args.imgsz, args.imgsz)), *aug,
                          T.ToTensor(), T.Normalize(_MEAN, _STD)])
    eval_tf = T.Compose([T.Resize((args.imgsz, args.imgsz)),
                         T.ToTensor(), T.Normalize(_MEAN, _STD)])
    data = Path(args.data)
    sets = {s: AtopyDataset(data / s, index, num_classes,
                            train_tf if s == "train" else eval_tf, split=s)
            for s in ("train", "val", "test")}

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = {t: torch.tensor([b[1][t] for b in batch]) for t in TASK_NAMES}
        return imgs, labels

    def dl(ds, sh):
        return DataLoader(ds, batch_size=args.batch, shuffle=sh, num_workers=args.workers,
                          pin_memory=True, drop_last=sh, collate_fn=collate)
    return sets["train"], dl(sets["train"], True), dl(sets["val"], False), dl(sets["test"], False)


# ============================================================================= 손실
def corn_loss(logits, targets, num_classes):
    """CORN(Shi et al. 2021) 조건부 순서형 손실. 헤드 출력이 K-1 개 로짓."""
    total = logits.new_zeros(())
    n = 0
    for i in range(num_classes - 1):
        mask = targets > (i - 1)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        total = total + F.binary_cross_entropy_with_logits(
            logits[mask, i], (targets[mask] > i).float(), reduction="sum")
        n += cnt
    return total / max(n, 1)


def corn_predict(logits):
    return (torch.cumprod(torch.sigmoid(logits), dim=1) > 0.5).sum(dim=1)


class UncertaintyWeighter(nn.Module):
    """Kendall et al. 2018 — homoscedastic uncertainty 로 5축 loss 자동 가중."""

    def __init__(self, task_names):
        super().__init__()
        self.log_var = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in task_names})

    def forward(self, losses):
        total = 0.0
        for t, l in losses.items():
            s = self.log_var[t]
            total = total + 0.5 * torch.exp(-s) * l + 0.5 * s
        return total

    def sigmas(self):
        return {t: float(torch.exp(0.5 * s).item()) for t, s in self.log_var.items()}


def make_criterions(train_ds, args, device):
    counts = train_ds.class_counts()
    crits = {}
    for t in TASK_NAMES:
        w = None
        if args.class_weight:
            c = torch.tensor(counts[t], dtype=torch.float)
            w = (c.sum() / (c + 1e-6)) ** args.cw_power
            w = (w / w.mean()).clamp(max=args.cw_clip).to(device)
        crits[t] = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    return crits


def combine(losses, args, weighter):
    """IBB 배치는 일부 축만 역전파하므로 losses 가 부분집합일 수 있다.
    전 축이 들어오면 기존 식과 동일하다."""
    if args.mtl == "uncertainty":
        return weighter(losses)                      # 들어온 축의 sigma 만 갱신된다
    sym = [losses[t] for t in TASK_NAMES if t != "severity" and t in losses]
    if "severity" not in losses:
        return sum(sym) / len(sym) if sym else None
    if not sym:
        return losses["severity"]
    return args.iga_weight * losses["severity"] \
        + (1.0 - args.iga_weight) * (sum(sym) / len(sym))


# ============================================================================= 루프
def make_scaler(use_amp):
    """torch<2.3 은 torch.amp.GradScaler 가 없다(학습 서버가 2.2.2)."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def run_epoch(model, loader, crits, num_classes, device, args, optimizer=None,
              use_amp=False, weighter=None, scaler=None, tasks=None):
    train = optimizer is not None
    model.train(train)
    scaler = scaler if scaler is not None else make_scaler(use_amp)
    tot, n = 0.0, 0
    preds = {t: [] for t in TASK_NAMES}
    trues = {t: [] for t in TASK_NAMES}
    for img, labels in loader:
        img = img.to(device, non_blocking=True)
        labels = {t: v.to(device) for t, v in labels.items()}
        with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=use_amp):
            out = model(img)
            losses = {t: (corn_loss(out[t], labels[t], num_classes[t]) if args.loss == "corn"
                          else crits[t](out[t], labels[t])) for t in (tasks or TASK_NAMES)}
            loss = combine(losses, args, weighter)
        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        bs = img.size(0)
        tot += float(loss.item()) * bs
        n += bs
        for t in TASK_NAMES:
            p = corn_predict(out[t].float()) if args.loss == "corn" else out[t].argmax(1)
            preds[t] += p.cpu().tolist()
            trues[t] += labels[t].cpu().tolist()
    return tot / max(n, 1), all_metrics(trues, preds, num_classes, TASK_NAMES)


def run_ibb_epoch(model, loader, sampler, crits, num_classes, device, args,
                  optimizer, scaler, use_amp, weighter):
    """IBB 학습 에폭 — 배치마다 샘플러가 정한 스텝/태스크의 loss 만 역전파한다.

    Step1(lv:*) 배치: 그 태스크 하나의 loss 만 (논문 Alg.1 의 forall q: phi^q).
    Step2(co:*) 배치: 그 엣지 두 태스크의 loss 만. 이 모델에는 코퓰러 theta 가 없어
        Step2 는 '조인트 층화 샘플링'으로 동작한다(sampler.py 상단 참고).

    train 지표는 리샘플된 분포 위의 값이라 의미가 없어 loss 만 집계한다.
    """
    model.train(True)
    phases = sampler.plan()                 # 소비 시점과 무관하게 스케줄을 먼저 확정
    tot, nb, stat = 0.0, 0, {}
    for bi, (img, labels) in enumerate(loader):
        if bi >= len(phases):
            raise RuntimeError(f"IBB 스케줄 불일치: 배치 {bi + 1} > 스케줄 {len(phases)}")
        name, tasks = phases[bi]
        img = img.to(device, non_blocking=True)
        labels = {t: v.to(device) for t, v in labels.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(img)
            losses = {t: (corn_loss(out[t], labels[t], num_classes[t]) if args.loss == "corn"
                          else crits[t](out[t], labels[t])) for t in tasks}
            loss = combine(losses, args, weighter)
        if loss is None:
            continue
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        v = float(loss.detach())
        tot += v
        nb += 1
        a = stat.setdefault(name, [0.0, 0])
        a[0] += v
        a[1] += 1
    return tot / max(nb, 1), {k: v[0] / v[1] for k, v in stat.items()}


def set_warmup(model, on):
    for p in model.mbn.shared_parameters():
        p.requires_grad = not on


def fmt(m, key):
    return " ".join(f"{t[:4]}={m[t][key]:.3f}" for t in TASK_NAMES)


# ============================================================================= 진단
def collect_gate_stats(model, loader, device, use_amp):
    """test set 전체 평균 CFEN 게이트. 채널별 평균을 먼저 낸 뒤 그 벡터의 std 를 본다 —
    std≈0 이면 게이트가 채널을 구분하지 못하고 상대 특징맵의 상수배로만 작동한다는 뜻
    (cls_mbn 의 균등 붕괴와 같은 실패 모드, 여기선 쌍마다 독립이라 쌍별로 갈린다)."""
    if not model.pairs:
        return None
    from model import _key
    model.eval()
    acc, n = {}, 0
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(img)
            bs = img.size(0)
            for g, h in model.pairs:
                gate = model.cfen[_key(g, h)].last_gate.float().mean(dim=(2, 3))   # (B,C)
                s = gate.sum(0)
                k = f"{g}<-{h}"
                acc[k] = s if k not in acc else acc[k] + s
            n += bs
    return {k: {"mean": float((v / n).mean()), "std": float((v / n).std())}
            for k, v in acc.items()}


# ============================================================================= main
def main():
    args = parse_args()
    if args.imgsz % 16:
        raise SystemExit(
            f"[args] --imgsz 는 16의 배수여야 한다(받은 값 {args.imgsz}).\n"
            "  MbN side branch 가 stride8 특징을 pool(2,2)로 절반 냈다가 Upsample x2 로 되돌리므로,\n"
            "  H/8 이 홀수면 Branch1 출력과 1픽셀 어긋난다(예: 300 -> 37 vs 36 에서 덧셈 실패).")
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    groups = GROUP_PRESETS[args.groups]
    iga_group = IGA_GROUP[args.groups]
    name = args.name or (f"amb_{args.groups}_{args.sfen}_"
                         f"{'aspp' if not args.no_aspp else 'noaspp'}"
                         + ("_ibb" if args.ibb else ""))
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.labels_csv or (Path(args.data) / "labels.csv")
    index, num_classes, levels = build_label_space(csv_path)
    tr_ds, tr_ld, va_ld, te_ld = build_loaders(args, num_classes, index)

    ordinal = args.loss == "corn"
    model = AtopyMaMNet(num_classes, groups, iga_group=iga_group,
                        pretrained=(args.init == "vgg16"), sfen_mode=args.sfen,
                        sfen_k=args.sfen_k, reduction=args.reduction,
                        use_sfen=not args.no_sfen, cfen_pairs=args.cfen,
                        use_aspp=not args.no_aspp, iga_skip=args.iga_skip,
                        head_hidden=args.head_hidden, dropout=args.dropout,
                        ordinal=ordinal).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    mbn_params = sum(p.numel() for p in model.mbn.parameters()) / 1e6
    pairs = ", ".join(f"{g}<-{h}" for g, h in model.pairs) or "(없음)"
    print(f"[model] AtopyMaMNet  groups={args.groups}({len(groups)}분기)  sfen={args.sfen}"
          f"{'(off)' if args.no_sfen else ''}  aspp={'off' if args.no_aspp else f'on@{iga_group}'}"
          f"  init={args.init}")
    print(f"        CFEN 쌍: {pairs}")
    print(f"        params={n_params:.1f}M (MbN {mbn_params:.1f}M)")
    print(model.describe())

    ibb_ld = ibb_sampler = None
    if args.ibb:
        Y = labels_from_dataset(tr_ds)
        ibb_sampler, ibb_info = build_ibb_sampler(
            Y, num_classes, args.batch, alpha=args.ibb_alpha, cap=args.ibb_cap,
            min_cell=args.ibb_min_cell, edge_v=args.ibb_edge_v,
            step2_ratio=args.ibb_step2_ratio, step1_mult=args.ibb_step1_mult,
            edge_loss=args.ibb_edge_loss, seed=args.seed)
        _, scored = select_edges(Y, args.ibb_edge_v, num_classes)
        print("[ibb] 전체 쌍 연관강도: "
              + " | ".join(f"{a[:4]}-{b[:4]} {v:.2f}" for a, b, v in scored))
        # 샘플러가 배치 경계에 맞춰 정확한 개수의 인덱스를 내므로 drop_last 불필요.
        ibb_ld = DataLoader(tr_ds, batch_size=args.batch, sampler=ibb_sampler,
                            num_workers=args.workers, collate_fn=collate_batch, pin_memory=True)

    crits = make_criterions(tr_ds, args, device)
    weighter = UncertaintyWeighter(TASK_NAMES).to(device) if args.mtl == "uncertainty" else None
    comb = ("Kendall σ 자동가중" if weighter is not None
            else f"{args.iga_weight:.2f}*IGA + {1 - args.iga_weight:.2f}*증상4평균")
    print(f"[loss] head={'CORN(순서형)' if ordinal else 'CE'}  결합={comb}")

    use_amp = device.type == "cuda"
    scaler = make_scaler(use_amp)
    warmup = args.warmup_epochs if args.init == "vgg16" else 0
    set_warmup(model, warmup > 0)

    def new_opt(lr, params=None):
        ps = params if params is not None else [p for p in model.parameters() if p.requires_grad]
        if weighter is not None:
            ps = ps + list(weighter.parameters())
        return torch.optim.Adam(ps, lr=lr)

    opt = new_opt(args.warmup_lr if warmup else args.lr)
    sched = None if warmup else torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=args.decay_drop, patience=args.rlrop_patience, min_lr=args.min_lr)

    def save(path, vm, ep):
        torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": vm,
                    "epoch": ep, "num_classes": num_classes, "levels": levels,
                    "groups": groups, "iga_group": iga_group,
                    "weighter": weighter.state_dict() if weighter is not None else None}, path)

    best, best_ep, no_improve, ep = -1e9, 0, 0, 0
    for ep in range(1, args.epochs + 1):
        if warmup and ep == warmup + 1:                      # MbN 해동
            set_warmup(model, False)
            opt = new_opt(args.lr, list(model.parameters()))
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=args.decay_drop, patience=args.rlrop_patience,
                min_lr=args.min_lr)
            no_improve = 0                                   # 해동 직후는 한 번 튄다
            print(f"[warmup] {warmup}ep 종료 -> MbN 해동, lr={args.lr:g}", flush=True)

        use_ibb = args.ibb and ep > args.ibb_warmup
        ibb_stat = None
        if use_ibb:
            if args.ibb_alpha_end is not None:               # alpha 어닐링(자연분포로 복귀)
                span = max(1, args.epochs - args.ibb_warmup - 1)
                f = min(1.0, (ep - args.ibb_warmup - 1) / span)
                ibb_sampler.set_alpha(
                    args.ibb_alpha + (args.ibb_alpha_end - args.ibb_alpha) * f)
            tr_loss, ibb_stat = run_ibb_epoch(model, ibb_ld, ibb_sampler, crits, num_classes,
                                              device, args, opt, scaler, use_amp, weighter)
        else:
            tr_loss, _ = run_epoch(model, tr_ld, crits, num_classes, device, args, opt,
                                   use_amp, weighter, scaler)
        va_loss, vm = run_epoch(model, va_ld, crits, num_classes, device, args, None,
                                use_amp, weighter, scaler)
        score = vm["score"]
        if sched is not None:
            sched.step(score)
        phase = "warmup" if ep <= warmup else ("ibb" if use_ibb else "main")
        print(f"[Ep {ep:>3}/{args.epochs}] ({phase}) tr={tr_loss:.3f} va={va_loss:.3f} "
              f"| score={score:.3f} (mean_QWK={vm['mean_qwk']:.3f} joint={vm['joint_ac']:.3f}) "
              f"lr={opt.param_groups[0]['lr']:.2g}")
        print(f"    QWK: {fmt(vm, 'qwk')}")
        print(f"    ±1 : {fmt(vm, 'acc1')}   acc: {fmt(vm, 'acc')}", flush=True)
        if weighter is not None:
            sg = weighter.sigmas()
            print("    σ  : " + " ".join(f"{t[:4]}={sg[t]:.2f}" for t in TASK_NAMES), flush=True)
        if ibb_stat:
            print("    ibb: " + " ".join(f"{k}={v:.3f}" for k, v in sorted(ibb_stat.items())),
                  flush=True)

        save(out_dir / "last.pt", vm, ep)
        if score > best:
            best, best_ep, no_improve = score, ep, 0
            save(out_dir / "best.pt", vm, ep)
            print(f"    -> best 갱신 (score={best:.3f})", flush=True)
        else:
            no_improve += 1
            if ep > warmup and args.patience > 0 and no_improve >= args.patience:
                print(f"[early stop] val score {args.patience}ep 개선 없음 "
                      f"(best ep{best_ep}={best:.3f}) -> ep{ep} 중단", flush=True)
                break

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    _, tm = run_epoch(model, te_ld, crits, num_classes, device, args, None, use_amp,
                      weighter, scaler)
    print(f"\n[TEST] (best ep{ckpt['epoch']})  score={tm['score']:.3f}  "
          f"mean_QWK={tm['mean_qwk']:.3f}  joint_ac={tm['joint_ac']:.3f}")
    print(f"    QWK: {fmt(tm, 'qwk')}")
    print(f"    ±1 : {fmt(tm, 'acc1')}   acc: {fmt(tm, 'acc')}")
    for t in TASK_NAMES:
        print(format_confusion(tm[t]["cm"], t, levels[t]))

    gates = collect_gate_stats(model, te_ld, device, use_amp)
    if gates:
        print("\n[CFEN 진단] 쌍별 채널게이트(test 평균). std 가 0 에 가까우면 그 방향은")
        print("  게이트가 채널을 못 가려 cross-feature 가 상대 특징맵의 상수배일 뿐이라는 뜻.")
        print("  라벨 상관(스피어만) 예상: lich<-acute(0.38)는 살고 acute<-lich(0.21)는 죽는다.")
        for k, v in sorted(gates.items(), key=lambda kv: -kv[1]["std"]):
            print(f"  {k:<16} mean={v['mean']:.3f} std={v['std']:.4f}")

    torch.save({**ckpt, "test_metrics": tm, "gate_stats": gates}, out_dir / "best.pt")
    (out_dir / "test_report.json").write_text(json.dumps(
        {"args": vars(args), "test": tm, "val": ckpt["val_metrics"], "gate_stats": gates,
         "levels": levels, "num_classes": num_classes}, ensure_ascii=False, indent=2))
    (out_dir / "done.txt").write_text(
        f"groups={args.groups} sfen={args.sfen} cfen={args.cfen} aspp={not args.no_aspp} "
        f"iga_skip={args.iga_skip} init={args.init} imgsz={args.imgsz} batch={args.batch} "
        f"loss={args.loss} mtl={args.mtl} ibb={int(args.ibb)}"
        + (f" ibb_alpha={args.ibb_alpha} ibb_cap={args.ibb_cap} "
           f"ibb_edge_v={args.ibb_edge_v} ibb_warmup={args.ibb_warmup}" if args.ibb else "")
        + f" params={n_params:.2f}M\n"
        f"stopped_epoch={ep} best_epoch={ckpt['epoch']} val_best_score={best:.4f} "
        f"test_score={tm['score']:.4f} test_mean_qwk={tm['mean_qwk']:.4f} "
        f"test_joint_ac={tm['joint_ac']:.4f} "
        + " ".join(f"test_{t}_qwk={tm[t]['qwk']:.4f}" for t in TASK_NAMES) + "\n")
    print(f"\n[done] {out_dir}  test score={tm['score']:.3f} (mean_QWK={tm['mean_qwk']:.3f})")


if __name__ == "__main__":
    main()
