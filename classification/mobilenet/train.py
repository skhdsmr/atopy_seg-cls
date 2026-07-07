"""
아토피 멀티태스크 분류 학습 — MobileNetV4 (timm) 백본.

- convnext_tiny/train.py 와 동일한 파이프라인(원본 이미지, 순서형 등급, QWK 지표).
- 백본만 MobileNetV4: --version(conv/hybrid) + --size(small/medium/large) 로
  timm 모델명 mobilenetv4_{version}_{size} 을 구성한다.
  (유효 조합: conv=small/medium/large, hybrid=medium/large)
- 공유 모듈(dataset, metrics)은 classification/ 루트에서 import.

직접 실행: python3 train.py --version conv --size medium --views both --epochs 50
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent            # .../classification/mobilenet
sys.path.insert(0, str(ROOT.parent))              # classification/ 공유모듈(dataset, metrics)
ATOPY = ROOT.parent.parent / "atopy"              # .../ogw/atopy

from dataset import (AtopyClsDataset, ImageDirClsDataset, build_label_index,  # noqa: E402
                     TASK_NAMES, NUM_CLASSES)
from metrics import per_task_metrics                          # noqa: E402
from model import MultiTaskNet                                # noqa: E402

OGW = ROOT.parent.parent                          # .../ogw
TRAIN_DIRS = {"front": (ATOPY / "TS_아토피_정면", ATOPY / "TL_아토피_정면"),
              "side":  (ATOPY / "TS_아토피_측면", ATOPY / "TL_아토피_측면")}
VAL_DIRS = {"front": (ATOPY / "VS_아토피_정면", ATOPY / "VL_아토피_정면"),
            "side":  (ATOPY / "VS_아토피_측면", ATOPY / "VL_아토피_측면")}
# source != atopy 일 때: images/{train,val} 폴더만 있는 서브셋. 라벨은 atopy JSON 조회.
SOURCE_DIRS = {"face":   OGW / "dataset_face",
               "lesion": OGW / "dataset_lesion"}

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description="MobileNetV4 멀티태스크 분류")
    p.add_argument("--version", choices=["conv", "hybrid"], default="conv",
                   help="MobileNetV4 계열: conv 또는 hybrid")
    p.add_argument("--size", choices=["small", "medium", "large"], default="medium",
                   help="모델 크기(conv=small/medium/large, hybrid=medium/large)")
    p.add_argument("--views", choices=["front", "side", "both"], default="both",
                   help="source=atopy 일 때만 적용. face/lesion 은 폴더 자체 스플릿 사용")
    p.add_argument("--source", choices=["atopy", "face", "lesion"], default="atopy",
                   help="atopy=원본 전체, face=dataset_face, lesion=dataset_lesion 서브셋")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone_lr_scale", type=float, default=1.0,
                   help="backbone LR = lr * scale. 과적합 억제 시 0.1~0.3 (사전학습 억제)")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", default=None)
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def select_pairs(views):
    keys = ["front", "side"] if views == "both" else [views]
    return [TRAIN_DIRS[k] for k in keys], [VAL_DIRS[k] for k in keys]


def build_loaders(args):
    train_tf = T.Compose([
        T.Resize((args.imgsz, args.imgsz)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    val_tf = T.Compose([
        T.Resize((args.imgsz, args.imgsz)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    if args.source == "atopy":
        tr_pairs, va_pairs = select_pairs(args.views)
        tr = AtopyClsDataset(tr_pairs, train_tf)
        va = AtopyClsDataset(va_pairs, val_tf)
    else:
        # dataset_face/dataset_lesion: 이미지 폴더 스플릿 + atopy JSON 라벨 조회
        ds_root = SOURCE_DIRS[args.source]
        index = build_label_index(ATOPY)
        print(f"[dataset] source={args.source} ({ds_root})  atopy 라벨인덱스 {len(index)}개")
        tr = ImageDirClsDataset([ds_root / "images" / "train"], index, train_tf)
        va = ImageDirClsDataset([ds_root / "images" / "val"], index, val_tf)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        labels = {t: torch.tensor([b[1][t] for b in batch]) for t in TASK_NAMES}
        return imgs, labels

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch, shuffle=sh,
                                   num_workers=args.workers, pin_memory=True,
                                   drop_last=sh, collate_fn=collate)
    return tr, dl(tr, True), dl(va, False)


def make_criterions(train_ds, use_weight, smoothing, device):
    counts = train_ds.class_counts()
    crits = {}
    for t in TASK_NAMES:
        w = None
        if use_weight:
            c = torch.tensor(counts[t], dtype=torch.float)
            w = (c.sum() / (c + 1e-6))
            w = (w / w.sum() * len(c)).to(device)
        crits[t] = nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing)
    return crits


def run_epoch(model, loader, crits, device, optimizer=None, use_amp=False):
    train = optimizer is not None
    model.train(train)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    tot, n = 0.0, 0
    preds = {t: [] for t in TASK_NAMES}
    trues = {t: [] for t in TASK_NAMES}
    for img, labels in loader:
        img = img.to(device)
        labels = {t: v.to(device) for t, v in labels.items()}
        with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=use_amp):
            out = model(img)
            loss = sum(crits[t](out[t], labels[t]) for t in TASK_NAMES)
        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        tot += loss.item() * img.size(0); n += img.size(0)
        for t in TASK_NAMES:
            preds[t] += out[t].argmax(1).cpu().tolist()
            trues[t] += labels[t].cpu().tolist()
    metrics = {t: per_task_metrics(trues[t], preds[t], NUM_CLASSES[t]) for t in TASK_NAMES}
    return tot / max(n, 1), metrics


def fmt(metrics, key):
    return " ".join(f"{t[:4]}={metrics[t][key]:.3f}" for t in TASK_NAMES)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_name = f"mobilenetv4_{args.version}_{args.size}"
    tag = args.views if args.source == "atopy" else args.source
    name = args.name or f"cls_{model_name}_{tag}"
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_ds, tr_ld, va_ld = build_loaders(args)
    model = MultiTaskNet(model_name, NUM_CLASSES,
                         embed_dim=args.embed_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {model_name} + neck({args.embed_dim})  params={n_params:.1f}M")
    crits = make_criterions(tr_ds, args.class_weight, args.label_smoothing, device)
    print(f"[loss] CE {'(역빈도가중)' if args.class_weight else ''} "
          f"label_smoothing={args.label_smoothing}")
    # 차등 LR: 사전학습 backbone 은 낮게, neck+head 는 기본 LR (과적합 억제)
    bb_params = list(model.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": args.lr * args.backbone_lr_scale},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    print(f"[optim] AdamW  head_lr={args.lr:.1e}  backbone_lr={args.lr*args.backbone_lr_scale:.1e}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"

    best = -1e9
    for ep in range(1, args.epochs + 1):
        tr_loss, _ = run_epoch(model, tr_ld, crits, device, opt, use_amp)
        va_loss, vm = run_epoch(model, va_ld, crits, device, None, use_amp)
        sched.step()
        mean_qwk = np.mean([vm[t]["qwk"] for t in TASK_NAMES])
        print(f"[Ep {ep:>3}/{args.epochs}] tr_loss={tr_loss:.3f} va_loss={va_loss:.3f} "
              f"| mean_QWK={mean_qwk:.3f}")
        print(f"    QWK : {fmt(vm, 'qwk')}")
        print(f"    ±1  : {fmt(vm, 'acc1')}   acc: {fmt(vm, 'acc')}", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "model_name": model_name,
                    "val_metrics": vm, "epoch": ep}, out_dir / "last.pt")
        if mean_qwk > best:
            best = mean_qwk
            torch.save({"model": model.state_dict(), "args": vars(args), "model_name": model_name,
                        "val_metrics": vm, "epoch": ep}, out_dir / "best.pt")
            print(f"    -> best 갱신 (mean_QWK={best:.3f})", flush=True)
    print(f"[done] {out_dir}  best mean_QWK={best:.3f}")


if __name__ == "__main__":
    main()
