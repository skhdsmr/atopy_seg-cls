"""구진(papulation) 싱글태스크 + 면적 스칼라 — dataset_rmask 전용.

입력 구성 (슬라이드 '3. bbox crop + 면적 스칼라'):
    병변 마스크 -> 정사각 bbox 크롭(+margin) -> 백본 256-d ⊕ 스칼라 4 -> neck -> 구진 헤드

스칼라 4개는 **크롭 전** 마스크(dataset_rmask/labels, rmask 이미지 전체가 프레임)에서 계산한다.
    ① area  = 병변 px / 프레임 px
    ② bbox  = 병변 bbox px / 프레임 px          (margin 적용 전 bbox)
    ③ fill  = 병변 px / bbox px
    ④ ncomp = log1p(연결요소 수, 8-이웃)

--scalar z 가 핵심 옵션이다. dataset_all_final 은 원본 해상도(512/1024)가 곧 촬영 출처라
출처마다 면적 분포가 전혀 다르다(면적비 평균 0.63 vs 0.20). 그대로 합쳐 넣으면 출처 내부의
'등급↑ -> 면적↑' 관계가 상쇄된다(심슨의 역설). 그래서 출처별 train 통계로 z-score 한다.
z-score 된 스칼라는 출처마다 평균0/분산1 이라 스칼라만으로는 출처를 알 수 없다 — 출처 플래그를
직접 넣으면 'Severe 가 1024 에 몰려 있다'는 라벨 편향을 외울 수 있어서 일부러 넣지 않는다.

    --scalar none : 스칼라 없음(bbox 크롭만, 대조군)
    --scalar raw  : 원값 그대로 + 마스크없음 플래그
    --scalar z    : 출처(원본 해상도)별 z-score + 마스크없음 플래그  (기본)

빈 마스크(병변 0px, dataset_rmask 에서 242장 전부 1024 출처) 처리:
    - z-score 통계(평균/표준편차)는 **마스크가 있는** train 표본만으로 낸다. 빈 마스크를
      넣으면 스칼라 0 이 1024 출처 통계를 끌어내려(area 0.105±0.118) z 가 사실상
      '마스크 비었나' 신호로 변질된다.
    - 빈 마스크 표본의 스칼라 4개는 0(=출처 평균)으로 두고, 5번째 입력 no_mask=1 로 따로
      알린다. 크롭은 전체 이미지로 폴백한다.

early stop 은 val loss, best 선택은 val 구진 QWK 로 서로 분리한다.

사용 예:
    python3 train.py --scalar z --model pvtv2b0 --imgsz 512 --patience 5
"""
import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
TASK = "papulation"
GRADES = ["None", "Mild", "Moderate", "Severe"]
K = len(GRADES)
SPLITS = ("train", "val", "test")
SCALARS = ("area", "bbox", "fill", "ncomp")
MODELS = {"pvtv2b0": "pvt_v2_b0", "pvtv2b1": "pvt_v2_b1", "pvtv2b2": "pvt_v2_b2",
          "effb0": "efficientnet_b0", "convnext_tiny": "convnext_tiny"}
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- 공통
def build_backbone(name, pretrained=True):
    """-> (backbone, feature_dim). 차원은 더미 forward 로 잡는다(num_features 와 다른 모델 대비)."""
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    m.eval()
    with torch.no_grad():
        feat = m(torch.zeros(2, 3, 224, 224)).shape[1]
    return m, feat


def build_transforms(imgsz):
    train_tf = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(), T.Normalize(_MEAN, _STD)])
    eval_tf = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor(), T.Normalize(_MEAN, _STD)])
    return train_tf, eval_tf


def corn_loss(logits, targets, num_classes, weight=None):
    """CORN 순서형 손실: i 번째 이진 문제('등급 > i')는 '등급 > i-1' 표본만으로 학습."""
    total, n = logits.new_zeros(()), 0
    for i in range(num_classes - 1):
        m = targets > (i - 1)
        if not bool(m.any()):
            continue
        w = weight[targets[m]] if weight is not None else None
        total = total + F.binary_cross_entropy_with_logits(
            logits[m, i], (targets[m] > i).float(), weight=w, reduction="sum")
        n += int(m.sum())
    return total / max(n, 1)


def corn_predict(logits):
    return (torch.cumprod(torch.sigmoid(logits), dim=1) > 0.5).sum(dim=1)


def class_weights(counts, power=0.5, clip_max=3.0):
    """역빈도^power, 평균 1 정규화 후 clip."""
    c = np.asarray(counts, dtype=np.float64)
    if power <= 0:
        return np.ones_like(c, dtype=np.float32)
    inv = np.where(c > 0, c.sum() / np.maximum(c, 1.0), 0.0) ** power
    w = np.where(c > 0, inv / inv[c > 0].mean(), 1.0)
    return np.clip(w, 1.0 / clip_max, clip_max).astype(np.float32)


def confusion(y, p, k):
    C = np.zeros((k, k), dtype=int)
    for a, b in zip(y, p):
        C[a, b] += 1
    return C


def qwk(y, p, k):
    O = confusion(y, p, k).astype(float)
    if O.sum() == 0:
        return 0.0
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (w * E).sum()
    return 0.0 if den < 1e-9 else float(1 - (w * O).sum() / den)


def per_task_metrics(y, p, k):
    y, p = np.asarray(y, int), np.asarray(p, int)
    if len(y) == 0:
        return {"n": 0, "qwk": 0.0, "acc": 0.0, "acc1": 0.0, "mae": 0.0, "f1": 0.0}
    f1s = []
    for c in range(k):
        tp, fp, fn = ((y == c) & (p == c)).sum(), ((y != c) & (p == c)).sum(), ((y == c) & (p != c)).sum()
        if tp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"n": int(len(y)), "qwk": qwk(y, p, k), "acc": float((y == p).mean()),
            "acc1": float((np.abs(y - p) <= 1).mean()), "mae": float(np.abs(y - p).mean()),
            "f1": float(np.mean(f1s))}


def parse_args():
    p = argparse.ArgumentParser(description="구진 싱글태스크 + 면적 스칼라 (dataset_rmask)")
    p.add_argument("--data", default="/home/work/ogw/dataset_rmask")
    p.add_argument("--src_root", default="/home/work/ogw/dataset_all_final/images",
                   help="원본 이미지. 해상도(=출처)를 여기서 읽는다")
    p.add_argument("--scalar", choices=["none", "raw", "z"], default="z")
    p.add_argument("--model", choices=list(MODELS), default="pvtv2b0")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--no_square", dest="square", action="store_false", default=True,
                   help="정사각 크롭 끄기(원본 비율 bbox -> 리사이즈로 찌그러짐)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=5, help="0=끄기")
    p.add_argument("--stop_on", choices=["loss", "qwk"], default="loss",
                   help="early stop 카운트 기준. best 선택은 항상 val QWK")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay(예 0.998). >0 이면 평가/저장을 EMA 가중치로 한다")
    p.add_argument("--weights", default="",
                   help="도메인 사전학습 ckpt. 'backbone.*' 키를 timm 백본에 이식(strict)")
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--class_weight_power", type=float, default=0.5)
    p.add_argument("--class_weight_clip", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="")
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--no_amp", dest="amp", action="store_false", default=True)
    return p.parse_args()


# ----------------------------------------------------------------------------- 스칼라
def mask_scalars(mask):
    """bool (H,W) -> (area, bbox, fill, log1p ncomp), 전부 크롭 전 프레임 기준."""
    a = int(mask.sum())
    if a == 0:
        return (0.0, 0.0, 0.0, 0.0)
    ys, xs = np.where(mask)
    bb = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    n = ndimage.label(mask, structure=np.ones((3, 3)))[1]
    return (a / mask.size, bb / mask.size, a / bb, float(np.log1p(n)))


def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 127


def build_records(args):
    """split 별 [(img, mask, label, src_res, scalars)] + 출처별 train 통계."""
    data = Path(args.data)
    labels = {}
    with open(data / "images" / "labels.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            g = (row.get(TASK) or "").strip()
            labels[row["stem"]] = GRADES.index(g) if g in GRADES else -1
    recs = {}
    for split in SPLITS:
        out = []
        for ip in sorted((data / "images" / split).glob("*.png")):
            y = labels.get(ip.stem, -1)
            if y < 0:
                continue
            mp = data / "labels" / split / ip.name
            with Image.open(Path(args.src_root) / split / ip.name) as im:
                src = im.size[1]                         # 원본 높이 = 출처(512/1024)
            m = load_mask(mp)
            out.append({"img": ip, "mask": mp, "y": y, "src": src, "empty": not m.any(),
                        "s": np.array(mask_scalars(m), dtype=np.float32)})
        recs[split] = out
    # 출처별 z-score 통계: train 만(누수 방지), 빈 마스크 제외(0 이 통계를 오염시킴)
    stats = {}
    for src in sorted({r["src"] for r in recs["train"]}):
        S = np.stack([r["s"] for r in recs["train"] if r["src"] == src and not r["empty"]])
        stats[src] = (S.mean(0), S.std(0) + 1e-6)
    return recs, stats


def scalar_vec(rec, mode, stats):
    """-> [스칼라 4개, no_mask]. 빈 마스크면 스칼라는 0(z 에선 출처 평균), no_mask=1."""
    flag = np.float32(rec["empty"])
    if rec["empty"]:
        s = np.zeros(len(SCALARS), np.float32)
    elif mode == "raw":
        s = rec["s"]
    else:
        mu, sd = stats[rec["src"]]
        s = ((rec["s"] - mu) / sd).astype(np.float32)
    return np.append(s, flag).astype(np.float32)


# ----------------------------------------------------------------------------- 데이터
def square_box(mask, margin, square):
    """union bbox + margin. square 면 긴 변 기준 정사각(중심 유지).
    프레임 밖은 PIL crop 이 검정으로 채운다 — rmask 배경이 이미 검정이라 일관된다."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = (y1 - y0) * margin, (x1 - x0) * margin
    y0, y1, x0, x1 = y0 - my, y1 + my, x0 - mx, x1 + mx
    if square:
        s = max(y1 - y0, x1 - x0)
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        y0, y1, x0, x1 = cy - s / 2, cy + s / 2, cx - s / 2, cx + s / 2
    else:
        h, w = mask.shape
        y0, x0, y1, x1 = max(0, y0), max(0, x0), min(h, y1), min(w, x1)
    return tuple(int(round(v)) for v in (x0, y0, x1, y1))


class PapDataset(Dataset):
    def __init__(self, recs, transform, mode, stats, margin, square):
        self.recs, self.tf, self.mode, self.stats = recs, transform, mode, stats
        self.margin, self.square = margin, square

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        im = Image.open(r["img"]).convert("RGB")
        box = square_box(load_mask(r["mask"]), self.margin, self.square)
        crop = im if box is None else im.crop(box)       # 빈 마스크 -> 전체 폴백
        s = (np.zeros(0, np.float32) if self.mode == "none"
             else scalar_vec(r, self.mode, self.stats))
        return self.tf(crop), torch.from_numpy(s), r["y"], r["src"]


# ----------------------------------------------------------------------------- 모델
class PapNet(nn.Module):
    """백본(256) ⊕ 스칼라(n_scalar) -> neck -> CORN (K-1) 로짓."""

    def __init__(self, backbone, n_scalar, embed_dim, dropout, pretrained=True):
        super().__init__()
        self.backbone, feat = build_backbone(backbone, pretrained=pretrained)
        self.neck = nn.Sequential(nn.Linear(feat + n_scalar, embed_dim),
                                  nn.BatchNorm1d(embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(embed_dim, K - 1)

    def load_backbone(self, path):
        """ckpt 의 backbone.* 를 timm 백본에 strict 로 이식. 헤드는 구조가 달라 새로 학습."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(bb, strict=True)
        return {"ckpt": str(path), "n_backbone": len(bb), "epoch": ck.get("epoch"),
                "ckpt_backbone": (ck.get("args") or {}).get("backbone")}

    def forward(self, x, s):
        f = self.backbone(x)
        f = f.flatten(1) if f.ndim > 2 else f
        return self.head(self.neck(torch.cat([f, s], dim=1)))


# ----------------------------------------------------------------------------- 루프
class EMA:
    """가중치 지수이동평균. 1400장 규모에서 에폭마다 val QWK 가 ±0.1 흔들리는 걸 눌러 준다.
    BN running stat 같은 buffer 는 평균내지 않고 그대로 복사한다."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
        for e, m in zip(self.model.buffers(), model.buffers()):
            e.copy_(m)


def run_epoch(model, loader, cls_w, device, opt=None, scaler=None, ema=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    ys, ps, srcs = [], [], []
    for x, s, y, src in loader:
        x, s, y = x.to(device, non_blocking=True), s.to(device), y.to(device)
        with torch.set_grad_enabled(train), torch.autocast("cuda", enabled=scaler is not None):
            out = model(x, s)
            loss = corn_loss(out.float(), y, K, weight=cls_w)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward() if scaler else loss.backward()
            (scaler.step(opt), scaler.update()) if scaler else opt.step()
            if ema is not None:
                ema.update(model)
        tot += float(loss.detach()) * len(y)
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(corn_predict(out.float()).cpu().numpy())
        srcs.append(np.asarray(src))
    return tot / max(n, 1), np.concatenate(ys), np.concatenate(ps), np.concatenate(srcs)


def report(y, p, src):
    m = per_task_metrics(y, p, K)
    m["far"] = int((np.abs(y - p) >= 2).sum())
    m["confusion"] = confusion(y, p, K).tolist()
    m["by_src"] = {int(s): per_task_metrics(y[src == s], p[src == s], K)
                   for s in np.unique(src)}
    return m


def fmt(m):
    by = "  ".join(f"src{s} QWK {v['qwk']:.3f}(n={v['n']})" for s, v in m["by_src"].items())
    return (f"QWK {m['qwk']:.4f}  acc {m['acc']:.3f}  ±1 {m['acc1']:.3f}  "
            f"MAE {m['mae']:.3f}  F1 {m['f1']:.3f}  far {m['far']}  | {by}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = args.name or (f"pap_rmask_{args.scalar}_{args.model}_r{args.imgsz}"
                         f"{'_sq' if args.square else ''}_m{args.margin}")
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    recs, stats = build_records(args)
    for split in SPLITS:
        ys = np.array([r["y"] for r in recs[split]])
        srcs = np.array([r["src"] for r in recs[split]])
        empty = {int(s): sum(r["empty"] for r in recs[split] if r["src"] == s)
                 for s in np.unique(srcs)}
        print(f"[data] {split:5s} n={len(ys)}  등급분포 {np.bincount(ys, minlength=K).tolist()}  "
              f"출처 {dict(zip(*np.unique(srcs, return_counts=True)))}  빈마스크 {empty}")
    if args.scalar == "z":
        for src, (mu, sd) in stats.items():
            print(f"[z-stat] src{src}: " + "  ".join(
                f"{k} {a:.3f}±{b:.3f}" for k, a, b in zip(SCALARS, mu, sd)))

    train_tf, eval_tf = build_transforms(args.imgsz)
    mk = lambda split, tf, sh: DataLoader(  # noqa: E731
        PapDataset(recs[split], tf, args.scalar, stats, args.margin, args.square),
        batch_size=args.batch, shuffle=sh, drop_last=sh, num_workers=args.workers,
        pin_memory=True)
    tr_ld, va_ld, te_ld = mk("train", train_tf, True), mk("val", eval_tf, False), \
        mk("test", eval_tf, False)

    n_scalar = 0 if args.scalar == "none" else len(SCALARS) + 1     # + no_mask
    model = PapNet(MODELS[args.model], n_scalar, args.embed_dim, args.dropout)
    pre = None
    if args.weights:
        pre = model.load_backbone(args.weights)
        print(f"[pretrained] 백본 {pre['n_backbone']} 텐서 이식 <- {pre['ckpt']} "
              f"(ckpt backbone={pre['ckpt_backbone']}, ep {pre['epoch']})")
    else:
        print("[pretrained] timm ImageNet")
    model.to(device)
    counts = np.bincount([r["y"] for r in recs["train"]], minlength=K)
    cw = class_weights(counts, args.class_weight_power, args.class_weight_clip)
    cls_w = torch.tensor(cw, device=device)
    print(f"[model] {args.model}  scalar={args.scalar}({n_scalar})  square={args.square}  "
          f"margin={args.margin}  class_w={np.round(cw, 2).tolist()}")
    print(f"[rule] epochs={args.epochs}  early stop=val {args.stop_on} "
          f"(patience {args.patience}) / best=val QWK  EMA={args.ema or 'off'}")

    bb = [p for n_, p in model.named_parameters() if n_.startswith("backbone.")]
    rest = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr * args.backbone_lr_scale},
                             {"params": rest, "lr": args.lr}], weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    ema = EMA(model, args.ema) if args.ema > 0 else None
    evm = ema.model if ema else model                    # 평가·저장 대상

    best_qwk, best_ep, best_loss, no_improve, history = -1e9, 0, 1e9, 0, []
    stop_best = -1e9                                     # stop_on 지표(클수록 좋게 부호 맞춤)
    for ep in range(1, args.epochs + 1):
        tr_loss, *_ = run_epoch(model, tr_ld, cls_w, device, opt, scaler, ema)
        va_loss, y, p, src = run_epoch(evm, va_ld, cls_w, device, scaler=scaler)
        sched.step()
        vm = report(y, p, src)
        history.append({"ep": ep, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_qwk": vm["qwk"], "lr": opt.param_groups[-1]["lr"]})
        tag = ""
        if vm["qwk"] > best_qwk:
            best_qwk, best_ep = vm["qwk"], ep
            torch.save({"model": evm.state_dict(), "args": vars(args), "epoch": ep,
                        "val_metrics": vm, "pretrained_from": pre,
                        "z_stats": {k: [a.tolist(), b.tolist()] for k, (a, b) in stats.items()}},
                       out_dir / "best.pt")
            tag += " *best(QWK)"
        best_loss = min(best_loss, va_loss)
        cur = vm["qwk"] if args.stop_on == "qwk" else -va_loss
        if cur > stop_best + args.min_delta:
            stop_best, no_improve = cur, 0
        else:
            no_improve += 1
            tag += f" ({args.stop_on} 정체 {no_improve}/{args.patience})"
        print(f"[ep {ep:3d}] train {tr_loss:.4f}  val {va_loss:.4f}  {fmt(vm)}  "
              f"lr {opt.param_groups[-1]['lr']:.1e}{tag}", flush=True)
        if args.patience and no_improve >= args.patience:
            print(f"[early stop] val {args.stop_on} {args.patience} epoch 정체")
            break

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    evm.load_state_dict(ck["model"])
    _, y, p, src = run_epoch(evm, te_ld, cls_w, device, scaler=scaler)
    tm = report(y, p, src)
    print(f"\n=== TEST (best ep{best_ep}, val QWK {best_qwk:.4f}) ===\n  {fmt(tm)}")
    print("  confusion (행=정답 None/Mild/Moderate/Severe):")
    for row in tm["confusion"]:
        print("   ", row)
    (out_dir / "test_report.json").write_text(json.dumps({
        "args": vars(args), "best_epoch": best_ep, "stopped_epoch": ep, "pretrained_from": pre,
        "val_metrics": ck["val_metrics"], "test_metrics": tm, "history": history,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"완료 -> {out_dir}")


if __name__ == "__main__":
    main()
