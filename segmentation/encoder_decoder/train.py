"""
U-Net++ (EfficientNet-b3, ImageNet 사전학습) 학습 스크립트.

- 모델만 새로(model.py), 데이터/증강/손실/지표는 segmentation/ 의 공유 모듈을 재사용한다.
    dataset.AtopySegDataset : YOLO 폴리곤 -> on-the-fly 0/1 마스크
    augment.build_train_tf  : 온라인 증강(ImageNet 정규화 -> 사전학습 인코더와 일치)
    losses.BCEDiceLoss/Tversky, metrics.seg_scores
- 사전학습 인코더 특성 반영:
    * 기본 lr 을 UNeXt(1e-3)보다 낮춘 1e-4 (미세조정).
    * --encoder_lr_scale 로 인코더 LR 을 디코더보다 낮게 줄 수 있음(과적합 억제).
    * --freeze_encoder 로 초기 N epoch 인코더 동결(디코더 워밍업) 옵션.
- val Dice 기준 best 저장 + 종료 후 test 평가. (train_unext.py 와 동일한 로깅 형식)

직접 실행: python3 train.py --data ../../dataset_lesion_merged --epochs 100 --batch 16
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent                 # .../segmentation/encoder_decoder
REPO = ROOT.parent.parent                              # .../ogw
SHARED = ROOT.parent                                   # .../segmentation (공유 모듈 위치)
sys.path.insert(0, str(SHARED))

from augment import (build_train_tf, build_train_crop_tf, build_val_tf,  # noqa: E402
                     FgConditionalTransform)
from dataset import AtopySegDataset                    # noqa: E402
from losses import BCEDiceLoss, BCETverskyLoss, BCEFocalTverskyLoss  # noqa: E402
from metrics import seg_scores                         # noqa: E402
from tiling import tiled_logits                        # noqa: E402
from model import build_model                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="U-Net++ / EfficientNet-b3 세그 학습")
    p.add_argument("--data", type=str, default=str(REPO / "dataset_lesion_merged"),
                   help="dataset_* 루트 (images/labels/{train,val,test})")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=512, help="32의 배수")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="디코더/헤드 기준 LR(사전학습이라 UNeXt 1e-3보다 낮춤)")
    p.add_argument("--encoder_lr_scale", type=float, default=1.0,
                   help="인코더 LR = lr * scale. 과적합 시 0.1~0.3 권장")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_encoder", type=int, default=0,
                   help="처음 N epoch 동안 인코더 동결(디코더 워밍업). 0=사용 안 함")
    # --- 메모리/속도: AMP(혼합정밀) + gradient accumulation ---
    p.add_argument("--amp", dest="amp", action="store_true", default=True,
                   help="mixed precision(기본 on). 메모리 ~절반 + A100서 더 빠름")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation 스텝. 실효 배치 = batch * accum "
                        "(고해상도서 물리 배치를 줄이고 실효 배치를 유지)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--name", type=str, default="unetpp_effb3")
    p.add_argument("--project", type=str, default=str(ROOT / "runs"))
    p.add_argument("--seed", type=int, default=42)
    # --- 모델 선택 ---
    p.add_argument("--encoder", type=str, default="efficientnet-b3",
                   help="smp 인코더(efficientnet-b3/b0, resnet34 ...). "
                        "HRNet 은 timm 경유 'tu-hrnet_w18/w32/w48'")
    p.add_argument("--decoder", type=str, default="unetpp",
                   choices=["unetpp", "unet", "manet"])
    p.add_argument("--decoder_attention", type=str, default="none",
                   choices=["none", "scse"],
                   help="scse=디코더 attention 게이트(unet/unetpp 만, 작은 병변 집중). manet 은 자체 attention")
    p.add_argument("--no_pretrained", action="store_true",
                   help="ImageNet 가중치 없이 from-scratch(대조군)")
    # --- 손실 ---
    p.add_argument("--loss", choices=["bcedice", "tversky", "focaltversky"],
                   default="bcedice")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Tversky alpha. alpha>beta=FP벌점(과탐↓), alpha<beta=FN벌점(미탐↓). beta=1-alpha")
    p.add_argument("--ft_gamma", type=float, default=1.333,
                   help="Focal Tversky 지수 gamma(>1: 어려운 이미지 집중). loss=focaltversky 에서만 사용")
    # --- 증강 on/off ---
    p.add_argument("--aug", dest="aug", action="store_true", default=True)
    p.add_argument("--no_aug", dest="aug", action="store_false")
    # --- crop 모드 (작은 병변용): 학습=네이티브 crop, 평가=타일드 전체이미지 ---
    p.add_argument("--crop_size", type=int, default=0,
                   help="0=끄기(기존 resize 방식). >0 이면 네이티브에서 이 크기로 crop 학습 "
                        "+ val/test 는 타일드 추론으로 전체이미지 Dice 평가")
    p.add_argument("--crop_pos_ratio", type=float, default=0.7,
                   help="병변 포함 crop 비율(나머지는 배경 포함 랜덤 crop)")
    p.add_argument("--crop_fg_thresh", type=float, default=0.0,
                   help="조건부 crop 임계(비율). 0=끄기(모든 이미지 crop). "
                        ">0 이면 fg가 이 값 미만(예:0.05)인 희소 이미지만 crop, "
                        "나머지 밀집은 full-image 리사이즈로 학습")
    p.add_argument("--tile_stride", type=int, default=0,
                   help="타일드 추론 stride(0이면 crop_size*3/4 로 자동, 오버랩 25%%)")
    return p.parse_args()


# --- 재현성 (train_unext.py 와 동일 규칙) -----------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    tf = getattr(info.dataset, "transform", None)
    if tf is not None and hasattr(tf, "set_random_seed"):
        tf.set_random_seed(worker_seed)


def build_loaders(args):
    crop = args.crop_size > 0
    if crop:
        # 학습: 네이티브 crop / 평가: 원해상도(리사이즈 X) -> 타일드 추론
        if args.aug:
            crop_tf = build_train_crop_tf(crop_size=args.crop_size,
                                          pos_ratio=args.crop_pos_ratio, seed=args.seed)
            if args.crop_fg_thresh > 0:
                # fg 조건부: 희소만 crop, 밀집은 full(=crop_size로 리사이즈)
                full_tf = build_train_tf(img_size=args.crop_size, seed=args.seed)
                train_tf = FgConditionalTransform(crop_tf, full_tf, args.crop_fg_thresh)
            else:
                train_tf = crop_tf
        else:
            train_tf = build_val_tf(img_size=None, seed=args.seed)
        val_tf = build_val_tf(img_size=None, seed=args.seed)
    else:
        train_tf = (build_train_tf(img_size=args.imgsz, seed=args.seed) if args.aug
                    else build_val_tf(args.imgsz, seed=args.seed))
        val_tf = build_val_tf(img_size=args.imgsz, seed=args.seed)

    tr = AtopySegDataset(root=args.data, split="train", transform=train_tf)
    va = AtopySegDataset(root=args.data, split="val", transform=val_tf)
    te = AtopySegDataset(root=args.data, split="test", transform=val_tf)

    g = torch.Generator()
    g.manual_seed(args.seed)

    def dl(ds, shuffle, bs=None):
        return DataLoader(ds, batch_size=bs or args.batch, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=shuffle,
                          worker_init_fn=seed_worker, generator=g)

    # crop 모드의 val/test 는 원해상도 -> 이미지당 타일드 추론이라 batch=1
    eval_bs = 1 if crop else None
    if not crop:
        mode = 'resize=%d' % args.imgsz
    elif args.crop_fg_thresh > 0:
        mode = 'crop=%d(조건부 fg<%.0f%%, pos=%.1f)+tiled' % (
            args.crop_size, args.crop_fg_thresh * 100, args.crop_pos_ratio)
    else:
        mode = 'crop=%d(pos=%.1f)+tiled' % (args.crop_size, args.crop_pos_ratio)
    print(f"[data] train={len(tr)} val={len(va)} test={len(te)}  "
          f"(from {args.data}, aug={'ON' if args.aug else 'OFF'}, {mode})")
    return dl(tr, True), dl(va, False, eval_bs), dl(te, False, eval_bs)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False, tile=0, stride=0):
    """tile>0 이면 원해상도 전체이미지를 타일드 추론(이미지당 batch=1)해 Dice 계산.
    tile=0 이면 기존 방식(리사이즈된 전체 이미지 1회 forward)."""
    model.eval()
    agg = {"dice": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0}
    n = 0
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        if tile > 0:
            # loader batch=1 가정. 전체이미지 타일드 추론.
            logits = tiled_logits(model, img, tile=tile, stride=stride, use_amp=use_amp)
        else:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(img)
        s = seg_scores(logits.float(), mask)
        bs = img.size(0)
        for k in agg:
            agg[k] += s[k] * bs
        n += bs
    return {k: v / max(n, 1) for k, v in agg.items()}


def set_encoder_requires_grad(model, flag):
    for p in model.encoder.parameters():
        p.requires_grad = flag


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_loader, va_loader, te_loader = build_loaders(args)

    dec_att = None if args.decoder_attention == "none" else args.decoder_attention
    model = build_model(
        encoder_name=args.encoder,
        encoder_weights=None if args.no_pretrained else "imagenet",
        decoder=args.decoder,
        decoder_attention=dec_att,
        num_classes=1, in_channels=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {args.decoder}{'+' + dec_att if dec_att else ''} + {args.encoder}  "
          f"pretrained={'no' if args.no_pretrained else 'imagenet'}  "
          f"params={n_params:.2f}M  device={device}")

    if args.loss == "focaltversky":
        criterion = BCEFocalTverskyLoss(alpha=args.alpha, gamma=args.ft_gamma)
        print(f"[loss] BCE+FocalTversky  alpha={args.alpha:.2f} "
              f"beta={1-args.alpha:.2f} gamma={args.ft_gamma:.2f}")
    elif args.loss == "tversky":
        criterion = BCETverskyLoss(alpha=args.alpha)
        print(f"[loss] BCE+Tversky  alpha={args.alpha:.2f} beta={1-args.alpha:.2f}")
    else:
        criterion = BCEDiceLoss()
        print("[loss] BCE+Dice")

    # 인코더/디코더 차등 LR (사전학습 인코더 미세조정)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr * args.encoder_lr_scale},
        {"params": dec_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    print(f"[optim] AdamW  decoder_lr={args.lr:.2e}  "
          f"encoder_lr={args.lr * args.encoder_lr_scale:.2e}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[mem] AMP={'on' if use_amp else 'off'}  accum={args.accum}  "
          f"실효배치={args.batch * args.accum}")

    # crop 모드면 평가는 타일드(전체이미지). stride 기본 = crop*3/4 (오버랩 25%)
    eval_tile = args.crop_size
    eval_stride = args.tile_stride or (int(args.crop_size * 0.75) if args.crop_size else 0)
    if eval_tile > 0:
        print(f"[eval] 타일드 추론  tile={eval_tile}  stride={eval_stride}")

    if args.freeze_encoder > 0:
        set_encoder_requires_grad(model, False)
        print(f"[warmup] 인코더 동결 {args.freeze_encoder} epoch")

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        if args.freeze_encoder > 0 and epoch == args.freeze_encoder + 1:
            set_encoder_requires_grad(model, True)
            print(f"[warmup] epoch {epoch}: 인코더 동결 해제")

        model.train()
        running, n = 0.0, 0
        optimizer.zero_grad()
        for step, (img, mask, _) in enumerate(tr_loader):
            img, mask = img.to(device), mask.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running += loss.item() * img.size(0)
            n += img.size(0)
        scheduler.step()
        train_loss = running / max(n, 1)

        val = evaluate(model, va_loader, device, use_amp=use_amp,
                       tile=eval_tile, stride=eval_stride)
        print(
            f"[Epoch {epoch:>3}/{args.epochs}] "
            f"train: loss={train_loss:.4f} lr={scheduler.get_last_lr()[-1]:.2e} | "
            f"val: Dice={val['dice']:.4f} IoU={val['iou']:.4f} "
            f"R={val['recall']:.4f} P={val['precision']:.4f}",
            flush=True,
        )

        torch.save({"epoch": epoch, "model": model.state_dict(), "val": val,
                    "args": vars(args)}, out_dir / "checkpoint_last.pth")
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "val": val,
                        "args": vars(args)}, out_dir / "checkpoint_best.pth")
            print(f"    -> best 갱신 (val Dice={best_dice:.4f})", flush=True)

    ckpt = torch.load(out_dir / "checkpoint_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, te_loader, device, use_amp=use_amp,
                    tile=eval_tile, stride=eval_stride)
    print(f"\n=== TEST metrics (best ckpt, epoch {ckpt['epoch']}) ===")
    print(f"Dice={test['dice']:.4f} IoU={test['iou']:.4f} "
          f"Recall={test['recall']:.4f} Precision={test['precision']:.4f}")
    print(f"[done] 결과: {out_dir}")


if __name__ == "__main__":
    main()
