"""
UNeXt 학습 스크립트 (아토피 병변 semantic segmentation) + 온라인 증강.

- 데이터: 기존 dataset_face(YOLO 폴리곤)를 on-the-fly 로 0/1 마스크로 변환해 사용.
          val/test 는 split_all.py 가 atopy VL/VS 이름으로 이미 고정한 split 을 그대로 씀.
- 증강: augment.py 의 파이프라인을 '주입'해 매 배치 실시간 랜덤 변형(온라인 증강).
        train 에만 적용, val/test 는 Resize+Normalize 만. --no_aug 로 끄고 비교 가능.
- 재현성: random/numpy/torch 시드 + DataLoader worker_init_fn/generator 까지 고정.
- 매 epoch: train loss + val 성능(Dice/IoU/Recall/Precision) 출력, val Dice 기준 best 저장.
- 학습 종료 후 test split 최종 평가.

사전 준비: torch, numpy, pillow, albumentations (모두 설치돼 있음)
직접 실행 : python3 train_unext.py --epochs 100 --batch 8 --imgsz 512
비교(증강X): python3 train_unext.py --no_aug --name unext_face_noaug
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))  # segmentation/ 공유모듈
from archs import UNext
from augment import build_train_tf, build_train_crop_tf, build_val_tf
from dataset import AtopySegDataset
from losses import BCEDiceLoss, BCETverskyLoss, BCEFocalTverskyLoss
from metrics import seg_scores
from tiling import tiled_logits

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent                                   # .../ogw


def parse_args():
    p = argparse.ArgumentParser(description="UNeXt segmentation 학습(+온라인 증강)")
    p.add_argument("--data", type=str, default=str(REPO / "dataset_face"),
                   help="dataset_* 루트 (images/labels/{train,val,test} 구조)")
    p.add_argument("--epochs", type=int, default=100, help="학습 epoch 수")
    p.add_argument("--batch", type=int, default=8, help="배치 크기")
    p.add_argument("--imgsz", type=int, default=512,
                   help="입력 크기(32의 배수). UNeXt는 경량이라 512 권장")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW 초기 학습률")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    p.add_argument("--workers", type=int, default=8, help="DataLoader 워커 수")
    p.add_argument("--device", type=str, default="cuda", help="cuda 또는 cpu")
    p.add_argument("--name", type=str, default="unext_face", help="실험 이름")
    p.add_argument("--project", type=str, default=str(ROOT / "runs"),
                   help="결과 저장 상위 폴더")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    # --- 손실 함수 선택 ---
    p.add_argument("--loss", choices=["bcedice", "tversky", "focaltversky"],
                   default="bcedice",
                   help="bcedice(baseline) / tversky(오탐 벌점) / focaltversky(어려운 이미지 집중)")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Tversky alpha. alpha>beta=FP벌점(과탐↓), alpha<beta=FN벌점(미탐↓). beta=1-alpha")
    p.add_argument("--ft_gamma", type=float, default=1.333,
                   help="Focal Tversky 지수 gamma(>1: 어려운 이미지 집중). loss=focaltversky 에서만 사용")
    # --- 온라인 증강 on/off (기본 on) ---
    p.add_argument("--aug", dest="aug", action="store_true", default=True,
                   help="train 온라인 증강 사용(기본 on)")
    p.add_argument("--no_aug", dest="aug", action="store_false",
                   help="증강 끄기(비교용 baseline)")
    # --- crop 모드 (작은 병변용): 학습=네이티브 crop, 평가=타일드 전체이미지 ---
    p.add_argument("--crop_size", type=int, default=0,
                   help="0=끄기(기존 resize). >0 이면 네이티브 crop 학습 + val/test 타일드 평가")
    p.add_argument("--crop_pos_ratio", type=float, default=0.7,
                   help="병변 포함 crop 비율(나머지는 배경 포함 랜덤 crop)")
    p.add_argument("--tile_stride", type=int, default=0,
                   help="타일드 추론 stride(0이면 crop_size*3/4)")
    return p.parse_args()


# --- 재현성 유틸 -------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 완전 결정론이 필요하면 아래 주석 해제(속도 일부 희생)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader 워커별 난수 고정 (num_workers>0 재현성).

    torch.initial_seed() 는 워커마다 다르지만 generator 시드에서 파생돼 결정론적.
    albumentations 2.x 는 자체 RNG라 전역 시드로 안 잡히므로, 이 워커가 들고 있는
    Dataset.transform 을 워커별 시드로 직접 re-seed 한다(워커 간 상관 제거 + 재현).
    """
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
        train_tf = (build_train_crop_tf(crop_size=args.crop_size,
                                        pos_ratio=args.crop_pos_ratio, seed=args.seed)
                    if args.aug else build_val_tf(img_size=None, seed=args.seed))
        val_tf = build_val_tf(img_size=None, seed=args.seed)   # 원해상도 -> 타일드
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

    eval_bs = 1 if crop else None   # 타일드는 이미지당 처리 -> batch=1
    print(f"[data] train={len(tr)} val={len(va)} test={len(te)}  "
          f"(from {args.data}, aug={'ON' if args.aug else 'OFF'}, "
          f"{'crop=%d+tiled' % args.crop_size if crop else 'resize=%d' % args.imgsz})")
    return dl(tr, True), dl(va, False, eval_bs), dl(te, False, eval_bs)


@torch.no_grad()
def evaluate(model, loader, device, tile=0, stride=0):
    """tile>0 이면 원해상도 전체이미지를 타일드 추론(batch=1)해 Dice 계산."""
    model.eval()
    agg = {"dice": 0.0, "iou": 0.0, "recall": 0.0, "precision": 0.0}
    n = 0
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        if tile > 0:
            logits = tiled_logits(model, img, tile=tile, stride=stride)
        else:
            logits = model(img)
        s = seg_scores(logits, mask)
        bs = img.size(0)
        for k in agg:
            agg[k] += s[k] * bs
        n += bs
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_loader, va_loader, te_loader = build_loaders(args)

    # crop 모드면 입력 크기는 crop_size, 평가는 타일드
    model_imgsz = args.crop_size if args.crop_size > 0 else args.imgsz
    eval_tile = args.crop_size
    eval_stride = args.tile_stride or (int(args.crop_size * 0.75) if args.crop_size else 0)
    if eval_tile > 0:
        print(f"[eval] 타일드 추론  tile={eval_tile}  stride={eval_stride}")

    model = UNext(num_classes=1, input_channels=3, img_size=model_imgsz).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] UNeXt  params={n_params:.2f}M  device={device}")

    if args.loss == "focaltversky":
        criterion = BCEFocalTverskyLoss(alpha=args.alpha, gamma=args.ft_gamma)
        print(f"[loss] BCE+FocalTversky  alpha={args.alpha:.2f} "
              f"beta={1-args.alpha:.2f} gamma={args.ft_gamma:.2f}")
    elif args.loss == "tversky":
        criterion = BCETverskyLoss(alpha=args.alpha)
        print(f"[loss] BCE+Tversky  alpha={args.alpha:.2f} beta={1-args.alpha:.2f}")
    else:
        criterion = BCEDiceLoss()
        print("[loss] BCE+Dice (baseline)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for img, mask, _ in tr_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            running += loss.item() * img.size(0)
            n += img.size(0)
        scheduler.step()
        train_loss = running / max(n, 1)

        val = evaluate(model, va_loader, device, tile=eval_tile, stride=eval_stride)
        print(
            f"[Epoch {epoch:>3}/{args.epochs}] "
            f"train: loss={train_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} | "
            f"val: Dice={val['dice']:.4f} IoU={val['iou']:.4f} "
            f"R={val['recall']:.4f} P={val['precision']:.4f}",
            flush=True,
        )

        torch.save({"epoch": epoch, "model": model.state_dict(), "val": val},
                   out_dir / "checkpoint_last.pth")
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "val": val},
                       out_dir / "checkpoint_best.pth")
            print(f"    -> best 갱신 (val Dice={best_dice:.4f})", flush=True)

    # 학습 종료 후 best 로 test 평가
    ckpt = torch.load(out_dir / "checkpoint_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, te_loader, device, tile=eval_tile, stride=eval_stride)
    print(f"\n=== TEST metrics (best ckpt, epoch {ckpt['epoch']}) ===")
    print(f"Dice={test['dice']:.4f} IoU={test['iou']:.4f} "
          f"Recall={test['recall']:.4f} Precision={test['precision']:.4f}")
    print(f"[done] 결과: {out_dir}")


if __name__ == "__main__":
    main()
