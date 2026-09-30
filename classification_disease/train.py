"""질환 6-way 단일라벨 분류 학습 (건선/아토피/여드름/정상/주사/지루).

중증도(IGA/EASI)나 징후 태그는 다루지 않는다 — 오직 '어떤 질환인가'만.
classification_topk/train.py 에서 바뀐 지점:
    라벨    : 태그 4 + IGA 5등급 -> 질환 6클래스 배타적          (dataset.py)
    출력층  : sigmoid Q차원 + CORN 헤드 -> softmax K차원 1개      (model.py)
    loss    : BCE + CORN 가중합 -> CrossEntropy 하나
    후처리  : threshold 학습 -> 불필요(argmax 로 확정)
    선택기준: val combo -> val macro_f1
    지표    : mAP/QWK -> 혼동행렬 + 클래스별 P/R/F1 + 출처별 정확도  (metrics.py)
그대로 유지: 백본, ImageNet fine-tuning, 차등 LR, cosine 스케줄, early stop, 증강.

test 는 제공된 Validation(새 케이스 + 새 출처)이라 학습·선택에 절대 쓰지 않는다.

사전 준비: python3 make_dataset_disease.py
직접 실행:
    python3 train.py --model effb0                  # 정면+측면 전부
    python3 train.py --model effb0 --angle front    # 정면만(실사 얼굴)
    python3 train.py --model effb0 --angle side     # 측면만(피부 접사)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent            # classification_disease
OGW = ROOT.parent
sys.path.insert(0, str(ROOT))

from dataset import (DISEASES, DiseaseDataset, NUM_DISEASES,             # noqa: E402
                     label_index_from_csv)
from metrics import (confusion, format_confusion, format_groups,          # noqa: E402
                     format_report, group_accuracy, per_class_metrics,
                     summarize)
from model import DiseaseNet                                              # noqa: E402

# 온디바이스 후보. 주석의 파라미터 수는 이 프로젝트 구성(6-way head + embed_dim 512)
# 기준 실측이라, ImageNet 1000-way head 로 세는 공개 표의 숫자보다 조금 작다.
MODELS = {
    "mnv3s":    "mobilenetv3_small_100",   # 2.05M
    "mnv4s":    "mobilenetv4_conv_small",  # 3.15M
    "efflite0": "tf_efficientnet_lite0",   # 4.03M
    "effb0":    "efficientnet_b0",         # 4.67M
    "efflite1": "tf_efficientnet_lite1",   # 4.80M
    "mnv3l":    "mobilenetv3_large_100",   # 4.86M
    "efflite2": "tf_efficientnet_lite2",   # 5.47M
    "efflite3": "tf_efficientnet_lite3",   # 7.58M
    "mnv4m":    "mobilenetv4_conv_medium",  # 9.09M
    "efflite4": "tf_efficientnet_lite4",   # 12.39M
    # --- 온디바이스 예산 밖. 상한(천장) 확인용 참고 백본 ---
    # 지금 병목이 '용량 부족'인지 '데이터/라벨의 천장'인지 가르는 대조군이다.
    # 여기서도 test macro-F1 이 비슷하게 멈추면 백본을 키워도 소용없다는 뜻.
    # 배포 후보로 고를 거라면 mnv4mh(10M) 정도가 상한이고 나머지는 25M+ 라 무리다.
    "pvt_b0":   "pvt_v2_b0",               #  3.55M  트랜스포머지만 mnv4s 급 크기
    "mnv4mh":   "mobilenetv4_hybrid_medium",  # 10.45M  MobileNetV4-M + attention 블록
    "pvt_b1":   "pvt_v2_b1",               # 13.76M  피라미드 ViT (SRA)
    "pvt_b2":   "pvt_v2_b2",               # 25.12M
    "cnxt_t":   "convnext_tiny",           # 28.22M  ConvNeXt-T
    "mnv4l":    "mobilenetv4_conv_large",  # 31.97M
}
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description="질환 6-way 분류")
    p.add_argument("--model", choices=list(MODELS), default="effb0")
    p.add_argument("--data", default=str(OGW / "dataset_disease"),
                   help="dataset_disease 루트({train,val,test}/*.png + labels.csv)")
    p.add_argument("--angle", choices=["both", "front", "side"], default="both",
                   help="both=정면+측면. 두 각도는 해상도·구도·케이스가 전부 달라 "
                        "사실상 다른 도메인이므로 front/side 단독 성능도 따로 볼 것")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--class_weight", action="store_true",
                   help="역빈도 클래스 가중. 클래스가 균형(각 900장)이라 기본은 off")
    p.add_argument("--select", choices=["macro_f1", "accuracy", "balanced_accuracy",
                                        "macro_auroc"], default="macro_f1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", default=None)
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_transforms(imgsz):
    """topk/crop 실험과 동일한 증강 — 비교 가능성을 위해 그대로 유지.

    수직 뒤집기는 넣지 않는다. 얼굴은 위아래가 정해진 구조라 뒤집으면 실제로 존재하지
    않는 입력이 된다. 색 지터는 약하게만 — 홍반(붉은기)이 질환 판별의 핵심 단서라
    색을 세게 흔들면 라벨과 어긋난다.
    """
    train_tf = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    eval_tf = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    return train_tf, eval_tf


def build_loaders(args):
    train_tf, eval_tf = build_transforms(args.imgsz)
    data = Path(args.data)
    csv_path = data / "labels.csv"
    if not csv_path.exists():
        print(f"[에러] labels.csv 없음: {csv_path}\n"
              f"       python3 make_dataset_disease.py 를 먼저 실행할 것.", file=sys.stderr)
        sys.exit(1)
    index = label_index_from_csv(csv_path)

    def mk(split, tf):
        return DiseaseDataset(data / split, index, tf, angle=args.angle)

    tr, va, te = mk("train", train_tf), mk("val", eval_tf), mk("test", eval_tf)
    for ds, nm in ((tr, "train"), (va, "val"), (te, "test")):
        if len(ds) == 0:
            print(f"[에러] {nm} 샘플이 0개. 이미지 다운로드가 끝났는지, --src 가 맞는지 확인.",
                  file=sys.stderr)
            sys.exit(1)

    def dl(ds, sh):
        return DataLoader(ds, batch_size=args.batch, shuffle=sh, num_workers=args.workers,
                          pin_memory=True, drop_last=sh)
    return tr, va, te, dl(tr, True), dl(va, False), dl(te, False)


def run_epoch(model, loader, criterion, device, optimizer=None, use_amp=False, scaler=None):
    """반환: (mean_loss, y_true(N,), y_pred(N,), scores(N,K)). scores 는 softmax 확률."""
    train = optimizer is not None
    model.train(train)
    if scaler is None:
        scaler = (torch.amp.GradScaler("cuda", enabled=use_amp)
                  if hasattr(torch.amp, "GradScaler")
                  else torch.cuda.amp.GradScaler(enabled=use_amp))
    tot, n = 0.0, 0
    ys, ps, ss = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        tot += loss.item() * y.size(0); n += y.size(0)
        lg = logits.detach().float()
        ys.append(y.detach().cpu().numpy())
        ps.append(lg.argmax(1).cpu().numpy())
        ss.append(torch.softmax(lg, dim=1).cpu().numpy())
    return (tot / max(n, 1), np.concatenate(ys), np.concatenate(ps), np.concatenate(ss))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_name = MODELS[args.model]
    name = args.name or f"dis_{args.model}_r{args.imgsz}_{args.angle}"
    out_dir = Path(args.project) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_ds, va_ds, te_ds, tr_ld, va_ld, te_ld = build_loaders(args)
    model = DiseaseNet(model_name, NUM_DISEASES, embed_dim=args.embed_dim,
                       dropout=args.dropout, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {args.model} ({model_name})  params={n_params:.1f}M  "
          f"classes={NUM_DISEASES}  angle={args.angle}")

    counts = tr_ds.class_counts()
    print(f"{'train 분포':12s} " + "  ".join(f"{d}={c}" for d, c in zip(DISEASES, counts)))
    w = None
    if args.class_weight:
        c = torch.tensor(counts, dtype=torch.float)
        w = c.sum() / (c + 1e-6)
        w = (w / w.sum() * len(c)).to(device)
        print(f"{'클래스 가중':12s} " + "  ".join(f"{d}={v:.2f}" for d, v in zip(DISEASES, w.tolist())))
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    print(f"[loss] CrossEntropy  label_smoothing={args.label_smoothing}  "
          f"class_weight={'on' if args.class_weight else 'off'}")
    print(f"[select] val {args.select}  |  test 는 최종 1회만 (새 케이스 + 새 출처)")

    bb_ids = {id(p) for n_, p in model.named_parameters() if n_.startswith("backbone")}
    groups = [
        {"params": [p for p in model.parameters() if id(p) in bb_ids],
         "lr": args.lr * args.backbone_lr_scale},
        {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": args.lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"

    best, best_ep, no_improve = -1e9, 0, 0
    for ep in range(1, args.epochs + 1):
        tr_loss, *_ = run_epoch(model, tr_ld, criterion, device, opt, use_amp)
        va_loss, vy, vp, vs = run_epoch(model, va_ld, criterion, device, None, use_amp)
        sched.step()
        vsum = summarize(vy, vp, vs, DISEASES)
        score = vsum[args.select]
        print(f"[Ep {ep:>3}/{args.epochs}] tr={tr_loss:.3f} va={va_loss:.3f} "
              f"| {args.select}={score:.3f} | acc={vsum['accuracy']:.3f} "
              f"bal_acc={vsum['balanced_accuracy']:.3f} macroF1={vsum['macro_f1']:.3f} "
              f"macroAUROC={vsum['macro_auroc']:.3f}", flush=True)

        ckpt = {"model": model.state_dict(), "args": vars(args), "model_name": model_name,
                "diseases": DISEASES, "val_summary": vsum, "epoch": ep}
        torch.save(ckpt, out_dir / "last.pt")
        if score > best + args.min_delta:
            best, best_ep, no_improve = score, ep, 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"    -> best 갱신 ({args.select}={best:.3f})", flush=True)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[early stop] val {args.select} {args.patience}ep 개선 없음 "
                      f"(best ep{best_ep}={best:.3f}) -> ep{ep} 중단", flush=True)
                break

    # --- best 복원 후 test 1회 ---
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    _, ty, tp, ts = run_epoch(model, te_ld, criterion, device, None, use_amp)

    per_class = per_class_metrics(ty, tp, ts, DISEASES)
    tsum = summarize(ty, tp, ts, DISEASES)
    cm = confusion(ty, tp, NUM_DISEASES)
    print("\n" + format_report(per_class, tsum, DISEASES,
                               f"[TEST] 제공 Validation (새 케이스 + 새 출처)  best_ep={ckpt['epoch']}"))
    print(f"\n[TEST] 혼동행렬 (행=정답, 열=예측)\n{format_confusion(cm, DISEASES)}")

    # 지름길 진단 — 출처/각도별로 쪼갠다. 여기서 편차가 크면 accuracy 를 믿지 말 것.
    by_source = group_accuracy(ty, tp, te_ds.meta("source"))
    print("\n" + format_groups(by_source, "[TEST] 출처(prefix)별 정확도"))
    if args.angle == "both":
        by_angle = group_accuracy(ty, tp, te_ds.meta("angle"))
        print("\n" + format_groups(by_angle, "[TEST] 각도별 정확도"))
        print("  참고: 측면은 전부 512px 피부 접사(Z4), 정면은 1024px 실사 얼굴이다. "
              "측면만 유독 높으면 질환이 아니라 도메인 단서를 본 것일 수 있다.")

    vsum = ckpt["val_summary"]
    gap = vsum[args.select] - tsum[args.select]
    print(f"\n[val -> test 격차] {args.select}: val={vsum[args.select]:.3f} "
          f"test={tsum[args.select]:.3f} (차이 {gap:+.3f})")
    print("  val 은 '같은 출처의 새 케이스', test 는 '다른 출처의 새 케이스'다. "
          "격차가 크면 출처 과적합이다.")

    torch.save({**ckpt, "test_summary": tsum, "test_per_class": per_class,
                "test_by_source": by_source}, out_dir / "best.pt")
    (out_dir / "test_report.json").write_text(json.dumps(
        {"epoch": ckpt["epoch"], "angle": args.angle, "val_summary": vsum,
         "test_summary": tsum, "per_class": per_class,
         "confusion": cm.tolist(), "classes": DISEASES,
         "by_source": by_source}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "done.txt").write_text(
        f"model={args.model} imgsz={args.imgsz} angle={args.angle} epochs={args.epochs} "
        f"class_weight={args.class_weight} label_smoothing={args.label_smoothing}\n"
        f"stopped_epoch={ep} best_epoch={ckpt['epoch']} val_best_{args.select}={best:.4f} "
        f"test_acc={tsum['accuracy']:.4f} test_macro_f1={tsum['macro_f1']:.4f} "
        f"test_macro_auroc={tsum['macro_auroc']:.4f}\n")
    print(f"\n[done] {out_dir}  val_best={best:.3f}  "
          f"test_acc={tsum['accuracy']:.3f} test_macroF1={tsum['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
