"""합성 이미지 라벨 일치율 — '생성기가 붙인 질환 라벨'을 분류기가 동의하는지 본다.

무엇을 재는가:
    GAN/SD 가 "이건 지루"라고 생성한 이미지를 실제 데이터로 학습한 6-way 분류기에
    넣어 같은 질환이 나오는지 센다. 일치율이 낮으면 그 합성 이미지는 라벨이 틀린
    것이고, 증강에 쓰면 라벨 노이즈를 주입하는 셈이 된다.

기준선(baseline):
    같은 분류기가 '실제 test 정면'에서 내는 점수. 합성 점수는 이 근처여야 정상이다.
    기준선보다 크게 높으면 합성이 현실보다 쉽다는 뜻이고(교과서적 과장, 애매한 경계
    사례 없음), 낮으면 라벨이 어긋난 것이다. 둘 다 증강 가치를 떨어뜨린다.

주의: 분류기 자체가 정답이 아니다. 이 지표는 '실제 데이터 분포와의 일치'를 재는
      대리 지표일 뿐이므로 반드시 기준선과 나란히 읽을 것.

사용법:
    python3 eval_label_agreement.py --run dis_effb0_r512_front
    python3 eval_label_agreement.py --run dis_effb0_r512_front --per_class 2000
    python3 eval_label_agreement.py --run dis_effb0_r512_front --arm gan --arm sd_ft
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent            # classification_disease
OGW = ROOT.parent
sys.path.insert(0, str(ROOT))

from dataset import (DISEASES, DiseaseDataset, NUM_DISEASES,             # noqa: E402
                     label_index_from_csv)
from metrics import confusion, per_class_metrics, summarize              # noqa: E402
from model import DiseaseNet                                             # noqa: E402
from train import build_transforms                                       # noqa: E402

# 이름 -> (이미지 디렉터리, labels.csv). 실제 test 정면이 기준선이다.
ARMS = {
    "real_test": (OGW / "dataset_disease" / "test", OGW / "dataset_disease" / "labels.csv"),
    "gan":       (OGW / "gan" / "synth_front" / "images",
                  OGW / "gan" / "synth_front" / "labels.csv"),
    "sd_lora":   (OGW / "diffusion" / "synth_lora_front" / "images",
                  OGW / "diffusion" / "synth_lora_front" / "labels.csv"),
    "sd_ft":     (OGW / "diffusion" / "synth_ft_front" / "images",
                  OGW / "diffusion" / "synth_ft_front" / "labels.csv"),
}
ARM_LABEL = {"real_test": "실제 test 정면 (기준선)", "gan": "GAN",
             "sd_lora": "SD-LoRA", "sd_ft": "SD-전체FT"}


def parse_args():
    p = argparse.ArgumentParser(description="합성 이미지 라벨 일치율",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--run", default="dis_effb0_r512_front",
                   help="runs/ 아래 런 이름(또는 경로)")
    p.add_argument("--ckpt", default="best", choices=["best", "last"])
    p.add_argument("--arm", action="append", dest="arms", choices=list(ARMS),
                   help="반복 지정. 기본은 전부(real_test/gan/sd_lora/sd_ft)")
    p.add_argument("--per_class", type=int, default=2000,
                   help="합성 arm 에서 클래스당 최대 장수(stem 순 앞에서부터). "
                        "arm 마다 생성량이 달라(2000/3500) 맞춰야 비교가 된다. "
                        "0=전부. 실제 test 는 600장 전부를 쓴다")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--project", default=str(ROOT / "runs"))
    p.add_argument("--out", default=None, help="JSON 저장 경로. 기본은 런 폴더 안")
    return p.parse_args()


def load_model(run_dir, ckpt_name, device):
    """반환: (model, cfg). 전처리(imgsz)를 학습 당시 값으로 재현한다."""
    ckpt_path = run_dir / f"{ckpt_name}.pt"
    if not ckpt_path.exists():
        print(f"[에러] 체크포인트 없음: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})
    cfg = {
        "imgsz": targs.get("imgsz", 224),
        "angle": targs.get("angle", "both"),
        "embed_dim": targs.get("embed_dim", 512),
        "dropout": targs.get("dropout", 0.5),
        "model_name": ckpt.get("model_name", targs.get("model", "?")),
        "epoch": ckpt.get("epoch", None),
    }
    model = DiseaseNet(cfg["model_name"], NUM_DISEASES, embed_dim=cfg["embed_dim"],
                       dropout=cfg["dropout"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def cap_per_class(ds, n):
    """클래스당 n 장만 남긴다(현재 정렬 순서 = stem 순 앞에서부터)."""
    if not n:
        return ds
    kept, seen = [], defaultdict(int)
    for s in ds.samples:
        y = s[1]
        if seen[y] < n:
            kept.append(s)
            seen[y] += 1
    ds.samples = kept
    return ds


@torch.no_grad()
def predict(model, loader, device):
    """반환: (y_true(N,), y_pred(N,), scores(N,K)). scores 는 softmax 확률."""
    ys, ps, ss = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        lg = logits.float()
        ys.append(y.numpy())
        ps.append(lg.argmax(1).cpu().numpy())
        ss.append(torch.softmax(lg, dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ss)


def eval_arm(name, model, cfg, a, device):
    """arm 하나 평가. 반환: 리포트 dict."""
    img_dir, csv_path = ARMS[name]
    if not img_dir.is_dir() or not csv_path.exists():
        print(f"[skip] {name}: 경로 없음 ({img_dir})")
        return None
    _, eval_tf = build_transforms(cfg["imgsz"])
    ds = DiseaseDataset(img_dir, label_index_from_csv(csv_path), eval_tf, angle="front")
    if name != "real_test":                      # 실제 test 는 600장 전부가 곧 기준선
        ds = cap_per_class(ds, a.per_class)
    if len(ds) == 0:
        print(f"[skip] {name}: 정면 샘플 0개")
        return None
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                        pin_memory=True)
    y, p, s = predict(model, loader, device)
    return {
        "arm": name,
        "label": ARM_LABEL[name],
        "images": str(img_dir),
        "n": int(len(y)),
        "summary": summarize(y, p, s, DISEASES),
        "per_class": per_class_metrics(y, p, s, DISEASES),
        "confusion": confusion(y, p, NUM_DISEASES).tolist(),
    }


def print_tables(reports):
    """스크린샷과 같은 두 표: 요약 지표 + 클래스별 recall."""
    w = max(len(r["label"]) for r in reports) + 2
    print("\n1. 라벨 일치율")
    print(f"  {'데이터':{w}s} {'n':>6s} {'acc':>7s} {'macro-F1':>9s} {'macro-AUROC':>12s}")
    print("  " + "-" * (w + 38))
    for r in reports:
        m = r["summary"]
        print(f"  {r['label']:{w}s} {r['n']:6d} {m['accuracy']:7.3f} "
              f"{m['macro_f1']:9.3f} {m['macro_auroc']:12.3f}")

    print("\n2. 클래스별 recall (생성 라벨을 분류기가 동의한 비율)")
    hdr = "".join(f"{r['label'][:9]:>11s}" for r in reports)
    print(f"  {'클래스':6s}{hdr}")
    print("  " + "-" * (6 + 11 * len(reports)))
    for c in DISEASES:
        row = "".join(f"{r['per_class'][c]['recall']:11.3f}" for r in reports)
        print(f"  {c:6s}{row}")

    base = next((r for r in reports if r["arm"] == "real_test"), None)
    if base:
        print("\n3. 기준선 대비 (합성 acc - 실제 test 정면 acc)")
        b = base["summary"]["accuracy"]
        for r in reports:
            if r["arm"] == "real_test":
                continue
            d = r["summary"]["accuracy"] - b
            print(f"  {r['label']:{w}s} {d:+7.3f}")
        print("  * 0 근처가 정상. 크게 낮으면 라벨이 틀린 것(노이즈 주입), "
              "크게 높으면 합성이 현실보다 쉬운 것(경계 사례 없음).")


def main():
    a = parse_args()
    run_dir = Path(a.run) if Path(a.run).is_dir() else Path(a.project) / a.run
    if not run_dir.is_dir():
        print(f"[에러] 런 없음: {run_dir}", file=sys.stderr)
        sys.exit(1)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, a.ckpt, device)
    print(f"[분류기] {run_dir.name}  {cfg['model_name']}  r{cfg['imgsz']}  "
          f"angle={cfg['angle']}  ckpt={a.ckpt}(epoch {cfg['epoch']})")

    arms = a.arms or list(ARMS)
    reports = [r for r in (eval_arm(n, model, cfg, a, device) for n in arms) if r]
    if not reports:
        print("[에러] 평가된 arm 이 없다.", file=sys.stderr)
        sys.exit(1)
    print_tables(reports)

    out = Path(a.out) if a.out else run_dir / "label_agreement.json"
    out.write_text(json.dumps({
        "classifier": {"run": run_dir.name, "ckpt": a.ckpt, **cfg},
        "per_class_cap": a.per_class,
        "arms": reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
