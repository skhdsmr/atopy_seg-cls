"""train split 안에서만 5-fold 층화 분할 — teacher OOF soft-label 생성 전용.

val/test(각 고정 ~1,000장)는 건드리지 않는다 — 폴더 자체가 train 에서 빠져 있으므로
이 스크립트가 보는 대상은 애초에 train/ 뿐이다.

5개 teacher(홍반/구진/찰상/태선화/IGA)가 전부 같은 분할을 써야 한다(README §2) —
사진 한 장이 홍반 teacher 에서는 fold 1, IGA teacher 에서는 fold 3 이면 소견별 soft
label 이 '무엇을 학습 중 못 봤는지' 서로 어긋난다. 그래서 이 스크립트를 한 번만
돌려 JSON 으로 고정하고, train_single.py --folds_json 으로 5개 teacher 모두 재사용한다.

층화 기준은 iga_grade(총평) 하나로 한다. 5축을 동시에 층화하면 등급조합 셀이
train_single.py가 요구하는 fold당 표본수보다 훨씬 잘게 쪼개져 StratifiedKFold가
감당하지 못한다(등급조합 5*4*4*4*4=1280가지 vs 표본 7,000여개).

사용법:
    python3 make_folds.py --data /path/to/dataset --out runs/folds.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from kd_common import IGA_TASK, KDDataset, build_label_index, resolve_split_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="images/{train,val,test}/ + labels.csv 데이터 루트")
    ap.add_argument("--labels_csv", default="", help="비우면 --data 아래에서 자동 탐색")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index, info = build_label_index(args.data, args.labels_csv or None)
    img_dir = resolve_split_dir(args.data, "train")
    ds = KDDataset(img_dir, None, index)   # mask_dir=None: fold 배정에는 crop 방식이 무관
    if len(ds) == 0:
        sys.exit(f"[에러] train 샘플이 0개다: {img_dir}")

    stems = [p.stem for p, _, _ in ds.samples]
    y = np.array([rec[IGA_TASK] for _, _, rec in ds.samples], dtype=int)

    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        sys.exit("[에러] scikit-learn 이 필요하다: pip install scikit-learn")

    fold = np.full(len(y), -1, dtype=int)
    known = y >= 0
    n_known = int(known.sum())
    if n_known >= args.n_folds:
        # 등급별 표본이 n_folds 개 미만이면 StratifiedKFold 가 에러를 낸다 — 그런
        # 희소 등급은 자동으로 마지막 인자(빈도 무시)로 죽지 않도록 시도/폴백한다.
        try:
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                  random_state=args.seed)
            known_idx = np.nonzero(known)[0]
            for k, (_, te) in enumerate(skf.split(np.zeros(n_known), y[known])):
                fold[known_idx[te]] = k
        except ValueError as e:
            print(f"[경고] 층화 실패({e}) -> IGA 라벨이 있는 표본도 무작위 KFold 로 대체.",
                 file=sys.stderr)
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            known_idx = np.nonzero(known)[0]
            for k, (_, te) in enumerate(kf.split(known_idx)):
                fold[known_idx[te]] = k
    # IGA 라벨이 없는 표본(비병변 등)은 층화 대상이 아니므로 라운드로빈으로 고르게.
    rng = np.random.RandomState(args.seed)
    unknown_idx = np.nonzero(~known)[0]
    rng.shuffle(unknown_idx)
    for i, idx in enumerate(unknown_idx):
        fold[idx] = i % args.n_folds
    assert (fold >= 0).all(), "fold 배정 누락"

    mapping = {s: int(f) for s, f in zip(stems, fold)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "data": str(Path(args.data).resolve()), "n_folds": args.n_folds,
        "seed": args.seed, "stratify": IGA_TASK, "n_samples": len(stems),
        "fold": mapping,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = np.bincount(fold, minlength=args.n_folds)
    print(f"[make_folds] {len(stems)} stems -> {args.n_folds} folds, 크기={counts.tolist()}")
    print(f"[make_folds] -> {out}")
    print("[make_folds] 5개 teacher(erythema/papulation/excoriation/lichenification/"
         "iga_grade) 학습에 전부 이 파일을 --folds_json 으로 줄 것.")


if __name__ == "__main__":
    main()
