"""dataset_face 에서 질환별 N장을 뽑아 수동 마스크 라벨링용 시드셋을 만든다.

self-training 의 출발점이다. 60장을 손으로 칠해 seg 모델을 학습시키고, 그 모델로
나머지 5340장에 의사라벨을 붙여 다시 학습하는 게 목표다. 그래서 이 60장은 '적당히
고른 60장'이면 안 된다 — 여기서 편향이 생기면 self-training 이 그 편향을 5400장으로
증폭한다. 두 가지를 강제한다:

  1) 케이스 키(subject)당 최대 1장. dataset_face 는 평균 1.21장/키라서 그냥 뽑으면
     같은 원본 케이스에서 파생된 닮은 이미지가 2~3장 들어온다. 10장 중 3장이 사실상
     한 장이면 시드의 실효 크기가 7장으로 줄어든다.
  2) 출처(prefix)별 비례 할당(최대잔여법). 질환 안에서도 출처가 심하게 쏠려 있어
     (건선 train: H0 700 / H1 0 이 아니라 섞여 있음) 무작위로 뽑으면 10장이 한
     출처에 몰릴 수 있다. 그러면 모델이 '그 출처의 촬영 특성'을 병변으로 배운다.

split 은 dataset_face 의 것을 그대로 쓴다. 질환별 10장을 8:1:1 로 나눌 때, 뽑아 놓고
셋으로 자르는 게 아니라 train 에서 8장 / val 에서 1장 / test 에서 1장을 각각 뽑는다.
dataset_face 의 split 은 (질환, 케이스 키) 단위로 짜여 있어 교집합이 0이고 test 는
train 에 없는 출처로 채워져 있다. 거기서 그대로 가져오면 그 성질을 공짜로 물려받는다.
직접 잘라 나누면 seg 용 val/test 가 train 과 같은 케이스·같은 출처가 돼서, 의사라벨
품질을 재야 할 val/test 가 실제보다 후하게 나온다.
val 1장 / test 1장은 통계로 쓸 크기가 아니다 — self-training 이 무너지는지(마스크가
사라지거나 화면을 덮는지) 보는 눈금이다. 숫자로 쓰려면 --per_class 를 올려야 한다.

출력은 label_editor.py 가 그대로 인식하는 규약이다(폴더명이 '_pred' 로 끝나고
images/ labels/ 를 가지면 편집 대상으로 자동 발견됨):

    dataset_face_seed_pred/
      images/<split>/<질환>__<stem>.png    원본 하드링크
      labels/<split>/<질환>__<stem>.txt    빈 파일(여기에 폴리곤이 쌓인다)
      pred/<split>/<질환>__<stem>.png      [원본|기존마스크] 2-up (있을 때만, 참고용)
      manifest.csv                         case,split,stem,disease,source,subject,has_mask

파일명에 질환을 붙이는 이유: 편집기 사이드바에 파일명만 나오는데, 병변을 칠하려면
그게 여드름인지 지루인지 알아야 한다(눈으로는 구분이 안 되는 게 이 데이터의 전제다).
원래 stem 은 manifest.csv 로 되돌린다 — export_seed_masks.py 가 그걸 쓴다.

사용법:
    python3 make_seed_set.py                      # 질환별 10장을 8:1:1 로
    python3 make_seed_set.py --per_class 20       # 질환별 20장 -> 16:2:2
    python3 make_seed_set.py --ratio 10:0:0       # 전부 train 에서 (예전 동작)
    python3 make_seed_set.py --seed 1             # 겹치지 않는 다른 60장(시드셋 확장)
"""
import argparse
import csv
import hashlib
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent          # repo 루트
DISEASES = ["건선", "아토피", "여드름", "정상", "주사", "지루"]
SPLITS = ["train", "val", "test"]

# 이미 마스크가 있는 곳들 — 참고 오버레이(pred/)를 만들 때만 쓴다. 라벨은 비운다.
MASK_DIRS = [ROOT / "atopy_crop_masks"]


def parse_args():
    p = argparse.ArgumentParser(description="수동 마스크 라벨링용 시드셋 생성")
    p.add_argument("--src", default=str(ROOT / "dataset_face"))
    p.add_argument("--out", default=str(ROOT / "dataset_face_seed_pred"),
                   help="label_editor 가 찾으려면 이름이 '_pred' 로 끝나야 한다")
    p.add_argument("--per_class", type=int, default=10,
                   help="질환별 총 장수(모든 split 합)")
    p.add_argument("--ratio", default="8:1:1",
                   help="train:val:test 비율. per_class 에 맞춰 최대잔여법으로 정수화")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="hardlink", choices=["hardlink", "symlink", "copy"])
    p.add_argument("--no_pred", action="store_true",
                   help="기존 마스크 참고 오버레이(pred/)를 만들지 않는다")
    p.add_argument("--clean", action="store_true", help="기존 출력 폴더를 비우고 새로 만든다")
    return p.parse_args()


def _rng(seed, *key):
    """(seed, key) 에서 유도한 독립 RNG. md5 를 쓰는 건 파이썬 hash() 가
    PYTHONHASHSEED 에 따라 프로세스마다 달라지기 때문 — 같은 인자면 항상 같은 60장."""
    h = hashlib.md5(("|".join([str(seed), *map(str, key)])).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def split_counts(per_class, ratio, splits):
    """'8:1:1' + per_class -> split 별 장수. 최대잔여법으로 합을 per_class 에 맞춘다."""
    w = [float(x) for x in ratio.split(":")]
    if len(w) != len(splits) or sum(w) <= 0:
        raise SystemExit(f"[에러] --ratio 는 '{':'.join('n' for _ in splits)}' 형식이어야 한다: {ratio}")
    raw = [per_class * x / sum(w) for x in w]
    cnt = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - int(raw[i])))[:per_class - sum(cnt)]:
        cnt[i] += 1
    return dict(zip(splits, cnt))


def pick(rows, per_class, seed, split):
    """질환별로 per_class 장 선택. 키당 1장 + 출처 비례 할당."""
    # 질환 -> 출처 -> 키 -> [stem...]
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        tree[r["disease"]][r["source"]][r["subject"]].append(r["stem"])

    chosen = []
    for dis in DISEASES:
        by_src = tree.get(dis)
        if not by_src:
            print(f"[pick] 경고: {split}/{dis} 행이 없음 -> 건너뜀")
            continue

        # 출처별 할당량: 키 개수에 비례 배분 후 최대잔여법으로 합을 per_class 에 맞춘다.
        keys = {s: sorted(v) for s, v in by_src.items()}
        tot = {s: len(k) for s, k in keys.items()}
        grand = sum(tot.values())
        target = min(per_class, grand)
        raw = {s: tot[s] * target / grand for s in tot}
        quota = {s: min(int(raw[s]), tot[s]) for s in tot}
        for s in sorted(tot, key=lambda s: (-(raw[s] - int(raw[s])), s)):
            if sum(quota.values()) >= target:
                break
            if quota[s] < tot[s]:
                quota[s] += 1
        # 큰 출처가 다 못 받은 몫을 여유 있는 출처로 넘긴다(1장짜리 출처가 많을 때).
        while sum(quota.values()) < target:
            cands = [s for s in tot if quota[s] < tot[s]]
            if not cands:
                break
            s = max(cands, key=lambda s: (tot[s] - quota[s], s))
            quota[s] += 1

        got = []
        for s in sorted(quota):
            ks = list(keys[s])
            _rng(seed, split, dis, s).shuffle(ks)
            for k in ks[: quota[s]]:
                stems = sorted(by_src[s][k])
                # 같은 키 안에서도 어느 장을 쓸지는 고정 RNG 로 정한다.
                got.append(_rng(seed, split, dis, s, k).choice(stems))
        if len(got) < per_class:
            print(f"[pick] 경고: {split}/{dis} {len(got)}장만 확보"
                  f"(키 {grand}개, 요청 {per_class})")
        chosen.extend(sorted(got))
    return chosen


def find_mask(stem):
    """기존 마스크 PNG 경로(없으면 None). 참고 오버레이용."""
    for d in MASK_DIRS:
        for sp in ("train", "val", "test"):
            p = d / sp / f"{stem}.png"
            if p.is_file():
                return p
    return None


def make_pred_png(img_path, mask_path, dst):
    """[원본 | 마스크 겹친 원본] 을 가로로 붙인다 — label_editor 의 '예측 보기' 포맷."""
    img = Image.open(img_path).convert("RGB")
    m = np.array(Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)) > 127
    a = np.array(img)
    ov = a.copy()
    ov[m] = (0.45 * a[m] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
    out = np.concatenate([a, ov], axis=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst)


def place(src_img, dst, mode):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(os.path.relpath(src_img.resolve(), dst.parent), dst)
    elif mode == "hardlink":
        try:
            os.link(src_img, dst)
        except OSError:
            shutil.copy2(src_img, dst)
    else:
        shutil.copy2(src_img, dst)


def main():
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    if not out.name.endswith("_pred") and "_pred_" not in out.name:
        print(f"[경고] '{out.name}' 은 '_pred' 로 끝나지 않아 label_editor 가 못 찾는다")
    if args.clean and out.exists():
        shutil.rmtree(out)

    with open(src / "labels.csv", encoding="utf-8") as f:
        allrows = list(csv.DictReader(f))
    counts = split_counts(args.per_class, args.ratio, SPLITS)
    print(f"[plan] 질환별 {args.per_class}장 = " +
          " / ".join(f"{sp} {n}" for sp, n in counts.items()))

    man, n_pred = [], 0
    for split, n in counts.items():
        if n <= 0:
            continue
        rows = [r for r in allrows if r["split"] == split]
        if not rows:
            raise SystemExit(f"[에러] {src}/labels.csv 에 split={split} 행이 없음")
        info = {r["stem"]: r for r in rows}
        for stem in pick(rows, n, args.seed, split):
            r = info[stem]
            case = f"{r['disease']}__{stem}"
            img = src / split / f"{stem}.png"
            if not img.is_file():
                print(f"[경고] 이미지 없음 -> 제외: {img}")
                continue
            place(img, out / "images" / split / f"{case}.png", args.mode)

            lbl = out / "labels" / split / f"{case}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            if not lbl.exists():
                lbl.write_text("")                # 빈 라벨 = 아직 안 칠함

            mask = find_mask(stem)
            if mask and not args.no_pred:
                make_pred_png(img, mask, out / "pred" / split / f"{case}.png")
                n_pred += 1
            man.append({"case": case, "split": split, "stem": stem,
                        "disease": r["disease"], "source": r["source"],
                        "subject": r["subject"], "has_mask": int(mask is not None)})

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["case", "split", "stem", "disease", "source",
                               "subject", "has_mask"])
        w.writeheader()
        w.writerows(sorted(man, key=lambda r: (SPLITS.index(r["split"]),
                                               DISEASES.index(r["disease"]), r["stem"])))

    report(out, man, n_pred, counts)


def report(out, man, n_pred, counts):
    """분포 + 무결성 점검. 학습 전에 눈으로 볼 것."""
    print(f"\n[out] {out}  이미지 {len(man)}장, 참고 오버레이 {n_pred}장")
    print(f"\n=== split × 질환 (장수) ===")
    print(f"{'질환':6s} " + " ".join(f"{sp:>6s}" for sp in SPLITS) + f"{'합계':>7s}")
    for d in DISEASES:
        c = [sum(1 for r in man if r["disease"] == d and r["split"] == sp) for sp in SPLITS]
        if not sum(c):
            continue
        print(f"{d:6s} " + " ".join(f"{v:6d}" for v in c) + f"{sum(c):7d}")

    print(f"\n=== 질환별 출처 분포 (전체) ===")
    for d in DISEASES:
        g = [r for r in man if r["disease"] == d]
        if not g:
            continue
        c = Counter(r["source"] for r in g)
        print(f"{d:6s} {dict(sorted(c.items(), key=lambda x: -x[1]))}")

    print(f"\n=== 무결성 점검 (모두 0 이어야 정상) ===")
    dup = len(man) - len({r["subject"] for r in man})
    print(f"중복 케이스 키: {dup}")
    grp = {sp: {(r["disease"], r["subject"]) for r in man if r["split"] == sp}
           for sp in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = grp[a] & grp[b]
        print(f"{a} ∩ {b} 그룹 누수: {len(ov)}" + ("  <-- 누수!" if ov else ""))

    n_mask = sum(r["has_mask"] for r in man)
    print(f"\n기존 마스크 보유(참고 오버레이 제공): {n_mask}/{len(man)}장")
    print(f"\n다음: python3 {ROOT}/streamlit/label_editor.py  ->  http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
