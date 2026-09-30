"""TSTR(Train Synthetic, Test Real) 용 데이터셋 조립.

합성만으로 학습하고 실제로 시험한다. train.py 는 {train,val,test}/*.png + labels.csv
를 그대로 먹으므로, 여기서는 **심볼릭 링크만 걸어** 그 규약을 맞춘 폴더를 만든다
(이미지 복사 없음 — 21,000장 × 4세트를 복사할 이유가 없다).

두 가지 프로토콜
    A: val/test 를 원래대로 둔다 (val 600, test 600).
       TRTR 기준선(runs/dis_effb0_r512_front, test 600장)과 **같은 test 집합**이라
       "TSTR 이 TRTR 의 몇 %인가"를 그대로 계산할 수 있다. 비율은 여기서만 나온다.
    B: 실제 train 정면 4,200장을 전부 val/test 로 옮긴다 (val 2,700, test 2,700).
       합성만으로 학습하면 실제 train 이 놀고 있으니 평가에 쓰는 것이다. test 가
       4.5배가 되어 추정 분산이 줄고 케이스 다양성이 넓어진다.
       **다만 B 로는 비율을 계산하면 안 된다** — 그 4,200장으로 학습한 TRTR 모델은
       이 test 를 이미 본 셈이라 누출이고, 누출 없는 TRTR 기준선을 만들 수 없다.
       B 는 'TSTR 절대 성능을 더 정확히' 재는 용도다.

B 에서 옮긴 행은 source 뒤에 '*' 를 붙인다
    출처가 train 과 test 사이에 겹친다(H1/H2/H3/H5/H7/H9). 표시를 안 하면 train.py
    의 출처별 정확도 표에서 '원래 test'와 '옮겨온 train'이 같은 칸에 합쳐져 버린다.
    두 집합은 성격이 다르다 — 원래 test 는 새 출처·새 케이스고, 옮겨온 train 은
    학습 출처와 같은 분포다. 섞으면 절대 수치가 올라가므로 반드시 갈라 봐야 한다.

subject 로 묶어 나눈다
    한 사람의 사진이 val 과 test 에 나눠 들어가면 두 집합이 독립이 아니다.
    질환별로 균형을 맞추면서 subject 단위로 배분한다.

사용법:
    python3 make_tstr_dataset.py                       # 4세트 전부
    python3 make_tstr_dataset.py --arm gan --protocol A
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OGW = ROOT.parent
REAL = OGW / "dataset_disease"
DISEASES = ["건선", "아토피", "여드름", "정상", "주사", "지루"]

SYNTH = {
    "gan":  OGW / "gan" / "synth_front",
    "sdft": OGW / "diffusion" / "synth_ft_front",
}
HEADER = ["split", "stem", "disease", "angle", "source", "subject"]


def real_rows(angle="정면"):
    with open(REAL / "labels.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["angle"] == angle]


def synth_rows(arm):
    with open(SYNTH[arm] / "labels.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_by_subject(rows, pinned=None, seed=0):
    """train 정면 행을 subject 단위로 절반씩(val/test) 나눈다. 질환별로 균형을 맞춘다.

    subject 하나가 여러 질환을 갖는 경우(15명)는 첫 질환 기준으로 배정한다 — 그
    사람의 모든 사진이 한쪽으로만 가면 되는 것이라 균형이 몇 장 어긋나는 건 무해하다.

    pinned: 원래 val/test 에 이미 등장하는 subject -> 그 split. 원본 데이터셋 자체에
    split 을 가로지르는 subject 가 5명 있어서, 그냥 나누면 같은 사람이 val 과 test
    양쪽에 앉는다. 이미 있는 쪽으로 붙여 준다.
    """
    pinned = pinned or {}
    by_sub = defaultdict(list)
    for r in rows:
        by_sub[r["subject"]].append(r)
    subs = sorted(by_sub)
    random.Random(seed).shuffle(subs)
    # 질환별로 번갈아 담아 양쪽 클래스 분포를 맞춘다.
    quota = defaultdict(int)
    to_val = set()
    for s in subs:
        d = by_sub[s][0]["disease"]
        n = len(by_sub[s])
        if s in pinned:
            if pinned[s] == "val":
                to_val.add(s); quota[d] += n
            else:
                quota[d] -= n
        elif quota[d] <= 0:
            to_val.add(s); quota[d] += n
        else:
            quota[d] -= n
    val = [r for s in subs if s in to_val for r in by_sub[s]]
    test = [r for s in subs if s not in to_val for r in by_sub[s]]
    return val, test


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def build(arm, protocol, out_root, seed):
    out = Path(out_root) / f"{arm}_{protocol}"
    for s in ("train", "val", "test"):
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        # 다시 만들 때 예전 링크를 반드시 지운다. 배분이 조금이라도 바뀌면 옛 링크가
        # 남아 같은 사진이 val 과 test 양쪽에 앉는다 — 덮어쓰기만으로는 안 지워진다.
        for old in d.glob("*.png"):
            old.unlink()
    rows_out = []

    # --- train = 합성 전량
    simg = SYNTH[arm] / "images"
    for r in synth_rows(arm):
        link(simg / f"{r['stem']}.png", out / "train" / f"{r['stem']}.png")
        rows_out.append({**r, "split": "train"})
    n_train = len(rows_out)

    # --- val/test = 실제
    rr = real_rows()
    orig = {s: [r for r in rr if r["split"] == s] for s in ("train", "val", "test")}
    plan = {"val": list(orig["val"]), "test": list(orig["test"])}
    if protocol == "B":
        pinned = {r["subject"]: s for s in ("val", "test") for r in orig[s]}
        mv_val, mv_test = split_by_subject(orig["train"], pinned, seed)
        for split, mv in (("val", mv_val), ("test", mv_test)):
            for r in mv:
                plan[split].append({**r, "source": r["source"] + "*"})

    for split, rows in plan.items():
        for r in rows:
            link(REAL / r["split"] / f"{r['stem']}.png", out / split / f"{r['stem']}.png")
            rows_out.append({**r, "split": split})

    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, HEADER, extrasaction="ignore")
        w.writeheader(); w.writerows(rows_out)

    n = {s: sum(1 for r in rows_out if r["split"] == s) for s in ("train", "val", "test")}
    moved = sum(1 for r in rows_out if r["source"].endswith("*"))
    print(f"[tstr] {arm}_{protocol}: train(합성) {n['train']}  val {n['val']}  test {n['test']}"
          f"{f'  (옮긴 실제 train {moved}장 포함)' if moved else ''}  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="TSTR 데이터셋 조립",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--arm", action="append", choices=list(SYNTH), help="기본 전부")
    ap.add_argument("--protocol", action="append", choices=["A", "B"], help="기본 둘 다")
    ap.add_argument("--out", default=str(OGW / "dataset_tstr"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for arm in (args.arm or list(SYNTH)):
        if not (SYNTH[arm] / "labels.csv").exists():
            print(f"[tstr] {arm}: {SYNTH[arm]} 없음. 건너뜀"); continue
        for p in (args.protocol or ["A", "B"]):
            build(arm, p, args.out, args.seed)


if __name__ == "__main__":
    main()
