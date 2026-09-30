"""dataset_face 생성 — dataset_disease 에서 정면(angle=정면)만 추려낸 파생 데이터셋.

측면을 빼는 이유는 두 각도가 사실상 다른 도메인이기 때문이다. 얼굴 정면은 병변이
좌우 대칭으로 다 보이지만 측면은 절반이 가려지고 코/뺨 그림자가 병변처럼 보인다.
정면만 쓰면 데이터는 절반이 되는 대신 입력 분포가 한 가지로 좁아진다.

split 은 절대 다시 뽑지 않는다. dataset_disease 의 split 을 그대로 물려받는다:
  - 그 split 은 (질환, 케이스 키) 단위로 짜서 train/val/test 그룹 교집합이 0이다.
    부분집합을 떠도 교집합은 0 이하로만 갈 수 있으니 누수 없음이 보존된다.
  - 여기서 다시 뽑으면 dataset_disease 로 낸 결과와 숫자를 나란히 놓을 수 없다.
    두 데이터셋의 차이가 '각도' 때문인지 '다른 split' 때문인지 못 가른다.
각도는 (질환, 각도)별로 층화해 뽑았으므로 정면만 떼도 split 비율이 그대로 유지된다
(train 4200 / val 600 / test 600, 질환마다 정확히 700/100/100).

출력 규약은 dataset_disease 와 동일 — 자립형이라 그대로 학습에 넣을 수 있다:
    <OUT>/train/*.png   <OUT>/val/*.png   <OUT>/test/*.png
    <OUT>/labels.csv    split,stem,disease,angle,source,subject
    <OUT>/stats.txt     분포 + 케이스 키 누수 점검 리포트

기본은 하드링크다. 같은 파일시스템이라 용량을 안 먹으면서(정면만 해도 ~5GB),
심볼릭 링크와 달리 dataset_disease 를 지우거나 옮겨도 안 깨진다.

사용법:
    python3 make_dataset_face.py                  # ../dataset_disease -> ../dataset_face
    python3 make_dataset_face.py --mode copy      # 독립 사본(다른 장비로 옮길 때)
    python3 make_dataset_face.py --angle 측면 --out ../dataset_side
    python3 make_dataset_face.py --labels_only    # CSV/stats 만 (배치 전 점검)
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

import make_dataset_disease as mdd

ROOT = Path(__file__).resolve().parent
OGW = ROOT.parent
SPLITS = mdd.SPLITS


def parse_args():
    p = argparse.ArgumentParser(description="dataset_disease -> 각도 하나만 남긴 파생 데이터셋")
    p.add_argument("--src", default=str(OGW / "dataset_disease"),
                   help="원본 데이터셋 루트. labels.csv 와 train/val/test 폴더가 있어야 한다")
    p.add_argument("--out", default=str(OGW / "dataset_face"))
    p.add_argument("--angle", default="정면", choices=mdd.ANGLES,
                   help="남길 각도(기본 정면)")
    p.add_argument("--mode", default="hardlink", choices=["hardlink", "symlink", "copy"],
                   help="이미지 배치 방식(기본 hardlink — 용량 0, 원본 이동에도 안전)")
    p.add_argument("--labels_only", action="store_true",
                   help="이미지는 건드리지 않고 labels.csv/stats.txt 만 생성")
    p.add_argument("--clean", action="store_true",
                   help="기존 <OUT>/{train,val,test} 를 비우고 새로 만든다")
    return p.parse_args()


def read_rows(src, angle):
    """원본 labels.csv 에서 해당 각도 행만. 반환 행은 mdd.write_stats 가 쓰는 스키마."""
    csv_path = Path(src) / "labels.csv"
    if not csv_path.exists():
        raise SystemExit(f"[에러] labels.csv 없음: {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        allrows = list(csv.DictReader(f))
    rows = [r for r in allrows if r["angle"] == angle]
    if not rows:
        raise SystemExit(f"[에러] angle=={angle} 인 행이 없음. 원본의 angle 값을 확인할 것")
    print(f"[read] {csv_path}  전체 {len(allrows)}행 -> {angle} {len(rows)}행")
    return rows


def place_images(rows, src, out, mode, clean):
    """split 폴더로 이미지 배치. 반환: 원본 이미지가 실제로 있던 행만."""
    for sp in SPLITS:
        d = Path(out) / sp
        if clean and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    placed, missing, fellback = [], [], 0
    for r in rows:
        s = Path(src) / r["split"] / f"{r['stem']}.png"
        if not s.exists():
            missing.append(r["stem"])
            continue
        dst = Path(out) / r["split"] / f"{r['stem']}.png"
        if not dst.exists():
            if mode == "symlink":
                os.symlink(os.path.relpath(s.resolve(), dst.parent), dst)
            elif mode == "hardlink":
                try:
                    os.link(s, dst)
                except OSError:
                    # 파일시스템이 하드링크를 막으면(원본이 심볼릭 링크인 경우 포함) 복사로.
                    shutil.copy2(s, dst)
                    fellback += 1
            else:
                shutil.copy2(s, dst)
        placed.append(r)

    if missing:
        print(f"[image] 경고: 원본 PNG 없음 {len(missing)}건 -> 제외. 예: {missing[:5]}")
    if fellback:
        print(f"[image] 하드링크 실패 {fellback}건은 복사로 대체")
    print(f"[image] {mode} 완료: {len(placed)}건 -> {out}/{{train,val,test}}")
    return placed


def main():
    args = parse_args()
    out = Path(args.out)
    if Path(args.src).resolve() == out.resolve():
        raise SystemExit("[에러] --src 와 --out 이 같다. 원본을 덮어쓸 뻔했다")
    out.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.src, args.angle)
    if args.labels_only:
        print("[image] --labels_only -> 이미지 배치 생략")
    else:
        rows = place_images(rows, args.src, out, args.mode, args.clean)
        if not rows:
            raise SystemExit("[에러] 배치된 이미지가 0건")

    csv_path = out / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "disease", "angle", "source", "subject"])
        for r in sorted(rows, key=lambda r: (SPLITS.index(r["split"]), r["disease"], r["stem"])):
            w.writerow([r["split"], r["stem"], r["disease"], r["angle"],
                        r["source"], r["subject"]])
    print(f"[out] {csv_path}  ({len(rows)}행)")

    # 각도가 하나뿐이니 리포트의 각도 열도 하나로 줄인다(0으로 찬 열을 안 찍도록).
    mdd.ANGLES = [args.angle]
    mdd.write_stats(out, rows)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
