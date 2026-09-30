"""label_editor 로 칠한 폴리곤 라벨 -> 이진 마스크 PNG. self-training 의 입력.

label_editor 는 YOLO 폴리곤 txt(정규화 좌표)로 저장한다. seg 학습 코드는
atopy_crop_masks 규약(L 모드, 0/255, 원본과 같은 크기)의 PNG 를 먹으므로 여기서 변환한다.
파일명은 manifest.csv 를 따라 '<질환>__<stem>' 에서 원래 <stem> 으로 되돌린다.

'아직 안 칠한 것' 과 '병변이 없어서 빈 마스크' 를 구분하는 게 이 스크립트의 핵심이다.
정상 클래스는 빈 마스크가 정답이라 라벨 파일이 비어 있는 게 정상인데, 손도 안 댄
파일도 똑같이 비어 있다. 둘을 파일 내용으로는 못 가른다.
label_editor 는 처음 저장할 때 <case>.txt.orig 백업을 만든다(사이드바 ✎ 표시와 같은
신호). 그래서 .txt.orig 존재 = '사람이 한 번 이상 저장함' 으로 읽고, 그것만 내보낸다.
즉 정상 이미지도 편집기에서 한 번 저장(Ctrl+S)해야 빈 마스크로 나온다.

사용법:
    python3 export_seed_masks.py --progress      # 진행률만 보고 끝
    python3 export_seed_masks.py                 # 완료분 -> ../dataset_face_seed_masks/
    python3 export_seed_masks.py --all           # 미작업분도 빈 마스크로 강제 출력
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DISEASES = ["건선", "아토피", "여드름", "정상", "주사", "지루"]


def parse_args():
    p = argparse.ArgumentParser(description="폴리곤 라벨 -> 이진 마스크 PNG")
    p.add_argument("--seed_dir", default=str(ROOT / "dataset_face_seed_pred"))
    p.add_argument("--out", default=str(ROOT / "dataset_face_seed_masks"))
    p.add_argument("--progress", action="store_true", help="진행률만 출력")
    p.add_argument("--all", action="store_true",
                   help="저장 이력(.txt.orig)이 없는 케이스도 내보낸다(빈 마스크가 됨)")
    return p.parse_args()


def read_polys(path):
    """txt -> [[(x,y),...]] 정규화 좌표. 정점 3개 미만은 버린다(편집기와 같은 규칙)."""
    polys = []
    if not path.is_file():
        return polys
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        n = [float(x) for x in v[1:]]
        pts = [(n[i], n[i + 1]) for i in range(0, len(n) - 1, 2)]
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def main():
    args = parse_args()
    seed_dir, out = Path(args.seed_dir), Path(args.out)
    with open(seed_dir / "manifest.csv", encoding="utf-8") as f:
        man = list(csv.DictReader(f))

    stat = defaultdict(Counter)
    todo = []
    for r in man:
        lbl = seed_dir / "labels" / r["split"] / f"{r['case']}.txt"
        done = lbl.with_suffix(".txt.orig").exists()
        polys = read_polys(lbl)
        stat[r["disease"]]["전체"] += 1
        stat[r["disease"]]["완료" if done else "미작업"] += 1
        if polys:
            stat[r["disease"]]["폴리곤有"] += 1
        if done or args.all:
            todo.append((r, polys))

    print(f"{'질환':6s} {'완료':>5s} {'미작업':>6s} {'폴리곤有':>8s} {'전체':>5s}")
    tot = Counter()
    for d in DISEASES:
        c = stat.get(d)
        if not c:
            continue
        tot.update(c)
        print(f"{d:6s} {c['완료']:5d} {c['미작업']:6d} {c['폴리곤有']:8d} {c['전체']:5d}")
    print(f"{'합계':6s} {tot['완료']:5d} {tot['미작업']:6d} {tot['폴리곤有']:8d} {tot['전체']:5d}")

    if args.progress:
        return
    if not todo:
        raise SystemExit("\n[중단] 내보낼 게 없다. 편집기에서 저장(Ctrl+S)한 케이스가 없음.")

    n_empty = 0
    for r, polys in todo:
        img = seed_dir / "images" / r["split"] / f"{r['case']}.png"
        w, h = Image.open(img).size
        m = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(m)
        for pts in polys:
            dr.polygon([(x * w, y * h) for x, y in pts], fill=255)
        if not polys:
            n_empty += 1
        dst = out / r["split"] / f"{r['stem']}.png"          # 원래 stem 으로 복원
        dst.parent.mkdir(parents=True, exist_ok=True)
        m.save(dst)

    px = [np.array(Image.open(out / r["split"] / f"{r['stem']}.png")).mean() / 255
          for r, _ in todo]
    print(f"\n[out] {out}  마스크 {len(todo)}장 (빈 마스크 {n_empty}장)")
    print(f"      전경 비율 평균 {np.mean(px) * 100:.1f}%  중앙값 {np.median(px) * 100:.1f}%")


if __name__ == "__main__":
    main()
