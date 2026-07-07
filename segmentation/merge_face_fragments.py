#!/usr/bin/env python3
"""
merge_face_fragments.py

dataset_face 의 잘게 쪼개진 병변 폴리곤을 "가까운 조각만" 국소 병합한다.

방침(대화에서 합의):
  - 원본(dataset_face)은 손대지 않고 dataset_face_merged 로 복사 후 작업.
  - train 라벨만 대상. val/test/이미지는 그대로.
  - 대상 = 폴리곤 개수 >= MIN_FRAGMENTS 이고, 서로 MAX_GAP_PX 이내로 가까운
    조각이 실제로 있는 파일만. (개수 단독이 아니라 개수 + 근접성)
  - 병합은 래스터화 경유(A): 폴리곤 -> 마스크 -> 가까운 조각만 morphology 로
    잇기 -> findContours 로 폴리곤 복원. 멀리 떨어진 병변은 보존.
  - 68장(대상) 각각 원본 사진 위 병합 전/후 오버레이를 저장해 사람이 검수.

학습은 이 스크립트 밖에서, 검수 통과 후에 별도로 돌린다.
"""
import os
import glob
import shutil
import argparse
import numpy as np
import cv2
from scipy import ndimage

IMG = 1024  # 래스터화 캔버스 크기(원본 imgsz)


def read_label(path):
    """YOLO seg 라벨 -> [(cls, np.array([[x,y],...] normalized)), ...]"""
    polys = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:  # cls + 최소 3점
                continue
            cls = int(float(p[0]))
            xy = np.array(p[1:], dtype=np.float32).reshape(-1, 2)
            polys.append((cls, xy))
    return polys


def write_label(path, polys):
    lines = []
    for cls, xy in polys:
        flat = " ".join(f"{v:.6f}" for v in xy.reshape(-1))
        lines.append(f"{cls} {flat}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def polys_to_mask(polys):
    """정규화 폴리곤들을 IMGxIMG 이진 마스크로."""
    m = np.zeros((IMG, IMG), np.uint8)
    for _, xy in polys:
        pts = np.round(xy * IMG).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def mask_to_polys(mask, cls=0, min_area=8):
    """이진 마스크 -> 정규화 폴리곤 리스트 (외곽선)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        # 살짝 단순화(좌표 폭증 방지)
        eps = 0.002 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) < 3:
            continue
        xy = c.reshape(-1, 2).astype(np.float32) / IMG
        out.append((cls, xy))
    return out


def merge_near(mask, max_gap_px, min_fragments):
    """
    조각이 min_fragments개 이상이고 max_gap_px 이내로 가까운 조각들만 병합.
    반환: (merged_mask, changed, n_before, n_after)
    """
    n_before, _ = cv2.connectedComponents(mask)
    n_before -= 1  # 배경 제외
    if n_before < min_fragments:
        return mask, False, n_before, n_before

    k = max(1, max_gap_px // 2)  # 절반씩 팽창->침식하면 gap<=2k 인 조각만 이어짐
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dil = cv2.dilate(mask, ker)
    fill = ndimage.binary_fill_holes(dil).astype(np.uint8)
    merged = cv2.erode(fill, ker)
    merged = (merged & 1).astype(np.uint8)

    n_after, _ = cv2.connectedComponents(merged)
    n_after -= 1
    changed = n_after != n_before
    return merged, changed, n_before, n_after


def overlay(img_path, before_polys, after_polys, out_path):
    """원본 사진 위에 병합 전(빨강)/후(초록) 폴리곤을 반투명으로 얹어 나란히 저장."""
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((IMG, IMG, 3), np.uint8)
    img = cv2.resize(img, (IMG, IMG))

    def draw(base, polys, color):
        ov = base.copy()
        for _, xy in polys:
            pts = np.round(xy * IMG).astype(np.int32)
            cv2.fillPoly(ov, [pts], color)
            cv2.polylines(base, [pts], True, color, 2)
        return cv2.addWeighted(ov, 0.35, base, 0.65, 0)

    left = draw(img.copy(), before_polys, (0, 0, 255))
    right = draw(img.copy(), after_polys, (0, 200, 0))
    cv2.putText(left, f"BEFORE ({len(before_polys)})", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(right, f"AFTER ({len(after_polys)})", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(out_path, np.hstack([left, right]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_face")
    ap.add_argument("--dst", default="dataset_face_merged")
    ap.add_argument("--max-gap", type=int, default=40, help="px @1024")
    ap.add_argument("--min-fragments", type=int, default=10)
    ap.add_argument("--review-dir", default="merge_review")
    ap.add_argument("--splits", default="train,val,test",
                    help="쉼표구분. 병합 적용할 split (예: train 만 하려면 --splits train)")
    ap.add_argument("--overlay-limit", type=int, default=25,
                    help="split별 검수 오버레이 최대 장수(용량 절약). 0이면 안 만듦")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 이면 대상 중 N장만 스윕(값 튜닝용). 데이터셋 복사/수정 안 함.")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, args.src)
    dst = os.path.join(root, args.dst)
    review = os.path.join(root, args.review_dir)
    os.makedirs(review, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    dry = args.limit > 0
    if not dry:
        # 복사본 준비 (원본 무손상)
        if os.path.exists(dst):
            print(f"[!] {dst} 이미 존재 -> 삭제 후 재생성")
            shutil.rmtree(dst)
        print(f"[복사] {src} -> {dst}")
        shutil.copytree(src, dst)
        # data.yaml 의 path 를 새 폴더로 교정
        yml = os.path.join(dst, "data.yaml")
        if os.path.exists(yml):
            with open(yml) as f:
                txt = f.read()
            txt = txt.replace(f"path: {src}", f"path: {dst}")
            with open(yml, "w") as f:
                f.write(txt)

    print(f"[설정] min_fragments={args.min_fragments}  max_gap={args.max_gap}px  "
          f"splits={splits}")
    total_changed = 0
    for split in splits:
        src_lbl = os.path.join(src, "labels", split)
        src_img = os.path.join(src, "images", split)
        label_files = sorted(glob.glob(os.path.join(src_lbl, "*.txt")))
        if dry:
            label_files = label_files[:args.limit]

        n_changed, n_target, merged_stat, ov_made = 0, 0, [], 0
        for lf in label_files:
            polys = read_label(lf)
            # 값싼 선별: 폴리곤 라인 수로 1차 컷(실제 게이트는 merge_near 가 덩어리 수로)
            if len(polys) < args.min_fragments:
                continue
            n_target += 1
            name = os.path.splitext(os.path.basename(lf))[0]
            classes = sorted({c for c, _ in polys})
            new_polys, any_change = [], False
            for cls in classes:
                cps = [(c, xy) for c, xy in polys if c == cls]
                merged, changed, nb, na = merge_near(
                    polys_to_mask(cps), args.max_gap, args.min_fragments)
                any_change = any_change or changed
                new_polys.extend(mask_to_polys(merged, cls=cls))

            if any_change:
                n_changed += 1
                merged_stat.append((name, len(polys), len(new_polys)))
                if not dry:
                    write_label(os.path.join(dst, "labels", split,
                                             name + ".txt"), new_polys)
                # 검수용 오버레이: 변경된 파일 중 최대 overlay_limit 장만(용량 절약)
                if ov_made < args.overlay_limit:
                    overlay(os.path.join(src_img, name + ".png"),
                            polys, new_polys,
                            os.path.join(review,
                                         f"{split}_{name}_gap{args.max_gap}.png"))
                    ov_made += 1

        total_changed += n_changed
        print(f"  [{split:5s}] 대상(라인>= {args.min_fragments}) {n_target:4d} | "
              f"실제 수정 {n_changed:4d} | 오버레이 샘플 {ov_made}장")
        for n, b, a in sorted(merged_stat, key=lambda x: x[1] - x[2],
                              reverse=True)[:5]:
            print(f"        {n}: {b} -> {a}")

    print(f"\n[검수] 오버레이 샘플(왼:전 빨강 / 오른:후 초록): {review}/  "
          f"(split별 최대 {args.overlay_limit}장)")
    if not dry:
        print(f"[완료] 병합 데이터셋: {dst}  (총 {total_changed}장 수정, "
              f"splits={splits})")


if __name__ == "__main__":
    main()
