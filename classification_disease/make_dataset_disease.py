"""dataset_disease 생성 — skin_dataset 의 라벨 JSON -> split 폴더 + labels.csv.

중증도/징후는 쓰지 않는다. JSON 에서 질환 분류에 필요한 것만 뽑는다:
    diagnosis_info.diagnosis_name  -> disease (라벨)
    photograph.file_path           -> angle (정면/측면)
    identifier                     -> stem -> subject/source 파생

출력(atopy_crop_masks / dataset_topk 와 동일 규약 — 자립형):
    <OUT>/train/*.png   <OUT>/val/*.png   <OUT>/test/*.png
    <OUT>/labels.csv    split,stem,disease,angle,source,subject
    <OUT>/stats.txt     분포 + 케이스 키 누수 점검 리포트

split 규약 — 데이터 조사 결과에 근거한 설계:
    test  = 제공된 Validation 전체. train 과 케이스 키 겹침이 0이고 출처(prefix) 분포도
            달라서 '새 케이스 + 새 출처' 조건을 만족하는 유일한 진짜 held-out 이다.
            튜닝에 쓰면 이 성질이 사라지므로 최종 1회만 본다.
    train/val = 제공된 Training 을 케이스 키 단위로 분할. 파일 단위로 자르면 안 된다 —
            한 키가 여러 장을 갖고 있어(건선 정면 800장/498키) 같은 원본 케이스가
            양쪽에 들어가면 val 이 부풀려진다.
    val 은 (질환, 각도, 출처)별로 층화해서 뗀다. 즉 val 은 '같은 출처의 새 케이스'를,
    test 는 '다른 출처의 새 케이스'를 잰다. 둘의 격차가 곧 출처 과적합의 크기다.

전체 stem 10,800개가 충돌 없이 고유해서 split 폴더에 평탄하게 모아도 안전하다.

사용법:
    python3 make_dataset_disease.py                    # ../skin_dataset -> ../dataset_disease
    python3 make_dataset_disease.py --link             # 복사 대신 심볼릭 링크(2GB+ 절약)
    python3 make_dataset_disease.py --val_frac 0.15
    python3 make_dataset_disease.py --labels_only      # 이미지 없이 CSV 만(다운로드 중)
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # classification_disease
OGW = ROOT.parent

# 라벨 순서 고정 — 인덱스가 체크포인트/리포트에 박히므로 절대 바꾸지 말 것.
DISEASES = ["건선", "아토피", "여드름", "정상", "주사", "지루"]
ANGLES = ["정면", "측면"]
SPLITS = ["train", "val", "test"]


def parse_args():
    p = argparse.ArgumentParser(description="dataset_disease 생성 (질환 6-way)")
    p.add_argument("--src", default=str(OGW / "skin_dataset"),
                   help="데이터 루트. 하위에 Training/ Validation/ 이 있는 구조")
    p.add_argument("--out", default=str(OGW / "dataset_disease"))
    p.add_argument("--val_frac", type=float, default=0.125,
                   help="Training 에서 val 로 뗄 케이스 키 비율(기본 0.125 -> 800장당 ~100장)")
    p.add_argument("--link", action="store_true",
                   help="이미지 복사 대신 심볼릭 링크(디스크 절약, SRC 이동 시 깨짐)")
    p.add_argument("--labels_only", action="store_true",
                   help="이미지는 건드리지 않고 labels.csv/stats.txt 만 생성(다운로드 중 점검용)")
    p.add_argument("--keep_test_leaks", action="store_true",
                   help="test 와 케이스 키가 겹치는 train/val 행을 버리지 않는다. "
                        "겹침이 의도된 데이터(한 케이스의 두 질환)로 확인되면 켤 것")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def subject_of(stem):
    """'H1_500531_P14_L0' -> 'H1_500531'. 앞 두 필드가 원본 케이스 키다.

    '같은 얼굴'이 아니다 — 이 데이터는 합성이라 키를 공유하는 파일도 서로 다른 얼굴로
    생성돼 있다(H2_21172_P5 와 P2 를 열어보면 머리·배경·인상이 전부 다르다. 1200장
    전부 픽셀 고유로 완전중복도 없다). 공유하는 건 병변 케이스 쪽이다 — 같은 키의
    이미지들은 같은 원본 케이스에서 파생돼 병변 양상이 서로 닮아 있다.

    그래도 이 키로 묶어서 split 해야 한다. 얼굴 기억(identity leak) 때문이 아니라
    같은 원본 케이스가 train/val 에 갈라져 들어가면 val 이 부풀려지기 때문이다.
    """
    return "_".join(stem.split("_")[:2])


def source_of(stem):
    """'H1_500531_P14_L0' -> 'H1'. 촬영 출처(기관/장비 추정)."""
    return stem.split("_")[0]


def image_path_for(json_path):
    """라벨 JSON 경로 -> 대응 이미지 경로.

    02.라벨링데이터/TL_아토피_정면/x.json -> 01.원천데이터/TS_아토피_정면/x.png
    (Validation 은 VL_ -> VS_). 규칙이 안 맞으면 None.
    """
    parts = list(json_path.parts)
    for i, seg in enumerate(parts):
        if seg.startswith("02."):
            parts[i] = seg.replace("02.", "01.", 1).replace("라벨링데이터", "원천데이터")
            break
    else:
        return None
    for i, seg in enumerate(parts):
        if seg.startswith("TL_"):
            parts[i] = "TS_" + seg[3:]
        elif seg.startswith("VL_"):
            parts[i] = "VS_" + seg[3:]
    return Path(*parts).with_suffix(".png")


def read_label(json_path):
    """JSON -> dict 또는 None(파싱 실패/정의 밖 라벨). 필요한 필드만 뽑는다."""
    try:
        ann = json.loads(json_path.read_text(encoding="utf-8"))["annotations"][0]
    except (json.JSONDecodeError, KeyError, IndexError, OSError):
        return None
    disease = ann.get("diagnosis_info", {}).get("diagnosis_name")
    if disease not in DISEASES:
        return None
    # 각도는 폴더명이 아니라 JSON 안의 file_path('아토피/측면/x.png')에서 읽는다 —
    # 폴더명 오배치가 있어도 라벨 원본을 따라간다.
    fp = ann.get("photograph", {}).get("file_path", "")
    angle = next((a for a in ANGLES if f"/{a}/" in fp), None)
    stem = ann.get("identifier") or json_path.stem
    return {"stem": stem, "disease": disease, "angle": angle}


def scan(src):
    """src 하위 라벨 JSON 전부 -> [row]. base 는 폴더(Training/Validation)로 정한다."""
    rows, skipped, no_angle = [], 0, 0
    for jp in sorted(Path(src).rglob("*.json")):
        parts = set(jp.parts)
        if "Training" in parts:
            base = "train"
        elif "Validation" in parts:
            base = "test"
        else:
            continue
        rec = read_label(jp)
        if rec is None:
            skipped += 1
            continue
        if rec["angle"] is None:
            no_angle += 1
            continue
        rows.append({**rec, "base": base, "img": image_path_for(jp),
                     "subject": subject_of(rec["stem"]), "source": source_of(rec["stem"])})
    if skipped:
        print(f"[scan] 건너뜀(라벨 파싱 실패/정의 밖 질환): {skipped}")
    if no_angle:
        print(f"[scan] 건너뜀(각도 판별 불가): {no_angle}")
    return rows


def _stratum_rng(seed, key):
    """층마다 독립적인 RNG. 층 키에서 시드를 유도해 다른 층의 존재에 영향받지 않게 한다.

    하나의 Random(seed) 를 층들이 돌려 쓰면, 나중에 질환이 추가돼 층이 늘어날 때
    RNG 소비 순서가 밀려서 '기존 질환의 split 까지 재배치'된다. 그러면 다운로드가
    끝나기 전 4클래스로 돌린 결과와 6클래스 결과를 비교할 수 없다.
    md5 를 쓰는 건 파이썬 hash() 가 프로세스마다 달라지기(PYTHONHASHSEED) 때문 —
    같은 입력이면 언제 어디서 돌려도 같은 split 이 나와야 한다.
    """
    h = hashlib.md5(f"{seed}|{'|'.join(key)}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _select_val_keys(keys_by_source, target, seed, stratum):
    """한 (질환, 각도) 안에서 val 로 보낼 키 집합을 고른다. 파일 수 합이 정확히 target.

    키를 쪼개지 않으면서 파일 수를 정확히 맞추는 게 목표다. 둘은 상충하지 않는다 —
    1장짜리 키가 폴더마다 309~800개로 넉넉해서 마지막 몇 장을 그걸로 채우면 된다.

    출처(prefix) 비율은 최대잔여법으로 출처별 할당량을 먼저 정해 맞춘다. 할당량만큼
    그리디로 채우고, 키 크기 때문에 남는 부족분은 '할당량 대비 가장 덜 찬 출처'부터
    메운다. 반환: (선택된 키 집합, 못 채운 부족분).
    """
    totals = {s: sum(v.values()) for s, v in keys_by_source.items()}
    grand = sum(totals.values())
    if grand <= target:
        return set(), target

    # 출처별 할당량 — 비례배분 후 최대잔여법으로 합을 정확히 target 에 맞춘다.
    raw = {s: totals[s] * target / grand for s in totals}
    quota = {s: int(raw[s]) for s in totals}
    short = target - sum(quota.values())
    for s in sorted(totals, key=lambda s: (-(raw[s] - int(raw[s])), s))[:short]:
        quota[s] += 1

    chosen, got, pool = set(), {}, {}
    for s, keys in keys_by_source.items():
        ks = sorted(keys)
        _stratum_rng(seed, stratum + (s,)).shuffle(ks)
        run, left = 0, []
        for k in ks:
            n = keys[k]
            # run + n < totals[s] : 그 출처가 통째로 val 로 가서 train 이 비는 걸 막는다.
            if run + n <= quota[s] and run + n < totals[s]:
                chosen.add(k)
                run += n
            else:
                left.append((k, n))
        got[s] = run
        pool[s] = left

    deficit = target - sum(got.values())
    while deficit > 0:
        cands = [(s, k, n) for s in pool for (k, n) in pool[s]
                 if n <= deficit and got[s] + n < totals[s]]
        if not cands:
            break                                  # 1장짜리 키가 동나면 여기서 멈춘다
        # 덜 찬 출처 우선, 그 안에서는 부족분을 크게 메우는 키 우선(빨리 수렴).
        s, k, n = max(cands, key=lambda c: (quota[c[0]] - got[c[0]], c[2]))
        chosen.add(k)
        got[s] += n
        deficit -= n
        pool[s] = [(kk, nn) for kk, nn in pool[s] if kk != k]
    return chosen, deficit


def split_train_val(rows, val_frac, seed):
    """Training 행을 (질환, 각도)마다 정확히 val_frac 비율로 분할. 출처 비율까지 맞춘다.

    그룹 단위는 '케이스 키'가 아니라 (질환, 케이스 키)다. 같은 키라도 질환이 다르면
    묶지 않는다는 뜻인데, 근거는 두 가지다:
      - 키는 사람이 아니다. 같은 키를 공유해도 생성된 얼굴이 서로 다르다(H2_21172_P5/P2,
        H2_2820 아토피/건선 모두 육안 확인). 그래서 질환을 가로질러 묶어도 막아지는
        identity leak 이 애초에 없다.
      - 질환이 다르면 병변 자체가 다르다. 같은 키를 묶는 이유가 '같은 원본 케이스라
        병변이 닮았다'인데, 그 논리는 질환 안에서만 성립한다.
    전역으로 묶으면 오히려 해롭다 — 질환을 가로지르는 키가 19개 있어(대부분 건선+지루)
    건선 층의 추첨이 지루 쪽 개수를 흔들고, 그러면 폴더당 정확히 100장이 불가능해지며
    주사/지루를 나중에 추가할 때 기존 건선 split 까지 재배치된다.

    같은 (질환, 키)의 모든 파일은 여전히 반드시 한쪽에만 들어간다.
    """
    # (질환, 각도) -> 출처 -> 키 -> 파일 수
    by_stratum = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        if r["base"] == "train":
            by_stratum[(r["disease"], r["angle"])][r["source"]][r["subject"]] += 1

    val_groups = set()
    for (dis, ang), by_src in sorted(by_stratum.items()):
        n_files = sum(sum(c.values()) for c in by_src.values())
        target = round(n_files * val_frac)
        chosen, deficit = _select_val_keys(by_src, target, seed, (dis, ang))
        if deficit:
            print(f"[split] 경고: {dis} {ang} val 목표 {target}장 중 {deficit}장 못 채움 "
                  f"(키 크기 제약). 실제 {target - deficit}장")
        val_groups.update((dis, k) for k in chosen)

    for r in rows:
        r["split"] = ("val" if (r["disease"], r["subject"]) in val_groups else "train") \
            if r["base"] == "train" else "test"
    return rows


def drop_test_leaks(rows):
    """(질환, 케이스 키)가 test 와 train/val 양쪽에 있으면 train/val 쪽 행을 버린다.

    split_train_val 과 같은 그룹 정의를 쓴다. 질환이 다른데 키만 같은 건(H2_2820:
    Training=아토피, Validation=건선) 겹침으로 치지 않는다 — 두 PNG 를 열어 확인한
    결과 서로 다른 사람이고, 질환이 다르면 병변도 달라서 누수가 아니기 때문이다.

    현재 데이터에선 걸리는 게 없다(제공된 Training/Validation 이 질환별로 이미 분리돼
    있음). 그래도 남겨 둔다 — 데이터가 갱신될 때 조용히 새어드는 걸 잡는 안전망이다.
    """
    test_groups = {(r["disease"], r["subject"]) for r in rows if r["split"] == "test"}
    kept, dropped = [], []
    for r in rows:
        if r["split"] != "test" and (r["disease"], r["subject"]) in test_groups:
            dropped.append(r)
        else:
            kept.append(r)
    if dropped:
        print(f"[leak] test 와 (질환, 키)가 겹치는 train/val 행 {len(dropped)}건 제거:")
        for r in dropped[:10]:
            print(f"       {r['stem']:24s} {r['split']:5s} {r['disease']} {r['angle']}")
        if len(dropped) > 10:
            print(f"       ... 외 {len(dropped) - 10}건")
    return kept


def place_images(rows, out, link):
    """split 폴더로 이미지 복사(또는 링크). 반환: 이미지가 실제로 놓인 행만."""
    for sp in SPLITS:
        (Path(out) / sp).mkdir(parents=True, exist_ok=True)
    placed, missing = [], 0
    for r in rows:
        src_img = r["img"]
        if src_img is None or not src_img.exists():
            missing += 1
            continue
        dst = Path(out) / r["split"] / f"{r['stem']}.png"
        if not dst.exists():
            if link:
                os.symlink(os.path.relpath(src_img.resolve(), dst.parent), dst)
            else:
                shutil.copy2(src_img, dst)
        placed.append(r)
    if missing:
        pct = missing / max(len(rows), 1) * 100
        print(f"[image] 경고: 원본 이미지 없음 {missing}건 ({pct:.1f}%) -> 제외. "
              f"다운로드가 진행 중이면 정상. 완료 후 다시 실행할 것.")
    print(f"[image] {'링크' if link else '복사'} 완료: {len(placed)}건 -> {out}/{{train,val,test}}")
    return placed


def write_stats(out, rows):
    """분포 + 누수 점검 리포트. 학습 전에 이 파일을 반드시 눈으로 볼 것."""
    lines = []
    n_by = Counter((r["split"], r["disease"], r["angle"]) for r in rows)
    lines.append("=== split × 질환 × 각도 (파일 수) ===")
    lines.append(f"{'split':6s} {'질환':6s} " + " ".join(f"{a:>7s}" for a in ANGLES) + f"{'합계':>8s}")
    for sp in SPLITS:
        for d in DISEASES:
            c = [n_by[(sp, d, a)] for a in ANGLES]
            lines.append(f"{sp:6s} {d:6s} " + " ".join(f"{v:7d}" for v in c) + f"{sum(c):8d}")

    lines.append("\n=== (질환, 케이스 키) 그룹 수 ===")
    grp = defaultdict(set)
    for r in rows:
        grp[r["split"]].add((r["disease"], r["subject"]))
    for sp in SPLITS:
        n_img = sum(1 for r in rows if r["split"] == sp)
        s = len(grp[sp])
        lines.append(f"{sp:6s}: 파일 {n_img:5d}  그룹 {s:5d}  평균 {n_img / max(s, 1):.2f} 장/그룹")

    lines.append("\n=== 그룹 누수 점검 (0 이어야 정상) ===")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = grp[a] & grp[b]
        lines.append(f"{a} ∩ {b}: {len(ov)}" + ("  <-- 누수!" if ov else "  OK"))

    lines.append("\n=== split × 출처(prefix) ===")
    for sp in SPLITS:
        c = Counter(r["source"] for r in rows if r["split"] == sp)
        lines.append(f"{sp:6s}: {dict(sorted(c.items(), key=lambda x: -x[1]))}")

    lines.append("\n=== 질환별 출처 구성 (test) — 출처가 질환을 알려주면 지름길 학습 위험 ===")
    for d in DISEASES:
        c = Counter(r["source"] for r in rows if r["split"] == "test" and r["disease"] == d)
        lines.append(f"{d:6s}: {dict(sorted(c.items(), key=lambda x: -x[1]))}")

    txt = "\n".join(lines)
    (Path(out) / "stats.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)


def main():
    args = parse_args()
    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"[에러] --src 없음: {src}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = scan(src)
    if not rows:
        raise SystemExit(f"[에러] 라벨 JSON 을 못 찾음: {src}/**/*.json")
    print(f"[scan] 라벨 {len(rows)}건")

    rows = split_train_val(rows, args.val_frac, args.seed)
    if not args.keep_test_leaks:
        rows = drop_test_leaks(rows)
    if args.labels_only:
        print("[image] --labels_only -> 이미지 배치 생략")
    else:
        rows = place_images(rows, out, args.link)
        if not rows:
            raise SystemExit("[에러] 배치된 이미지가 0건. 원천데이터 다운로드를 확인할 것.")

    csv_path = out / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "disease", "angle", "source", "subject"])
        for r in sorted(rows, key=lambda r: (SPLITS.index(r["split"]), r["disease"],
                                             r["angle"], r["stem"])):
            w.writerow([r["split"], r["stem"], r["disease"], r["angle"],
                        r["source"], r["subject"]])
    print(f"[out] {csv_path}  ({len(rows)}행)")
    write_stats(out, rows)
    print(f"\n[done] {out}  — 학습: bash run.sh")


if __name__ == "__main__":
    main()
