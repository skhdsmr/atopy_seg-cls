"""2-step IBB(Iterative Balanced Batch) 샘플러 — OCNN-IT 의 'IT' 부분.

`/home/work/Code/cls_sev/sampler.py` 를 이 폴더 규약(dataset.build_label_space 로 등급 수를
런타임에 정함)으로 옮긴 것이다. 알고리즘은 손대지 않았고, 모듈 전역 NUM_CLASSES 의존만
호출부 인자(num_classes)로 바꿨다 — 실데이터 IGA 가 1~4(4등급)라 등급 수가 데이터마다
달라지기 때문이다.

왜 이 실험이 필요한가: 이 저장소에서 태스크 의존성을 특징/라벨 층에서 다룬 실험
(CANet, CCNN pairwise, cls_mbn 의 CFEN)은 전부 baseline 을 못 넘겼는데, **데이터 층**
에서 다룬 IBB 샘플러만 4개 백본 전부에서 넘겼다(최고 .5630). 그래서 그룹분기·CA/SA·ASPP
(특징 층)로 번 것과 샘플러(데이터 층)로 번 것이 더해지는지를 따로 확인해야 한다.

Walecki+ 2017 "Deep Structured Learning for Facial AU Intensity Estimation" Alg.1 의
축소판이다. 논문 Alg.1 은 3-step(subject / AU-level / AU-cooccurrence)이지만:

  - subject 축 제거: 이 데이터는 환자 1명당 이미지가 1장 남짓이고 split 이 이미 ID
    단위로 분리돼 있어(train∩val=0) subject 균등 배치가 uniform 과 사실상 같다.
    대신 앞 --ibb_warmup 에폭을 자연분포로 돌려 백본 W 를 워밍업한다.
  - 남은 두 축으로 2-step:

  Step 1 (marginal) : 태스크 하나를 골라 그 태스크의 '등급'이 균등해지도록 뽑은 배치.
                      해당 태스크 head 의 loss 만 역전파(논문의 ∀q: φ^q 갱신에 대응).
  Step 2 (pairwise) : 상관이 강한 태스크 쌍(edge)을 골라 그 쌍의 '등급조합 셀'이
                      균등해지도록 뽑은 배치. 쌍에 속한 두 태스크의 loss 만 역전파.

두 스텝은 **배치 단위로 인터리브**된다. 블록으로 분리하면 BatchNorm running stats 와
AdamW 모멘텀이 스텝이 바뀔 때마다 흔들린다.

한계(정직하게): 이 모델에는 논문의 코퓰러 pairwise 파라미터 θ 가 없다(그건 CCNN).
따라서 Step 2 는 θ 전용 갱신이 아니라 '조인트 층화 샘플링'으로 동작한다 — 주변분포
지름길 학습(예: '대부분 Moderate 니까 다 Moderate')을 억제하는 정규화 효과를 노리는
것이지 논문 Step 3 그 자체는 아니다. 이 저장소가 구현하는 것은 OCNN(-IT) 까지다.

균등화는 '완전균등'이 아니라 alpha 지수 완화(기본 sqrt) + 반복배수 상한(cap)이다.
train 의 iga_grade=Clear 는 n=2 수준이라 완전균등이면 에폭당 수백 배 반복 -> 2장 암기가
된다.

UNKNOWN(-1) 라벨은 그 축에서만 반복배수 0 이 되어 빠진다 — 라벨이 있는 축에서는
그대로 쓰인다.

단독 실행하면 학습 없이 분포/엣지/반복배수 진단표만 출력한다:
    python3 sampler.py --data /home/work/ogw/dataset_all_final/images
"""
import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from torch.utils.data import Sampler

from dataset import TASK_NAMES


# ----------------------------------------------------------------------------- 라벨 배열화
def labels_from_dataset(ds):
    """SevDataset -> {task: np.ndarray(int64)}. UNKNOWN(-1) 포함."""
    return ds.label_arrays()


def active_tasks(Y):
    """라벨이 하나라도 있는 축만. 전부 UNKNOWN 인 축은 샘플러가 다룰 게 없다."""
    return [t for t in TASK_NAMES if t in Y and (Y[t] >= 0).any()]


# ----------------------------------------------------------------------------- 엣지 선별
def cramers_v(ya, yb, Ka, Kb):
    """두 순서형 변수의 연관 강도(0~1). 논문의 θ 가지치기에 대응하는 엣지 선별 기준.

    한쪽이라도 UNKNOWN 인 표본은 뺀다 — -1 을 등급으로 세면 없는 셀이 생겨 V 가 왜곡된다.
    """
    m = (ya >= 0) & (yb >= 0)
    ya, yb = ya[m], yb[m]
    n = len(ya)
    if n == 0:
        return 0.0
    obs = np.zeros((Ka, Kb), dtype=np.float64)
    np.add.at(obs, (ya, yb), 1.0)
    ra, rb = obs.sum(1, keepdims=True), obs.sum(0, keepdims=True)
    exp = ra @ rb / n
    nz = exp > 0
    chi2 = float(((obs[nz] - exp[nz]) ** 2 / exp[nz]).sum())
    k = min((ra > 0).sum(), (rb > 0).sum())
    return math.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0.0


def select_edges(Y, v_min, num_classes, tasks=None):
    """연관 강도가 v_min 이상인 태스크 쌍만 (내림차순). 반환: (채택 목록, 전체 점수)."""
    tasks = tasks or active_tasks(Y)
    scored = [(a, b, cramers_v(Y[a], Y[b], num_classes[a], num_classes[b]))
              for a, b in combinations(tasks, 2)]
    scored.sort(key=lambda e: -e[2])
    return [e for e in scored if e[2] >= v_min], scored


# ----------------------------------------------------------------------------- 반복배수 계산
def _cap_normalize(r, cap, iters=50):
    """샘플별 기대 반복배수 r 을 mean(r)=1, max(r)<=cap 이 되도록 조정(총합 보존).

    총합 보존이 핵심이다 — cap 에 걸려 잘린 몫을 버리면 에폭당 실효 표본 수가 줄어든다.
    잘린 만큼을 아직 여유 있는 샘플에 비례 재분배한다.
    """
    r = np.asarray(r, dtype=np.float64).copy()
    s = r.sum()
    if s <= 0:
        return r
    r *= len(r) / s
    for _ in range(iters):
        over = r > cap
        if not over.any():
            break
        excess = float((r[over] - cap).sum())
        r[over] = cap
        free = ~over & (r > 0)
        if not free.any():
            break
        r[free] += excess * r[free] / r[free].sum()
    return r


def level_repeat(y, K, alpha, cap):
    """Step 1: 등급 빈도의 alpha 승 역가중 -> 샘플별 기대 반복배수. UNKNOWN 은 0."""
    known = y >= 0
    cnt = np.bincount(y[known], minlength=K).astype(np.float64)
    w = np.where(cnt > 0, np.power(np.maximum(cnt, 1.0), -alpha), 0.0)
    r = np.zeros(len(y), dtype=np.float64)
    r[known] = w[y[known]]
    return _cap_normalize(r, cap)


def pair_repeat(ya, yb, Ka, Kb, alpha, cap, min_cell):
    """Step 2: 등급조합 셀 빈도의 alpha 승 역가중. min_cell 미만 셀은 제외(반복배수 0).

    최소 셀이 n=1 인 쌍이 있어 셀 완전균등은 최대 수십~백 배 반복이 된다 -> 희소 셀은
    Step 1 과 자연분포 에폭에만 맡기고 Step 2 에서는 뺀다.
    한쪽이라도 UNKNOWN 인 표본은 조합 자체가 정의되지 않으므로 제외한다.
    """
    known = (ya >= 0) & (yb >= 0)
    cell = np.zeros((Ka, Kb), dtype=np.float64)
    np.add.at(cell, (ya[known], yb[known]), 1.0)
    w = np.where(cell >= min_cell, np.power(np.maximum(cell, 1.0), -alpha), 0.0)
    r = np.zeros(len(ya), dtype=np.float64)
    r[known] = w[ya[known], yb[known]]
    return _cap_normalize(r, cap)


# ----------------------------------------------------------------------------- 스케줄
@dataclass
class Phase:
    name: str                 # 로그용 이름
    step: int                 # 1=등급균등, 2=쌍-공존균등
    tasks: tuple              # 이 배치에서 역전파할 태스크
    n_batches: int
    build: object = field(repr=False, default=None)   # alpha -> 반복배수 ndarray


def _interleave(a, b):
    """두 배치 목록을 비율에 맞춰 고르게 섞는다(Step1 블록/Step2 블록 분리 방지)."""
    na, nb = len(a), len(b)
    out, i, j = [], 0, 0
    while i < na or j < nb:
        if j >= nb or (i < na and (i + 1) * max(nb, 1) <= (j + 1) * max(na, 1)):
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out


def _roundrobin_order(slot_list):
    """[0,0,1,1,2,2] -> 0,1,2,0,1,2 순서가 되도록 하는 인덱스 순열."""
    buckets = {}
    for k, v in enumerate(slot_list):
        buckets.setdefault(v, []).append(k)
    order, keys = [], list(buckets)
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                order.append(buckets[k].pop(0))
    return order


class IBBSampler(Sampler):
    """배치 단위로 Step1/Step2 를 번갈아 내보내는 인덱스 샘플러.

    학습 루프가 에폭 시작에 plan() 을 불러 스케줄(배치별 (이름, 태스크))을 확정하고,
    DataLoader 는 그 다음 __iter__ 로 인덱스를 받아간다. 배치 i 에서 역전파할 태스크는
    batch_phases[i] 다.
    """

    def __init__(self, phases, batch_size, seed=0, alpha=0.5):
        self.phases = phases
        self.batch_size = batch_size
        self.seed = seed
        self.alpha = alpha
        self.epoch = 0
        self.batch_phases = []
        self._pending = None
        self._cache = (None, None)

    @property
    def n_batches(self):
        return sum(p.n_batches for p in self.phases)

    def __len__(self):
        return self.n_batches * self.batch_size

    def set_alpha(self, alpha):
        """에폭별 alpha 어닐링용. alpha->0 이면 자연분포로 복귀한다."""
        self.alpha = alpha

    def _probs(self):
        if self._cache[0] == self.alpha:
            return self._cache[1]
        probs = []
        for p in self.phases:
            r = p.build(self.alpha)
            s = r.sum()
            probs.append(r / s if s > 0 else np.full(len(r), 1.0 / len(r)))
        self._cache = (self.alpha, probs)
        return probs

    def plan(self):
        """다음 에폭의 인덱스와 배치별 (이름, 태스크) 스케줄을 **지금** 확정한다.

        학습 루프에서 iter(loader) 직후에 batch_phases 를 읽는 방식은 num_workers 에
        의존한다 — workers>0 이면 DataLoader 가 __init__ 에서 prefetch 하며 샘플러를
        소비하지만, workers=0 이면 첫 next() 전까지 소비하지 않아 첫 에폭의 스케줄이
        비어 있다. 그래서 소비 시점과 무관하게 루프가 먼저 plan() 을 부르고, __iter__
        는 그 결과를 그대로 내보낸다.
        """
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        probs = self._probs()
        # 페이즈별 배치 슬롯 -> 스텝 내 라운드로빈 -> 두 스텝 인터리브
        slots = {1: [], 2: []}
        for pi, p in enumerate(self.phases):
            slots[p.step].extend([pi] * p.n_batches)
        s1 = [slots[1][k] for k in _roundrobin_order(slots[1])]
        s2 = [slots[2][k] for k in _roundrobin_order(slots[2])]
        order = _interleave(s1, s2)

        idx, bph = [], []
        for pi in order:
            p, pr = self.phases[pi], probs[pi]
            idx.extend(rng.choice(len(pr), size=self.batch_size, replace=True, p=pr).tolist())
            bph.append((p.name, p.tasks))
        self.batch_phases = bph
        self._pending = idx
        return bph

    def __iter__(self):
        if self._pending is None:
            self.plan()
        idx, self._pending = self._pending, None
        return iter(idx)


# ----------------------------------------------------------------------------- 빌더
def build_ibb_sampler(Y, num_classes, batch_size, *, alpha=0.5, cap=4.0, min_cell=5, edge_v=0.25,
                      step2_ratio=0.5, step1_mult=1.0, edge_loss="pair", seed=0,
                      tasks=None, verbose=True):
    """라벨 배열 dict -> (IBBSampler, info dict).

    step1_mult: Step1 은 배치를 태스크 5개가 나눠 쓰므로 mult=1 이면 head 하나당
    업데이트 수가 자연분포 에폭의 1/5 이 된다(공유 trunk 는 매 배치 갱신되므로 영향 적음).
    head 수렴이 느리면 5.0 으로 올려 태스크마다 온전한 1 에폭치를 준다(에폭 비용도 5배).
    """
    tasks = tasks or active_tasks(Y)
    if not tasks:
        raise ValueError("IBB: 라벨이 있는 축이 하나도 없다.")
    n = len(Y[tasks[0]])
    steps1 = max(1, round(math.ceil(n / batch_size) * step1_mult))
    steps2 = max(1, round(math.ceil(n / batch_size) * step2_ratio))
    edges, scored = select_edges(Y, edge_v, num_classes, tasks)
    if not edges:                                    # 엣지가 하나도 안 남으면 Step 1 만
        steps2 = 0

    phases = []
    per_task = max(1, steps1 // len(tasks))
    for t in tasks:
        y, K = Y[t], num_classes[t]
        phases.append(Phase(f"lv:{t}", 1, (t,), per_task,
                            build=lambda a, y=y, K=K: level_repeat(y, K, a, cap)))
    if steps2:
        per_edge = max(1, steps2 // len(edges))
        for a, b, v in edges:
            ya, yb, Ka, Kb = Y[a], Y[b], num_classes[a], num_classes[b]
            pair = tuple(tasks) if edge_loss == "all" else (a, b)
            phases.append(Phase(f"co:{a[:4]}-{b[:4]}", 2, pair, per_edge,
                                build=lambda al, ya=ya, yb=yb, Ka=Ka, Kb=Kb:
                                pair_repeat(ya, yb, Ka, Kb, al, cap, min_cell)))

    sampler = IBBSampler(phases, batch_size, seed=seed, alpha=alpha)
    info = {"edges": edges, "scored": scored, "n_batches": sampler.n_batches,
            "steps1": per_task * len(tasks),
            "steps2": sampler.n_batches - per_task * len(tasks),
            "natural": math.ceil(n / batch_size)}
    if verbose:
        print(f"[ibb] alpha={alpha} cap={cap}x min_cell={min_cell} edge_V>={edge_v} "
              f"edge_loss={edge_loss}")
        if edges:
            print(f"[ibb] 엣지 {len(edges)}개: "
                  + ", ".join(f"{a[:4]}-{b[:4]}(V={v:.2f})" for a, b, v in edges))
        else:
            print(f"[ibb] 엣지 없음(V>={edge_v} 인 쌍이 하나도 없다) -> Step1 만 돈다. "
                  f"--ibb_edge_v 를 낮추거나 그대로 두고 marginal 균등만 쓸 것.")
        print(f"[ibb] 에폭당 배치 {info['n_batches']} (step1_mult={step1_mult}) "
              f"(Step1 {info['steps1']} + Step2 {info['steps2']}, "
              f"자연분포 에폭={info['natural']})")
    return sampler, info


# ----------------------------------------------------------------------------- 진단 리포트
def report(data, labels_csv, split, alpha, cap, min_cell, edge_v):
    """학습 없이 분포/엣지/반복배수만 출력. IBB 노브를 고르기 전에 한 번 볼 것."""
    from pathlib import Path

    from dataset import IMG_EXTS, build_label_space

    data = Path(data)
    index, num_classes, levels = build_label_space(labels_csv or (data / "labels.csv"))
    if split != "all":
        stems = {p.stem for p in (data / split).iterdir() if p.suffix.lower() in IMG_EXTS}
        index = {k: v for k, v in index.items() if k in stems}
    Y = {t: np.array([r[t] for r in index.values()], dtype=np.int64) for t in TASK_NAMES}
    tasks = active_tasks(Y)
    n = len(index)
    print(f"# split={split}  N={n}\n")

    print("## 축1 등급 분포 / 기대 반복배수 (alpha=%.2f, cap=%.1fx)" % (alpha, cap))
    for t in tasks:
        known = Y[t] >= 0
        cnt = np.bincount(Y[t][known], minlength=num_classes[t])
        nk = int(known.sum())
        r = level_repeat(Y[t], num_classes[t], alpha, cap)
        print(f"  {t}  (라벨있음 {nk}/{n})")
        for i, name in enumerate(levels[t]):
            if cnt[i] == 0:
                print(f"    {name:13s} n=   0  (없음)")
                continue
            rep = r[Y[t] == i][0]
            print(f"    {name:13s} n={cnt[i]:4d} ({100 * cnt[i] / max(nk, 1):4.1f}%)  "
                  f"균등={nk / num_classes[t] / cnt[i]:6.1f}x -> 적용={rep:5.2f}x")

    print("\n## 축2 엣지 연관강도(Cramer's V) / 셀 커버리지")
    edges, scored = select_edges(Y, edge_v, num_classes, tasks)
    keep = {(a, b) for a, b, _ in edges}
    for a, b, v in scored:
        m = (Y[a] >= 0) & (Y[b] >= 0)
        cell = np.zeros((num_classes[a], num_classes[b]))
        np.add.at(cell, (Y[a][m], Y[b][m]), 1.0)
        used = int((cell >= min_cell).sum())
        mark = "채택" if (a, b) in keep else "제외"
        print(f"  [{mark}] {a[:6]:6s}-{b[:6]:6s} V={v:.3f}  "
              f"셀 {cell.size:2d} / 비어있지않음 {int((cell > 0).sum()):2d} / "
              f">={min_cell} {used:2d}  최대셀={int(cell.max()):4d}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IBB 샘플러 진단 리포트(학습 없음)")
    ap.add_argument("--data", default="../dataset_all_final/images")
    ap.add_argument("--labels_csv", default=None)
    ap.add_argument("--split", default="train", help="train/val/test/all")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=4.0)
    ap.add_argument("--min_cell", type=int, default=5)
    ap.add_argument("--edge_v", type=float, default=0.25)
    a = ap.parse_args()
    report(a.data, a.labels_csv, a.split, a.alpha, a.cap, a.min_cell, a.edge_v)
