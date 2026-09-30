"""2-step IBB(Iterative Balanced Batch) 샘플러 — Walecki+ 2017 Alg.1 의 축소판(B안).

논문 Alg.1 은 3-step(subject / AU-level / AU-cooccurrence)이지만, 이 데이터는
환자 1명당 이미지가 1.06장(고유 ID 1691 / 이미지 1800)이고 split 이 이미 ID 단위로
분리돼 있어 subject 축 배치는 uniform 샘플링과 사실상 동일하다 -> Step 1 제거.
남은 두 축으로 2-step 을 구성한다:

  Step 1 (marginal) : 태스크 하나를 골라 그 태스크의 '등급'이 균등해지도록 뽑은 배치.
                      해당 태스크 head 의 loss 만 역전파(논문의 forall q: phi^q 갱신에 대응).
  Step 2 (pairwise) : 상관이 강한 태스크 쌍(edge)을 골라 그 쌍의 '등급조합 셀'이
                      균등해지도록 뽑은 배치. 쌍에 속한 두 태스크의 loss 만 역전파.

주의(B안의 한계): 이 모델에는 논문의 코퓰러 pairwise 파라미터 theta 가 없다. 따라서
Step 2 는 theta 전용 갱신이 아니라 '조인트 층화 샘플링'으로 동작한다 — 주변분포
지름길 학습을 억제하는 정규화 효과를 노리는 것이지, 논문의 Step 3 그 자체는 아니다.

균등화는 '완전균등'이 아니라 alpha 지수 완화(기본 sqrt) + 반복배수 상한(cap)이다.
train 의 severity=Clear 는 n=2 라 완전균등 시 에폭당 140배 반복 -> 2장 암기가 되므로.

단독 실행하면 학습 없이 분포/엣지/반복배수 진단표만 출력한다:
    python3 sampler.py --data ../dataset_all_final/images/labels.csv
"""
import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from torch.utils.data import Sampler

from dataset import TASK_NAMES, NUM_CLASSES


# ----------------------------------------------------------------------------- 라벨 배열화
def labels_to_arrays(label_dicts):
    """[{task: idx}, ...] -> {task: np.ndarray(int)}."""
    return {t: np.asarray([d[t] for d in label_dicts], dtype=np.int64) for t in TASK_NAMES}


def labels_from_dataset(ds):
    """crop 데이터셋(samples=(img, mask, labels))에서 라벨 배열을 뽑는다."""
    return labels_to_arrays([s[-1] for s in ds.samples])


# ----------------------------------------------------------------------------- 엣지 선별
def cramers_v(ya, yb, Ka, Kb):
    """두 순서형 변수의 연관 강도(0~1). 논문의 theta 가지치기에 대응하는 엣지 선별 기준."""
    n = len(ya)
    obs = np.zeros((Ka, Kb), dtype=np.float64)
    np.add.at(obs, (ya, yb), 1.0)
    ra, rb = obs.sum(1, keepdims=True), obs.sum(0, keepdims=True)
    exp = ra @ rb / n
    nz = exp > 0
    chi2 = float(((obs[nz] - exp[nz]) ** 2 / exp[nz]).sum())
    k = min((ra > 0).sum(), (rb > 0).sum())
    return math.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0.0


def select_edges(Y, v_min):
    """연관 강도가 v_min 이상인 태스크 쌍만 (내림차순)."""
    scored = [(a, b, cramers_v(Y[a], Y[b], NUM_CLASSES[a], NUM_CLASSES[b]))
              for a, b in combinations(TASK_NAMES, 2)]
    scored.sort(key=lambda e: -e[2])
    return [e for e in scored if e[2] >= v_min], scored


# ----------------------------------------------------------------------------- 반복배수 계산
def _cap_normalize(r, cap, iters=50):
    """샘플별 기대 반복배수 r 을 mean(r)=1, max(r)<=cap 이 되도록 조정(총합 보존)."""
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
    """Step 1: 등급 빈도의 alpha 승 역가중 -> 샘플별 기대 반복배수."""
    cnt = np.bincount(y, minlength=K).astype(np.float64)
    w = np.where(cnt > 0, np.power(np.maximum(cnt, 1.0), -alpha), 0.0)
    return _cap_normalize(w[y], cap)


def pair_repeat(ya, yb, Ka, Kb, alpha, cap, min_cell):
    """Step 2: 등급조합 셀 빈도의 alpha 승 역가중. min_cell 미만 셀은 제외(반복배수 0).

    최소 셀이 n=1 인 쌍이 있어 셀 완전균등은 최대 100배 반복이 된다 -> 희소 셀은
    Step 1 과 자연분포 에폭에만 맡기고 Step 2 에서는 뺀다.
    """
    cell = np.zeros((Ka, Kb), dtype=np.float64)
    np.add.at(cell, (ya, yb), 1.0)
    w = np.where(cell >= min_cell, np.power(np.maximum(cell, 1.0), -alpha), 0.0)
    return _cap_normalize(w[ya, yb], cap)


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


class IBBSampler(Sampler):
    """배치 단위로 Step1/Step2 를 번갈아 내보내는 인덱스 샘플러.

    DataLoader 는 에폭마다 iter(sampler) 를 부르고, 그 시점에 배치별 (이름, 태스크)
    스케줄이 self.batch_phases 에 채워진다. 학습 루프는 배치 순번으로 이를 조회한다.
    """

    def __init__(self, phases, batch_size, seed=0, alpha=0.5):
        self.phases = phases
        self.batch_size = batch_size
        self.seed = seed
        self.alpha = alpha
        self.epoch = 0
        self.batch_phases = []
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

    def __iter__(self):
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
        return iter(idx)


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


# ----------------------------------------------------------------------------- 빌더
def build_ibb_sampler(Y, batch_size, *, alpha=0.5, cap=4.0, min_cell=5, edge_v=0.25,
                      step2_ratio=0.5, step1_mult=1.0, edge_loss="pair", seed=0, verbose=True):
    """라벨 배열 dict -> (IBBSampler, info dict).

    step1_mult: Step1 은 배치를 태스크 5개가 나눠 쓰므로 mult=1 이면 head 하나당
    업데이트 수가 자연분포 에폭의 1/5 이 된다(공유 trunk 는 매 배치 갱신되므로 영향 적음).
    head 수렴이 느리면 5.0 으로 올려 태스크마다 온전한 1 에폭치를 준다(에폭 비용도 5배).
    """
    n = len(next(iter(Y.values())))
    steps1 = max(1, round(math.ceil(n / batch_size) * step1_mult))
    steps2 = max(1, round(math.ceil(n / batch_size) * step2_ratio))
    edges, scored = select_edges(Y, edge_v)
    if not edges:                                    # 엣지가 하나도 안 남으면 Step 1 만
        steps2 = 0

    phases = []
    per_task = max(1, steps1 // len(TASK_NAMES))
    for t in TASK_NAMES:
        y, K = Y[t], NUM_CLASSES[t]
        phases.append(Phase(f"lv:{t}", 1, (t,), per_task,
                            build=lambda a, y=y, K=K: level_repeat(y, K, a, cap)))
    if steps2:
        per_edge = max(1, steps2 // len(edges))
        for a, b, v in edges:
            ya, yb, Ka, Kb = Y[a], Y[b], NUM_CLASSES[a], NUM_CLASSES[b]
            tasks = tuple(TASK_NAMES) if edge_loss == "all" else (a, b)
            phases.append(Phase(f"co:{a[:4]}-{b[:4]}", 2, tasks, per_edge,
                                build=lambda al, ya=ya, yb=yb, Ka=Ka, Kb=Kb:
                                pair_repeat(ya, yb, Ka, Kb, al, cap, min_cell)))

    sampler = IBBSampler(phases, batch_size, seed=seed, alpha=alpha)
    info = {"edges": edges, "scored": scored, "n_batches": sampler.n_batches,
            "steps1": per_task * len(TASK_NAMES),
            "steps2": sampler.n_batches - per_task * len(TASK_NAMES)}
    if verbose:
        print(f"[ibb] alpha={alpha} cap={cap}x min_cell={min_cell} edge_V>={edge_v} "
              f"edge_loss={edge_loss}")
        print(f"[ibb] 엣지 {len(edges)}개: "
              + ", ".join(f"{a[:4]}-{b[:4]}(V={v:.2f})" for a, b, v in edges))
        print(f"[ibb] 에폭당 배치 {info['n_batches']} (step1_mult={step1_mult}) "
              f"(Step1 {info['steps1']} + Step2 {info['steps2']}, 자연분포 에폭={steps1})")
    return sampler, info


# ----------------------------------------------------------------------------- 진단 리포트
def _report(csv_path, split, alpha, cap, min_cell, edge_v):
    import csv as _csv
    from dataset import TASKS, _LABEL2IDX
    rows = [r for r in _csv.DictReader(open(csv_path, encoding="utf-8"))
            if split in ("all", r.get("split"))]
    dicts = []
    for r in rows:
        try:
            dicts.append({t: _LABEL2IDX[t][r[t]] for t in TASK_NAMES})
        except KeyError:
            continue
    Y = labels_to_arrays(dicts)
    n = len(dicts)
    print(f"# {csv_path}  split={split}  N={n}\n")

    print("## 축1 등급 분포 / 기대 반복배수 (alpha=%.2f, cap=%.1fx)" % (alpha, cap))
    for t in TASK_NAMES:
        cnt = np.bincount(Y[t], minlength=NUM_CLASSES[t])
        r = level_repeat(Y[t], NUM_CLASSES[t], alpha, cap)
        print(f"  {t}")
        for i, name in enumerate(TASKS[t]):
            if cnt[i] == 0:
                print(f"    {name:13s} n=   0  (없음)"); continue
            rep = r[Y[t] == i][0]
            print(f"    {name:13s} n={cnt[i]:4d} ({100*cnt[i]/n:4.1f}%)  "
                  f"균등={n/NUM_CLASSES[t]/cnt[i]:6.1f}x -> 적용={rep:5.2f}x")

    print("\n## 축2 엣지 연관강도(Cramer's V) / 셀 커버리지")
    edges, scored = select_edges(Y, edge_v)
    keep = {(a, b) for a, b, _ in edges}
    for a, b, v in scored:
        cell = np.zeros((NUM_CLASSES[a], NUM_CLASSES[b]))
        np.add.at(cell, (Y[a], Y[b]), 1.0)
        used = int((cell >= min_cell).sum())
        mark = "채택" if (a, b) in keep else "제외"
        print(f"  [{mark}] {a[:6]:6s}-{b[:6]:6s} V={v:.3f}  "
              f"셀 {cell.size:2d} / 비어있지않음 {int((cell>0).sum()):2d} / "
              f">={min_cell} {used:2d}  최대셀={int(cell.max()):4d}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IBB 샘플러 진단 리포트(학습 없음)")
    ap.add_argument("--data", default="../dataset_all_final/images/labels.csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=4.0)
    ap.add_argument("--min_cell", type=int, default=5)
    ap.add_argument("--edge_v", type=float, default=0.25)
    a = ap.parse_args()
    _report(a.data, a.split, a.alpha, a.cap, a.min_cell, a.edge_v)
