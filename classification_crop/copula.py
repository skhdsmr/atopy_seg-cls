"""Frank 코퓰러 pairwise 항 — 저자 구현(RWalecki/copula_ordinal_regression) 대조 포팅.

대조 파일: copulas.py(frank), statistics.py(log_prob/node_potn/edge_potn),
           BASE.py(_cdf/_pdf), COR.py(_loss/predict/_init_para)

저자 구현에서 그대로 가져온 것
  - theta 파라미터화 : theta = (sigmoid(raw) - 0.5) * CUT,  CUT=25 -> theta in (-12.5, +12.5)
                       raw 초기값 0.01 (theta=0 은 독립 특이점이라 저자도 0 을 피한다)
  - 노드 포텐셜      : -log P(y^q = l)                                   (statistics.node_potn)
  - 엣지 포텐셜      : -log[ C(u1,v1) + C(u0,v0) - C(u0,v1) - C(u1,v0) ] (statistics.edge_potn)
  - CDF 레이아웃     : pdf 를 누적하고 앞에 0 을 붙인 K+1 개. cdf[l+1]=F(l), cdf[l]=F(l-1)
  - log_prob 안정화  : realmin=1e-20 로 [realmin, 1-realmin] 클리핑 + NaN/inf -> realmin
  - 손실 결합        : w_nodes * mean(노드) + (1-w_nodes) * mean(엣지)    (COR._loss)
                       저자 기본 w_nodes=0.1 (즉 pairwise 에 0.9). 저자도 튜닝 대상으로 둔다.
  - 추론 가중        : 노드/엣지에 같은 w_nodes 가중을 적용해 MAP        (COR.predict)
  - shared_copula    : True=엣지당 theta 스칼라 1개, False=엣지당 (L,L) 행렬 (COR._init_para)

  - 다중출처 theta  : 논문 Eq(12) "marginal 은 공유, pairwise 는 데이터셋별로 따로".
                       CopulaPairwise(n_sources=S) 면 엣지당 theta 를 S 벌 두고 표본의
                       출처(_src)로 골라 쓴다. unary(CNN+head)는 그대로 공유된다.
                       근거(합성 1800장 train, Cramer's V): 구진-태선화가 정면 0.120 /
                       측면 0.274 인데 합치면 0.180 으로 희석되고, severi-lichen 은
                       정면 0.145 / 측면 0.142 인데 합치면 0.071 로 **사라진다**.
                       논문이 AU17 에서 관찰한 맥락 의존성과 같은 현상이다.

이 파일에서 저자와 다르게 한 것(의도적, 이유 명시)
  1) frank() 를 수치적으로 안정한 동치 형태로 다시 씀. 저자 식은 |theta| 가 CUT/2 에 가까우면
     1 + U*V/D 가 통째로 0 으로 반올림돼 log(0) 이 된다(저자는 C<=0 을 0 으로 눌러 log_prob 의
     realmin 이 받아내게 두지만, 그 구간의 값이 실제로 뭉개진다). frank_ref() 로 원식을 남겨
     두고 테스트에서 동치를 확인한다.
  2) 주변 CDF 를 저자의 선형 threshold 모델(BASE._cdf) 대신 CNN 의 CORN head 에서 만든다.
     CORN 은 sigmoid 누적곱이라 F 가 단조증가로 구조적으로 보장돼 순서형 제약을 그대로 만족한다.
     -> 논문 3.1절 ordinal unary 를 CORN 이 담당. (CNN 연결부는 이 저장소 고유)
  3) 결합 추론을 AD3 근사(저자 pystruct.inference_ad3) 대신 전수 열거로 한다.
     논문/저자는 AU 10+ 라 2^10 평가가 필요하지만 여기는 5*4*4*4*4=1280 조합뿐이라
     정확한 MAP 을 낼 수 있다.
"""
import torch
import torch.nn as nn

CUT = 25.0             # 저자 copulas.frank 의 cut. theta 범위 = (-CUT/2, +CUT/2)
RAW_INIT = 0.01        # 저자 COR._init_para 의 theta 초기값
REALMIN = 1e-20        # 저자 statistics.log_prob 의 realmin


# ----------------------------------------------------------------- 저자 statistics.log_prob
def log_prob(P, realmin=REALMIN):
    """저자 log_prob 포팅: [realmin, 1-realmin] 클리핑 + NaN/inf -> realmin."""
    P = torch.nan_to_num(P, nan=realmin, posinf=1.0, neginf=realmin)
    return torch.log(P.clamp(realmin, 1.0 - realmin))


# ----------------------------------------------------------------- 주변 CDF (CNN 연결부)
def cdf_from_corn(logits):
    """CORN 로짓 (B,K-1) -> 주변 CDF (B,K+1) = [0, F(0), ..., F(K-2), 1].

    저자 edge_potn 의 `cumsum(pdf) 앞에 0 붙이기`와 같은 레이아웃:
    레벨 l 에 대해 상한 = cdf[:, l+1] = F(l),  하한 = cdf[:, l] = F(l-1).
    """
    cum = torch.cumprod(torch.sigmoid(logits.float()), dim=1)      # P(y>0..K-2)
    F = 1.0 - cum
    zero = torch.zeros_like(F[:, :1])
    one = torch.ones_like(F[:, :1])
    return torch.cat([zero, F, one], dim=1)


def cdf_to_pdf(cdf):
    """(B,K+1) -> (B,K) 레벨별 확률."""
    return cdf[:, 1:] - cdf[:, :-1]


# ----------------------------------------------------------------- 저자 copulas.frank
def theta_from_raw(raw, cut=CUT):
    """저자 frank() 의 d 변환: theta = (sigmoid(raw) - 0.5) * cut."""
    return (torch.sigmoid(raw) - 0.5) * cut


def frank_ref(u, v, th):
    """저자 원식 그대로(식 9). 동치 확인용 — 큰 |th| 에서는 상쇄로 정확도를 잃는다."""
    U = torch.exp(-th * u) - 1.0
    V = torch.exp(-th * v) - 1.0
    D = torch.exp(-th) - 1.0
    C = 1.0 + U * V / D
    C = torch.where(C <= 0, torch.zeros_like(C), C)
    return -(1.0 / th) * torch.log(C)


def _frank_pos(u, v, th):
    """th > 0 전용. 저자 식을 전개해 상쇄가 없는 형태로:
         1 + U*V/D = num/den,
         num = e^{-th*u}(1-e^{-th*v}) + e^{-th*v}(1-e^{-th(1-v)}),  den = 1-e^{-th}
       두 항 모두 th>0, u,v in [0,1] 에서 음이 아니라 log 가 안전하다.
    """
    num = torch.exp(-th * u) * (-torch.expm1(-th * v)) \
        + torch.exp(-th * v) * (-torch.expm1(-th * (1.0 - v)))
    den = -torch.expm1(-th)
    return -(torch.log(num.clamp_min(REALMIN)) - torch.log(den.clamp_min(REALMIN))) / th


def frank(u, v, th, eps=1e-6):
    """Frank 코퓰러. frank_ref 와 동치이되 |th| 전 구간에서 안정."""
    ath = th.abs().clamp_min(eps)
    pos = _frank_pos(u, v, ath)
    neg = u - _frank_pos(u, 1.0 - v, ath)        # C_{-th}(u,v) = u - C_th(u, 1-v)
    c = torch.where(th >= 0, pos, neg)
    return torch.where(th.abs() < eps, u * v, c)  # th->0 은 독립


def indep(u, v, th=None):
    """저자 copulas.indep."""
    return u * v


COPULAS = {"frank": frank, "indep": indep}


# ----------------------------------------------------------------- 저자 statistics.*_potn
def node_potn(cdf, y=None):
    """저자 node_potn. y 주면 (B,) 그 라벨의 -log P, 안 주면 (B,K) 전체."""
    pdf = cdf_to_pdf(cdf)
    if y is None:
        return -log_prob(pdf)
    return -log_prob(pdf.gather(1, y.view(-1, 1)).squeeze(1))


def edge_potn(cdf_r, cdf_s, th, y_r=None, y_s=None, copula=frank):
    """저자 edge_potn. y 주면 (B,) 그 조합의 -log P, 안 주면 (B,Kr,Ks) 전체.

    조합식은 저자와 동일: P = C(u1,v1) + C(u0,v0) - C(u0,v1) - C(u1,v0).
    th 가 (Kr,Ks) 행렬이면 저자 shared_copula=False (셀마다 theta) 에 해당한다.
    """
    # th 의 차원으로 종류를 가른다: 0=스칼라(엣지 공통), 1=(B,) 표본별(다중출처),
    # 2=(Kr,Ks) 셀별(저자 shared_copula=False). 세 경우가 섞일 일은 없다.
    if y_r is not None:
        u0 = cdf_r.gather(1, y_r.view(-1, 1)).squeeze(1)          # F(l_r - 1)
        u1 = cdf_r.gather(1, (y_r + 1).view(-1, 1)).squeeze(1)    # F(l_r)
        v0 = cdf_s.gather(1, y_s.view(-1, 1)).squeeze(1)
        v1 = cdf_s.gather(1, (y_s + 1).view(-1, 1)).squeeze(1)
        d = th if th.ndim <= 1 else th[y_r, y_s]
        P = copula(u1, v1, d) + copula(u0, v0, d) - copula(u0, v1, d) - copula(u1, v0, d)
        return -log_prob(P)
    u = cdf_r.unsqueeze(2)                                        # (B,Kr+1,1)
    v = cdf_s.unsqueeze(1)                                        # (B,1,Ks+1)
    if th.ndim == 1:                                              # 표본별 theta (B,)
        d = th.view(-1, 1, 1)
        j = copula(u, v, d)
        return -log_prob(j[:, 1:, 1:] + j[:, :-1, :-1] - j[:, :-1, 1:] - j[:, 1:, :-1])
    d = th if th.ndim == 0 else th.unsqueeze(0)
    if th.ndim == 0:
        j = copula(u, v, d)
        P = j[:, 1:, 1:] + j[:, :-1, :-1] - j[:, :-1, 1:] - j[:, 1:, :-1]
    else:                                                          # 셀별 theta
        P = (copula(u[:, 1:], v[:, :, 1:], d) + copula(u[:, :-1], v[:, :, :-1], d)
             - copula(u[:, :-1], v[:, :, 1:], d) - copula(u[:, 1:], v[:, :, :-1], d))
    return -log_prob(P)


# ----------------------------------------------------------------- 모듈
class CopulaPairwise(nn.Module):
    """엣지별 코퓰러 파라미터. 저자 COR 의 theta 에 해당.

    shared=True  : 엣지당 스칼라 1개              (저자 shared_copula=True, 기본)
    shared=False : 엣지당 (Kr,Ks) 행렬            (저자 shared_copula=False)
    """

    def __init__(self, edges, task_names, num_classes, shared=True, cut=CUT, copula="frank",
                 n_sources=1, source_names=None):
        super().__init__()
        self.edges = [(a, b) for a, b, *_ in edges]
        self.task_names = list(task_names)
        self.cut = float(cut)
        self.shared = bool(shared)
        self.copula = COPULAS[copula]
        self.n_sources = int(n_sources)
        self.source_names = list(source_names or [str(i) for i in range(self.n_sources)])
        if self.n_sources > 1 and not shared:
            raise ValueError("다중출처 theta 는 shared=True 에서만 지원한다"
                             "(셀별 theta x 출처는 표본당 파라미터가 과해진다).")
        if shared:
            self.raw = nn.ParameterList(
                [nn.Parameter(torch.full((self.n_sources,), RAW_INIT)) for _ in self.edges])
        else:
            self.raw = nn.ParameterList(
                [nn.Parameter(torch.full((num_classes[a], num_classes[b]), RAW_INIT))
                 for a, b in self.edges])

    def theta(self, i):
        """엣지 i 의 theta. shared 면 (S,), 셀별이면 (Kr,Ks)."""
        return theta_from_raw(self.raw[i], self.cut)

    def theta_for(self, i, src=None):
        """표본별 theta. src=(B,) 출처 인덱스. 단일출처면 기존과 같은 스칼라를 돌려준다."""
        th = self.theta(i)
        if not self.shared:
            return th                                   # (Kr,Ks) 셀별
        if src is None:
            return th[0] if self.n_sources == 1 else th.mean()
        return th[src]                                  # (B,)

    def thetas(self, src_id=None):
        """로그용 대표값. src_id 를 주면 그 출처의 값만."""
        out = []
        for i in range(len(self.edges)):
            th = self.theta(i)
            if self.shared:
                out.append(float(th[src_id]) if src_id is not None else float(th.mean()))
            else:
                out.append(float(th.mean()))
        return out

    def edge_index(self, a, b):
        try:
            return self.edges.index((a, b))
        except ValueError:
            return self.edges.index((b, a))

    def edge_loss(self, cdfs, labels, edge_ids=None, detach_marginals=False, src=None):
        """저자 COR._loss 의 NCJLL 항: 엣지별 -log P(y^r,y^s) 의 평균."""
        if detach_marginals:                      # IBB Step3: theta 만 갱신
            cdfs = {t: c.detach() for t, c in cdfs.items()}
        ids = range(len(self.edges)) if edge_ids is None else edge_ids
        terms = []
        for i in ids:
            a, b = self.edges[i]
            terms.append(edge_potn(cdfs[a], cdfs[b], self.theta_for(i, src),
                                   labels[a], labels[b], self.copula).mean())
        return torch.stack(terms).mean()


def node_loss(cdfs, labels, task_names=None):
    """저자 COR._loss 의 NCLL 항: 태스크별 -log P(y^q=l) 평균 dict."""
    ts = task_names or list(cdfs)
    return {t: node_potn(cdfs[t], labels[t]).mean() for t in ts}


def joint_decode(cdfs, pw, num_classes, task_names, w_nodes=0.1, src=None, _cache={}):
    """전수 열거 MAP. 저자 COR.predict 와 동일하게 노드/엣지에 w_nodes 가중을 적용한다.

    저자는 inference_ad3(-node*w, -edge*(1-w)) 로 근사 추론하지만, 여기는 조합이
    1280 개뿐이라 전수 열거로 정확한 MAP 을 낸다.
    """
    key = tuple(task_names)
    if key not in _cache:
        _cache[key] = torch.cartesian_prod(*[torch.arange(num_classes[t]) for t in task_names])
    dev = cdfs[task_names[0]].device
    grid = _cache[key].to(dev)                                   # (M,Q)
    score = 0.0
    for qi, t in enumerate(task_names):
        score = score + w_nodes * (-node_potn(cdfs[t]))[:, grid[:, qi]]
    if pw is not None:
        we = 1.0 - w_nodes
        for i, (a, b) in enumerate(pw.edges):
            ep = -edge_potn(cdfs[a], cdfs[b], pw.theta_for(i, src), copula=pw.copula)  # (B,Ka,Kb)
            score = score + we * ep[:, grid[:, task_names.index(a)],
                                    grid[:, task_names.index(b)]]
    best = score.argmax(dim=1)
    return {t: grid[best, qi] for qi, t in enumerate(task_names)}
