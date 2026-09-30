// ============================================================
// 어파인 변환 추정 + RANSAC (순수 JS, 외부 의존성 없음)
//
// react-native-fast-opencv 에는 estimateAffine2D/estimateAffinePartial2D 가
// 없으므로(제공 함수는 findHomographyFromMatches 뿐) 직접 구현한다.
// 직접 구현하면 인라이어 마스크·재투영 오차를 그대로 손에 쥘 수 있어
// 정합 신뢰도 4축 계산에 오히려 유리하다.
//
// 표기: A = [a,b,c, d,e,f] (2x3, row-major)
//        x' = a*x + b*y + c
//        y' = d*x + e*y + f
//
// 이 파일은 네이티브 의존성이 없으므로 node 로 단독 테스트할 수 있다.
// ============================================================

export const IDENTITY = [1, 0, 0, 0, 1, 0];

export function applyAffine(A, x, y) {
  return [A[0] * x + A[1] * y + A[2], A[3] * x + A[4] * y + A[5]];
}

export function invertAffine(A) {
  const det = A[0] * A[4] - A[1] * A[3];
  if (!isFinite(det) || Math.abs(det) < 1e-12) return null;
  const ia = A[4] / det, ib = -A[1] / det;
  const id = -A[3] / det, ie = A[0] / det;
  return [ia, ib, -(ia * A[2] + ib * A[5]), id, ie, -(id * A[2] + ie * A[5])];
}

// B∘A (먼저 A 를 적용하고 그 결과에 B 를 적용)
export function composeAffine(B, A) {
  return [
    B[0] * A[0] + B[1] * A[3], B[0] * A[1] + B[1] * A[4], B[0] * A[2] + B[1] * A[5] + B[2],
    B[3] * A[0] + B[4] * A[3], B[3] * A[1] + B[4] * A[4], B[3] * A[2] + B[4] * A[5] + B[5],
  ];
}

// 균등 스케일 + 평행이동 (크롭/리사이즈 보정용)
export function scaleTranslate(s, tx, ty) {
  return [s, 0, tx, 0, s, ty];
}

// ------------------------------------------------------------
// 3x3 선형계 풀이 (Cramer). m 은 row-major 9개.
// ------------------------------------------------------------
function solve3x3(m, r) {
  const det =
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (!isFinite(det) || Math.abs(det) < 1e-10) return null;

  const d0 =
    r[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (r[1] * m[8] - m[5] * r[2]) +
    m[2] * (r[1] * m[7] - m[4] * r[2]);
  const d1 =
    m[0] * (r[1] * m[8] - m[5] * r[2]) -
    r[0] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * r[2] - r[1] * m[6]);
  const d2 =
    m[0] * (m[4] * r[2] - r[1] * m[7]) -
    m[1] * (m[3] * r[2] - r[1] * m[6]) +
    r[0] * (m[3] * m[7] - m[4] * m[6]);

  return [d0 / det, d1 / det, d2 / det];
}

// ------------------------------------------------------------
// 대응점 3쌍으로 어파인 정확해 (RANSAC 최소 표본)
//   src/dst: [{x,y}, ...],  i0..i2: 표본 인덱스
// ------------------------------------------------------------
export function affineFrom3(src, dst, i0, i1, i2) {
  const p = [src[i0], src[i1], src[i2]];
  const q = [dst[i0], dst[i1], dst[i2]];
  const m = [p[0].x, p[0].y, 1, p[1].x, p[1].y, 1, p[2].x, p[2].y, 1];
  const row1 = solve3x3(m, [q[0].x, q[1].x, q[2].x]);   // null = 3점이 거의 일직선
  if (!row1) return null;
  const row2 = solve3x3(m, [q[0].y, q[1].y, q[2].y]);
  if (!row2) return null;
  return [row1[0], row1[1], row1[2], row2[0], row2[1], row2[2]];
}

// ------------------------------------------------------------
// 인라이어 전체에 대한 최소자승 재적합 (정규방정식 3x3 두 번)
// ------------------------------------------------------------
export function affineLeastSquares(src, dst, idxs) {
  if (!idxs || idxs.length < 3) return null;
  let sxx = 0, sxy = 0, syy = 0, sx = 0, sy = 0, n = 0;
  let bx0 = 0, bx1 = 0, bx2 = 0, by0 = 0, by1 = 0, by2 = 0;
  for (const i of idxs) {
    const { x, y } = src[i];
    const { x: X, y: Y } = dst[i];
    sxx += x * x; sxy += x * y; syy += y * y; sx += x; sy += y; n += 1;
    bx0 += x * X; bx1 += y * X; bx2 += X;
    by0 += x * Y; by1 += y * Y; by2 += Y;
  }
  const m = [sxx, sxy, sx, sxy, syy, sy, sx, sy, n];
  const row1 = solve3x3(m, [bx0, bx1, bx2]);
  if (!row1) return null;
  const row2 = solve3x3(m, [by0, by1, by2]);
  if (!row2) return null;
  return [row1[0], row1[1], row1[2], row2[0], row2[1], row2[2]];
}

// ------------------------------------------------------------
// 결정론적 난수 (LCG)
//   세션마다 같은 입력이면 같은 정합 결과가 나와야 추세가 흔들리지 않는다.
//   Math.random 을 쓰면 같은 사진을 재분석할 때 면적값이 미세하게 달라진다.
// ------------------------------------------------------------
function lcg(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ------------------------------------------------------------
// RANSAC 어파인 추정
//   src -> dst 로 보내는 A 와 인라이어 인덱스를 반환.
//   threshold: 재투영 오차 허용치(dst 좌표계 px)
// ------------------------------------------------------------
export function ransacAffine(src, dst, opts = {}) {
  const {
    threshold = 3.0,
    maxIters = 3000,
    minIters = 100,
    confidence = 0.995,
    seed = 20240101,
  } = opts;

  const n = src.length;
  if (n < 3) return null;

  const thr2 = threshold * threshold;
  const rand = lcg(seed);
  let best = null;          // { A, idxs, err }
  let iters = maxIters;

  for (let it = 0; it < Math.max(minIters, Math.min(iters, maxIters)); it++) {
    // 서로 다른 3점 표본
    const i0 = (rand() * n) | 0;
    let i1 = (rand() * n) | 0;
    let i2 = (rand() * n) | 0;
    if (i1 === i0) i1 = (i1 + 1) % n;
    if (i2 === i0 || i2 === i1) i2 = (i2 + 2) % n;
    if (i2 === i0 || i2 === i1) continue;

    const A = affineFrom3(src, dst, i0, i1, i2);
    if (!A) continue;

    // 표본에서 나온 변환이 이미 비상식적이면 조기 폐기(연산 절약)
    const det = A[0] * A[4] - A[1] * A[3];
    if (!isFinite(det) || det <= 0) continue;

    const idxs = [];
    let sse = 0;
    for (let i = 0; i < n; i++) {
      const [px, py] = applyAffine(A, src[i].x, src[i].y);
      const dx = px - dst[i].x, dy = py - dst[i].y;
      const e2 = dx * dx + dy * dy;
      if (e2 <= thr2) { idxs.push(i); sse += e2; }
    }

    if (!best || idxs.length > best.idxs.length ||
        (idxs.length === best.idxs.length && sse < best.sse)) {
      best = { A, idxs, sse };
      // 인라이어 비율에 맞춰 필요한 반복 횟수를 줄인다(표준 적응형 RANSAC)
      const w = idxs.length / n;
      if (w > 0) {
        const denom = Math.log(1 - Math.pow(w, 3));
        if (denom < 0) iters = Math.min(maxIters, Math.ceil(Math.log(1 - confidence) / denom));
      }
    }
  }

  if (!best || best.idxs.length < 3) return null;

  // 인라이어 전체로 재적합 -> 재투영 오차 재계산
  const refined = affineLeastSquares(src, dst, best.idxs) || best.A;
  let sse = 0;
  const idxs = [];
  for (let i = 0; i < n; i++) {
    const [px, py] = applyAffine(refined, src[i].x, src[i].y);
    const dx = px - dst[i].x, dy = py - dst[i].y;
    const e2 = dx * dx + dy * dy;
    if (e2 <= thr2) { idxs.push(i); sse += e2; }
  }
  if (idxs.length < 3) return { A: best.A, inliers: best.idxs, reprojErr: Math.sqrt(best.sse / best.idxs.length) };

  return {
    A: refined,
    inliers: idxs,
    reprojErr: Math.sqrt(sse / idxs.length),   // 인라이어 RMS 재투영 오차
  };
}

// ------------------------------------------------------------
// 변환 분해 (물리적 타당성 판정용)
//   QR 분해: M = R(theta) * [[sx, sx*shear],[0, sy]]
// ------------------------------------------------------------
export function decomposeAffine(A) {
  const a = A[0], b = A[1], c = A[3], d = A[4];
  const sx = Math.hypot(a, c);
  const det = a * d - b * c;
  const sy = sx > 1e-9 ? det / sx : 0;
  const rotationDeg = (Math.atan2(c, a) * 180) / Math.PI;
  const shear = sx > 1e-9 ? (a * b + c * d) / (sx * sx) : 0;
  return {
    scaleX: sx,
    scaleY: sy,
    shearDeg: (Math.atan(shear) * 180) / Math.PI,
    rotationDeg,
    det,
    aspect: sy !== 0 ? Math.abs(sx / sy) : Infinity,
  };
}

// ------------------------------------------------------------
// 인라이어 공간 분포
//   한 구석에만 몰린 인라이어는 국소적으로만 맞는 변환을 만든다.
//   grid x grid 로 나눠 몇 칸이 채워졌는지 + 좌표 표준편차(정규화)를 본다.
// ------------------------------------------------------------
export function spatialCoverage(points, width, height, grid = 3) {
  if (!points.length) return { coverage: 0, spread: 0 };
  const cells = new Set();
  let mx = 0, my = 0;
  for (const p of points) {
    const gx = Math.min(grid - 1, Math.max(0, ((p.x / width) * grid) | 0));
    const gy = Math.min(grid - 1, Math.max(0, ((p.y / height) * grid) | 0));
    cells.add(gy * grid + gx);
    mx += p.x; my += p.y;
  }
  mx /= points.length; my /= points.length;
  let v = 0;
  for (const p of points) v += (p.x - mx) ** 2 + (p.y - my) ** 2;
  const rms = Math.sqrt(v / points.length);
  return {
    coverage: cells.size / (grid * grid),
    spread: rms / Math.hypot(width, height),   // 0~약0.4
  };
}
