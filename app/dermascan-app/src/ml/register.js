// ============================================================
// 기준 정합 (온디바이스) — 근거리 사진을 등록 시점의 원거리 기준 사진에 맞춘다
//
// 흐름:
//   1) 서버에서 받은 '기준 ROI 크롭'(원거리 사진의 병변 주변을 잘라 512로 맞춘 것)과
//      근거리 사진을 같은 작업 해상도로 두어 스케일 차를 미리 줄인다.
//      -> ORB 는 스케일 차에 약하고 fast-opencv 에 SIFT/AKAZE 가 없으므로
//         이 정규화가 매칭 성공률을 좌우한다.
//   2) 양쪽에서 '병변 영역을 제외'한 마스크로 ORB 특징점을 뽑는다.
//        - 근거리: 분할 마스크(segmentMask)를 팽창시켜 제외
//        - 기준:   등록 시 사용자가 지정한 ROI 사각형을 제외
//      병변은 세션마다 변하므로 정합 기준이 될 수 없다. 안 변하는 주변 피부만 쓴다.
//   3) kNN 매칭 + Lowe ratio test -> RANSAC 어파인 추정(affine.js)
//   4) 신뢰도 4축 + ROI 담김 정도를 규칙 기반으로 조합해 3단계 등급 판정
//   5) 기준 좌표계에서 상대 면적 산출 (ROI 격자를 역변환해 근거리 마스크를 샘플링)
//
// 변환 모델은 어파인으로 고정한다. 호모그래피는 자유도가 높아 특징이 적은
// 피부에서 과적합되고, 유사변환은 기울임을 못 잡는다.
// ============================================================
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';

import * as cv from './cv';
import {
  ransacAffine, composeAffine, scaleTranslate, invertAffine,
  applyAffine, decomposeAffine, spatialCoverage,
} from './affine';
import { now } from './perf';

// 매칭 작업 해상도(긴 변). 서버가 내려주는 기준 크롭도 같은 값으로 맞춰져 있다.
export const WORK = 512;

// 병변 마스크를 이만큼 팽창시켜 제외한다(경계 특징점이 매칭에 섞이지 않도록).
const LESION_PAD_PX = 8;

// 상대 면적 격자 해상도. 기준 ROI 를 GRID x GRID 로 나눠 세므로
// 세션이 달라도 분모가 항상 같다 -> 면적 추이가 흔들리지 않는다.
const AREA_GRID = 256;

const ORB_FEATURES = 3000;
const RATIO_TEST = 0.75;
const RANSAC_THRESH_PX = 3.0;

// ------------------------------------------------------------
// 신뢰도 문턱값 — 데이터가 쌓이면 여기만 조정한다.
//   high : 정상 채택
//   mid  : 값은 쓰되 추세에서 가중치를 낮춤(참고용 표시)
//   그 외 : 채택하지 않고 재촬영 유도
// ------------------------------------------------------------
export const REG_TH = {
  high: { inliers: 15, inlierRatio: 0.35, reprojErr: 3.0, coverage: 5 / 9, roiCoverage: 0.9 },
  mid: { inliers: 8, inlierRatio: 0.20, reprojErr: 6.0, coverage: 3 / 9, roiCoverage: 0.7 },
  // 물리적 타당성(어느 등급이든 벗어나면 즉시 low)
  plausible: { relScaleMin: 0.4, relScaleMax: 2.5, maxAspect: 1.3, maxShearDeg: 15, maxRotationDeg: 45 },
};

// ------------------------------------------------------------
// 이미지 -> 작업 해상도 base64 (가로 기준 WORK, 비율 유지)
// ------------------------------------------------------------
async function loadWorkImage(uri) {
  const ctx = ImageManipulator.manipulate(uri).resize({ width: WORK });
  const img = await ctx.renderAsync();
  const saved = await img.saveAsync({ compress: 0.92, format: SaveFormat.JPEG, base64: true });
  return { base64: saved.base64, width: saved.width, height: saved.height };
}

// ------------------------------------------------------------
// 분리형 최대필터(팽창). 0/1 배열을 제자리에서 반경 r 만큼 부풀린다.
// ------------------------------------------------------------
function dilate(src, w, h, r) {
  if (r <= 0) return src;
  const tmp = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let k = -r; k <= r && !v; k++) {
        const xx = x + k;
        if (xx >= 0 && xx < w && src[y * w + xx]) v = 1;
      }
      tmp[y * w + x] = v;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let k = -r; k <= r && !v; k++) {
        const yy = y + k;
        if (yy >= 0 && yy < h && tmp[yy * w + x]) v = 1;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

// 근거리 작업 이미지 크기의 '매칭 허용' 마스크(255=사용). 병변은 제외.
function buildNearKeepMask(segMask, segSize, w, h) {
  const lesion = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(segSize - 1, ((y / h) * segSize) | 0);
    for (let x = 0; x < w; x++) {
      const sx = Math.min(segSize - 1, ((x / w) * segSize) | 0);
      lesion[y * w + x] = segMask[sy * segSize + sx];
    }
  }
  const grown = dilate(lesion, w, h, LESION_PAD_PX);
  const keep = new Uint8Array(w * h);
  for (let i = 0; i < keep.length; i++) keep[i] = grown[i] ? 0 : 1;
  return keep;
}

// 기준 크롭에서 ROI(병변) 사각형을 제외한 마스크
function buildCropKeepMask(roiInCrop, w, h) {
  const keep = new Uint8Array(w * h).fill(1);
  const x0 = Math.max(0, Math.floor(roiInCrop.x - LESION_PAD_PX));
  const y0 = Math.max(0, Math.floor(roiInCrop.y - LESION_PAD_PX));
  const x1 = Math.min(w, Math.ceil(roiInCrop.x + roiInCrop.w + LESION_PAD_PX));
  const y1 = Math.min(h, Math.ceil(roiInCrop.y + roiInCrop.h + LESION_PAD_PX));
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) keep[y * w + x] = 0;
  }
  return keep;
}

// ------------------------------------------------------------
// 신뢰도 등급 판정 (규칙 기반)
// ------------------------------------------------------------
export function gradeRegistration(m) {
  const reasons = [];
  const p = REG_TH.plausible;

  // 축 3) 물리적 타당성 — 벗어나면 다른 축이 좋아도 신뢰할 수 없다
  if (!(m.det > 0)) reasons.push('변환이 뒤집힘(오매칭)');
  if (m.relScale < p.relScaleMin || m.relScale > p.relScaleMax) {
    reasons.push(m.relScale < p.relScaleMin ? '기준보다 너무 멀리서 찍힘' : '기준보다 너무 가까이서 찍힘');
  }
  if (m.aspect > p.maxAspect) reasons.push('가로세로가 심하게 눌림');
  if (Math.abs(m.shearDeg) > p.maxShearDeg) reasons.push('기울임이 과도함');
  if (Math.abs(m.rotationDeg) > p.maxRotationDeg) reasons.push('회전이 과도함');
  if (reasons.length) return { grade: 'low', reasons };

  const pass = (t) =>
    m.inliers >= t.inliers &&
    m.inlierRatio >= t.inlierRatio &&
    m.reprojErr <= t.reprojErr &&
    m.coverage >= t.coverage &&
    m.roiCoverage >= t.roiCoverage;

  if (pass(REG_TH.high)) return { grade: 'high', reasons: [] };

  if (pass(REG_TH.mid)) {
    // 중간 등급은 '무엇이 부족했는지'를 남겨 재촬영 안내에 쓴다
    const t = REG_TH.high;
    if (m.inliers < t.inliers || m.inlierRatio < t.inlierRatio) reasons.push('일치하는 피부 특징이 적음');
    if (m.reprojErr > t.reprojErr) reasons.push('정합 오차가 다소 큼');
    if (m.coverage < t.coverage) reasons.push('일치점이 한쪽에 치우침');
    if (m.roiCoverage < t.roiCoverage) reasons.push('기준 부위 일부가 화면 밖');
    return { grade: 'medium', reasons };
  }

  if (m.inliers < REG_TH.mid.inliers || m.inlierRatio < REG_TH.mid.inlierRatio) {
    reasons.push('기준 사진과 같은 부위로 보이지 않음');
  }
  if (m.reprojErr > REG_TH.mid.reprojErr) reasons.push('정합 오차가 큼');
  if (m.coverage < REG_TH.mid.coverage) reasons.push('일치점이 한쪽에만 몰림');
  if (m.roiCoverage < REG_TH.mid.roiCoverage) reasons.push('기준 부위가 화면에 다 담기지 않음');
  return { grade: 'low', reasons };
}

// ------------------------------------------------------------
// 기준 좌표계에서의 상대 면적
//   기준 ROI 를 AREA_GRID x AREA_GRID 격자로 훑으며, 각 점을 역변환해
//   근거리 분할 마스크를 샘플링한다. 분모(ROI 격자 수)가 세션과 무관하게
//   고정이므로 세션 간 비교가 성립한다.
// ------------------------------------------------------------
function measureArea(anchorAffine, roiAnchorPx, segMask, segSize, near) {
  const inv = invertAffine(anchorAffine);   // 기준 px -> 근거리 원본 px
  if (!inv) return null;

  let lesion = 0, covered = 0;
  const total = AREA_GRID * AREA_GRID;
  for (let j = 0; j < AREA_GRID; j++) {
    const ay = roiAnchorPx.y + ((j + 0.5) / AREA_GRID) * roiAnchorPx.h;
    for (let i = 0; i < AREA_GRID; i++) {
      const ax = roiAnchorPx.x + ((i + 0.5) / AREA_GRID) * roiAnchorPx.w;
      const [nx, ny] = applyAffine(inv, ax, ay);
      if (nx < 0 || ny < 0 || nx >= near.width || ny >= near.height) continue;  // 화면 밖
      covered++;
      const sx = Math.min(segSize - 1, ((nx / near.width) * segSize) | 0);
      const sy = Math.min(segSize - 1, ((ny / near.height) * segSize) | 0);
      if (segMask[sy * segSize + sx]) lesion++;
    }
  }

  return {
    areaRatio: lesion / total,            // 기준 ROI 면적 대비 병변 비율(고정 분모)
    roiCoverage: covered / total,         // ROI 중 근거리 사진에 실제로 담긴 비율
    areaPxAnchor: Math.round((lesion / total) * roiAnchorPx.w * roiAnchorPx.h),
  };
}

// ------------------------------------------------------------
// 메인 진입점
//   anchor: 서버 GET /sites/{sid}/anchor/crop 응답
//   near  : { uri, width, height }  (takePictureAsync 결과)
//   seg   : segmentMask() 결과 { mask, size }
// ------------------------------------------------------------
export async function registerToAnchor({ anchor, near, seg }) {
  const t0 = now();
  const mats = [];
  const track = (m) => { mats.push(m); return m; };

  try {
    // (1) 양쪽을 같은 작업 해상도로 — 스케일 차 정규화
    const nearWork = await loadWorkImage(near.uri);
    const nearMat = track(cv.matFromBase64(nearWork.base64));
    const nearGray = track(cv.toGray(nearMat));
    const cropMat = track(cv.matFromBase64(anchor.cropUrl));
    const cropGray = track(cv.toGray(cropMat));
    const tPre = now();

    // (2) 병변 제외 마스크
    const nearKeep = buildNearKeepMask(seg.mask, seg.size, nearWork.width, nearWork.height);
    const cropKeep = buildCropKeepMask(anchor.roiInCrop, anchor.cropWidth, anchor.cropHeight);
    const nearMask = track(cv.maskMatFromBytes(nearKeep, nearWork.width, nearWork.height));
    const cropMask = track(cv.maskMatFromBytes(cropKeep, anchor.cropWidth, anchor.cropHeight));

    // (3) ORB + ratio test
    const nearFeat = cv.detectORB(nearGray, nearMask, ORB_FEATURES);
    const cropFeat = cv.detectORB(cropGray, cropMask, ORB_FEATURES);
    track(nearFeat.descriptors); track(cropFeat.descriptors);

    if (nearFeat.count < 10 || cropFeat.count < 10) {
      return fail('피부 특징점이 너무 적습니다 (매끈한 부위/초점 확인)', {
        nearKeypoints: nearFeat.count, cropKeypoints: cropFeat.count,
      });
    }

    const matches = cv.matchWithRatioTest(nearFeat.descriptors, cropFeat.descriptors, RATIO_TEST);
    const tMatch = now();
    if (matches.length < 6) {
      return fail('기준 사진과 일치하는 부분을 찾지 못했습니다', { matches: matches.length });
    }

    const src = matches.map((m) => nearFeat.points[m.queryIdx]);
    const dst = matches.map((m) => cropFeat.points[m.trainIdx]);

    // (4) RANSAC 어파인
    const est = ransacAffine(src, dst, { threshold: RANSAC_THRESH_PX });
    if (!est) return fail('정합 변환을 추정하지 못했습니다', { matches: matches.length });
    const tRansac = now();

    // (5) 좌표계 합성: 근거리 원본 px -> 작업 px -> 기준 크롭 px -> 기준 원본 px
    const nearScale = nearWork.width / near.width;
    const anchorAffine = composeAffine(
      anchor.cropToAnchor,
      composeAffine(est.A, scaleTranslate(nearScale, 0, 0)),
    );

    // (6) 상대 면적 (기준 좌표계)
    const roiPx = {
      x: anchor.roi.x * anchor.anchorWidth,
      y: anchor.roi.y * anchor.anchorHeight,
      w: anchor.roi.w * anchor.anchorWidth,
      h: anchor.roi.h * anchor.anchorHeight,
    };
    const area = measureArea(anchorAffine, roiPx, seg.mask, seg.size, near);
    if (!area) return fail('면적 계산에 실패했습니다 (변환 특이)', {});

    // (7) 신뢰도 4축 + ROI 담김 정도
    const inlierPts = est.inliers.map((i) => src[i]);
    const { coverage, spread } = spatialCoverage(inlierPts, nearWork.width, nearWork.height, 3);
    const d = decomposeAffine(est.A);
    // 근거리 사진이 ROI 를 꽉 채워 찍혔다면 기대 스케일은 roi 폭 / 근거리 폭
    const expectedScale = anchor.roiInCrop.w / nearWork.width;
    const metrics = {
      matches: matches.length,
      inliers: est.inliers.length,
      inlierRatio: est.inliers.length / matches.length,
      reprojErr: est.reprojErr,
      coverage,
      spread,
      scaleX: d.scaleX,
      scaleY: d.scaleY,
      aspect: d.aspect,
      shearDeg: d.shearDeg,
      rotationDeg: d.rotationDeg,
      det: d.det,
      relScale: expectedScale > 0 ? d.scaleX / expectedScale : 0,
      roiCoverage: area.roiCoverage,
    };

    const { grade, reasons } = gradeRegistration(metrics);

    return {
      ok: grade !== 'low',
      grade,
      reasons,
      affine: anchorAffine,
      metrics,
      areaRatio: area.areaRatio,
      areaPxAnchor: area.areaPxAnchor,
      timings: {
        preprocessMs: tPre - t0,
        matchMs: tMatch - tPre,
        ransacMs: tRansac - tMatch,
        totalMs: now() - t0,
      },
    };
  } finally {
    cv.release(...mats);
    cv.clearBuffers();
  }
}

function fail(message, metrics) {
  return {
    ok: false,
    grade: 'low',
    reasons: [message],
    affine: null,
    metrics: metrics || {},
    areaRatio: null,
    areaPxAnchor: null,
  };
}

// 서버 저장/표시에 필요한 최소 필드만 추린다(캡처 업로드 payload).
export function toCaptureMetrics(reg) {
  if (!reg) return null;
  const m = reg.metrics || {};
  return {
    alignGrade: reg.grade,
    alignInliers: m.inliers ?? null,
    alignRatio: m.inlierRatio != null ? Number(m.inlierRatio.toFixed(3)) : null,
    alignReproj: m.reprojErr != null ? Number(m.reprojErr.toFixed(2)) : null,
    alignCoverage: m.coverage != null ? Number(m.coverage.toFixed(3)) : null,
    alignRoiCoverage: m.roiCoverage != null ? Number(m.roiCoverage.toFixed(3)) : null,
    alignReasons: reg.reasons || [],
    areaRatio: reg.areaRatio != null ? Number(reg.areaRatio.toFixed(5)) : null,
    affine: reg.affine ? reg.affine.map((v) => Number(v.toFixed(6))) : null,
  };
}
