// ============================================================
// react-native-fast-opencv 어댑터
//
// 이 파일이 네이티브 OpenCV 를 만지는 유일한 지점이다. 라이브러리가
// 버전마다 객체 생성 API(Mat.create / OpenCV.createObject)와 enum export 를
// 다르게 노출하므로, 그 차이를 여기서만 흡수하고 register.js 는 순수 로직만
// 다루게 한다. 라이브러리 업그레이드로 깨지면 이 파일만 보면 된다.
//
// 설치(개발 클라이언트 재빌드 필요):
//   npx expo install react-native-fast-opencv
//   eas build --profile development --platform android
// ============================================================
import * as FOCV from 'react-native-fast-opencv';
import UPNG from 'upng-js';

import { bytesToBase64 } from './bytes';

const OpenCV = FOCV.OpenCV;

// OpenCV 상수는 값이 고정이라 enum export 가 없으면 숫자로 대체해도 안전하다.
const C = {
  COLOR_BGR2GRAY: FOCV.ColorConversionCodes?.COLOR_BGR2GRAY ?? 6,
  COLOR_BGRA2GRAY: FOCV.ColorConversionCodes?.COLOR_BGRA2GRAY ?? 10,
  INTER_AREA: FOCV.InterpolationFlags?.INTER_AREA ?? 3,
  THRESH_BINARY: FOCV.ThresholdTypes?.THRESH_BINARY ?? 0,
  NORM_HAMMING: FOCV.NormTypes?.NORM_HAMMING ?? 6,
  CV_8U: FOCV.DataTypes?.CV_8U ?? 0,
  CV_8UC3: FOCV.DataTypes?.CV_8UC3 ?? 16,
};

export function isAvailable() {
  return !!OpenCV && typeof OpenCV.ORB_create === 'function';
}

function assertAvailable() {
  if (!isAvailable()) {
    throw new Error(
      'react-native-fast-opencv 를 찾을 수 없습니다. ' +
      '`npx expo install react-native-fast-opencv` 후 개발 클라이언트를 재빌드하세요.'
    );
  }
}

// ------------------------------------------------------------
// 객체 생성 (버전별 API 흡수)
// ------------------------------------------------------------
function createMat(rows, cols, type) {
  if (typeof FOCV.Mat?.create === 'function') return FOCV.Mat.create(rows, cols, type);
  if (typeof OpenCV.createObject === 'function') {
    return OpenCV.createObject(FOCV.ObjectType?.Mat ?? 'mat', rows, cols, type);
  }
  throw new Error('Mat 생성 API 를 찾을 수 없습니다 (cv.js 어댑터 갱신 필요)');
}

function createSize(w, h) {
  if (typeof FOCV.Size?.create === 'function') return FOCV.Size.create(w, h);
  if (typeof OpenCV.createObject === 'function') {
    return OpenCV.createObject(FOCV.ObjectType?.Size ?? 'size', w, h);
  }
  throw new Error('Size 생성 API 를 찾을 수 없습니다 (cv.js 어댑터 갱신 필요)');
}

// base64(JPEG/PNG) -> Mat. data URI 접두사는 떼고 넘긴다.
export function matFromBase64(b64) {
  assertAvailable();
  const raw = b64.startsWith('data:') ? b64.slice(b64.indexOf(',') + 1) : b64;
  return FOCV.Mat.createFromBase64(raw);
}

// 벡터/키포인트/매치 접근자 — 호스트 객체 형태가 버전마다 조금씩 다르다.
function vecSize(v) {
  if (!v) return 0;
  if (typeof v.size === 'function') return v.size();
  if (typeof v.size === 'number') return v.size;
  if (typeof v.length === 'number') return v.length;
  return 0;
}

function kpPoint(kp) {
  const p = kp?.pt ?? kp;
  const x = p?.x, y = p?.y;
  if (typeof x !== 'number' || typeof y !== 'number') {
    throw new Error('KeyPoint 좌표를 읽지 못했습니다 (cv.js 어댑터 갱신 필요)');
  }
  return { x, y };
}

// ------------------------------------------------------------
// 기본 연산
// ------------------------------------------------------------
export function toGray(src) {
  const dst = createMat(0, 0, C.CV_8U);
  // 디코드 결과가 3채널/4채널 어느 쪽이든 동작하도록 순서대로 시도
  try {
    OpenCV.cvtColor(src, dst, C.COLOR_BGR2GRAY);
  } catch (e) {
    OpenCV.cvtColor(src, dst, C.COLOR_BGRA2GRAY);
  }
  return dst;
}

export function resize(src, width, height) {
  const dst = createMat(0, 0, C.CV_8U);
  OpenCV.resize(src, dst, createSize(width, height), 0, 0, C.INTER_AREA);
  return dst;
}

// 0/255 이진 마스크 Mat 만들기.
//   fast-opencv 에 raw 버퍼로 Mat 을 만드는 안정적인 경로가 없어서,
//   이미 의존성에 있는 upng-js 로 PNG 인코딩 후 디코드시킨다(수 ms 수준).
export function maskMatFromBytes(bytes, width, height) {
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const v = bytes[i] ? 255 : 0;
    rgba[i * 4] = v; rgba[i * 4 + 1] = v; rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
  }
  const png = UPNG.encode([rgba.buffer], width, height, 0);
  const mat = matFromBase64(bytesToBase64(png));
  const gray = toGray(mat);
  const bin = createMat(0, 0, C.CV_8U);
  OpenCV.threshold(gray, bin, 127, 255, C.THRESH_BINARY);
  release(mat, gray);
  return bin;
}

// ------------------------------------------------------------
// ORB 검출 + 기술자
//   mask: 255 인 영역에서만 특징점을 뽑는다(=병변 제외에 사용)
// ------------------------------------------------------------
export function detectORB(grayMat, maskMat, nfeatures = 3000) {
  assertAvailable();
  // scaleFactor 1.2 / nlevels 8 = OpenCV 기본. 원거리↔근거리 스케일 차는
  // register.js 에서 미리 정규화하므로 여기서 피라미드를 더 키우지 않는다.
  const orb = OpenCV.ORB_create(nfeatures);
  const { keypoints, descriptors } = maskMat
    ? OpenCV.detectAndCompute(orb, grayMat, maskMat)
    : OpenCV.detectAndCompute(orb, grayMat);

  const n = vecSize(keypoints);
  const pts = new Array(n);
  for (let i = 0; i < n; i++) pts[i] = kpPoint(keypoints.get(i));
  return { points: pts, descriptors, count: n };
}

// ------------------------------------------------------------
// kNN 매칭 + Lowe ratio test
//   crossCheck 는 knn 과 함께 쓸 수 없으므로 false 로 둔다.
//   반환: [{ queryIdx, trainIdx, distance }, ...]
// ------------------------------------------------------------
export function matchWithRatioTest(descQuery, descTrain, ratio = 0.75) {
  assertAvailable();
  const matcher = OpenCV.BFMatcher_create(C.NORM_HAMMING, false);
  const knn = OpenCV.knnMatchBF(matcher, descQuery, descTrain, 2);

  const out = [];
  const n = vecSize(knn);
  for (let i = 0; i < n; i++) {
    const pair = knn.get(i);
    if (vecSize(pair) < 2) continue;
    const m0 = pair.get(0), m1 = pair.get(1);
    if (!m0 || !m1) continue;
    if (m0.distance < ratio * m1.distance) {
      out.push({ queryIdx: m0.queryIdx, trainIdx: m0.trainIdx, distance: m0.distance });
    }
  }
  return out;
}

export function release(...mats) {
  for (const m of mats) {
    try { m?.release?.(); } catch { /* 이미 해제됨 */ }
  }
}

// 프레임 단위 버퍼 정리(제공되는 버전에서만 동작)
export function clearBuffers() {
  try { OpenCV?.clearBuffers?.(); } catch { /* noop */ }
}
