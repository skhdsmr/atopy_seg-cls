// ============================================================
// 온디바이스 병변 분할 (segmentation) — react-native-fast-tflite
//
// 서버 /predict(분류+Grad-CAM) 를 대체한다. 순수 온디바이스 순전파:
//   1) 이미지를 모델 입력크기(정사각)로 리사이즈 (expo-image-manipulator)
//   2) JPEG 디코드 -> ImageNet 정규화 Float32 NHWC [1,H,W,3]
//   3) tflite 추론 -> logits [1,H,W,1]
//   4) sigmoid + threshold -> 0/1 마스크
//   5) 반투명 마스크 PNG(data URI) 오버레이 생성 + 병변 면적 비율 계산
//
// 학습/변환과 동일한 전처리 규약(segmentation/augment.py, export_tflite.py):
//   Resize((imgsz,imgsz)) -> Normalize(ImageNet mean/std)
// ============================================================
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { decode as decodeJpeg } from 'jpeg-js';
import UPNG from 'upng-js';
import { loadTensorflowModel } from 'react-native-fast-tflite';

import { MODELS, DEFAULT_MODEL } from './models.gen';
import { ACTIVE_MODEL } from './modelSelect';
import { base64ToBytes, bytesToBase64 } from './bytes';
import { now, usedMemoryMB, stats } from './perf';

// modelSelect.js 에서 고른 모델을 레지스트리에서 찾는다(없으면 기본 모델로 대체).
const MODEL_KEY = MODELS[ACTIVE_MODEL] ? ACTIVE_MODEL : DEFAULT_MODEL;
const ENTRY = MODELS[MODEL_KEY];
if (!ENTRY) {
  throw new Error('번들된 모델이 없습니다. `python app/model/export_tflite.py --batch` 를 먼저 실행하세요.');
}
const META = ENTRY.meta;

export const MODEL_NAME = MODEL_KEY;       // 결과 화면 표시용
const IMGSZ = META.imgsz;                  // 체크포인트 해상도(자동)
const MEAN = META.mean;                    // ImageNet
const STD = META.std;
const THR = META.threshold ?? 0.5;

// 오버레이 색상(반투명 빨강)
const MASK_RGBA = [239, 68, 68, 132];

// 모델은 한 번만 로드해서 재사용
let _modelPromise = null;
function getModel() {
  if (!_modelPromise) {
    // float32 입출력 모델(react-native-fast-tflite 는 Float32Array 만 입력 가능).
    _modelPromise = loadTensorflowModel(ENTRY.model);
  }
  return _modelPromise;
}

// 앱 시작 시 미리 로드해두면 첫 추론 지연을 줄일 수 있다(옵션).
export function preloadModel() {
  getModel().catch(() => {});
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// imageUri -> { width, height, rgba(Uint8Array RGBA) } (모델 입력크기로 리사이즈된 픽셀)
async function loadPixels(imageUri) {
  // SDK 54 신 ImageManipulator API (GuidedCaptureScreen 과 동일 관용구)
  const ctx = ImageManipulator.manipulate(imageUri).resize({ width: IMGSZ, height: IMGSZ });
  const img = await ctx.renderAsync();
  const saved = await img.saveAsync({ compress: 1, format: SaveFormat.JPEG, base64: true });
  const jpegBytes = base64ToBytes(saved.base64);
  const { width, height, data } = decodeJpeg(jpegBytes, { useTArray: true });
  return { width, height, rgba: data };
}

// RGBA 픽셀 -> 정규화 Float32 NHWC 입력텐서
function toInputTensor(rgba) {
  const n = IMGSZ * IMGSZ;
  const input = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4] / 255;
    const g = rgba[i * 4 + 1] / 255;
    const b = rgba[i * 4 + 2] / 255;
    input[i * 3] = (r - MEAN[0]) / STD[0];
    input[i * 3 + 1] = (g - MEAN[1]) / STD[1];
    input[i * 3 + 2] = (b - MEAN[2]) / STD[2];
  }
  return input;
}

// logits(Float32Array, 길이 IMGSZ*IMGSZ) -> { overlayUrl, coverage }
function buildOverlay(logits) {
  const n = IMGSZ * IMGSZ;
  const rgba = new Uint8Array(n * 4); // 기본 전부 투명(0)
  let fg = 0;
  for (let i = 0; i < n; i++) {
    if (sigmoid(logits[i]) >= THR) {
      rgba[i * 4] = MASK_RGBA[0];
      rgba[i * 4 + 1] = MASK_RGBA[1];
      rgba[i * 4 + 2] = MASK_RGBA[2];
      rgba[i * 4 + 3] = MASK_RGBA[3];
      fg++;
    }
  }
  // UPNG.encode(imgs[], w, h, cnum=0 -> 무손실 truecolor+alpha)
  const png = UPNG.encode([rgba.buffer], IMGSZ, IMGSZ, 0);
  const overlayUrl = `data:image/png;base64,${bytesToBase64(png)}`;
  return { overlayUrl, coverage: fg / n };
}

// 메인 진입점: imageUri -> 분할 결과 + 성능 계측
// 반환: { overlayUrl, coverage, detected, timings{preprocessMs,inferenceMs,postprocessMs,totalMs},
//         memory{beforeMB,afterMB,deltaMB} , imgsz }
export async function segment(imageUri) {
  // 모델이 아직 로드 안 됐는지(=이번 호출에서 로드되는지) 판단 -> 로드 증가량이 유의미한지 표시용
  const wasLoaded = _modelPromise != null;

  // (1) 모델 로드 '전' 기준 메모리
  const memBase = await usedMemoryMB();

  const t0 = now();
  const model = await getModel();             // 최초 1회만 실제 로드(이후 캐시)
  const tLoad = now();
  // (2) 모델 로드 '후' 메모리 -> (2)-(1) = 모델 가중치 상주 메모리
  const memAfterLoad = await usedMemoryMB();

  const { rgba } = await loadPixels(imageUri);
  const input = toInputTensor(rgba);
  const tPre = now();

  const outputs = await model.run([input]);
  const tInf = now();

  const logits = outputs[0];                   // NHWC [1,IMGSZ,IMGSZ,1] 평탄화 Float32Array
  if (!logits || logits.length < IMGSZ * IMGSZ) {
    throw new Error(`모델 출력 크기 이상: ${logits ? logits.length : 'null'} (기대 ${IMGSZ * IMGSZ})`);
  }
  const { overlayUrl, coverage } = buildOverlay(logits);
  const tPost = now();

  // (3) 추론 '후' 메모리 -> (3)-(2) = 추론 시 활성화/버퍼 증가량
  const memAfterInfer = await usedMemoryMB();

  const timings = {
    modelLoadMs: tLoad - t0,                   // 최초만 유의미(캐시 후엔 ~0)
    preprocessMs: tPre - tLoad,
    inferenceMs: tInf - tPre,
    postprocessMs: tPost - tInf,
    totalMs: tPost - t0,
  };
  const d = (a, b) => (a != null && b != null ? a - b : null);
  const memory = {
    freshLoad: !wasLoaded,                      // 이번 호출에서 모델이 처음 로드됐는가
    baseMB: memBase,                            // (1) 로드 전
    afterLoadMB: memAfterLoad,                  // (2) 로드 후
    afterInferMB: memAfterInfer,                // (3) 추론 후
    loadDeltaMB: d(memAfterLoad, memBase),      // (2)-(1) 모델 가중치 상주
    inferDeltaMB: d(memAfterInfer, memAfterLoad), // (3)-(2) 추론 활성화
    totalDeltaMB: d(memAfterInfer, memBase),    // (3)-(1) 전체 증가
  };
  console.log('[segment] timings(ms)=', JSON.stringify(timings), ' memory(MB)=', JSON.stringify(memory));

  return { overlayUrl, coverage, detected: coverage > 0, timings, memory, imgsz: IMGSZ, modelName: MODEL_KEY };
}

// ------------------------------------------------------------
// 정합/면적 계산용: 오버레이 PNG 인코딩 없이 0/1 마스크만 뽑는다.
//   register.js 가 (a) 특징점 매칭에서 병변을 제외하고 (b) 기준 좌표계로
//   면적을 환산하는 데 쓴다. PNG 인코딩을 건너뛰므로 segment() 보다 빠르다.
// 반환: { mask: Uint8Array(IMGSZ*IMGSZ, 0|1), size: IMGSZ, coverage }
// ------------------------------------------------------------
export async function segmentMask(imageUri) {
  const model = await getModel();
  const { rgba } = await loadPixels(imageUri);
  const outputs = await model.run([toInputTensor(rgba)]);
  const logits = outputs[0];
  const n = IMGSZ * IMGSZ;
  if (!logits || logits.length < n) {
    throw new Error(`모델 출력 크기 이상: ${logits ? logits.length : 'null'} (기대 ${n})`);
  }
  const mask = new Uint8Array(n);
  let fg = 0;
  for (let i = 0; i < n; i++) {
    if (sigmoid(logits[i]) >= THR) { mask[i] = 1; fg++; }
  }
  return { mask, size: IMGSZ, coverage: fg / n };
}

// 벤치마크: N회 반복하며 (a) 추론만 과 (b) 전처리+추론+후처리 전체 를 각각 계측한다.
// 반환: { runs, inference:{...}, full:{...}, memory{beforeMB,peakMB,deltaMB}, imgsz }
export async function benchmark(imageUri, runs = 20) {
  const model = await getModel();

  // 워밍업(첫 호출 초기화 비용 제외): 전체 파이프라인 1회 + 추론 1회
  {
    const { rgba } = await loadPixels(imageUri);
    const input = toInputTensor(rgba);
    const o = await model.run([input]);
    buildOverlay(o[0]);
    await model.run([input]);
  }

  const memBefore = await usedMemoryMB();
  let peak = memBefore;
  const inferSamples = [];   // 추론(model.run)만
  const fullSamples = [];    // 전처리 + 추론 + 후처리 전체
  for (let i = 0; i < runs; i++) {
    const t0 = now();
    const { rgba } = await loadPixels(imageUri);      // 전처리(매 회 재수행 = 실제 파이프라인)
    const input = toInputTensor(rgba);
    const t1 = now();
    const outputs = await model.run([input]);         // 추론
    const t2 = now();
    buildOverlay(outputs[0]);                          // 후처리
    const t3 = now();
    inferSamples.push(t2 - t1);
    fullSamples.push(t3 - t0);
    if ((i & 3) === 0) {
      const m = await usedMemoryMB();
      if (m != null && (peak == null || m > peak)) peak = m;
    }
  }
  const memAfter = await usedMemoryMB();
  if (memAfter != null && (peak == null || memAfter > peak)) peak = memAfter;

  return {
    runs,
    inference: stats(inferSamples),
    full: stats(fullSamples),
    memory: {
      beforeMB: memBefore,
      peakMB: peak,
      deltaMB: memBefore != null && peak != null ? peak - memBefore : null,
    },
    imgsz: IMGSZ,
  };
}
