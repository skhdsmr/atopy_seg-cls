// ============================================================
// 온디바이스 아토피 '분류' — react-native-fast-tflite
//
// segment.js 와 짝을 이루는 분류 파이프라인. 순수 온디바이스 순전파:
//   1) 이미지를 모델 입력크기(정사각)로 리사이즈 (expo-image-manipulator)
//   2) JPEG 디코드 -> ImageNet 정규화 Float32 NHWC [1,H,W,3]
//   3) tflite 추론 -> 이어붙인 logits [1, sum(taskSizes)]
//   4) meta.tasks 의 offset/size 로 잘라 태스크별 softmax -> 등급(index)+확신도
//
// 태스크(순서형 등급): severity(IGA)/erythema/papulation/excoriation/lichenification.
// 전처리 규약은 학습/변환(export_cls_tflite.py)과 동일: Resize -> ImageNet 정규화.
// ============================================================
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { decode as decodeJpeg } from 'jpeg-js';
import { loadTensorflowModel } from 'react-native-fast-tflite';

import { CLS_MODELS, DEFAULT_CLS_MODEL } from './models.cls.gen';
import { ACTIVE_CLS_MODEL } from './modelSelectCls';
import { base64ToBytes } from './bytes';
import { now, usedMemoryMB, stats } from './perf';

// modelSelectCls.js 에서 고른 모델을 레지스트리에서 찾는다(없으면 기본 모델로 대체).
const MODEL_KEY = CLS_MODELS[ACTIVE_CLS_MODEL] ? ACTIVE_CLS_MODEL : DEFAULT_CLS_MODEL;
const ENTRY = CLS_MODELS[MODEL_KEY];
if (!ENTRY) {
  throw new Error('번들된 분류 모델이 없습니다. `python app/model/export_cls_tflite.py --batch` 를 먼저 실행하세요.');
}
const META = ENTRY.meta;

export const CLS_MODEL_NAME = MODEL_KEY;    // 결과 화면 표시용
const IMGSZ = META.imgsz;
const MEAN = META.mean;
const STD = META.std;
const TASKS = META.tasks;                   // [{name,title,title_ko,labels,offset,size}]
const NUM_OUTPUTS = META.num_outputs;

// 모델은 한 번만 로드해서 재사용
let _modelPromise = null;
function getModel() {
  if (!_modelPromise) {
    _modelPromise = loadTensorflowModel(ENTRY.model);
  }
  return _modelPromise;
}

// 앱 시작 시 미리 로드해두면 첫 추론 지연을 줄일 수 있다(옵션).
export function preloadClsModel() {
  getModel().catch(() => {});
}

// 수치 안정 softmax (구간 [lo,hi))
function softmax(arr, lo, hi) {
  let max = -Infinity;
  for (let i = lo; i < hi; i++) if (arr[i] > max) max = arr[i];
  let sum = 0;
  const out = new Array(hi - lo);
  for (let i = lo; i < hi; i++) {
    const e = Math.exp(arr[i] - max);
    out[i - lo] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

// imageUri -> 모델 입력크기로 리사이즈된 RGBA 픽셀
async function loadPixels(imageUri) {
  const ctx = ImageManipulator.manipulate(imageUri).resize({ width: IMGSZ, height: IMGSZ });
  const img = await ctx.renderAsync();
  const saved = await img.saveAsync({ compress: 1, format: SaveFormat.JPEG, base64: true });
  const jpegBytes = base64ToBytes(saved.base64);
  const { data } = decodeJpeg(jpegBytes, { useTArray: true });
  return data; // RGBA Uint8Array
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

// logits(Float32Array 길이 NUM_OUTPUTS) -> 태스크별 예측 배열
function parsePredictions(logits) {
  return TASKS.map((t) => {
    const probs = softmax(logits, t.offset, t.offset + t.size);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return {
      task: t.name,
      title: t.title,
      titleKo: t.title_ko,
      index: best,                          // 예측 등급(0=가장 낮음)
      label: t.labels[best],                // 예: 'Moderate'
      confidence: probs[best],              // 해당 등급 확률
      probs,                                // 등급별 확률 전체
      labels: t.labels,
    };
  });
}

// 메인 진입점: imageUri -> { predictions, severity, timings, memory, imgsz, modelName }
// predictions: 태스크별 {task,title,titleKo,index,label,confidence,probs,labels}
// severity: 편의용, severity(IGA) 태스크 예측(없으면 첫 태스크)
export async function classify(imageUri) {
  const wasLoaded = _modelPromise != null;
  const memBase = await usedMemoryMB();

  const t0 = now();
  const model = await getModel();
  const tLoad = now();
  const memAfterLoad = await usedMemoryMB();

  const rgba = await loadPixels(imageUri);
  const input = toInputTensor(rgba);
  const tPre = now();

  const outputs = await model.run([input]);
  const tInf = now();

  const logits = outputs[0];                // 평탄화 Float32Array [NUM_OUTPUTS]
  if (!logits || logits.length < NUM_OUTPUTS) {
    throw new Error(`모델 출력 크기 이상: ${logits ? logits.length : 'null'} (기대 ${NUM_OUTPUTS})`);
  }
  const predictions = parsePredictions(logits);
  const tPost = now();

  const memAfterInfer = await usedMemoryMB();

  const timings = {
    modelLoadMs: tLoad - t0,
    preprocessMs: tPre - tLoad,
    inferenceMs: tInf - tPre,
    postprocessMs: tPost - tInf,
    totalMs: tPost - t0,
  };
  const d = (a, b) => (a != null && b != null ? a - b : null);
  const memory = {
    freshLoad: !wasLoaded,
    baseMB: memBase,
    afterLoadMB: memAfterLoad,
    afterInferMB: memAfterInfer,
    loadDeltaMB: d(memAfterLoad, memBase),
    inferDeltaMB: d(memAfterInfer, memAfterLoad),
    totalDeltaMB: d(memAfterInfer, memBase),
  };
  console.log('[classify] timings(ms)=', JSON.stringify(timings), ' memory(MB)=', JSON.stringify(memory));

  const severity = predictions.find((p) => p.task === 'severity') || predictions[0];
  return { predictions, severity, timings, memory, imgsz: IMGSZ, modelName: MODEL_KEY };
}

// 벤치마크: N회 반복하며 (a) 추론만 과 (b) 전처리+추론+후처리 전체 를 각각 계측한다.
export async function benchmarkClassify(imageUri, runs = 20) {
  const model = await getModel();

  // 워밍업
  {
    const rgba = await loadPixels(imageUri);
    const input = toInputTensor(rgba);
    const o = await model.run([input]);
    parsePredictions(o[0]);
    await model.run([input]);
  }

  const memBefore = await usedMemoryMB();
  let peak = memBefore;
  const inferSamples = [];
  const fullSamples = [];
  for (let i = 0; i < runs; i++) {
    const t0 = now();
    const rgba = await loadPixels(imageUri);
    const input = toInputTensor(rgba);
    const t1 = now();
    const outputs = await model.run([input]);
    const t2 = now();
    parsePredictions(outputs[0]);
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
