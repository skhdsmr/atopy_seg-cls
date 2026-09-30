// 온디바이스 성능 계측 헬퍼 (추론 속도 + 메모리)

// 단조 증가 고해상도 시계 (Hermes 는 performance.now 지원, 없으면 Date.now 폴백)
export const now = () =>
  (global.performance && typeof global.performance.now === 'function')
    ? global.performance.now()
    : Date.now();

// 앱 사용 메모리(MB). react-native-device-info 네이티브 모듈이
// 현재 dev build 에 포함돼 있지 않으면(구 빌드) null 을 돌려준다 -> 앱은 안 죽고 "—" 표시.
let DeviceInfo = null;
try {
  DeviceInfo = require('react-native-device-info').default;
} catch (e) {
  DeviceInfo = null;
}

export async function usedMemoryMB() {
  if (!DeviceInfo || typeof DeviceInfo.getUsedMemory !== 'function') return null;
  try {
    const bytes = await DeviceInfo.getUsedMemory();
    return bytes > 0 ? bytes / (1024 * 1024) : null;
  } catch (e) {
    return null;
  }
}

// 배열 통계 (ms)
export function stats(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const n = s.length;
  const mean = arr.reduce((a, b) => a + b, 0) / n;
  const pct = (p) => s[Math.min(n - 1, Math.floor(p * (n - 1)))];
  return { n, mean, median: pct(0.5), min: s[0], max: s[n - 1], p90: pct(0.9) };
}
