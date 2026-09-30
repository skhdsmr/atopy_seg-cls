// base64 <-> byte 변환 (RN/Hermes 에 atob/btoa/Buffer 가 없어도 동작하도록 순수 JS 구현)
const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const LOOKUP = (() => {
  const t = new Uint8Array(256);
  for (let i = 0; i < CHARS.length; i++) t[CHARS.charCodeAt(i)] = i;
  return t;
})();

// base64 문자열 -> Uint8Array
export function base64ToBytes(b64) {
  let len = b64.length;
  let pad = 0;
  if (b64[len - 1] === '=') pad++;
  if (b64[len - 2] === '=') pad++;
  const outLen = (len * 3) / 4 - pad;
  const out = new Uint8Array(outLen);
  let p = 0;
  for (let i = 0; i < len; i += 4) {
    const a = LOOKUP[b64.charCodeAt(i)];
    const b = LOOKUP[b64.charCodeAt(i + 1)];
    const c = LOOKUP[b64.charCodeAt(i + 2)];
    const d = LOOKUP[b64.charCodeAt(i + 3)];
    const chunk = (a << 18) | (b << 12) | (c << 6) | d;
    if (p < outLen) out[p++] = (chunk >> 16) & 0xff;
    if (p < outLen) out[p++] = (chunk >> 8) & 0xff;
    if (p < outLen) out[p++] = chunk & 0xff;
  }
  return out;
}

// Uint8Array / ArrayBuffer -> base64 문자열
export function bytesToBase64(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let out = '';
  const len = u8.length;
  for (let i = 0; i < len; i += 3) {
    const a = u8[i];
    const b = i + 1 < len ? u8[i + 1] : 0;
    const c = i + 2 < len ? u8[i + 2] : 0;
    const chunk = (a << 16) | (b << 8) | c;
    out += CHARS[(chunk >> 18) & 63];
    out += CHARS[(chunk >> 12) & 63];
    out += i + 1 < len ? CHARS[(chunk >> 6) & 63] : '=';
    out += i + 2 < len ? CHARS[chunk & 63] : '=';
  }
  return out;
}
