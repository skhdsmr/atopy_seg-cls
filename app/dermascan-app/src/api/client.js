// ============================================================
// Backend API client
// Just set the real server address in BASE_URL to switch from mock to live server.
// ============================================================
import { MOCK_HOSPITALS, MOCK_CHATS, MOCK_RECORDS } from '../data/mock';

// Local network mode: put the phone and PC on the same network (hotspot/Wi-Fi) and connect directly via the PC's Wi-Fi IP.
// The IP below is the PC's current Wi-Fi IPv4 (printed when running server/setup_local.ps1).
// This IP changes whenever the network changes, so update it each time.
export const BASE_URL = 'http://10.221.101.98:8000';
const USE_MOCK = !BASE_URL;

export const CHAT_FEE_KRW = 500;

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// Logged-in user identifier (email). Set by AuthContext on login/logout.
let _userId = 'demo';
export function setAuthUser(email) {
  _userId = email || 'demo';
}

// Shared fetch: injects the user header + timeout (default 15s).
// Without a timeout the request hangs for minutes when the server is unreachable, so always set one.
async function api(path, opts = {}, timeoutMs = 15000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(`${BASE_URL}${path}`, {
      ...opts,
      signal: ctrl.signal,
      headers: {
        'X-User-Id': _userId,
        ...(opts.body && typeof opts.body === 'string' ? { 'Content-Type': 'application/json' } : {}),
        ...(opts.headers || {}),
      },
    });
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('서버 응답 시간 초과 (연결 확인 필요)');
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

// ------------------------------------------------------------
// 1. 병변 분할(추론)은 온디바이스 tflite 로 이동했습니다 -> src/ml/segment.js
//    (기존 서버 /predict 분류 + Grad-CAM 은 제거)
// ------------------------------------------------------------

// ------------------------------------------------------------
// 2. Partner hospitals
// ------------------------------------------------------------
export async function getNearbyHospitals(coords) {
  if (USE_MOCK) {
    await wait(400);
    return MOCK_HOSPITALS;
  }
  const q = coords ? `?lat=${coords.lat}&lng=${coords.lng}` : '';
  const res = await api(`/hospitals${q}`);
  return res.json();
}

export async function registerHospital(hospitalId, register = true) {
  if (USE_MOCK) {
    await wait(300);
    return { ok: true, registered: register };
  }
  const res = await api(`/hospitals/${hospitalId}/register`, {
    method: register ? 'POST' : 'DELETE',
  });
  return res.json();
}

// ------------------------------------------------------------
// 3. Chat (doctor <-> patient, charged per message to the patient)
// ------------------------------------------------------------
export async function getChats() {
  if (USE_MOCK) {
    await wait(300);
    return MOCK_CHATS;
  }
  const res = await api('/chats');
  return res.json();
}

export async function sendMessage(chatId, text, senderRole) {
  if (USE_MOCK) {
    await wait(200);
    return {
      ok: true,
      message: { id: String(Date.now()), text, senderRole, ts: '방금' },
      charged: senderRole === 'patient' ? CHAT_FEE_KRW : 0,
    };
  }
  const res = await api(`/chats/${chatId}/messages`, {
    method: 'POST',
    body: JSON.stringify({ text, senderRole }),
  });
  return res.json();
}

// ------------------------------------------------------------
// 4. Symptom records (calendar + daily photos)
// ------------------------------------------------------------
export async function getRecords() {
  if (USE_MOCK) {
    await wait(300);
    return MOCK_RECORDS;
  }
  const res = await api('/records');
  return res.json();
}

export async function addRecord(record) {
  if (USE_MOCK) {
    await wait(300);
    return { ok: true, record };
  }
  const res = await api('/records', { method: 'POST', body: JSON.stringify(record) });
  return res.json();
}

// ------------------------------------------------------------
// 5. Visit appointments (server keeps the record; the actual notification is a local app notification)
// ------------------------------------------------------------
export async function addAppointment(hospitalId, date) {
  if (USE_MOCK) {
    await wait(200);
    return { ok: true };
  }
  const res = await api('/appointments', {
    method: 'POST',
    body: JSON.stringify({ hospitalId, date }),
  });
  return res.json();
}

// ------------------------------------------------------------
// 6. Monitoring: site + capture + CV measurement
// ------------------------------------------------------------
export async function getSites() {
  const res = await api('/sites');
  return res.json();
}

export async function createSite(site) {
  // site: { bodyPart, side, label, condition }
  const res = await api('/sites', { method: 'POST', body: JSON.stringify(site) });
  return res.json();
}

export async function deleteSite(siteId) {
  const res = await api(`/sites/${siteId}`, { method: 'DELETE' });
  return res.json();
}

// Anchor image for the site (guide overlay). referenceUrl=null if not registered yet.
// Also returns { width, height, roi } — roi is the lesion rectangle (normalized).
export async function getSiteReference(siteId) {
  const res = await api(`/sites/${siteId}/reference`);
  return res.json();
}

// Register the anchor photo (wide shot, taken once) + lesion ROI for a site.
// roi: { x, y, w, h } normalized to 0~1 on the captured image.
export async function setSiteAnchor(siteId, imageUri, roi) {
  const form = new FormData();
  form.append('file', { uri: imageUri, name: 'anchor.jpg', type: 'image/jpeg' });
  form.append('roi', JSON.stringify(roi));
  const res = await api(`/sites/${siteId}/anchor`, { method: 'POST', body: form }, 60000);
  if (!res.ok) throw new Error((await safeDetail(res)) || '기준 사진 등록 실패');
  return res.json();
}

// Matching target for on-device registration (ROI crop of the anchor + its transform).
// Fetch once per capture session and keep it in state — see src/ml/register.js.
export async function getAnchorCrop(siteId) {
  const res = await api(`/sites/${siteId}/anchor/crop`, {}, 30000);
  if (!res.ok) throw new Error((await safeDetail(res)) || '기준 크롭을 불러오지 못했습니다');
  return res.json();
}

// FastAPI puts the human-readable message in `detail`
async function safeDetail(res) {
  try { return (await res.json())?.detail; } catch { return null; }
}

export async function getCaptures(siteId) {
  const res = await api(`/sites/${siteId}/captures`);
  return res.json();
}

// Calendar: capture count per date { 'YYYY-MM-DD': count }
export async function getCaptureDates() {
  const res = await api('/captures/dates');
  return res.json();
}

// Calendar: all captures across sites for the selected date
export async function getCapturesByDate(date) {
  const res = await api(`/captures/by-date?date=${date}`);
  return res.json();
}

// Original capture image URL (for <Image source={{uri}}>)
export function captureImageUrl(captureId) {
  return `${BASE_URL}/captures/${captureId}/image`;
}

// Live preview check: low-res snapshot -> returns only brightness/exposure reasons (not saved).
// Short timeout: if the response is slow it's better to move on to the next sample.
export async function checkQuality(imageUri) {
  const form = new FormData();
  form.append('file', { uri: imageUri, name: 'preview.jpg', type: 'image/jpeg' });
  const res = await api('/quality', { method: 'POST', body: form }, 6000);
  if (!res.ok) throw new Error('프리뷰 점검 오류');
  return res.json();
}

// Upload a session capture. The server runs the quality/erythema gate; `metrics`
// carries the on-device registration + relative area + severity results.
export async function addCapture(siteId, imageUri, metrics = null) {
  const form = new FormData();
  form.append('file', { uri: imageUri, name: 'capture.jpg', type: 'image/jpeg' });
  if (metrics) form.append('metrics', JSON.stringify(metrics));
  const res = await api(`/sites/${siteId}/captures`, { method: 'POST', body: form }, 60000);
  if (!res.ok) throw new Error((await safeDetail(res)) || '촬영 분석 서버 오류');
  return res.json();
}
