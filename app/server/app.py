"""
DermaScan inference server (FastAPI)

Provides the POST /predict endpoint called by the app's src/api/client.js.
Reuses the predict logic from model/inference.py as-is.

Run:
    pip install fastapi uvicorn python-multipart
    uvicorn server.app:app --host 0.0.0.0 --port 8000

App connection:
    Set BASE_URL in dermascan-app/src/api/client.js to
    'http://<PC_IP>:8000' (on a real device, use the PC's IP on the same Wi-Fi)
"""
import io
import os
import sys
import gc
import json
import base64

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.cm as cm
from fastapi import FastAPI, UploadFile, File, Form, Header, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Add path so modules in the model/ directory can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
import timm  # noqa: E402

# Data layer / Kakao Local API / CV pipeline from the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import kakao  # noqa: E402
import cv_pipeline  # noqa: E402
import uuid  # noqa: E402

# Capture image storage path
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'captures')
os.makedirs(CAPTURE_DIR, exist_ok=True)

NUM_CLASSES = 5
CLASS_NAMES = ['CD', 'EC', 'OTHERS', 'SC', 'TC']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHT_CANDIDATES = [
    os.path.join(_HERE, '..', 'model', 'best_mobilenetv4.pth'),
    os.path.join(_HERE, '..', 'best_mobilenetv4.pth'),
    os.path.join(_HERE, 'best_mobilenetv4.pth'),
]
WEIGHT_PATH = next((p for p in _WEIGHT_CANDIDATES if os.path.isfile(p)), _WEIGHT_CANDIDATES[0])

app = FastAPI(title='DermaScan Inference')
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)
db.init_db()

# ---- Load model / Grad-CAM once ----
_model = None
_cam_state = {'act': None, 'grad': None}


def get_model():
    global _model
    if _model is None:
        m = timm.create_model('mobilenetv4_conv_medium', pretrained=False, num_classes=NUM_CLASSES)
        sd = torch.load(WEIGHT_PATH, map_location=DEVICE)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m.load_state_dict(sd)
        m.to(DEVICE).eval()
        # Grad-CAM hook (last block right before pooling)
        target = m.blocks[-1]
        target.register_forward_hook(lambda mod, i, o: _cam_state.update(act=o.detach()))
        target.register_full_backward_hook(lambda mod, gi, go: _cam_state.update(grad=go[0].detach()))
        _model = m
    return _model


def preprocess_size(img_pil, size=512):
    """Match the training data (512x512).
    Resize so the shorter side equals `size` (preserving aspect ratio), then center-crop a square.
    This also greatly reduces inference memory and overlay response size for high-resolution phone photos."""
    w, h = img_pil.size
    scale = size / min(w, h)
    img_pil = img_pil.resize((round(w * scale), round(h * scale)))
    w, h = img_pil.size
    left, top = (w - size) // 2, (h - size) // 2
    return img_pil.crop((left, top, left + size, top + size))


def run(img_pil):
    model = get_model()
    img_pil = preprocess_size(img_pil, 512)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    x = tf(img_pil).unsqueeze(0).to(DEVICE)

    model.zero_grad()
    out = model(x)
    probs = F.softmax(out, dim=1)[0]
    idx = int(out.argmax(1).item())
    out[0, idx].backward()

    # Grad-CAM
    g, a = _cam_state['grad'], _cam_state['act']
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam -= cam.min()
    vmax = np.percentile(cam, 99)
    if vmax > 0:
        cam = np.clip(cam / vmax, 0, 1)

    # Overlay (keep the original in low-activation regions)
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    heat = cm.jet(cam)[..., :3]
    thr, alpha = 0.2, 0.5
    cam_w = np.clip((cam - thr) / (1 - thr), 0, 1)[..., None]
    overlay = np.clip((1 - alpha * cam_w) * img_np + alpha * cam_w * heat, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    probs_dict = {c: float(p) for c, p in zip(CLASS_NAMES, probs.detach().cpu().numpy())}

    # Free gradient/activation memory for the next request
    model.zero_grad(set_to_none=True)
    _cam_state['grad'] = None
    _cam_state['act'] = None
    del x, out, g, a, w, cam
    gc.collect()

    return probs_dict, overlay_img


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    probs, overlay = run(img)

    # Encode as JPEG instead of PNG to greatly reduce response size (improves tunnel stability)
    buf = io.BytesIO()
    overlay.save(buf, format='JPEG', quality=85)
    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'probs': probs,
        'overlayUrl': f'data:image/jpeg;base64,{overlay_b64}',
    }


@app.get('/health')
def health():
    return {'status': 'ok', 'device': str(DEVICE)}


# ============================================================
# Hospital / record / chat / appointment API
# User identification via the X-User-Id header (app sends the login email; defaults to demo)
# ============================================================
def _uid(x_user_id):
    return x_user_id or 'demo'


@app.get('/hospitals')
def hospitals(lat: float = None, lng: float = None, x_user_id: str = Header(None)):
    uid = _uid(x_user_id)
    # If a Kakao REST key and coordinates are available, search for real nearby dermatology clinics
    if kakao.ENABLED and lat is not None and lng is not None:
        try:
            results = kakao.search_dermatology(lat, lng)
            if results:
                reg = db.get_registered_ids(uid)
                for r in results:
                    r['registered'] = r['id'] in reg
                return results
        except Exception as e:
            print('카카오 검색 실패, 시드 병원으로 대체:', e)
    # Fallback: seed partner hospitals
    return db.get_hospitals(uid, lat, lng)


@app.post('/hospitals/{hid}/register')
def register_hospital(hid: str, x_user_id: str = Header(None)):
    return db.set_registration(_uid(x_user_id), hid, True)


@app.delete('/hospitals/{hid}/register')
def unregister_hospital(hid: str, x_user_id: str = Header(None)):
    return db.set_registration(_uid(x_user_id), hid, False)


@app.get('/records')
def records(x_user_id: str = Header(None)):
    return db.get_records(_uid(x_user_id))


@app.post('/records')
def add_record(payload: dict = Body(...), x_user_id: str = Header(None)):
    return db.add_record(_uid(x_user_id), payload.get('date'), payload.get('severity', '경증'),
                         payload.get('note', ''), payload.get('photo'))


@app.get('/chats')
def chats(x_user_id: str = Header(None)):
    return db.get_chats(_uid(x_user_id))


@app.post('/chats/{cid}/messages')
def send_message(cid: str, payload: dict = Body(...), x_user_id: str = Header(None)):
    return db.add_message(_uid(x_user_id), cid, payload.get('text', ''),
                          payload.get('senderRole', 'patient'))


@app.get('/billing')
def billing(x_user_id: str = Header(None)):
    return db.get_billing(_uid(x_user_id))


@app.post('/appointments')
def appointments(payload: dict = Body(...), x_user_id: str = Header(None)):
    return db.add_appointment(_uid(x_user_id), payload.get('hospitalId'), payload.get('date'))


# ============================================================
# Monitoring: sites + captures + CV measurements
# ============================================================
@app.get('/sites')
def list_sites(x_user_id: str = Header(None)):
    return db.get_sites(_uid(x_user_id))


@app.post('/sites')
def create_site(payload: dict = Body(...), x_user_id: str = Header(None)):
    sid = 's_' + uuid.uuid4().hex[:10]
    return db.create_site(_uid(x_user_id), sid,
                          payload.get('bodyPart', ''), payload.get('side', ''),
                          payload.get('label', ''), payload.get('condition', ''))


@app.delete('/sites/{sid}')
def remove_site(sid: str, x_user_id: str = Header(None)):
    # Also clean up the stored image files
    folder = os.path.join(CAPTURE_DIR, sid)
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass
    return db.delete_site(_uid(x_user_id), sid)


@app.get('/sites/{sid}/captures')
def site_captures(sid: str):
    return db.get_captures(sid)


@app.get('/sites/{sid}/reference')
def site_reference(sid: str):
    """Anchor image (base64) for the guide overlay, plus its size and lesion ROI.
    Returns empty values instead of 204 when none exists."""
    a = db.get_site_anchor(sid) or {}
    path = a.get('path')
    if not path or not os.path.isfile(path):
        return {'referenceUrl': None, 'width': None, 'height': None, 'roi': None}
    with open(path, 'rb') as f:
        raw = f.read()
    w, h = a.get('width'), a.get('height')
    if not (w and h):
        try:
            w, h = Image.open(io.BytesIO(raw)).size
        except Exception:
            w, h = None, None
    b64 = base64.b64encode(raw).decode()
    return {'referenceUrl': f'data:image/jpeg;base64,{b64}',
            'width': w, 'height': h, 'roi': a.get('roi')}


# ------------------------------------------------------------
# Anchor: the wide reference photo, captured once when the site is registered.
#   Later sessions are close-ups registered against this frame on-device.
#   The server only stores it and serves the pre-cropped matching target —
#   all registration math runs in the app (src/ml/register.js).
# ------------------------------------------------------------
ANCHOR_MARGIN = 0.6    # crop = ROI expanded by 60% per side: the surrounding skin is what actually matches
CROP_WORK = 512        # must equal WORK in src/ml/register.js
MIN_ROI = 0.03         # reject degenerate ROIs (< 3% of a side)


def _parse_roi(roi_json):
    try:
        r = json.loads(roi_json)
        roi = {k: float(r[k]) for k in ('x', 'y', 'w', 'h')}
    except Exception:
        raise HTTPException(status_code=400, detail='roi 형식 오류 (x,y,w,h 정규화 값 필요)')
    if roi['w'] < MIN_ROI or roi['h'] < MIN_ROI:
        raise HTTPException(status_code=400, detail='선택한 병변 영역이 너무 작습니다')
    if not (0 <= roi['x'] and 0 <= roi['y'] and roi['x'] + roi['w'] <= 1.001
            and roi['y'] + roi['h'] <= 1.001):
        raise HTTPException(status_code=400, detail='병변 영역이 사진 밖으로 벗어났습니다')
    return roi


@app.post('/sites/{sid}/anchor')
async def set_anchor(sid: str, file: UploadFile = File(...), roi: str = Form(...)):
    """Register (or replace) the anchor photo + lesion ROI for a site."""
    parsed = _parse_roi(roi)
    raw = await file.read()

    # The anchor is measured once and every later session inherits its frame,
    # so a bad anchor poisons the whole trend — apply the quality gate strictly.
    if cv_pipeline.ENABLED:
        metrics = cv_pipeline.analyze(raw)
        if not metrics.get('qualityOk'):
            return {'ok': False, 'accepted': False, **metrics}
    else:
        metrics = {'qualityOk': True, 'reasons': [], 'note': 'OpenCV 미설치'}

    try:
        w, h = Image.open(io.BytesIO(raw)).size
    except Exception:
        raise HTTPException(status_code=400, detail='이미지 디코드 실패')

    folder = os.path.join(CAPTURE_DIR, sid)
    os.makedirs(folder, exist_ok=True)
    cid = 'c_' + uuid.uuid4().hex[:10]
    img_path = os.path.join(folder, f'{cid}.jpg')
    with open(img_path, 'wb') as f:
        f.write(raw)

    db.set_site_anchor(sid, img_path, parsed, w, h)

    from datetime import datetime
    date = datetime.now().strftime('%Y-%m-%d %H:%M')
    db.add_capture(cid, sid, date, img_path, metrics, kind='anchor')

    return {'ok': True, 'accepted': True, 'id': cid, 'date': date,
            'roi': parsed, 'width': w, 'height': h, **metrics}


@app.get('/sites/{sid}/anchor/crop')
def anchor_crop(sid: str):
    """Matching target for on-device registration: the ROI neighbourhood of the
    anchor, resized to CROP_WORK, plus the affine that maps it back to anchor pixels.

    Cropping here (instead of on the phone) keeps the device work small and, more
    importantly, normalizes the scale gap between the wide anchor and the close-up
    session photo — ORB is weak across large scale differences and fast-opencv
    ships no SIFT/AKAZE, so this normalization is what makes matching viable.
    """
    a = db.get_site_anchor(sid)
    if not a or not a.get('path') or not a.get('roi') or not os.path.isfile(a['path']):
        raise HTTPException(status_code=404, detail='기준 사진이 아직 등록되지 않았습니다')

    img = Image.open(a['path']).convert('RGB')
    W, H = img.size
    roi = a['roi']
    rx, ry, rw, rh = roi['x'] * W, roi['y'] * H, roi['w'] * W, roi['h'] * H

    cx, cy = rx + rw / 2, ry + rh / 2
    cw, ch = rw * (1 + 2 * ANCHOR_MARGIN), rh * (1 + 2 * ANCHOR_MARGIN)
    x0, y0 = max(0.0, cx - cw / 2), max(0.0, cy - ch / 2)
    x1, y1 = min(float(W), cx + cw / 2), min(float(H), cy + ch / 2)

    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    px, py = crop.size
    s = CROP_WORK / max(px, py)
    crop = crop.resize((max(1, round(px * s)), max(1, round(py * s))), Image.LANCZOS)

    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'cropUrl': f'data:image/jpeg;base64,{b64}',
        'cropWidth': crop.size[0], 'cropHeight': crop.size[1],
        # crop px -> anchor px  (2x3 row-major, consumed by src/ml/affine.js)
        'cropToAnchor': [1 / s, 0, int(x0), 0, 1 / s, int(y0)],
        # lesion rectangle inside the crop — excluded from feature matching
        'roiInCrop': {'x': (rx - int(x0)) * s, 'y': (ry - int(y0)) * s, 'w': rw * s, 'h': rh * s},
        'roi': roi,
        'anchorWidth': W, 'anchorHeight': H,
        'work': CROP_WORK,
    }


@app.post('/quality')
async def quality(file: UploadFile = File(...)):
    """Live preview check: measures brightness/exposure only (not stored). Aspect ratio is judged by the client."""
    raw = await file.read()
    if not cv_pipeline.ENABLED:
        return {'qualityOk': True, 'reasons': [], 'note': 'OpenCV 미설치'}
    try:
        return cv_pipeline.live_quality(raw)
    except Exception as e:
        return {'qualityOk': True, 'reasons': [], 'note': f'분석 생략: {e}'}


@app.get('/captures/dates')
def captures_dates(x_user_id: str = Header(None)):
    """Number of captures per date that has any (for calendar dot markers)."""
    return db.get_capture_dates(_uid(x_user_id))


@app.get('/captures/by-date')
def captures_by_date(date: str, x_user_id: str = Header(None)):
    """List of captures across all sites for the selected date (YYYY-MM-DD)."""
    return db.get_captures_by_date(_uid(x_user_id), date)


@app.get('/captures/{cid}/image')
def capture_image(cid: str):
    """Original capture image (JPEG). Responds as a file so <Image> can cache/load it directly."""
    path = db.get_capture_image_path(cid)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail='이미지 없음')
    return FileResponse(path, media_type='image/jpeg')


# On-device metric fields accepted from the app (src/ml/register.js toCaptureMetrics
# + the on-device severity classifier). Anything else in the payload is ignored.
CLIENT_METRIC_KEYS = {
    'alignGrade', 'alignInliers', 'alignRatio', 'alignReproj', 'alignCoverage',
    'alignRoiCoverage', 'alignReasons', 'areaRatio', 'affine', 'severity',
}


@app.post('/sites/{sid}/captures')
async def add_capture(sid: str, file: UploadFile = File(...), metrics: str = Form(None)):
    """Session capture (close-up).

    Registration and relative-area measurement happen on the device; the server
    stores the result and still runs its own quality/erythema gate so those two
    numbers stay comparable across sessions (same code, same working resolution).
    """
    anchor = db.get_site_anchor(sid)
    if not anchor or not anchor.get('path') or not anchor.get('roi'):
        raise HTTPException(status_code=409,
                            detail='기준 사진이 먼저 등록되어야 합니다 (부위 등록 시 1회 촬영)')

    raw = await file.read()

    # Server-side CV: quality gate + erythema only. Registration moved on-device.
    if cv_pipeline.ENABLED:
        result = cv_pipeline.analyze(raw)
    else:
        result = {'qualityOk': True, 'reasons': [], 'note': 'OpenCV 미설치'}

    # Merge the on-device registration / area / severity metrics
    if metrics:
        try:
            client = json.loads(metrics)
        except Exception:
            raise HTTPException(status_code=400, detail='metrics JSON 파싱 실패')
        result.update({k: v for k, v in client.items() if k in CLIENT_METRIC_KEYS})

    folder = os.path.join(CAPTURE_DIR, sid)
    os.makedirs(folder, exist_ok=True)
    cid = 'c_' + uuid.uuid4().hex[:10]
    img_path = os.path.join(folder, f'{cid}.jpg')
    with open(img_path, 'wb') as f:
        f.write(raw)

    from datetime import datetime
    date = datetime.now().strftime('%Y-%m-%d %H:%M')
    db.add_capture(cid, sid, date, img_path, result, kind='session')

    return {'id': cid, 'date': date, **result}
