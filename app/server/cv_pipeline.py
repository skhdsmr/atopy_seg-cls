"""Classical computer vision pipeline (OpenCV) — no trained model required.

Axis 4) Quality gate: brightness / blur / exposure
Axis 3) Post-hoc registration: ORB features + homography
Axis 5) Color & scale: erythema index (a*), reference (ArUco) scale detection skeleton

All measurements are computed with the same code on the server for trend consistency.

NOTE: session registration now runs on the device (dermascan-app/src/ml/register.js:
close-up -> anchor, affine + RANSAC + 4-axis confidence). register_to_reference below
is kept for server-side validation only — analyze() no longer calls it in the request
path. Use it to sanity-check the on-device numbers against the same image pair.
"""
import numpy as np

try:
    import cv2
    ENABLED = True
except Exception:
    ENABLED = False

# Working resolution (always normalized to the same size for measurement consistency)
WORK_SIZE = 1024

# Quality thresholds (empirical defaults — to be tuned as data accumulates)
TH = {
    'brightness_min': 50,    # Lower bound for mean luminance (too dark)
    'brightness_max': 210,   # Upper bound for mean luminance (too bright)
    'blur_min': 80.0,        # Lower bound for Laplacian variance (blurry below this)
    'clip_max': 0.15,        # Upper bound for over/underexposed clipped pixel ratio
}


def _to_work(img_bgr):
    h, w = img_bgr.shape[:2]
    s = WORK_SIZE / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (round(w * s), round(h * s)))
    return img_bgr


def decode(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError('이미지 디코드 실패')
    return _to_work(img)


# ------------------------------------------------------------
# Axis 4) Quality gate
# ------------------------------------------------------------
def quality_metrics(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # Sharpness (higher = crisper)
    total = gray.size
    under = float((gray < 10).sum()) / total
    over = float((gray > 245).sum()) / total
    clip = under + over

    reasons = []
    if brightness < TH['brightness_min']:
        reasons.append('너무 어두움')
    if brightness > TH['brightness_max']:
        reasons.append('너무 밝음')
    if blur < TH['blur_min']:
        reasons.append('흐릿함(초점)')
    if clip > TH['clip_max']:
        reasons.append('노출 과다/부족')

    return {
        'brightness': round(brightness, 1),
        'blur': round(blur, 1),
        'clip': round(clip, 3),
        'qualityOk': len(reasons) == 0,
        'reasons': reasons,
    }


# ------------------------------------------------------------
# Lightweight check for live preview: brightness/exposure only (resolution-independent → low-res snapshots OK)
#   Blur is sensitive to resolution, so it is excluded in real time and judged only on the actual capture.
# ------------------------------------------------------------
def live_quality(image_bytes):
    img = decode(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    total = gray.size
    under = float((gray < 10).sum()) / total
    over = float((gray > 245).sum()) / total
    clip = under + over

    reasons = []
    if brightness < TH['brightness_min']:
        reasons.append('너무 어두움')
    if brightness > TH['brightness_max']:
        reasons.append('너무 밝음')
    if clip > TH['clip_max']:
        reasons.append('노출 과다/부족')

    return {
        'brightness': round(brightness, 1),
        'clip': round(clip, 3),
        'qualityOk': len(reasons) == 0,
        'reasons': reasons,
    }


# ------------------------------------------------------------
# Axis 5) Erythema index (a* in Lab = red-green). Higher = redder.
# ------------------------------------------------------------
def erythema_index(img_bgr, mask=None):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    a = lab[:, :, 1].astype(np.float32) - 128.0  # 0~255 → centered at 0
    if mask is not None:
        vals = a[mask > 0]
    else:
        # Central 60% region only (reduces edge/background influence)
        h, w = a.shape
        vals = a[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)].ravel()
    return round(float(vals.mean()), 2)


# ------------------------------------------------------------
# Axis 3) Post-hoc registration: warp the current photo into the reference coordinate frame
# ------------------------------------------------------------
def register_to_reference(ref_bgr, cur_bgr):
    ref = _to_work(ref_bgr)
    cur = _to_work(cur_bgr)
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return {'aligned': cur, 'confidence': 0.0, 'inliers': 0}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)
    if len(matches) < 12:
        return {'aligned': cur, 'confidence': 0.0, 'inliers': len(matches)}

    src = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return {'aligned': cur, 'confidence': 0.0, 'inliers': 0}

    inliers = int(inlier_mask.sum())
    aligned = cv2.warpPerspective(cur, H, (ref.shape[1], ref.shape[0]))
    # Registration confidence: inliers / number of matches (0~1)
    conf = round(inliers / max(len(matches), 1), 2)
    return {'aligned': aligned, 'confidence': conf, 'inliers': inliers}


# ------------------------------------------------------------
# Axis 5) Scale reference detection (ArUco marker) — skeleton
#   Knowing the real side length of a printed ArUco marker enables mm/pixel conversion.
# ------------------------------------------------------------
def detect_scale(img_bgr, marker_mm=20.0):
    if not hasattr(cv2, 'aruco'):
        return None
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, adict)
        if ids is None or len(corners) == 0:
            return None
        c = corners[0].reshape(4, 2)
        side_px = np.linalg.norm(c[0] - c[1])
        if side_px <= 0:
            return None
        return round(marker_mm / float(side_px), 4)  # mm per pixel
    except Exception:
        return None


# ------------------------------------------------------------
# Full analysis (called when a capture is uploaded)
# ------------------------------------------------------------
def analyze(cur_bytes, ref_bytes=None, marker_mm=20.0):
    cur = decode(cur_bytes)
    q = quality_metrics(cur)

    target = cur
    registration = None
    if ref_bytes is not None:
        ref = decode(ref_bytes)
        reg = register_to_reference(ref, cur)
        registration = {'confidence': reg['confidence'], 'inliers': reg['inliers']}
        # If registration is confident enough, measure on the aligned image
        if reg['confidence'] >= 0.25:
            target = reg['aligned']

    result = {
        **q,
        'erythema': erythema_index(target),
        'scaleMmPerPx': detect_scale(cur, marker_mm),
        'registration': registration,
    }
    return result
