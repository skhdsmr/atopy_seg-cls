# DermaScan

AI-assisted skin-lesion **classification & detection** with a long-term, per-body-site **monitoring** workflow, partner-hospital discovery, doctor–patient chat, and symptom records — delivered as a React Native (Expo) mobile app backed by a FastAPI inference + data server.

The project combines two things:

1. **A trained CNN classifier** (MobileNetV4) that predicts one of five skin conditions from a photo and produces a **Grad-CAM** heatmap showing where the model looked.
2. **A classical computer-vision pipeline** (OpenCV, no training required) that powers reproducible **lesion monitoring** — same-framing capture, image quality gating, photo-to-photo registration, and an erythema (redness) index for tracking a lesion over time.

> ⚠️ **Medical disclaimer.** DermaScan is a research/MVP prototype for educational and screening-support purposes only. It is **not a medical device** and must not be used for diagnosis or treatment decisions. Always consult a qualified clinician.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Model](#model)
  - [Architecture & training](#architecture--training)
  - [Inference & Grad-CAM](#inference--grad-cam)
  - [Evaluation](#evaluation)
- [Mobile app](#mobile-app)
- [Backend server](#backend-server)
  - [Monitoring CV pipeline (5 axes)](#monitoring-cv-pipeline-5-axes)
  - [API endpoints](#api-endpoints)
- [Getting started](#getting-started)
- [Large assets & secrets](#large-assets--secrets)
- [Roadmap](#roadmap)

---

## Repository layout

```
hsbioMVP/
├── model/                     # CNN training / evaluation / single-image inference
│   ├── train_mobilenetv4.py   # transfer-learning training loop
│   ├── test_mobilenetv4.py    # test-set evaluation (accuracy, recall, F1, confusion matrix)
│   ├── inference.py           # single-image prediction + Grad-CAM visualization
│   └── best_mobilenetv4.pth   # trained weights (not in Git — see "Large assets")
├── SkinDisNet_balanced/       # dataset (not in Git — see "Large assets")
│   └── test/{CD,EC,OTHERS,SC,TC}/
├── server/                    # FastAPI inference + backend API
│   ├── app.py                 # routes: /predict, monitoring, hospitals, chat, records ...
│   ├── cv_pipeline.py         # classical-CV monitoring pipeline (OpenCV)
│   ├── db.py                  # SQLite data layer
│   ├── kakao.py               # Kakao Local API (nearby dermatology clinics)
│   ├── config.example.py      # copy to config.py and fill in your Kakao REST key
│   └── setup_local.ps1        # Windows → WSL port-forwarding helper
├── dermascan-app/             # React Native (Expo) mobile client
│   └── src/{screens,components,api,data,context,navigation,utils}
└── requirements.txt           # Python dependencies (model + server)
```

---

## Dataset

`SkinDisNet_balanced` is the curated set used for training/evaluation: **4 lesion types + 1 normal
type**. The lesion classes come from **SkinDisNet** (smartphone dermatology photos) and the normal
class (`OTHERS`) from **SCIN** ([Skin Condition Image Network](https://github.com/google-research-datasets/scin),
its *healthy* class). The class order is the alphabetical `ImageFolder` ordering
`['CD', 'EC', 'OTHERS', 'SC', 'TC']` and **must match** across training, evaluation, and inference.

Each class uses **500 images → 350 train / 75 val / 75 test** (2,500 total). Validation and test are
all original images; the **training** split is topped up with augmentation where originals are scarce:

| Class | Condition | Train (original) | Train (augmented) | Val | Test |
|-------|-----------|-----------------:|------------------:|----:|-----:|
| CD | Contact dermatitis | 327 | 23 | 75 | 75 |
| EC | Eczema | 316 | 34 | 75 | 75 |
| SC | Scabies | 193 | 157 | 75 | 75 |
| TC | Tinea corporis (ringworm) | 125 | 225 | 75 | 75 |
| OTHERS | No lesion / normal | 350 | 0 | 75 | 75 |

All images are 512×512 RGB, normalized with the standard ImageNet statistics
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`); no resizing at train time.

> The repo's `SkinDisNet_balanced/` ships the **test split** (75/class, 375 total); the full
> train/val splits are distributed separately (see [Large assets](#large-assets--secrets)).

📥 **Download the dataset:** [SkinDisNet_balanced (Google Drive)](https://drive.google.com/file/d/1Tn8scdVj86Fpim6BIWLxH7lEKrhJGJL3/view?usp=drive_link) — unzip into the project root.

---

## Model

### Why MobileNetV4

- **MobileNet** — Google's CNN family (2017) for mobile/edge devices, built on depthwise separable
  convolutions.
- **MobileNetV4** adds two key ideas:
  - **Universal Inverted Bottleneck (UIB)** — unifies Inverted Bottleneck, ConvNeXt, FFN, and a new
    Extra Depthwise variant into a single block.
  - **Mobile MQA** — an attention block tuned for mobile accelerators.
- Variant used: **MNV4-conv-medium (~9.7M parameters)** via
  [`timm`](https://github.com/huggingface/pytorch-image-models), ImageNet-pretrained — a good fit
  for a model that may eventually run on-device.

### Architecture & training

| Item | Value |
|------|-------|
| Backbone | `mobilenetv4_conv_medium` (timm), ImageNet-pretrained |
| Trainable layers | `conv_head` + `classifier` only (rest frozen — light fine-tuning to curb overfitting) |
| Input | 512×512 RGB, ImageNet normalization, no resize |
| Loss | Cross-entropy |
| Optimizer | AdamW (`lr=1e-4`, `weight_decay=1e-2`) |
| LR schedule | Cosine annealing (`T_max=150`, `eta_min=1e-6`) |
| Batch size | 8 × **gradient accumulation 4** = effective 32 |
| Epochs | 150 |
| Precision | Mixed precision (AMP / `autocast` + `GradScaler`) |
| Checkpointing | Best-by-val-accuracy and best-by-val-macro-F1 saved separately |

Train it with:

```bash
cd model
python train_mobilenetv4.py
```

This expects `SkinDisNet_balanced/{train,val}/` next to the script and writes checkpoints to
`SkinDisNet_Models/best_mobilenetv4_acc.pth` and `..._f1.pth`. The weight shipped to the server,
`model/best_mobilenetv4.pth`, is the selected best checkpoint.

📥 **Download the trained weights:** [best_mobilenetv4.pth (Google Drive)](https://drive.google.com/file/d/1VVVlYONUgVlwfTUstf51Wxg_otJ7HXuS/view?usp=drive_link) — place it under `model/`.

### Inference & Grad-CAM

`inference.py` runs a single image through the model and produces a **Grad-CAM** overlay that
highlights the lesion region the model attended to:

```bash
cd model
python inference.py path/to/image.jpg --out result.png --alpha 0.5 --thr 0.2
```

- Grad-CAM hooks the **last spatial block** (`model.blocks[-1]`), because `conv_head` runs after
  global pooling and would carry no spatial information.
- The heatmap is normalized (clipped at the 99th percentile) and blended proportionally to
  activation, so low-activation areas keep the original photo.
- The same logic is reused server-side by `POST /predict`, which returns class probabilities plus
  a base64 Grad-CAM overlay for the app.

### Evaluation

The best checkpoint (around epoch 93/150) reached **val accuracy 78.93%** and **val macro F1
0.7890**. On the held-out test set (75 images/class):

```bash
cd model
python test_mobilenetv4.py
```

| Metric | Value |
|--------|------:|
| Test loss | 0.9885 |
| Test accuracy | **74.13%** |
| Test macro F1 | **0.7462** |

**Per-class** (recall + F1):

| Class | Recall | F1 |
|-------|-------:|-----:|
| CD | 66.67% | 0.5882 |
| EC | 69.33% | 0.6667 |
| OTHERS | 94.67% | 0.9281 |
| SC | 76.00% | 0.8261 |
| TC | 64.00% | 0.7218 |

**Confusion matrix** (rows = actual, cols = predicted):

| actual ↓ \ pred → | CD | EC | OTHERS | SC | TC |
|-------------------|---:|---:|-------:|---:|---:|
| **CD** | 50 | 16 | 2 | 2 | 5 |
| **EC** | 18 | 52 | 2 | 1 | 2 |
| **OTHERS** | 1 | 2 | 71 | 0 | 1 |
| **SC** | 11 | 4 | 1 | 57 | 2 |
| **TC** | 15 | 7 | 2 | 3 | 48 |

The normal class (`OTHERS`) is separated cleanly (~95% recall); the remaining error is mostly
between the visually similar inflammatory classes **CD ↔ EC**, and **TC** is under-recalled —
consistent with its smaller pool of original images.

---

## Mobile app

`dermascan-app` is an [Expo](https://expo.dev/) (React Native) app. It talks to the backend via
`src/api/client.js`; if `BASE_URL` is empty it falls back to a **mock mode** for most features
(monitoring requires a live server).

| Tab / Flow | Screens | What it does |
|------------|---------|--------------|
| **Diagnosis** | `Diagnosis → Analyzing → Result` | Capture (in-app camera with front/back switch) or pick a photo → **on-device lesion segmentation (TFLite)** → mask overlay + estimated lesion-area ratio. Runs fully offline; no server call. |
| **Monitoring** | `Monitoring → AddSite → AnchorCapture / GuidedCapture / SiteTimeline` | Register a body site, shoot **one wide anchor photo** and mark the lesion ROI on it. Every later session is a **close-up only**, registered on-device against that anchor → **relative lesion area** trend + registration-confidence grade, with IGA/EASI shown alongside. **Calendar view** shows every site's photos for a chosen day |
| **Hospitals** | `Hospitals → HospitalDetail` | Nearby partner dermatology clinics (Kakao Local API), register your clinic, book a visit with a local reminder |
| **Records** | `Records` | Calendar-based daily photo + severity log |
| **Chat** | `ChatList → ChatRoom` | Doctor–patient messaging (patient is billed per message; doctor is free) |
| **Profile** | `Profile` | Account, role (patient/doctor), stats, notifications, billing |

Key app dependencies: `expo-camera`, `expo-image-manipulator`, `expo-image-picker`,
`expo-location`, `expo-notifications`, `react-navigation`, `react-native-calendars`,
`react-native-webview` (Kakao Map).

---

## Backend server

`server/app.py` is a **FastAPI** service that serves both model inference and the app's data
layer (hospitals, chat, records, monitoring). Data is stored in SQLite (`server/dermascan.db`);
the current user is identified by the `X-User-Id` header (login email).

### Monitoring CV pipeline (5 axes)

The monitoring feature is built on a classical-CV pipeline (`cv_pipeline.py`, OpenCV) so that all
measurements are computed identically on the server for trend consistency — **classical CV first,
CNN later**:

| Axis | Concern | Implementation |
|------|---------|----------------|
| 1 | Body-site metadata | Per-site slots (body part / side / condition) — `data/bodyMap.js` |
| 2 | Shape guidance | Anchor ROI crop shown as a semi-transparent framing guide — `GuidedCaptureScreen` |
| 3 | Registration | **Moved on-device** — close-up → anchor, affine + RANSAC (see below) |
| 4 | Quality gate | Brightness, blur (Laplacian variance), and over/under-exposure clipping |
| 5 | Color & scale | Erythema index (`a*` in CIE-Lab); ArUco-marker scale (mm/px) detection skeleton |

A lightweight `live_quality()` (brightness/exposure only, resolution-independent) backs the app's
real-time capture guidance via `POST /quality`.

### Anchor-based registration (on-device)

The wide photo is taken **once, at site registration** (`AnchorCaptureScreen`) together with a
lesion ROI drawn by the user. Every later session is a close-up that is registered against that
anchor frame **entirely on the phone** — no LiDAR/depth sensor, no server round-trip for the math:

1. **Scale normalization** — the server serves the anchor's ROI neighbourhood pre-cropped to 512 px
   (`GET /sites/{id}/anchor/crop`) along with the affine that maps it back to anchor pixels. ORB is
   weak across large scale gaps and `react-native-fast-opencv` ships no SIFT/AKAZE, so shrinking that
   gap up front is what makes matching work at all.
2. **Lesion exclusion** — the lesion changes between sessions and cannot anchor anything, so it is
   masked out of feature detection on both sides: the on-device segmentation mask (dilated) on the
   close-up, the ROI rectangle on the anchor. Only stable surrounding skin is matched.
3. **Affine + RANSAC** — ORB → kNN + Lowe ratio test → RANSAC affine
   ([`src/ml/affine.js`](dermascan-app/src/ml/affine.js), pure JS, seeded so the same photo always
   yields the same number). Homography over-fits on low-texture skin; a similarity transform can't
   absorb tilt.
4. **Confidence, 4 axes + ROI coverage** — inlier count/ratio, RMS reprojection error, transform
   plausibility (scale/aspect/shear/rotation/determinant), inlier spatial spread over a 3×3 grid, and
   how much of the anchor ROI actually made it into frame. Combined by rule into
   **high / medium / low** (thresholds: `REG_TH` in [`src/ml/register.js`](dermascan-app/src/ml/register.js)).
5. **Relative area** — the ROI is walked as a fixed 256×256 grid in *anchor* coordinates, each cell
   inverse-mapped into the close-up and sampled against the segmentation mask. The denominator never
   changes, so the value is comparable across sessions.

`high` is recorded normally, `medium` is recorded but drawn faded ("참고용", lower weight in the
trend), and **`low` is never uploaded** — a mis-registered point corrupts the trend permanently,
which is worse than a missing one. The server's `register_to_reference()` is kept for offline
validation of the on-device numbers against the same image pair.

### API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict` | Image → class probabilities + Grad-CAM overlay _(legacy classifier; the app now segments on-device and no longer calls this)_ |
| `GET` | `/health` | Health check (reports device: cpu/cuda) |
| `GET` | `/hospitals?lat=&lng=` | Nearby partner clinics, sorted by distance |
| `POST`/`DELETE` | `/hospitals/{id}/register` | Register / unregister your clinic |
| `GET`/`POST` | `/records` | Daily photo + severity log |
| `GET` | `/chats` · `POST /chats/{id}/messages` | Chat (patient billed per message) |
| `GET` | `/billing` | Accumulated charges |
| `POST` | `/appointments` | Record a visit booking |
| `GET`/`POST` | `/sites` · `DELETE /sites/{id}` | Monitoring sites (CRUD) |
| `GET` | `/sites/{id}/captures` | Captures for a site (area, registration grade, severity) |
| `GET` | `/sites/{id}/reference` | Anchor image + dimensions + lesion ROI |
| `POST` | `/sites/{id}/anchor` | Register the anchor photo + ROI (multipart: `file`, `roi`) — quality gate is strict here |
| `GET` | `/sites/{id}/anchor/crop` | ROI crop (512 px) + `cropToAnchor` affine — the on-device matching target |
| `POST` | `/sites/{id}/captures` | Upload a session capture (multipart: `file`, `metrics` = on-device registration/area/severity JSON). 409 if the site has no anchor yet |
| `POST` | `/quality` | Live preview check (brightness/exposure only; not stored) |
| `GET` | `/captures/dates` | Dates that have captures (calendar dots) |
| `GET` | `/captures/by-date?date=` | All sites' captures for one day |
| `GET` | `/captures/{id}/image` | Capture image (JPEG) |

---

## Getting started

### 1. Python environment (model + server)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# PyTorch with CUDA: install the matching build from https://pytorch.org
```

### 2. Run the inference/backend server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The mobile app reaches the server over the local network. On Windows + WSL, the API runs inside
WSL while the phone connects through the PC's Wi-Fi IP; `server/setup_local.ps1` (run as
Administrator) sets up the WSL→Windows port forwarding and prints the `BASE_URL` to use. See
[`server/README.md`](server/README.md) for the full local-network walkthrough.

### 3. Run the app

```bash
cd dermascan-app
npm install
npx expo start            # add --tunnel if the phone can't reach Metro over LAN
```

Open it with **Expo Go** (scan the QR), or press `a` / `i` / `w`. Set
`dermascan-app/src/api/client.js` → `BASE_URL` to the address printed by the setup script
(e.g. `http://<PC-IP>:8000`). The server is still required for **monitoring, hospitals,
chat, and records** — only the Diagnosis segmentation runs on-device.

### 4. On-device segmentation (TFLite)

The Diagnosis flow no longer calls the server. It runs a lesion-segmentation model
(**U-Net++ / EfficientNet-b0, 256×256, sigmoid**) fully on-device via
[`react-native-fast-tflite`](https://github.com/mrousavy/react-native-fast-tflite).

On-device **registration** additionally needs OpenCV via
[`react-native-fast-opencv`](https://github.com/lukaszkurantdev/react-native-fast-opencv)
(ORB / BFMatcher only — the affine+RANSAC estimator is our own JS, since the library exposes
no `estimateAffine2D`). Install it before building the dev client:

```bash
cd dermascan-app
npx expo install react-native-fast-opencv
```

All library calls are isolated in [`src/ml/cv.js`](dermascan-app/src/ml/cv.js); if a version bump
changes the object-creation API, that adapter is the only file to touch.

Because TFLite and OpenCV are native modules, **Expo Go cannot run them** — you need a custom
dev build (`expo-dev-client`). Two ways:

**A. Local machine with Android Studio / Xcode**

```bash
cd dermascan-app
npm install
npx expo run:android              # or: npx expo run:ios  (Android SDK + device/emulator required)
```

**B. EAS Build (cloud — for a headless server with no Android SDK)**

```bash
cd dermascan-app
npm install
npx expo install expo-dev-client  # (already in package.json)
eas login                         # free Expo account (expo.dev)
eas init                          # links the project (writes extra.eas.projectId)
eas build --platform android --profile development   # cloud-builds an installable APK
```

The build finishes with a URL/QR — download the APK and install it on your phone
(allow "install unknown apps"). Then serve the JS bundle from your dev machine and
scan the QR **from inside the installed dev-client app**:

```bash
npx expo start --dev-client --tunnel   # --tunnel: phone and server need not share a LAN
```

The Diagnosis (segmentation) flow runs **fully on-device**, so it works even without
the backend running. `eas.json` ships a ready `development` profile (internal APK).

**Model zoo & selection** — everything lives under `app/`. Put any number of
segmentation checkpoints in `app/model/checkpoint/*.pth` (the filename becomes the
model's name), convert them all at once, then pick which one the app runs by editing
a single line.

```bash
# one-time toolchain
pip install -r app/requirements-convert.txt

# convert EVERY app/model/checkpoint/*.pth -> assets/models/<name>.tflite + <name>.json
#   and (re)generate the model registry src/ml/models.gen.js
python app/model/export_tflite.py --batch [--sample /path/to/any.png]
```

Each `.pth` becomes a `<name>.tflite` (**float32 I/O** — required, see below) plus a
`<name>.json` (imgsz/encoder/normalization, read automatically from the checkpoint).
The converter then writes `src/ml/models.gen.js`, a static `require()` registry of
every bundled model.

**Selecting the active model** — edit one line in
[`src/ml/modelSelect.js`](dermascan-app/src/ml/modelSelect.js):

```js
export const ACTIVE_MODEL = 'effb0_unetpp_256';   // ← name from app/model/checkpoint
```

Reload the app (no dev-build rebuild, no re-conversion) and it runs that model,
auto-resizing input to the checkpoint's resolution. The result screen shows the
active model name + input size and its on-device latency/memory.

Notes:
- **float32 I/O is mandatory**: `react-native-fast-tflite` can only feed a
  `Float32Array`; a pure float16-I/O model (onnx2tf's `*_float16.tflite`) would
  size-mismatch and **crash the app natively**. The converter ships float32 I/O and
  asserts the dtype before finishing.
- **Every bundled model ships inside the APK.** Each EfficientNet-b0 model is ~24 MB
  regardless of resolution, so bundling many inflates the APK — keep only the `.pth`
  you actually want to compare in `app/model/checkpoint/`.
- `.tflite` files are git-ignored (generated artifacts); regenerate with `--batch`.
- Both **smp** models (U-Net++/U-Net/DeepLab/HRNet…) and **UNeXt** convert. UNeXt is a
  custom (non-smp) architecture — its definition lives self-contained at
  [`app/model/unext_arch.py`](model/unext_arch.py) and the converter auto-detects it
  from the checkpoint (keys `encoder1.*`/`final.*`), reading the input size from the
  filename's trailing number (e.g. `unext_512` → 512) when the checkpoint has no
  `args`. UNeXt is much lighter (~6 MB vs ~24 MB for EfficientNet-b0).

---

## Large assets & secrets

The following are **git-ignored** and must be obtained / created separately:

| Item | Why excluded | How to get it |
|------|--------------|---------------|
| `model/best_mobilenetv4.pth` | Binary model artifact | [Download from Google Drive](https://drive.google.com/file/d/1VVVlYONUgVlwfTUstf51Wxg_otJ7HXuS/view?usp=drive_link) and place it under `model/` |
| `SkinDisNet_balanced/` | Dataset | [Download from Google Drive](https://drive.google.com/file/d/1Tn8scdVj86Fpim6BIWLxH7lEKrhJGJL3/view?usp=drive_link) and unzip into the project root |
| `server/dermascan.db`, `server/data/captures/` | Runtime data / user uploads (privacy) | Auto-created by the server |
| `server/config.py`, `dermascan-app/src/config.js` | Contain API keys | Copy the matching `*.example.*` file and fill in your own keys |

After downloading, the layout should be `model/best_mobilenetv4.pth` and
`SkinDisNet_balanced/test/{CD,EC,OTHERS,SC,TC}/...` (plus `train/` and `val/` if you intend to train).

**API keys.** Kakao keys are read from the (git-ignored) config files:

```bash
cp server/config.example.py server/config.py                 # set KAKAO_REST_KEY (or use the env var)
cp dermascan-app/src/config.example.js dermascan-app/src/config.js   # set KAKAO_JS_KEY
```

Get a Kakao **REST API key** (server) and **JavaScript key** (map) from
[Kakao Developers](https://developers.kakao.com). For the JS key, register the map's
`KAKAO_MAP_BASE_URL` as a Web platform site domain.

---

## Roadmap

- **Non-face body sites**: the segmentation model is currently trained on face-only data
  (AI Hub 안면부 피부질환), so monitoring is limited to facial sites until the separately-trained
  body model is available. Registration/area code is site-agnostic and needs no change.
- **Learned matcher fallback**: sessions that fail ORB matching (smooth, low-texture skin) currently
  have no rescue path on-device. A pretrained matcher (SuperPoint/LightGlue) would need TFLite
  conversion; a printed marker fallback is the cheaper alternative.
- **Confidence classifier**: replace the rule-based grade with a small model once enough
  (metrics → was-it-actually-right) pairs have accumulated.
- **Itch annotation**: let users mark the actually-itchy spot on a captured lesion, so lesion
  location / severity can be refined after the fact.
- **Scale measurement**: complete the ArUco mm/px calibration for absolute (mm²) lesion size —
  the current area metric is relative to the anchor ROI.
- **CNN-assisted monitoring**: layer learned models on top of the classical-CV foundation.
- **Long-term goal**: per-site longitudinal monitoring of uremic pruritus and radiation dermatitis.

---

_Built as an MVP. Contributions and issues welcome._
