# DermaScan (React Native / Expo)

MVP app for AI-based skin-lesion detection/classification, per-site lesion monitoring, and clinic
linkage.

## Screens

| Tab | Screens | Features |
|-----|---------|----------|
| (Login) | `LoginScreen` | Google / Naver / Kakao social login (mock) |
| Diagnosis | `Diagnosis → Analyzing → Result` | In-app camera (front/back switch) or gallery upload → AI analysis → top conditions (≥10%) with symptoms + Grad-CAM overlay |
| Monitoring | `Monitoring → AddSite / GuidedCapture / SiteTimeline` | Register body sites; capture each with a ghost overlay / outline guide and live brightness·exposure·aspect-ratio guidance; track quality/erythema over time; calendar view of all sites' photos by day |
| Hospitals | `Hospitals → HospitalDetail` | Nearby partner-clinic recommendations, register your clinic, visit-booking reminders |
| Records | `Records` | Calendar-based daily photo / severity log |
| Chat | `ChatList → ChatRoom` | Doctor–patient chat (doctor/patient split; patient billed per message) |
| Profile | `Profile` | Account / stats / notifications / billing / logout |

## Running

```bash
cd dermascan-app
npm install
npx expo start
```

- Scan the QR with **Expo Go** on your phone, or press `a` (Android) / `i` (iOS) / `w` (web).
- Camera, gallery, and notifications work on a real device.

## Configuration (Kakao map key)

The hospital map uses the Kakao Maps JS SDK. Copy the template and add your key (the real
`config.js` is git-ignored):

```bash
cp src/config.example.js src/config.js   # set KAKAO_JS_KEY
```

Get a **JavaScript key** from [Kakao Developers](https://developers.kakao.com) and register the
`KAKAO_MAP_BASE_URL` value as the Web-platform site domain. Without a valid key, the map shows a
placeholder.

## Backend (inference server) connection

By default `src/api/client.js` points `BASE_URL` at a server. Most features have a **mock fallback**
when `BASE_URL` is empty, but **monitoring requires a live server**.

1. Run the inference server (from the project root):
   ```bash
   pip install fastapi uvicorn python-multipart
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```
2. Set `BASE_URL` in `src/api/client.js` to your PC IP:
   ```js
   export const BASE_URL = 'http://192.168.0.10:8000'; // PC on the same Wi-Fi
   ```

The server reuses the `model/inference.py` logic (MobileNetV4 + Grad-CAM) and returns class
probabilities and a base64 Grad-CAM overlay via `POST /predict`. See
[`../server/README.md`](../server/README.md) for the full local-network setup.

## Swap points for real integration (currently mock)

- **OAuth**: `src/context/AuthContext.js` → expo-auth-session (Google), Kakao/Naver SDK
- **Inference / hospitals / chat / records API**: `src/api/client.js` → real server calls when `BASE_URL` is set
- **Map**: Kakao Maps via WebView (`src/components/KakaoMap.js`)
- **Visit reminders**: `src/utils/notifications.js` (expo-notifications, already working)

## Patient vs doctor

Branches on `AuthContext`'s `user.role` (`'patient' | 'doctor'`). In chat, patients see a billing
confirmation modal on send (500 KRW per message) while doctors send for free (`ChatRoomScreen`).
