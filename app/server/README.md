# DermaScan inference & backend server (local-network setup)

Exposes the real MobileNetV4 model + backend (hospitals / records / chat / appointments /
monitoring) so the app (`dermascan-app`) can connect over a **local network** — direct access on the
same network, no public tunnel.

```
phone ── same network (hotspot/Wi-Fi) ──> PC Wi-Fi IP:8000 ──(port forward)──> WSL FastAPI:8000
```

The model / torch live in the WSL `.venv`, so **run the server inside WSL**. Because WSL2 is an
isolated virtual network, port 8000 arriving on Windows has to be forwarded into WSL.

## 1. Run the server (WSL terminal)

```bash
cd /mnt/c/hsbioMVP
source .venv/bin/activate
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

When `Uvicorn running on http://0.0.0.0:8000` appears it's ready. Leave this window open.

## 2. Port forwarding + firewall (Administrator PowerShell, once)

In an **Administrator** PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File c:\hsbioMVP\server\setup_local.ps1
```

Note the printed `app BASE_URL: http://<PC-IP>:8000`.

- Re-run this whenever the WSL IP changes — i.e. after `wsl --shutdown` or a PC reboot.

## 3. Set the app's BASE_URL

Set `BASE_URL` in `dermascan-app/src/api/client.js` to the address printed in step 2:

```js
export const BASE_URL = 'http://10.221.101.98:8000'; // your PC's current Wi-Fi IP
```

## 4. Connect the phone + run the app

- Put the phone and PC on the **same network** (connect the PC to the phone's hotspot, or use the
  same router).
- Start the app (PowerShell):
  ```powershell
  cd c:\hsbioMVP\dermascan-app
  npx expo start --tunnel --clear
  ```
  (Metro can connect over tunnel or LAN; the API still goes directly to `BASE_URL` above.)

## Health check

```bash
bash /mnt/c/hsbioMVP/server/test_predict.sh   # calls local :8000 directly from inside WSL
```

From a browser on the same network, `http://<PC-IP>:8000/health` returning 200 means the
connection works.

## Configuration (Kakao key)

Nearby-clinic search uses the Kakao Local API. Copy the template and add your REST key (the real
`config.py` is git-ignored):

```bash
cp server/config.example.py server/config.py     # set KAKAO_REST_KEY, or use the env var
```

If no valid key is present, `kakao.py` disables itself and `app.py` falls back to the seeded
hospital list.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict` | Image → class probabilities + Grad-CAM overlay |
| `GET` | `/health` | Health check (reports device: cpu/cuda) |
| `GET` | `/hospitals?lat=&lng=` | Nearby partner clinics, sorted by distance |
| `POST`/`DELETE` | `/hospitals/{id}/register` | Register / unregister your clinic |
| `GET`/`POST` | `/records` | Daily photo + severity log |
| `GET` | `/chats` · `POST /chats/{id}/messages` | Chat (patient billed 500 KRW per message) |
| `GET` | `/billing` | Accumulated charges |
| `POST` | `/appointments` | Record a visit booking |
| `GET`/`POST` | `/sites` · `DELETE /sites/{id}` | Monitoring sites (CRUD) |
| `GET` | `/sites/{id}/captures` | Captures for a site |
| `GET` | `/sites/{id}/reference` | Reference (ghost) image + dimensions |
| `POST` | `/sites/{id}/captures` | Upload a capture → quality / erythema / registration metrics |
| `POST` | `/quality` | Live preview check (brightness/exposure only; not stored) |
| `GET` | `/captures/dates` | Dates that have captures (calendar dots) |
| `GET` | `/captures/by-date?date=` | All sites' captures for one day |
| `GET` | `/captures/{id}/image` | Capture image (JPEG) |

The current user is identified by the `X-User-Id` header (login email); data is stored in
`dermascan.db` (SQLite).

## Files

- `app.py` — FastAPI server (inference + backend API)
- `cv_pipeline.py` — classical-CV monitoring pipeline (OpenCV)
- `db.py` — SQLite data layer + partner-hospital seed
- `kakao.py` — Kakao Local API client
- `config.example.py` — template; copy to `config.py` and add your key
- `setup_local.ps1` — Windows→WSL port forwarding / firewall setup (Administrator)
- `test_*.sh` — local smoke tests (`predict`, `kakao`, `monitor`)
- `dermascan.db` — SQLite DB (auto-created / updated)

## Notes

- If a GPU is available it is used automatically (check `device` in `/health`).
- To return to mock mode, set `BASE_URL = null` in `client.js` (note: monitoring requires a live
  server and has no mock).
- For production, deploy the server to a cloud host (Render / Railway / etc.) for a stable URL.
