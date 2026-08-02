# ImSwitch Static Control

A **zero-build** HTML/JS control panel for an ImSwitch microscope. Use it when you
don't have the compiled React frontend available — just open one file in a browser.

It talks to the same FastAPI backend as the React app, using the convention
`{protocol}://{host}:{httpPort}/imswitch/api/...`.

## What it can do

- **Livestream** via MJPEG (`<img>` tag, no WebSocket needed)
- Enter **IP / HTTP port / socket port** (persisted in `localStorage`)
- **Stage** control — X/Y/Z/A jog, step size, speed, absolute & relative moves, home, stop
- **Illumination** — channels are queried dynamically (`getLaserNames`) with per-channel
  on/off + intensity slider (range from `getLaserValueRanges`)
- **Snap & download** an image (PNG) and **snap to disk** (TIFF/PNG/…)
- Basic **autofocus** (range / resolution / defocus / algorithm)
- **Objective** switching (slots), speed, calibration
- **Load & display** the current setup JSON, with download
- **LED matrix** — all on/off, intensity, ring, halves, single-LED grid
- **ESP32** — reconnect, restart, connection check, and **connect the PS controller
  via `ps_act`** (editable JSON sent through `writeSerial`)

## Run it

No compilation. Either:

```bash
# from this directory
python3 -m http.server 8080
# then open http://localhost:8080
```

or just double-click `index.html` (serving via http.server is recommended so the
browser treats it as a normal web origin).

Then in the top bar set **Host**, **HTTP port** (default `8001`) and press
**Connect / Test**.

## Notes

- The backend enables CORS (`allow_origins=["*"]`), so a page served from
  `localhost:8080` can call the API on another host/port.
- **HTTPS / self-signed certs:** if ImSwitch runs with SSL, set protocol to `https`,
  then open `https://<host>:<port>/imswitch/api/version` once in a new tab and accept
  the certificate, otherwise the browser silently blocks the requests.
- The **socket port** field is stored for reference; this client intentionally uses
  only HTTP + MJPEG and does not open a WebSocket.
- The **LED matrix** and some device features only respond if the corresponding
  device is configured in the active setup; otherwise the panel shows "unavailable".
- The **`ps_act`** payload is fully editable — adjust the JSON if your UC2 firmware
  expects a different schema.
- Open the **Activity log** (bottom dock) to see every request/response.
