# Kiosk / touchscreen UI (`#/mobile`)

A reduced, touch-first shell for the FRAME microscope, designed for the
Raspberry Pi DSI touchscreen running chromium in kiosk mode. Bambu Lab /
Formlabs-style: near-black surfaces, one lime accent, large touch targets,
one page at a time behind a left icon rail.

## URL

The SPA is served by a plain `StaticFiles` mount (no SPA fallback), so the
kiosk route is hash-based and survives hard reloads without backend changes:

```
http://localhost:8001/imswitch/ui/index.html#/mobile
```

(or via the proxy: `http://<host>/imswitch/ui/index.html#/mobile`)

Point the chromium kiosk service at that URL, e.g.:

```
chromium-browser --kiosk --noerrdialogs --disable-infobars \
  --disable-pinch --overscroll-history-navigation=0 \
  "http://localhost:8001/imswitch/ui/index.html#/mobile"
```

The "Full UI" button at the bottom of the rail navigates back to `#/`, which
renders the regular desktop SPA (same bundle, same session).

## Pages

| Hash | Page | Backing APIs |
|---|---|---|
| `#/mobile` | Home: static GLB digital twin (live positions), status column, quick nav | `Frame3DViewer`, PositionSlice, UC2Slice, StorageSlice, `getMicroscopeStandName`, `getBoardTemperature` |
| `#/mobile/stage` | XY jog pad, Z column with ZEN-style focus view (tap to seek), safe homing, stop | `movePositionerXYZ`, `startFrameHoming`, `stopAllAxes`, HomingSlice |
| `#/mobile/lasers` | Per-laser on/off + intensity (sent on release) | ParameterRangeSlice (`getHardwareParameters`), `setLaserValue/Active` |
| `#/mobile/leds` | LED matrix patterns (all/ring/circle/halves), brightness, off | `LEDMatrixController/*` |
| `#/mobile/camera` | Basic MJPEG live stream, detector picker, snap | `LiveViewController/mjpeg_stream`, `startLiveView`, `startSnap` |
| `#/mobile/objective` | Slot cards with INSERTED state, tap to swap | `ObjectiveController/getstatus`, `moveToObjective` |
| `#/mobile/wifi` | Device-admin internet panel (iframe) | external `{host}/admin/panel/internet` |
| `#/mobile/system` | ESP32 info/actions, one-button master firmware update, storage, versions | `UC2ConfigController/*`, usbFlashSlice, StorageSlice, `/version` |

Pages whose backend controller is missing in the active setup are hidden from
the rail (`BackendCapabilitiesSlice`).

## Structure

- `MobileApp.jsx` — shell (rail + page switch), rendered by `App.jsx` when the
  hash matches; lives inside the existing Redux/WebSocket/Snackbar providers,
  so there is exactly one socket connection.
- `mobileTheme.js` — dedicated dark kiosk theme (do not reuse the desktop
  themes: those are density-optimised, the opposite of touch-friendly).
- `mobileRoutes.js` — tiny hash router (`useMobileRoute`).
- `components/` — page scaffold, status tiles, touch buttons, placeholder
  frames for explanatory images.
- `pages/` — one file per rail entry.

Live state (stage position, laser power, homing, storage, bus/E-stop, USB
flash progress) arrives through the existing `WebSocketHandler` signals — the
kiosk never polls for anything that is pushed.
