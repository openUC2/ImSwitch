# §4.B implementation notes

What landed, what's deferred, and what the frontend needs to do. Sequel to [STREAMING_REVIEW.md](STREAMING_REVIEW.md).

## Files touched

| File | What changed |
|---|---|
| `imswitch/imcontrol/controller/controllers/LiveViewController.py` | New `_JpegEncoder` helper (PyTurboJPEG → cv2 fallback); `StreamWorker._u16_to_u8_no_alloc` preallocated-buffer cast; JPEG/MJPEG workers use new encoder; WebRTC pre-resize moved into producer; `recv()` simplified to a wrap-only path. |
| `imswitch/imcontrol/model/interfaces/hikcamera.py` | `setBinning` rewritten (BinningSelector + GenICam standard names + legacy fallback + safe stop/resume); new `_trySetSdkFlip` (ReverseX/ReverseY); callback's `np.flip` gated on `_sdk_flip_active`. |
| `imswitch/imcontrol/model/managers/detectors/HikCamManager.py` | `setFlipImage` now also calls `_trySetSdkFlip`. |
| `imswitch/imcontrol/model/io/recording_service.py` | New dispatch thread + bounded queue (size 8) for the per-frame writer dispatch; `_on_new_frame` only copies + enqueues; `_append_tiff` keeps `TiffWriter` open per filepath and writes with zlib level 1. |
| `imswitch/imcontrol/model/writers/ome_tiff_writer.py` | `OMETiffWriter` ctor accepts `compression`/`compression_level` (default `"zlib"` level 1); `tif.write` passes them through; older-tifffile compat for `compressionargs`. |
| `imswitch/imcommon/framework/noqt.py` | `_handle_stream_frame` no longer `.copy()`'s the metadata dict per client (in-place `frame_id` mutation inside the per-frame loop). Bigger "pack once" win is deferred — needs a wire-format change. |

## What landed (§4.B)

### 4.B.8 — preallocated u8/u16 buffers
- New `StreamWorker._u16_to_u8_no_alloc(src)` helper. Two preallocated buffers per worker; reallocated only when the post-crop+subsample shape changes. Used by JPEG, MJPEG and WebRTC workers.
- Two passes over the data, zero per-frame allocations on the hot path.

### 4.B.9 — PyTurboJPEG with cv2 fallback
- New module-level `_JpegEncoder` class. On construction:
  - tries `from turbojpeg import TurboJPEG, TJPF_BGR, TJPF_GRAY` → uses libjpeg-turbo;
  - on any failure (ImportError / OSError / version mismatch) silently falls back to cv2.
- If a turbojpeg encode call fails at runtime, the encoder permanently demotes to cv2 for the rest of the session (one-shot warning log) — no per-frame fallback overhead.
- Both `JPEGStreamWorker` and `MJPEGStreamWorker` go through it.
- To enable: `pip install PyTurboJPEG` (or `pip install PyTurboJPEG[turbojpeg]`). On Debian/Pi 5 also: `sudo apt install libturbojpeg`.

### 4.B.10 — WebRTC pre-resize at the producer
- All the heavy work (dtype cast, crop, subsample, grayscale→RGB, target-size resize, even-dim alignment) moved from `recv()` into `_captureAndEmit`. The queue now holds rgb24 frames already sized for the encoder.
- `recv()` is now ~15 lines and just wraps the queued frame in `av.VideoFrame`. The aiortc event loop no longer competes with cv2 for CPU.
- Hard 720p cap added in the producer so software libx264 on the Pi 5 isn't asked to encode 9 MP frames.

### 4.B.11 — pack socket.io payload once (partial)
- The cheap part landed: `_handle_stream_frame` no longer makes a per-client `metadata.copy()`. The hot-loop now just mutates `frame_id` on the shared metadata dict (the loop is the only writer and runs serially per frame).
- The expensive part — sharing the JPEG bytes across all clients in a single msgpack pack — is deferred. It would need a wire format change (e.g. `[4-byte frame_id LE][msgpack header][image bytes]`) and a matching frontend update. Given the user has a single viewer (laptop), the current per-client `msgpack.packb` cost is approximately one 500 KB memcpy per emit, which is dominated by socket.io's own send. Revisit if multi-viewer becomes common.

### 4.B.12 — recording writes off the StreamWorker thread
- New private `_dispatch_thread` + bounded `queue.Queue(maxsize=8)` inside `RecordingService`.
- `_on_new_frame` (called on the StreamWorker thread via `sigUpdateImage`):
  1. early-return if no recorder is active (no copy paid);
  2. `frame.copy()` (~5 ms for 18 MB) to release the SDK-owned ring-buffer reference;
  3. `put_nowait` on the dispatch queue.
- The dispatch thread runs `MP4Writer.write_frame(...)` and `StreamingDataStoreAdapter.write_frame(...)` — so the synchronous H.264 encode and the streaming-adapter copy no longer block live preview.
- On queue overflow we drop the oldest pending frame (single drop, log every 100). Trade-off: prefers smooth preview over guaranteed recording rate. If you'd rather block when overrun, change to `put(block=True, timeout=...)`.

### 4.B.13 — TiffWriter kept open + cheap compression
- `OMETiffWriter` already kept a per-detector writer open inside `_writer_loop` — that part was fine.
- Added a `compression` kwarg with default `"zlib"` (level 1). On microscopy data this typically halves file size at ~1 GB/s — usually faster end-to-end than uncompressed because SD/USB is the bottleneck.
- Fallback for older `tifffile` versions that don't accept `compressionargs`.
- `BackgroundStorageWorker._append_tiff` (the path for `task_type='append_tiff'`) used to be `tiff.imwrite(append=True)` per call (open, walk IFD, append, close). Now it keeps a `tifffile.TiffWriter` per filepath in `self._tiff_appenders`, closes them all on `stop()`. Also defaults to zlib-1 compression.

### 4.B.16 — HIK SDK-side flip
- `CameraHIK._trySetSdkFlip(flipY, flipX)` pushes `ReverseY`/`ReverseX` via `MV_CC_SetBoolValue`. On success sets `_sdk_flip_active = True` so the callback skips `np.flip`. On rejection the callback's software path runs as before.
- `HikCamManager.setFlipImage` now calls this after updating `_camera.flipImage`.

### Bonus — `setBinning` fix
- `0x80000100` (MV_E_GC_GENERIC) on `BinningX`/`BinningY` was the camera model rejecting the legacy feature names. The MV-CS200-10UC follows the GenICam SFNC: `BinningHorizontal`/`BinningVertical`, gated by `BinningSelector` and optionally `BinningHorizontalMode`/`BinningVerticalMode`.
- New flow:
  1. stop streaming;
  2. `BinningSelector = "Sensor"` (if supported);
  3. `BinningHorizontalMode = BinningVerticalMode = "Sum"` (if supported);
  4. `BinningHorizontal`/`BinningVertical = N`, falling back to `BinningX`/`BinningY` if the new names are rejected;
  5. re-read `WidthMax`/`HeightMax` and update `Width`/`Height`;
  6. resume streaming.
- Both H and V failures are now logged as a single warning rather than the ambiguous pair.

## §4.C round (current)

> §4.C.17 (two-stage preview pipeline) was reverted on `main` and moved to the `fix/streaminglatency-twostep` branch. Notes for that change live with that branch.

### `setStreamParameters` no longer restarts streams

The old behaviour of stopping + starting the StreamWorker on every settings change tore down the WebRTC peer connection (which is bound to the worker's video track). The "had to apply twice" symptom: first apply restarted the worker but left the frontend with a stale RTC peer; second apply re-offered the SDP.

Now `setStreamParameters` mutates `worker._params` in-place. The worker reads each field on every iteration anyway (`throttle_ms`, `subsampling_factor`, `crop_size`, `max_width`). Side effect: the preview pipeline's subsample factor is updated live via `_sync_preview_sub()` without restarting the camera-side preview thread.

### `setBinning` — verify by readback, not by ret code

The Hik MV-CS200-10UC (and similar) returns `0x80000100` (`MV_E_GC_GENERIC`) on `BinningHorizontal`/`BinningVertical` **even when the binning is applied**. Previously this triggered both the "fall back to no binning" warning AND the success log line together — what you saw.

Now `setBinning` calls `MV_CC_GetIntValue("BinningHorizontal")` after the set and trusts the readback. Three outcomes:
- `applied`: readback matches the requested binning → info log only
- `partial`: readback shows one axis matches → warning
- neither: real failure → warning, `self.binning` falls back to `1`

### Frontend JPEG decode (msgpack `bin` → base64 string)

After §4.A the JPEG bytes come over the wire as a msgpack `bin` (decoded to `Uint8Array` by `@msgpack/msgpack`), but the React UI still consumes a base64 string. Conversion now happens in **one** place — `WebSocketHandler.js`, the `decoded.image` branch — where the raw `Uint8Array` is `btoa`-converted in 32 KB chunks. All downstream consumers (`LiveViewComponent`, `OverviewRegistrationWizard`, `SetLasersTab`, `TestHomingTab`, …) keep using `data:image/jpeg;base64,${liveViewImage}` and work unchanged.

`LiveViewComponent.js` additionally gained a `createImageBitmap` fast path: when the browser supports it (Chrome ≥ 63, Firefox ≥ 54, Safari ≥ 15) the JPEG decode runs off the main thread; the classic `HTMLImageElement` path is kept as a fallback. An in-flight `cancelled` flag drops stale decodes when frames arrive faster than the decoder can process them.

## Round 3: the actual Windows bottleneck

The event-loop policy fix from the previous round was correct but not sufficient. The real Windows-vs-Mac/Linux gap was the **ACK round-trip**, not the event loop.

`noqt.py` has a per-client backpressure scheme: the server can have up to `MAX_FRAME_LAG` frames in flight before it stops sending. Old value was `1` — meaning the server has to wait for the client to ACK frame N before it sends N+1. On Mac/Linux the RTT is ~5 ms, dwarfed by the 50–100 ms throttle. On Windows the RTT is ~80–100 ms (ProactorEventLoop + Defender), which dominates: at 80 ms throttle the effective cadence collapses to ~5 fps (the "LIVE • 5.0 FPS" you saw). Bumped to `3`, letting up to 4 frames pipeline; the throttle stays in charge of the cadence and Windows now matches Mac/Linux.

Cost: 3 extra in-flight JPEGs per client, ~1.5 MB peak at 500 KB each. Negligible.

Re: [github.com/Kludex/uvicorn#2749](https://github.com/Kludex/uvicorn/discussions/2749) — that thread is about `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` being silently overridden by uvicorn ≥ 0.36. Doesn't apply to us: we construct the loop ourselves with `asyncio.new_event_loop()` and pass the instance to `uvicorn.Config(loop=…)`. uvicorn uses the loop we give it without touching the policy. The new `print(f"Server event loop: {type(self._asyncio_loop).__name__}")` in `ImSwitchServer.run()` confirms this at boot — if you ever see `ProactorEventLoop` on Windows in that line, that thread's workaround (`loop="asyncio"`) becomes relevant.

## The MJPEG apply-twice bug

Root cause: `handleSubmitSettings` was dispatching `setImageFormat(newFormat)` **before** the backend stop+start. Sequence on a JPEG→MJPEG submit:

1. `setStreamParameters("mjpeg", …)` — backend just mutates its `_streamParams['mjpeg']`. The JPEG worker is still active.
2. `dispatch(setImageFormat("mjpeg"))` — Redux flips. React re-renders. `LiveViewControlWrapper` swaps `LiveViewComponent` → `MJPEGViewer`.
3. `<MJPEGViewer>` mounts and immediately fetches `/mjpeg_stream?detectorName=…`. The endpoint calls `getMJPEGWorker(detectorName)` — returns `None` because the active worker is still `JPEGStreamWorker`. Endpoint returns `{"status": "error"}` as JSON. The `<img>` can't render that, fails silently.
4. *Now* `stopLiveView()` + `startLiveView("mjpeg")` fire. MJPEG worker spins up, but the `<img>` already moved on.

Fixed by reordering: backend stop+start first, then Redux dispatch. By the time `MJPEGViewer` mounts the backend really is on MJPEG. Also factored the protocol-specific param/detector blobs into a single `paramsForDraft(draftSettings)` helper, cutting `handleSubmitSettings` from ~235 lines (5 nested ternaries for each of params + detector) to ~120.

## Why Windows is 3–4× slower than Mac/Linux (and the fix)

Symptom: Pi 5 → Ethernet → Chrome on Windows runs at 3–4 FPS. The same backend feeding Chrome on macOS or Linux runs at 20–30 FPS. Even running both halves on the same Windows host shows >1 s end-to-end latency.

**Root cause** is not the JPEG encoder or noqt — it's two interacting defaults in the Windows asyncio + uvicorn stack:

1. **Python 3.8+ on Windows defaults to `ProactorEventLoop`.** That loop is fine for plain HTTP but has a long tail of issues with python-socketio + uvicorn + websockets: slow ws upgrades, occasional silent drop-back to long-polling, and write-coalescing that translates into hundreds of ms of buffered latency at high frame rates. Linux/Mac uses `uvloop` (via uvicorn) which doesn't have these issues.
2. **uvicorn's `ws="auto"` can fall back to the pure-Python `wsproto`** if it can't load `websockets` cleanly. That's another 2–3× slowdown on the upgrade path.

**Fix (both applied in `ImSwitchServer.ServerThread`).**
- On `sys.platform == "win32"` we call `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())` *before* the loop is constructed.
- `uvicorn.Config(..., ws="websockets")` forces the C-accelerated websockets path.

The frontend `socket.io-client` already does `transports: ["websocket"]` (in `WebSocketHandler.js`), so the long-poll → ws upgrade dance is skipped entirely — we go straight to websocket.

If after this Windows still looks slower than Mac/Linux on identical hardware, the next thing to check is Windows Defender's real-time scanning on the Python process — it adds ~50 µs per socket write which compounds at 20 fps × ~50 KB JPEGs.

## On the >100% CPU you saw on the MacBook

A 3648×5472×3 (60 MP) RGB camera at `subsampling=4`, `throttle_ms=80` (12.5 fps), JPEG q80 is:
- subsample to 1368×912×3 ≈ 3.75 MP (the camera-side preview thread now does this off the StreamWorker)
- JPEG encode 3.75 MP RGB ≈ 5–8 ms with PyTurboJPEG, **25–40 ms with cv2.imencode**
- msgpack pack ≈ 5 ms per client
- copy + dispatch to recording (if recording is on) ≈ 30 ms for a 60 MB raw copy

If PyTurboJPEG isn't installed you fall to cv2 and burn ~50% of one core just on the JPEG step at 12.5 fps. Check the startup log for `JPEG encoder: PyTurboJPEG (libjpeg-turbo)` — if you see the `cv2.imencode fallback` line instead, that's almost certainly the bottleneck. `pip install PyTurboJPEG` + `brew install jpeg-turbo` on macOS.

Nothing in noqt itself regressed — the per-client `metadata.copy()` was *removed* in §4.B.11, not added.

## Follow-up: MJPEG stays put on submit; per-protocol settings panel

Three small things fell out of the first MJPEG round:

1. **The settings cascade was re-introduced when §4.C.17 was reverted.** Both `setStreamParameters` and `setDetectorStreamParameters` were back to `stopLiveView + startLiveView` on every change. Result: clicking Submit on a JPEG→MJPEG switch fired 4–5 starts in one second (you saw it as "I have to apply twice and manually restart"). The in-place mutation fix is reapplied here for both endpoints, so a Submit now produces *exactly one* stop+start (the format change itself, from the frontend's `handleSubmitSettings`), and same-protocol param changes propagate live with no restart.

2. **MJPEGViewer was rendering at the JPEG's natural size instead of filling its parent.** `maxWidth/maxHeight: 100%` only clamps an oversized natural image; with `subsampling=4` on a 60 MP sensor the JPEG comes out 1368×912, the parent flex container has no fixed height, so `height: 100%` collapses and the img stays at natural size in the middle of a much larger black box. Fixed by switching to `width: 100%; height: auto; maxHeight: 80vh; objectFit: contain` and a `position: relative` parent with no `height: 100%`.

3. **No MJPEG-specific settings panel in `StreamControlOverlay`.** Mirrored the JPEG block (quality / subsampling / throttle sliders) onto the `mjpeg` slice of `draftSettings`. Same backend knobs (`jpeg_quality`, `subsampling_factor`, `throttle_ms`) — the difference is purely the transport.

Also bumped the MJPEGViewer's nonce on `liveStreamState.imageFormat` change so a protocol switch always re-opens the multipart connection (otherwise the `<img>` keeps reading from the now-closed previous stream). `StreamPresets` got a parallel `mjpeg` case so preset application pushes MJPEG params to the backend.

## Frontend: MJPEG protocol + WebRTC hidden

Two changes to the stream-format selector in `StreamControlOverlay.js`:

- **WebRTC removed from the dropdown** (the `<MenuItem value="webrtc">` is commented out, not deleted). The backend `WebRTCStreamWorker`, the `webrtc_offer` SDP endpoint and the `WebRTCViewer.jsx` React component are all left intact. Switching it back on is one line in the JSX once the aiortc path is stable again.
- **MJPEG added as a new option** — labelled "MJPEG (HTTP) - Bypass Socket.IO". When selected, the live preview tile renders a new `MJPEGViewer` (`frontend/src/axon/MJPEGViewer.jsx`) that just does:

  ```jsx
  <img src="${ip}:${port}/imswitch/api/LiveViewController/mjpeg_stream?startStream=true&detectorName=…" />
  ```

  The browser's built-in `multipart/x-mixed-replace` decoder handles everything; the socket.io / MessagePack / msgpack-pack / `frame_ack` machinery is bypassed entirely. The trade-off is no per-frame metadata sidecar (no `frame_id`, no `server_timestamp`, no `pixel_size`) — anything that needs that should stick with the JPEG protocol.

  Why MJPEG matters now: it's the most robust fallback on Windows, especially before the event-loop fix above lands. A plain HTTP GET with `Connection: keep-alive` has no transport-upgrade dance, no asyncio loop quirks, no ack-loop backpressure to coordinate, and the FPS is bounded by the camera + the wire, not by socket.io's Python overhead per client.

  Click / double-click / image-load callbacks are forwarded to the same parent handlers (`onClick`, `onDoubleClick`, `onImageLoad`) so stage-on-double-click still works.

`LiveViewControlWrapper.js` routes the new `imageFormat === "mjpeg"` value to `MJPEGViewer`; `useWebGL` is gated to exclude both `mjpeg` and `webrtc` so the WebGL renderer isn't accidentally selected for either.

## ExperimentController async writes — done

`save_frame_ome` was synchronous: it called `ome_writer.write_frame(img, metadata)` on the workflow thread, blocking the next stage step for 20–100 ms per frame (much more on slow SD/USB targets). For fast wellplate scans the disk write was the single largest per-frame cost.

It now queues to a small purpose-built `FrameSaveWorker` (`imswitch/imcontrol/model/io/frame_save_worker.py`, ~120 lines):

- Single dedicated thread + bounded FIFO (`maxsize=32`). Per-writer ordering preserved because all writes go through the same thread.
- `submit(fn, *args, on_success=…, on_error=…, **kwargs)` API. The workflow thread queues `ome_writer.write_frame` and returns immediately.
- On queue overflow we **block** rather than drop — experiment data integrity matters more than smooth latency. The maxsize caps peak RAM at `32 × frame_size` (e.g. ~600 MB for 9 MP RGB; trim if running on a memory-tight Pi 5).
- `on_success(chunk_info)` fires on the saver thread — that's where `sigUpdateOMEZarrStore.emit(...)` happens. Signal emission is thread-safe (`noqt.SignalInstance.emit` already wraps a `run_coroutine_threadsafe`).
- Frame is `img.copy()`'d before queueing (~5–10 ms for 9 MP) so the camera SDK ring buffer can be reused while the writer thread takes its time.
- `stopExperiment` calls `_frame_saver.flush(timeout=30 s)` so the experiment doesn't "end" until queued tiles are on disk. If the timeout fires (stuck disk), a warning is logged and the saver thread keeps draining in the background.

The downstream `_write_individual_tiff` (per-frame TIFF + OME-XML) automatically inherits the async behaviour because it's called from inside `ome_writer.write_frame` — i.e. on the saver thread. No further plumbing needed for that path.

Why not reuse `RecordingService.BackgroundStorageWorker`? It would have meant exposing a singleton across two unrelated subsystems and adding a new `TaskType`, ~50 LOC for an ABI we'd then have to keep stable. The dedicated `FrameSaveWorker` is smaller, has no coupling to RecordingService, and only ships the API ExperimentController actually needs (`submit`/`flush`/`stop`).

## What's deferred / out of scope

- **§4.B.11 — full "pack once" with shared msgpack payload** — needs frontend change, marginal benefit at single viewer.
- **§4.B.14 — Tucsen SDK binning** — `TucsenCamManager.setBinning` is still a TODO stub. The Tucsen SDK exposes binning via `TUCAM_Capa_SetValue(handle, TUCAM_IDCAPA.TUIDC_BIN_???, value)` but the exact capability ID varies by model (`TUIDC_BINNING_SUM` / `TUIDC_BINNING_MEAN` / `TUIDC_RESOLUTION` on some). Without hardware to verify I'd risk breaking the live feed. Recommended follow-up: confirm the right cap ID on your Dhyana, then mirror the HIK flow (stop → set cap → re-read sensor size → start).
- **§4.B.15 — HIK live-preview-only binning toggle** — skipped per your instruction.
- **§4.C.17 — Two-stage preview pipeline (HIK + Tucsen)** — lives on `fix/streaminglatency-twostep`. Not merged to main yet.
- **§4.C.18 — multiprocessing shared memory + sidecar encoder** — biggest headroom, biggest risk; not started.
- **§4.C.19 — ffmpeg subprocess pipe** — not started.
- **§4.C.20 — WebRTC architecture refactor** — not started. The WebRTC dropdown option is hidden in the UI but the backend + viewer code is preserved.
- **§4.C.21 — Zarr-default for streaming-recording** — not started; recording defaults still TIFF.

## Frontend update needed

The §4.A base64 removal means `data.image` over socket.io is now `Uint8Array` instead of base64 string. Wherever you do:

```js
// BEFORE (base64 string)
img.src = `data:image/jpeg;base64,${data.image}`;
```

switch to:

```js
// AFTER (raw bytes via msgpack 'bin')
const blob = new Blob([data.image], { type: 'image/jpeg' });
const url = URL.createObjectURL(blob);
// revoke the previous URL to avoid leaking object-URLs every frame
if (img._lastUrl) URL.revokeObjectURL(img._lastUrl);
img._lastUrl = url;
img.src = url;
```

Or, if you'd rather decode directly into a `<canvas>` and skip the `<img>` round-trip:

```js
const blob = new Blob([data.image], { type: 'image/jpeg' });
const bitmap = await createImageBitmap(blob);
ctx.transferFromImageBitmap(bitmap);  // OffscreenCanvas
// or ctx.drawImage(bitmap, 0, 0)
bitmap.close();
```

`createImageBitmap` + `OffscreenCanvas` is usually the lowest-latency path in Chrome/Firefox; useful for high FPS.

### What `data.image` decodes to

With `msgpack-lite` / `@msgpack/msgpack` and `use_bin_type` on the server, MessagePack `bin` is decoded to a `Uint8Array` by default. If your decoder is configured to return `Buffer` or `ArrayBuffer` you may need a small adapter:

```js
const bytes = data.image instanceof Uint8Array
  ? data.image
  : new Uint8Array(data.image);
const blob = new Blob([bytes], { type: 'image/jpeg' });
```

### Don't forget to revoke object-URLs

Each `URL.createObjectURL` creates a long-lived handle that holds the Blob memory until you `revokeObjectURL` or reload the page. At 20 FPS that's 20 leaks/sec. Pattern:

```js
let lastUrl = null;
socket.on('frame', (data) => {
  const decoded = msgpack.decode(data);   // {metadata, image}
  const blob = new Blob([decoded.image], { type: 'image/jpeg' });
  const url = URL.createObjectURL(blob);
  img.onload = () => URL.revokeObjectURL(url);   // free after browser parses
  img.src = url;
  socket.emit('frame_ack', { frame_id: decoded.metadata.frame_id });
});
```

## How to opt into the new features per detector

Edit the camera's `defaultStreamSettings` in your setup JSON (e.g. `imswitch/_data/user_defaults/imcontrol_setups/example_tucsen.json`):

```json
"defaultStreamSettings": {
    "protocol": "jpeg",
    "jpeg_quality": 80,
    "subsampling_factor": 4,
    "throttle_ms": 50,
    "broadcast_frames": false
}
```

`broadcast_frames: false` turns off the per-frame fan-out to Histogram/FFT/Holo/ImageController for that detector. Keep it `true` (default) if you're using streaming-recording for that camera, since `RecordingService` listens on the same signal.

## Verifying PyTurboJPEG kicked in

On startup look for one of these log lines from `_JpegEncoder.__init__`:

- `JPEG encoder: PyTurboJPEG (libjpeg-turbo)` — you're on the fast path
- `PyTurboJPEG not available (<reason>); using cv2.imencode fallback` — install fix needed

To install on Pi 5:
```sh
sudo apt-get install -y libturbojpeg0
pip install PyTurboJPEG
```

(The PyPI package wraps the system `libturbojpeg`; on Debian/Ubuntu/Raspbian the apt package is `libturbojpeg0` or `libturbojpeg0-dev`.)

## Expected impact on the 9 MP path

After §4.B on top of §4.A, the per-frame budget at subsampling_factor=4 on a Pi 5 is approximately:

| Stage | After §4.A | After §4.B |
|---|---|---|
| u16 → u8 (right-shift) | 5–10 ms | 3–6 ms (preallocated) |
| JPEG encode (post-subsample) | 8–15 ms (cv2) | 4–8 ms (turbojpeg) |
| Per-client msgpack | 5–10 ms | 4–8 ms (no metadata copy) |
| Recording write (streaming) | 50–200 ms (inline) | ~5 ms (just enqueue + copy) |
| WebRTC `recv()` | 50–100 ms (resize inline) | 1–2 ms (wrap only) |
| TIFF write per frame | unbounded (full bandwidth) | ~50% of uncompressed at ~1 GB/s |
| **Live preview FPS at 9 MP** | **~15–25 Hz** | **~25–45 Hz** |

Recording-throttle: with the new dispatch queue + zlib-1 TIFF compression, the streaming recording can keep up with the live preview to ~25 Hz before queue overflow starts dropping frames. If you need higher, increase `_dispatch_queue` maxsize in `RecordingService.__init__` (8 → 32) at the cost of higher memory headroom (each pending frame holds an 18 MB copy at 9 MP).

## Things to spot-check before committing

1. **Binning on the MV-CS200-10UC**: after the fix, `setBinning(2)` should log `Binning set to 2x2, sensor size now <new_W>x<new_H>` with `<new_W> = WidthMax / 2`. The previous `WARNING [CameraHIK] BinningX set failed ret=0x80000100` should be gone (or downgraded to a single `debug`).
2. **`setFlipImage(True, False)`** should log `SDK flip set (ReverseY=True, ReverseX=False); software np.flip skipped in callback.` If you instead see `SDK flip not accepted ...`, the model doesn't expose ReverseY (rare) and the software path will still flip — same behaviour as before.
3. **PyTurboJPEG path**: log line on startup.
4. **Recording**: start a streaming recording at 9 MP and check the log for `RecordingDispatch queue full; dropping frames` — if you see it, lower `subsampling_factor` for the recorder path or raise `_dispatch_queue` maxsize.
5. **WebRTC**: the first frame should arrive faster (no per-frame cv2 resize on the asyncio loop). If anything looks wrong (corrupted colour, wrong size) it's most likely the producer-side resize step. Bisect by adding `subsampling_factor=1, max_width=1280` and checking the queue contents.
