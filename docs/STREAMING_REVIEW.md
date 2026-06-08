# Streaming pipeline review — large-sensor lag on Raspberry Pi 5

Scope: live preview + on-disk capture for HIK (`hikcamera.py` / `HikCamManager.py`) and Tucsen (`tucsencamera.py` / `TucsenCamManager.py`) on a Pi 5 host, observed from a laptop browser. Read-only review; no code edited.

Files of record:
- `imswitch/imcontrol/controller/controllers/LiveViewController.py` (the streaming pipeline)
- `imswitch/imcontrol/model/interfaces/hikcamera.py` (HIK SDK callback)
- `imswitch/imcontrol/model/interfaces/tucsencamera.py` (Tucsen SDK callback)
- `imswitch/imcontrol/model/io/recording_service.py` (snap + streaming TIFF/Zarr)
- `imswitch/imcontrol/model/io/ome_writers/single_tiff_writer.py` (BigTIFF append)
- `imswitch/imcommon/framework/noqt.py` (signal → socket.io fan-out, backpressure)
- `imswitch/imcommon/framework/binary_streaming.py` (LZ4/Zstd binary encoder)

---

## 1. The actual data path, end-to-end

```
[SDK callback thread] ── frame (np view over SDK buffer) ──┐
                                                           │
   HikCamera._on_frame         deque(maxlen=3)             │
   TucsenCamera._on_frame_cb   deque(maxlen=5)             │
                                                           ▼
[StreamWorker thread, per detector, per protocol]
   loop every throttle_ms (default 50ms → 20 Hz):
     1) detector.getLatestFrame()            ← reference to deque tail
     2) sigUpdateFrame.emit(...)             ← BROADCAST raw frame to every
        (psygnal, synchronous, in this thread)  controller listening on
                                                sigUpdateImage
     3) _captureAndEmit(frame, fid):
          - JPEG:   (frame/16).astype(u8) → crop → subsample → cv2.imencode
                    → base64 → msgpack-ready dict → sigStreamFrame.emit
          - MJPEG:  full-frame np.min/np.max → float normalise → u8
                    → crop → cv2.imencode → push to queue
          - Binary: u16 left-shift to fill range → subsample → LZ4
                    → sigStreamFrame.emit
          - WebRTC: u16→u8 (fast path uses >> 8) → push to queue
                    → recv() in aiortc loop does cv2.resize → av.VideoFrame

[noqt.SignalInstance.emit] (for sigUpdateStreamFrame)
   per ready socket.io client:
     - re-packs metadata + payload with msgpack
     - asyncio.run_coroutine_threadsafe → sio.emit('frame', bytes, to=sid)
   backpressure: client must ACK previous frame_id mod 256 before next is sent
                 (correct, rollover-safe)

[Recording path, in parallel]
   StreamWorker step 2 above also reaches RecordingService._on_new_frame
     if streaming-recording is active → _streaming_adapter.write_frame()
     runs SYNCHRONOUSLY on the StreamWorker thread
   Snap path: BackgroundStorageWorker thread (good, decoupled)
```

Two important observations from this map before getting into numbers:

- **Capture, broadcast, encode, msgpack, and per-client serialization all run on the same Python thread** as the StreamWorker (one per active stream). The GIL is held for almost the whole iteration. Anything that needs to live next to the worker (FastAPI, asyncio loop, MJPEG HTTP generator) shares the GIL with it.
- **Frame size is what scales**. The control flow doesn't change between a 2 MP and a 9 MP camera, but the cost of each step is linear in pixel count, and several steps allocate fresh buffers that are 2–8× the raw frame size.

---

## 2. Why it gets dramatically slower at 9 MP+

Below, "9 MP" means 3000×3000 (Tucsen Dhyana) or 4096×2160 (HIK 4K mono) at uint16 — roughly 18 MB per raw frame. All timings are budgetary estimates for Pi 5 (Cortex-A76 @ 2.4 GHz, NEON) with stock libjpeg-turbo and OpenCV; treat as order-of-magnitude, not benchmarks.

### 2.1 The biggest single offender: float64 conversion of full frames

[`LiveViewController.py:345-346`](imswitch/imcontrol/controller/controllers/LiveViewController.py:345)

```python
if frame.dtype == np.uint16:
    frame = (frame / 16).astype(np.uint8)
```

`frame / 16` in NumPy promotes the whole array to **float64**. For 9 MP × 8 bytes that is a 72 MB intermediate allocation, every frame, just for the division. Then `.astype(np.uint8)` allocates another 9 MB. The Pi 5 memory subsystem can do this, but it competes with the SDK DMA buffers and OpenCV's internal allocations.

Budget: ~80–150 ms per 9 MP frame on a Pi 5 for this one line.

The WebRTC worker has the right pattern at [`LiveViewController.py:514-519`](imswitch/imcontrol/controller/controllers/LiveViewController.py:514):

```python
if frame.dtype == np.uint16:
    frame = (frame >> 8).astype(np.uint8)
```

That's a NEON-friendly right-shift, no float intermediate, ~10–20 ms for the same frame. The JPEG worker doesn't use it.

### 2.2 The MJPEG worker is worse

[`LiveViewController.py:429-435`](imswitch/imcontrol/controller/controllers/LiveViewController.py:429)

```python
if frame.dtype != np.uint8:
    vmin = float(np.min(frame))
    vmax = float(np.max(frame))
    if vmax > vmin:
        frame = ((frame - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
```

That's two full-array scans (`np.min`, `np.max`), a subtract, a divide, a multiply, and a cast — at least 4 passes through 18 MB, mostly in float64. Budget: ~300–500 ms per 9 MP frame. This is why MJPEG feels especially slow on big cameras.

It is also semantically dubious: the displayed brightness flickers because vmin/vmax change frame to frame.

### 2.3 Work happens *before* subsampling, not after

[`LiveViewController.py:345-354`](imswitch/imcontrol/controller/controllers/LiveViewController.py:345):

```python
if frame.dtype == np.uint16:
    frame = (frame / 16).astype(np.uint8)         # ← full 9 MP
if self._params.crop_size > 0:
    frame = apply_center_crop(frame, ...)         # ← full 9 MP
if self._params.subsampling_factor > 1:
    frame = frame[::N, ::N]                       # ← view, free
```

With the default `subsampling_factor=4`, the user has *already asked* for a 1/16 area image. But the costly dtype cast runs on the full frame first. Reordering to `subsample → cast → crop` makes the dtype cast 16× cheaper for free.

Stride-slicing is a view, but [`cv2.imencode`](imswitch/imcontrol/controller/controllers/LiveViewController.py:362) will force a contiguous copy on a non-contiguous view; `np.ascontiguousarray` is missing in the JPEG worker (the binary worker calls it at line 259).

### 2.4 JPEG encoding the full frame

[`LiveViewController.py:362`](imswitch/imcontrol/controller/controllers/LiveViewController.py:362):

```python
success, encoded = self._cv2.imencode('.jpg', frame, encode_params)
```

`libjpeg-turbo` on Pi 5 with NEON: ~30–60 MP/s at quality 80. A 9 MP frame at q=80 ≈ 150–300 ms. With subsampling_factor=4 this drops to ≈10–20 ms. So subsampling more aggressively for preview is far cheaper than trying to encode the full frame.

Note: the Pi 5 has **no hardware JPEG encoder** (the legacy VideoCore JPEG path was removed in the Pi 5 SoC) and **no hardware H.264 encoder**. There is no shortcut beyond NEON-optimised libjpeg-turbo or libturbojpeg.

### 2.5 Base64 in the JPEG path

[`LiveViewController.py:365-367`](imswitch/imcontrol/controller/controllers/LiveViewController.py:365):

```python
import base64
jpeg_bytes = encoded.tobytes()
encoded_image = base64.b64encode(jpeg_bytes).decode('utf-8')
```

`base64.b64encode` of a ~500 KB JPEG: ~5–10 ms on Pi 5, and the result is 33% larger. It is then handed to msgpack in [`noqt.py:230-233`](imswitch/imcommon/framework/noqt.py:230), which packs it as a string. The binary worker uses raw bytes through msgpack (`use_bin_type=True`) and does *not* base64-encode (line 207-210), which is the correct pattern. The JPEG worker should do the same — there is no reason to base64 a payload that travels through a binary-safe transport (msgpack over socket.io's binary frame).

For 30 clients, this is the difference between ~500 KB × 30 = 15 MB/s and ~660 KB × 30 = 20 MB/s of socket.io output, plus the encode cost.

### 2.6 Per-client repackaging

[`noqt.py:201-241`](imswitch/imcommon/framework/noqt.py:201): the frame payload is `msgpack.packb(...)`'d **per client** because the per-client `frame_id` is embedded inside the packed metadata.

For a single laptop viewer this is fine. With several viewers (one main UI + helper tabs), the per-frame cost grows linearly. The packing is ~5–10 ms per 500 KB frame.

Better: pack the metadata and the payload separately, or emit a tiny header with `frame_id` and a shared opaque blob. Or accept that the frame-counter scheme could live in a header byte instead of in MessagePack, and emit one packed payload to all sids.

### 2.7 Unconditional fan-out of raw frames to every controller

[`LiveViewController.py:1014-1020`](imswitch/imcontrol/controller/controllers/LiveViewController.py:1014):

```python
worker.sigStreamFrame.connect(self._commChannel.sigUpdateStreamFrame)
worker.sigUpdateFrame.connect(self._commChannel.sigUpdateImage)
worker.enableFrameBroadcast(True)
```

`sigUpdateImage` is connected to **all of these**:
[`ImageController.update`](imswitch/imcontrol/controller/controllers/ImageController.py:36),
[`AlignXYController.update`](imswitch/imcontrol/controller/controllers/AlignXYController.py:15),
[`AlignAverageController.update`](imswitch/imcontrol/controller/controllers/AlignAverageController.py:14),
[`FFTController.update`](imswitch/imcontrol/controller/controllers/FFTController.py:29),
[`SquidStageScanController.update`](imswitch/imcontrol/controller/controllers/SquidStageScanController.py:35),
[`HistogrammController.update`](imswitch/imcontrol/controller/controllers/HistogrammController.py:24),
[`MichelsonTimeSeriesController.update`](imswitch/imcontrol/controller/controllers/MichelsonTimeSeriesController.py:131),
[`OffAxisHoloController.update`](imswitch/imcontrol/controller/controllers/OffAxisHoloController.py:233),
[`InLineHoloController.update`](imswitch/imcontrol/controller/controllers/InLineHoloController.py:173),
[`RecordingService._on_new_frame`](imswitch/imcontrol/model/io/recording_service.py:575) (when recording),
and any plugin controllers.

psygnal fires connected slots **synchronously on the emitting thread**. So every active stream iteration runs *all* connected `update()` methods inline. With 9 MP frames:

- A histogram is two passes through 18 MB — ~50–100 ms.
- An FFT preview at full size on a Pi 5 is hopeless — hundreds of ms or more, even before plotting.
- Hologram reconstruction controllers do whole-frame FFTs.

The code itself flags this at [`LiveViewController.py:201-205`](imswitch/imcontrol/controller/controllers/LiveViewController.py:201):

> WARNING: This can significantly impact performance if many controllers are connected.

…and then enables broadcasting unconditionally. This is almost certainly the single biggest reason the user perceives lag with larger cameras: every connected controller runs full-resolution Python work on each frame, on the streaming thread, while the stream is waiting to emit JPEGs.

### 2.8 Recording is on the same thread as streaming

[`recording_service.py:538-566`](imswitch/imcontrol/model/io/recording_service.py:538): when a streaming recording is active, `_on_new_frame` is invoked from `sigUpdateImage` (same StreamWorker thread) and calls `_streaming_adapter.write_frame(...)` *synchronously*. There is a `BackgroundStorageWorker` for snaps but the streaming-recording path bypasses it.

For 18 MB/frame, that means each write blocks the streaming thread for the duration of the TIFF write. SD-card throughput is the floor; even a USB SSD shares the bus with the camera (USB3 on Pi 5). Expect 50–200 ms blocking per frame, depending on storage.

### 2.9 BigTIFF append + per-frame open

[`single_tiff_writer.py:74-76`](imswitch/imcontrol/model/io/ome_writers/single_tiff_writer.py:74) keeps the `TiffWriter` open for the session — good. But [`recording_service.py:304-306`](imswitch/imcontrol/model/io/recording_service.py:304) is the wrong pattern:

```python
def _append_tiff(self, filepath: str, data: np.ndarray):
    tiff.imwrite(filepath, data, append=True)
```

`tiff.imwrite(..., append=True)` reopens the file, seeks to the end, rewrites the IFD chain, and closes it — every call. For 9 MP / 18 MB pages this is ≈3× the cost of a normal append (open + write + IFD bookkeeping + close). The `SingleTiffWriter` already uses the keep-open `tifffile.TiffWriter` pattern; the recording service path should too.

Also: no compression is requested for streaming TIFF. 30 FPS × 18 MB = 540 MB/s, far above SD card throughput and at the limit of cheap USB SSDs. `compression='zlib'` (level 1) gets ~2× on typical microscopy data; `compression='zstd'` (level 1, requires `imagecodecs` or `tifffile` ≥2023) gets ~3× at ~1 GB/s zstd throughput on a Pi 5 — usually faster end-to-end than no compression because the disk is the bottleneck.

### 2.10 HIK callback minor — `np.flip` produces a non-contiguous view

[`hikcamera.py:556-559`](imswitch/imcontrol/model/interfaces/hikcamera.py:556):

```python
if self.flipImage[0]:
    frame = np.flip(frame, axis=0)
if self.flipImage[1]:
    frame = np.flip(frame, axis=1)
```

`np.flip` returns a view with negative strides. Downstream code that needs a contiguous buffer (cv2, tifffile) will silently copy — a 9 MP copy in C-order is ~30–40 ms. This isn't dominant, but worth knowing: if you flip on the SDK side via `MV_CC_SetBoolValue("ReverseX"/"ReverseY", ...)` you save a copy per frame and the camera firmware does the work.

### 2.11 WebRTC's downscale runs on the wrong thread

[`LiveViewController.py:702-711`](imswitch/imcontrol/controller/controllers/LiveViewController.py:702): the `recv()` coroutine in the aiortc loop calls `cv2.resize` to downscale frames above 720p. That serializes the resize into the asyncio loop alongside the SDP/ICE work. Move the resize into `_captureAndEmit` (producer side) and put a *small* RGB u8 frame in the queue; recv just hands it to `av.VideoFrame`.

### 2.12 Throttle is measured in `time.time()`, but the cost dominates

The throttle is set to 50 ms (`throttle_ms`). If `_captureAndEmit` itself takes 400 ms at 9 MP, the worker can never hit 20 Hz; effective rate becomes 1/0.4 = 2.5 Hz. Lowering the throttle does nothing because the bottleneck is per-frame compute, not the timer.

---

## 3. What good throughput looks like on a Pi 5

Approximate budgets per 9 MP u16 frame, on a Pi 5, after each fix in §4:

| Stage | Today | After §4 quick wins | After §4 medium changes |
|---|---|---|---|
| u16 → u8 conversion | 80–150 ms (float64) | 10–20 ms (right_shift) | 5–10 ms (preallocated, in-place) |
| Subsample 4× | view (free) | view, applied first | view, applied first |
| JPEG encode (after subsample → ~2 MP equiv) | 60–120 ms (full frame) | 8–15 ms (subsampled) | 5–10 ms (PyTurboJPEG) |
| base64 | 5–10 ms | 0 (removed) | 0 |
| msgpack per client | 5–10 ms × n | 5–10 ms × n | 1–2 ms × n (shared blob) |
| Controller fan-out (sigUpdateImage) | 100–500 ms (all controllers) | 0 if disabled | <5 ms (throttled poll model) |
| Recording write (streaming) | 50–200 ms (blocking) | unchanged | 0 (async queue) |
| **Per-frame budget** | **~400–900 ms** | **~30–60 ms** | **~15–30 ms** |
| **Achievable preview FPS** | **~1.5–2.5 Hz** | **~15–25 Hz** | **~30–60 Hz** |

The numbers assume preview at subsampling_factor=4. If you must show the full sensor, divide everything but the encode by 16× back up.

---

## 4. Recommendations, ordered by impact ÷ effort

### A. Quick wins (one-line / few-line changes, no architectural risk)

1. **Replace `(frame/16).astype(np.uint8)` with `(frame >> 4).astype(np.uint8)` (12→8 bit) or `>> 8` (16→8 bit)** in `JPEGStreamWorker` and `MJPEGStreamWorker`. Eliminates the float64 intermediate. Single largest CPU saving on the JPEG path. ([`LiveViewController.py:346`](imswitch/imcontrol/controller/controllers/LiveViewController.py:346))
2. **Reorder operations: subsample → cast → crop**, not cast → crop → subsample. With subsampling_factor=4 the cast becomes 16× cheaper for free. ([`LiveViewController.py:345-354`](imswitch/imcontrol/controller/controllers/LiveViewController.py:345))
3. **Drop the `base64.b64encode` in the JPEG worker** and emit raw bytes through msgpack (`use_bin_type=True`), matching the binary worker. ([`LiveViewController.py:365-367`](imswitch/imcontrol/controller/controllers/LiveViewController.py:365), [`noqt.py:201-241`](imswitch/imcommon/framework/noqt.py:201))
4. **Replace MJPEG's `min/max → float → cast` normalisation with a fixed right-shift** (`(frame >> 4).astype(np.uint8)` for 12-bit, `>> 8` for 16-bit). The auto-stretch is also semantically harmful: it makes the preview flicker. If you want auto-stretch, compute it on a thumbnail. ([`LiveViewController.py:429-435`](imswitch/imcontrol/controller/controllers/LiveViewController.py:429))
5. **Set sensible per-detector defaults in the setup JSON** via `defaultStreamSettings` (the loader at [`LiveViewController.py:802`](imswitch/imcontrol/controller/controllers/LiveViewController.py:802) already supports this). For a 9 MP camera on a Pi 5, defaults like `{"protocol":"jpeg","jpeg_quality":75,"subsampling_factor":4,"throttle_ms":80,"max_width":1280}` are appropriate.
6. **Disable `enableFrameBroadcast(True)` unless a controller actually needs frames in real time.** The comment at [`LiveViewController.py:201-205`](imswitch/imcontrol/controller/controllers/LiveViewController.py:201) already warns against it. Either flip the default to off, or wire each consumer to subscribe explicitly. Histogram/FFT/Holo controllers can poll the cached frame at their own (slower) cadence — they don't need every frame.
7. **Add `np.ascontiguousarray(frame)` before `cv2.imencode`** in the JPEG and MJPEG workers (the binary worker already does this). Without it, a non-contiguous view from `np.flip` or stride slicing forces an internal copy inside OpenCV, often less efficient.

### B. Medium wins (small refactors, no SDK changes)

8. **Use `numpy.right_shift(src, n, out=preallocated_u8)` with a pre-allocated output buffer.** Cuts per-frame allocations to zero in the steady state. Combine with `cv2.imencode` releasing the GIL.
9. **Use `PyTurboJPEG` (libjpeg-turbo with the simple wrapper) instead of `cv2.imencode('.jpg')`.** Roughly 1.5–2× faster on ARM at the same quality. One-time `pip install PyTurboJPEG` + ~10 lines of integration. Keep cv2 as a fallback.
10. **Pre-resize at the producer** so the WebRTC `recv()` only converts to `av.VideoFrame`. The current code (`recv` does the resize) puts cv2.resize inside the asyncio loop. Move it to `_captureAndEmit`.
11. **Pack the socket.io payload once and only personalise a tiny header.** The per-client msgpack repackaging in `_handle_stream_frame` is wasted CPU; the frame body is identical for every client. The frame_id ack can live in a small leading struct (4 bytes) instead of inside the msgpack metadata.
12. **Move the streaming-recording write off the StreamWorker thread.** Reuse `BackgroundStorageWorker` (it already exists for snaps), or add a dedicated single-writer thread per detector that pulls from the camera's ring buffer at its own cadence. Decouple "what the user sees" from "what's saved to disk".
13. **Keep the TIFF writer open during streaming recording and enable cheap compression** (`tifffile.TiffWriter(..., bigtiff=True)`, then `writer.write(frame, compression='zlib', compressionargs={'level': 1})` or `compression='zstd'` if `imagecodecs` is available). Stop using `tiff.imwrite(..., append=True)` for per-frame appends. ([`recording_service.py:304`](imswitch/imcontrol/model/io/recording_service.py:304))
14. **For Tucsen, enable SDK-side binning** (`TUIDC_BINNING`) for live preview. `TucsenCamManager.setBinning` currently has a TODO at [`tucsencamera.py:695-700`](imswitch/imcontrol/model/interfaces/tucsencamera.py:695). Hardware binning reads less data off the sensor, cutting transfer + memcpy 4×–16× *before* it ever reaches NumPy. For acquisition you can switch back to 1×1.
15. **For HIK, enable `BinningHorizontal/BinningVertical` for live preview**, similarly. `HikCamManager` already supports binning at startup but does not expose a "preview vs capture" toggle. Adding a `setPreviewBinning(n)` method that calls `MV_CC_SetIntValue("BinningSelector", "Sensor")` + `BinningHorizontal/Vertical` would let the live view run at, say, 2×2 binned while snapshots run at 1×1.
16. **Move `np.flip` to the SDK** for HIK: `MV_CC_SetBoolValue("ReverseX"/"ReverseY", True)`. Saves a 9 MP copy per frame.

### C. Architectural shifts (bigger investment, biggest headroom)

17. **Two-stage pipeline: raw u16 → recorder, preview u8 → encoder.** Add a small `_preview_buffer` next to the existing `frame_buffer` in `CameraHIK` / `CameraTucsen`. The SDK callback fills both: the raw u16 page goes into `frame_buffer` (for recording / snap), and a downsampled u8 page goes into `_preview_buffer` (for streaming). The StreamWorker pulls from `_preview_buffer` and skips the whole cast/subsample dance. Recording pulls full-resolution from `frame_buffer` on its own cadence. This is the single most impactful refactor — it removes ~80% of the streaming-thread CPU.
18. **Use `multiprocessing.shared_memory` + a sidecar encoder process.** The Python GIL is fundamentally why one slow controller stalls the stream. A second process (linked by shared memory) for JPEG encoding bypasses the GIL entirely and can use a second core. `multiprocessing.shared_memory.SharedMemory` lets you avoid the pickle copy.
19. **Pipe frames into a long-running `ffmpeg` subprocess** for streaming. Something like `ffmpeg -f rawvideo -pix_fmt gray16le -s WxH -framerate 20 -i pipe:0 -vf "scale=1280:-2,format=gray8" -f mjpeg pipe:1` runs entirely in native code, uses ffmpeg's NEON paths, and is a separate process so it doesn't compete for the GIL. The Pi 5 software libx264 path is also good if you want H.264 over WebRTC.
20. **For WebRTC specifically**, accept that Pi 5 has no hardware H.264 encoder. The current `aiortc` path is software libx264 with all the Python overhead. If WebRTC is a hard requirement, run `aiortc` in a separate process linked by shared memory; otherwise consider sticking with MJPEG/binary for the Pi 5 and reserving WebRTC for hosts with `h264_v4l2m2m` or NVENC.
21. **For TIFF on the Pi 5, prefer Zarr (chunked, compressed, multi-writer-friendly) for streaming and produce TIFF as a post-process** — the OME-Zarr support is already there. A 9 MP frame compresses fast with Blosc/Zstd and Zarr writes are append-only into separate chunk files, which a Pi's I/O scheduler handles better than rewriting a giant TIFF directory each frame.

### D. Things to leave alone — they're already correct

- The **explicit ACK-based backpressure** in [`noqt.py:139-241`](imswitch/imcommon/framework/noqt.py:139) is well designed (rollover-safe, drops cleanly, applies per-client). Do not replace this with a naïve send loop.
- The **separation of binary/jpeg/mjpeg/webrtc workers** is good. The interface boundary is clean; the cost is purely in the body of `_captureAndEmit`.
- The **`StreamingResponse(multipart/x-mixed-replace)`** HTTP path for MJPEG is the right shape. The bottleneck is upstream of it.
- The **HIK SDK callback's zero-copy `np.frombuffer`** path is correct and you should not introduce extra copies there. The `.copy()` in Tucsen's `_convert_raw_to_numpy` is unavoidable because the SDK reuses the buffer asynchronously.
- The **`SingleTiffWriter` pattern** (keep `tifffile.TiffWriter` open across writes) at [`single_tiff_writer.py:74-76`](imswitch/imcontrol/model/io/ome_writers/single_tiff_writer.py:74) is what the streaming recorder should mimic.

---

## 5. Suggested measurement, before you change anything

The estimates above are budgets, not measurements. Before reordering production code, add a one-shot per-stage timer in `_captureAndEmit` for one camera and one minute of streaming, logging at INFO. Useful counters:

- time spent in `getLatestFrame()` (camera-side)
- time spent in dtype cast
- time spent in subsample
- time spent in `cv2.imencode`
- time spent in `base64.b64encode`
- time spent in `sigUpdateFrame.emit` (which is where the fan-out cost shows up)
- time spent in `sigStreamFrame.emit` (msgpack + sio.emit)
- queue depth on `_mjpeg_queue` / WebRTC `_frame_queue`

The numbers will tell you whether §2.1 (cast), §2.7 (broadcast), or §2.8 (recording) is dominant on your specific 9 MP camera, and which fix from §4 to do first. On a Pi 5 with the HIK MV-CS200-10UC (20 MP), I would predict §2.7 dominates; on the Tucsen Dhyana at 9 MP it's probably §2.1 + §2.4.

---

## 6. TL;DR

The lag scales with pixel count because **the streaming hot path runs full-frame Python work on a single thread**, with one unnecessary float64 conversion (§2.1), one unnecessary base64 encode (§2.5), one unnecessary fan-out to every controller (§2.7), and one unnecessary synchronous TIFF write (§2.8). None of these are camera-driver problems — the HIK and Tucsen interfaces both deliver frames zero-copy from the SDK and behave correctly. The bottleneck is in `LiveViewController._captureAndEmit`, in the unconditional `enableFrameBroadcast(True)`, and in the way `RecordingService._on_new_frame` piggybacks on the streaming signal.

If you make one change, make it §4.A.1 + §4.A.2 (right-shift + reorder). If you make two, add §4.A.6 (turn off the controller fan-out by default). If you make three, also do §4.B.12 (move recording off the streaming thread). Those three alone should restore smooth preview at 9 MP on the Pi 5.
