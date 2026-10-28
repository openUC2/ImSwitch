# STORM: in-the-loop detection, analysis (2026-10-14)

Goal for the next round: localise single molecules while acquiring, and use the localisation rate
as feedback, for example to steer the 405 nm activation laser. Doing this in the browser with
[webSMLM](https://github.com/HohlbeinLab/webSMLM) is one option. This page describes the local STORM
pipeline as it is now, webSMLM, and a proposal.

**Already done this round**
- The STORM Arkitekt app and the controller's Arkitekt service are removed (a61ace97, 0f8d77c4).
  Their rekuest startup hook used to make the whole controller fail to import wherever
  `arkitekt_next` was installed, so every STORM endpoint returned 404.
- The unused combined `STORMController.js` is removed (51b2066a).
- Saving was fixed (b2f612c3). It had stopped after the first frame since e00543c5, and `(frames,
  ids)` chunks broke the store set-up.

## 1. The local pipeline today (`controllers/STORMReconController.py`)

1. `startSTORMReconstructionLocal` creates `<data>/STORMController/<ts>/<session>/` and starts
   `startFastSTORMAcquisition`. That crops, initialises saving, and starts a worker thread.
2. The worker polls `detector.getChunk()` every 1 ms, crops each frame, and, with local processing
   on, calls `_enhancedProcessFrame` **in the same thread** that drains the camera.
3. Fitting is the bundled microEye `localize_frame`:
   - 8-bit LUT equalisation, then a bandpass filter;
   - a quantile threshold;
   - OpenCV `SimpleBlobDetector`;
   - a phasor fit that runs one `fft2` per ROI in a Python loop.

   An MLE fitter (numba) is present but never selected.
4. Results:
   - one TIFF and one CSV per frame;
   - every `update_rate` frames a `sigExperimentImageUpdate("STORM", ndarray)`;
   - every 20 frames a frame count (`sigNSTORMImageAcquired`).

**Weak points, from reading the code (not run)**
- **Settings are ignored.** The fit gets no threshold or ROI size, so the defaults 0.4 and 13
  apply. `fitting_method` and the pixel sizes are never used.
- **The "reconstruction" is not super-resolved.** It counts hits on the camera pixel grid.
  `saveImage` checks `sumReconstruction`, which is never set, so stopping returns no path.
- **The live image never reaches the UI.**
  - The server emits `signal_msgpack`, but `STORMControllerLocal.js` listens on the old `"signal"`
    event and expects `jdata.image`.
  - The payload is a float `ndarray.tolist()`, megabytes per update.
  - `sigSTORMLocalizationUpdate` and `sigSTORMStatisticsUpdate` are never emitted.
- **Not real-time.** Fitting and per-frame file writes share the thread that drains the camera.
- **The virtual `smlm` sample cannot test feedback.** Its blinking density is fixed (0.05) and
  does not depend on laser power.

## 2. webSMLM (v0.13.3, 2026-10-05; read, not run)

- **Licence:** MIT. Bundled UTIF.js and pako are MIT; the simulator (WASM) is BSD-3.
- **Form:** a single 2 MB `webSMLM.html` of classic scripts. It is an app, not a library (no npm
  package or ES module). It can be scripted through:
  - `window.webSMLM.{analyze, analyzeBatch, liveStream:{pushChunk, end, isActive}}`;
  - `?autorun=1&fileUrl=…`;
  - Playwright helpers in `tools/`.
- **In the browser:**
  - Detection: wavelet, DoG or box bandpass plus local maxima, in Web Workers.
  - Fitting: phasor, least squares and Poisson MLE (2-D, elliptical, astigmatic 3-D) on WebGPU,
    with a CPU fallback.
  - Also: drift correction (AIM, cross-correlation, COMET), NeNA, FRC, tracking, smFRET.
- **Input:** TIFF, ND2 and FITS files, or live ImageJ-style TIFF chunks.
- **Live protocol** (`tools/test_livestream_demo.py`):
  - The page connects **out** to a plain WebSocket server.
  - One binary message is one TIFF chunk (`tifffile.imwrite(buf, stack, imagej=True)`);
    `{"cmd":"stop"}` ends the session.
  - The page acks `{"cmd":"ack", framesReceived, chunkCount, ok}`. The acks are advisory: they do
    not slow the sender, and **they do not carry localisation counts.** `chunkLocs` is computed
    but not sent.
  - Marked experimental. No throughput figures are published. The GPU batch flush waits 400 ms
    (`gpuFlushMs`), which suggests chunks of 10–50 frames.

## 3. Proposal

1. **Frames to the browser.**
   - The Socket.IO `frame` stream does not fit: latest frame only, throttled, subsampled, ack-gated,
     its own format.
   - Instead, add a plain FastAPI WebSocket, e.g. `/imswitch/ws/websmlm`, that speaks webSMLM's
     live protocol.
   - The STORM worker hands every cropped frame to a bounded queue. A sender packs N frames into an
     ImageJ TIFF, stops while sent − acked > K, and counts dropped frames. Raw frames still go to
     disk.
2. **Hosting.** Serve a pinned copy of `webSMLM.html` from ImSwitch (same origin; no mixed content
   or self-signed wss problems), in a tab or a same-origin iframe.
3. **Feedback.**
   - Ask upstream for `chunkLocs` in the ack, and for a `?ws=…&connect=1` parameter.
   - ImSwitch keeps a smoothed localisations-per-frame value and runs an opt-in, clamped,
     rate-limited controller on the activation laser.
   - The controller holds or ramps down when acks go stale. Raising laser power is ask-first
     (AGENTS.md).
4. **Endpoints:**
   - `startSMLMStream(roi, framesPerChunk, maxInflight)`, `stopSMLMStream`;
   - `getSMLMStreamStatus` (sent, acked, dropped, localisations per frame);
   - `setActivationFeedback(enabled, laserName, targetLocsPerFrame, maxValue, gain, period)`,
     `getActivationFeedbackStatus`.
5. **Data rates** (uint16 frames):

   | Frames | Rate | Bandwidth |
   |---|---|---|
   | 256² | 100 fps | ≈ 13 MB/s |
   | 512² | 50 fps | ≈ 26 MB/s |
   | full 5 MP | 20 fps | ≈ 200 MB/s (not feasible) |

   So an ROI crop is mandatory, and Wi-Fi on the Pi is marginal. How many frames per second the
   browser can fit is unknown and needs measuring.
6. **Hybrid (recommended).** The feedback loop only needs spot counts, not fits. A server-side DoG
   plus local-maximum count on every k-th frame, in its own thread or process, closes the loop
   headless, so it keeps working without a browser tab open. webSMLM is then the live viewer and
   quality check (super-resolved image, drift, density). The microEye path is kept for offline
   use or removed. Making the virtual `smlm` density depend on laser power is a prerequisite for
   testing the loop.

## 4. Questions for Bene

1. What is the loop for: 405 nm feedback only, or also a live super-resolved image and quality
   checks?
2. Which camera, ROI and frame rate? Pi on Wi-Fi or Ethernet? Which machine runs the browser?
3. Must the loop keep running with no browser open during hours-long acquisitions?
4. Is it OK to bundle a pinned `webSMLM.html` (MIT, 2 MB)? Shall we send the upstream PRs (ack
   counts, auto-connect)?
5. Which laser activates, at what maximum power, with what ramp limits, and who confirms switching
   the loop on?
6. Keep the bundled microEye (numba/cv2; licence not checked) or drop it?
7. Now that nothing in ImSwitch imports `arkitekt_next` or `mikro_next` for STORM, can the
   `mikro-next` dependency go (`pyproject.toml:78`, `setup.py`)? That is a dependency change and
   needs your OK.
