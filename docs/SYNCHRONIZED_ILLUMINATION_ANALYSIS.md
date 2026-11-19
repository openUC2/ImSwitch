# Synchronized illumination: analysis (2026-10-14)

Goal for the next round: light only while the sensor integrates. Light on, trigger the camera, wait
the exposure, light off, then read out, as one timed unit, to reduce photobleaching. This page
describes what exists today, where it is open-loop, and the three ways to close the loop.

Paths: **FW** = `uc2-ESP` (branch `feature/strobed-sweep`), **UR** = `UC2-REST/uc2rest`,
**IS** = `imswitch/imcontrol`.

**Status of the claims below**
- Checked by hand: the normal-mode light-off ordering, `sendTrigger`'s payload, the missing
  LatestFrameStore, and the virtual-camera stale frames, now fixed.
- Read but not checked by hand: the rest of the file:line references.
- Not measured: the jitter figures (estimates).

## 1. Firmware today

### `stagescan` (`FW/main/src/motor/StageScan.cpp`)

The engine behind performance mode. Each frame runs `exposeFrame` (:284-291):

1. Light on through DeviceRouter. A local channel is an LEDC write. A CAN-remote channel is an SDO
   that the slave applies on its next 1 ms tick, and nothing confirms it was applied.
2. `vTaskDelay(tPre)`.
3. `fireTrigger`: the pin goes high for ≥ 1 ms, then low, then a `{"cam":1,"frame":n}` report.
4. `vTaskDelay(tPost)`.
5. Light off.

The task runs at priority 1 on a 1 kHz tick, so expect about 1 ms of jitter on every edge, and
several ms when the light sits on a CAN node.

**The firmware never knows the camera exposure.** `tPost` only means "ms after the trigger".

### `strobesweep` (`StrobeSweep.cpp`)

The stage moves at constant velocity. Each frame:
- the camera pulse comes from a hardware timer (`PulseOut`);
- the flash is a timer one-shot at `t + delayUs` (`LaserStrobe`), with handling time subtracted;
- the position is latched.

Over CAN, a SYNC frame fires the camera on TX_SUCCESS. The LED node arms its own timer from the SYNC
timestamp, and the motor node latches its position into TPDO3.

Expected jitter: µs locally, tens of µs over CAN. But the flash is a **full-on GPIO pulse with PWM
detached**:
- intensity can only be set through the width;
- width ≤ 1000 µs, delay ≤ 100 ms, duty ≤ 5 % (PinConfigDefault.h:288-291);
- one channel at a time.

So it cannot cover a millisecond exposure.

### Other firmware paths

- **Digital-out trigger table:** `delay()` in the main loop; a pulse is a blocking 10 ms.
- **Camera feedback:** none. `CAM_IO1_OUT` is declared but disabled, and no exposure-active input
  exists.

## 2. ImSwitch today, one frame per mode

| Mode | Light | Camera | Frame ↔ light | Open-loop parts |
|---|---|---|---|---|
| Normal wellplate (`experiment_normal_mode.py:661-718`) | Python `set_laser_power`: serial plus `sleep(0.04)` | software trigger (`snapSync`) or frame-number polling | ordering | `t_pre` sleep; **the light goes off only after `save_frame_ome`** (a post-step of the acquire, checked); `keepIlluminationOn="auto"` keeps one channel on through moves and autofocus |
| Aufgaben (`engine.py` `_acquire`, `host.py`) | `setValue` + `setEnabled` over serial | software trigger for the whole run | ordering | relies on the serial ack, which arrives before a CAN slave applies the value; light stays on during channel Z-offset moves; off before saving |
| Performance mode / fast scan | firmware stagescan | firmware TTL, camera on external trigger | camera frame id → metadata row; one missed trigger shifts every later frame | `tPre` (90 ms) with the light on; `tPost` (50 ms) unrelated to the exposure, which this mode never sets |
| StageMap strobed prescan | firmware timer flash (20 µs) | firmware TTL, exposure = `strobeWindowMs` (30 ms) | Hik trigger index or arrival time | flash placed late in the window so all rolling-shutter rows are open; delay, width and period calibrated empirically |
| Virtual microscope | — | — | — | snaps came from a queue of up to 5 old frames, and one shared light value went dark when another source was switched off; **both fixed** (682f2941) |

**Where the asynchrony comes from**
- **Light switching is unconfirmed.** It goes over serial (and SDO over CAN) with unknown latency
  and no "applied" acknowledgement.
- **Windows are made of sleeps.** Light windows are fixed sleeps (`tPre`, `tPost`, 40 ms), not
  handshakes.
- **Exposure is set in two places.** Camera exposure and the firmware light window are set
  separately, or the exposure not at all.
- **Software triggers are late.** They go through the SDK with ms latency, after the light is on.
- **Rolling shutters need more light time.** The light must cover exposure plus readout.
- **Nothing observes the exposure.** Frames are matched by order or id, never by an observed
  exposure.

The `LatestFrameStore` / `grabSynchronized` layer that earlier notes describe is **not on this
branch** (checked). `_beginTriggeredAcquisition` is copied across six controllers.

## 3. Cameras

| Camera | Trigger in | Strobe / exposure-active out |
|---|---|---|
| Hik (`hikcamera.py:1196-1356`) | software, Line0, `TriggerDelay`, `GlobalReset` | not implemented (no LineSelector / StrobeEnable) |
| Daheng | software, Line2 | SDK `StrobeSwitch` unused |
| AV / Basler | FrameStart on Line* | none |
| Toupcam | `OPTION_TRIGGER` | SDK IoControl strobe unused |
| Tucsen | capability only; live view forces sequence mode (unverified) | none |

Other gaps on the camera side:
- `setDetectorTriggerType` takes strings that differ per camera.
- No manager supports `ExposureMode=TriggerWidth`, where the trigger pulse width sets the exposure.

## 4. Three ways to close the loop

**A. The firmware owns the frame.**
- ImSwitch sends exposure (plus readout for rolling shutter) and intensity once. A hardware timer
  sequences light on (PWM preset, gated from the ISR), the camera trigger after a pre-roll, and
  light off after exposure + readout. One round trip per frame, or none inside a scan.
- **Changes:**
  - Firmware: generalise `LaserStrobe`/`PulseOut` to gated PWM, review the width and duty limits
    for safety, add an exposure field to stagescan and strobesweep.
  - uc2rest: the new payload fields.
  - ImSwitch: one source of truth for the exposure, pushed to camera and firmware; ideally camera
    `TriggerWidth` so one pulse sets both.
- **Precision:** µs locally, tens of µs over CAN (estimated).
- **Risks:** camera and firmware disagreeing on the exposure, LED driver rise time, laser safety
  with long pulses, the single shared timer.

**B. The camera's exposure-active output gates the light.** This is truly closed loop.
- Wire the camera output to the driver enable, or to an ESP32 GPIO interrupt that gates the
  selected channel.
- **Changes:**
  - Firmware: enable `CAM_IO1` as an input and add the ISR gate.
  - ImSwitch: a LineSelector / LineSource / Strobe API for Hik, Daheng, Toupcam and GenICam.
- **Precision:** sub-µs wired directly, a few µs through the ISR, 50 µs to 1 ms over CAN.
- **Risks:** wiring, opto-output polarity, cameras without an output line, late CAN-remote LEDs.

**C. ImSwitch sequences it in software, with handshakes.**
- The firmware acknowledges once the light is applied, CAN slaves included. ImSwitch then fires
  the software trigger, waits for the SDK's exposure-end event (or frame minus readout), and
  switches the light off before saving.
- **Changes:** the smallest of the three. It fixes the ordering bugs of section 2.
- **Precision:** ms to tens of ms. It is not tight.

A sensible order:
1. C first: it fixes the light-after-save and light-during-moves problems in normal mode and in
   Aufgaben.
2. A as the target for fluorescence.
3. B where a camera has an output line.

## 5. Bugs found on the way (not fixed this round)

1. `UR/digitalout.py` `sendTrigger` posts to `/digital_act` with `digitalid`/`digitalval`, which
   the firmware does not know (checked). As a result `UC2ConfigController.sendTrigger` and
   `ESP32LEDLaserManager.sendTrigger` do nothing.
2. `digitalout.set_trigger` sends JSON booleans, but the firmware's `getJsonInt` reads numbers
   only, so the trigger table never enables.
3. `motor.setTrigger` sends `"setTrig"`, which has no firmware handler. `stopStageScanning` calls
   `startStageScanning(stopped=1)`, which raises a TypeError.
4. `TriggerAcquisitionController.snap` returns the previous frame. `setSoftwareTrigger` sets a
   `buffer_size` that HikCamManager rejects.
5. Tucsen maps "Internal trigger" to continuous.
6. Performance mode:
   - `tPost` defaults to 50 ms regardless of exposure, and per-channel exposures are ignored.
   - It registers the camera-trigger callback on `UC2ConfigManager.ESP32` but unregisters it on
     `mStage._rs232manager._esp32`.
7. `startFastStageScanAcquisition` problems:
   - It stops acquisition and never restarts it or sets the trigger source.
   - Its timelapse wait exits at once.
   - In an Aufgaben job the camera stays in software-trigger mode, while `fast_stage_scan` expects
     TTL pulses (unverified).
8. StrobeSweep refuses to start during a stagescan, but not the other way round; both drive
   `CAMERA_TRIGGER_PIN`.

## 6. Open questions for the next round

- Which cameras have an output line on the openUC2 rigs (Hik models in use)?
- Global or rolling shutter on the target camera? This decides whether the light covers the
  exposure only or the exposure plus readout.
- Is light-off before readout needed for every frame, or only for fluorescence channels?
  Brightfield LEDs could stay on.
- Is a per-frame serial round trip acceptable (option A per frame), or must the firmware run
  whole sequences?
