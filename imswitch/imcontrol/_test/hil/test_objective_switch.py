"""Switching the objective changes the image, and light still reaches the camera through it.

MOVES REAL HARDWARE: turns the turret, applies the Z offset stored for the slot, and runs the
autofocus (moves Z). SWITCHES LIGHT.

The turret switches once; both tests read that one measurement. Afterwards the LED is off and
the turret is back on the slot it started from, also when the measurement failed.
"""
import contextlib
import io
import time

import pytest
from PIL import Image, ImageChops, ImageStat

import hil_client as client

SETTLE_S = 0.5                 # wait after the turret, the autofocus or a light has changed
MOVE_TIMEOUT = 120             # seconds the turret may take
AUTOFOCUS_TIMEOUT = 180
AUTOFOCUS_RANGE_UM = 100
AUTOFOCUS_STEP_UM = 10
AUTO_EXPOSURE_RESET_MS = 1500  # ImSwitch returns to manual exposure after this time
FRAME_RESIZE = 0.25            # the frames are compared at this share of their size
# The image must change by max(MIN_CHANGE, noise * NOISE_FACTOR) grey values.
# The noise is the largest difference between NOISE_SAMPLES frames of an unchanged scene.
MIN_CHANGE = 1.5
NOISE_FACTOR = 1.5
NOISE_SAMPLES = 6
NOISE_DELAY_S = 0.5


# --- objective turret and autofocus -----------------------------------------------------------

def objective_status():
    return client.get_json("/ObjectiveController/getstatus")


def switch_objective(slot):
    """Turn the turret to ``slot`` (with the slot's Z offset) and wait; returns the status.

    A busy turret or an unconfigured slot skips the test, any other problem fails it.
    """
    started = client.get_json("/ObjectiveController/moveToObjective",
                              {"slot": slot, "skipZ": False})
    if not started.get("accepted"):
        if started.get("reason") in ("slot_not_configured", "busy"):
            pytest.skip(f"objective slot {slot} unavailable: {started}")
        pytest.fail(f"objective slot {slot} refused: {started}")

    def finished():
        # moveToObjective only starts the move, so the status tells when this request is done.
        status = objective_status()
        done = (not status.get("isMovingObjective")
                and status.get("lastCompletedObjectiveRequestId") == started["requestId"])
        return status if done or status.get("lastObjectiveMoveError") else None

    status = client.wait_until(finished, MOVE_TIMEOUT)
    assert status, f"objective move to slot {slot} did not finish in {MOVE_TIMEOUT}s"
    assert not status.get("lastObjectiveMoveError"), (
        f"objective move to slot {slot} failed: {status.get('lastObjectiveMoveError')}")
    assert status.get("currentObjective") == slot, (
        f"objective reports slot {status.get('currentObjective')}, expected {slot}")
    time.sleep(SETTLE_S)
    return status


def autofocus():
    """Run the software autofocus (moves Z) and wait for it.

    An autofocus that does not work skips the test: then the rig cannot measure, which says
    nothing about the objective.
    """
    started = client.get_json("/AutofocusController/autoFocus", timeout=60, params={
        "rangez": AUTOFOCUS_RANGE_UM, "resolutionz": AUTOFOCUS_STEP_UM})
    if started.get("status") == "error":
        pytest.skip(f"autofocus did not start: {started.get('message')}")

    def finished():
        status = client.get_json("/AutofocusController/getAutofocusStatus")
        if status.get("state") in ("error", "aborted"):
            pytest.skip(f"autofocus ended as {status.get('state')!r}")
        done = not status.get("isRunning") and status.get("state") in ("finished", "idle")
        return status if done else None

    status = client.wait_until(finished, AUTOFOCUS_TIMEOUT)
    if status is None:
        client.get_json("/AutofocusController/stopAutofocus")
        pytest.fail(f"autofocus did not finish in {AUTOFOCUS_TIMEOUT}s")
    print(f"\nautofocus: z={status.get('currentZ')}, quality={status.get('lastQuality')}")
    time.sleep(SETTLE_S)


# --- light ------------------------------------------------------------------------------------

def set_light(name, on):
    """Switch one LED on (at the highest value its setup allows) or off."""
    value = client.setup()["lasers"][name]["valueRangeMax"] if on else 0
    client.get_json("/LaserController/setLaserValue", {"laserName": name, "value": value})
    client.get_json("/LaserController/setLaserActive", {"laserName": name, "active": on})
    time.sleep(SETTLE_S)


# --- camera -----------------------------------------------------------------------------------

def take_image(detector):
    """One small greyscale frame (PIL) of the acquisition detector; needs its live view."""
    response = client.get("/RecordingController/snapNumpyToFastAPI", timeout=60,
                          params={"detectorName": detector, "resizeFactor": FRAME_RESIZE})
    assert response.status_code == 200, response.text
    return Image.open(io.BytesIO(response.content)).convert("L")


def image_difference(first, second):
    """Mean absolute grey-value difference between two frames."""
    return ImageStat.Stat(ImageChops.difference(first, second)).mean[0]


def noise_baseline(detector):
    """Measure the camera noise with the scene unchanged.

    Returns (last frame, noise floor, change that counts as a real difference).
    """
    previous = take_image(detector)
    drifts = []
    for _ in range(NOISE_SAMPLES - 1):
        time.sleep(NOISE_DELAY_S)
        current = take_image(detector)
        drifts.append(image_difference(previous, current))
        previous = current
    noise_floor = max(drifts)
    return previous, noise_floor, max(MIN_CHANGE, noise_floor * NOISE_FACTOR)


@contextlib.contextmanager
def live_view(detector):
    """Keep the detector streaming inside a ``with`` block.

    Only a live view that was started here is stopped again.
    """
    # The HTTP status is 200 even when ImSwitch refuses, so only the "status" field tells.
    status = client.post_json("/LiveViewController/startLiveView", {"detectorName": detector},
                              timeout=60).get("status")
    if status == "long_exposure":
        pytest.skip(f"{detector}: exposure too long for live view")
    assert status in ("success", "already_running"), (
        f"{detector}: startLiveView answered {status!r}")
    time.sleep(0.4)
    try:
        yield
    finally:
        if status == "success":
            client.get_json("/LiveViewController/stopLiveView", {"detectorName": detector})


def auto_expose(detector):
    """One auto-exposure pass on the scene as it is now; needs a running live view."""
    client.get_json("/SettingsController/setDetectorExposureOnce",
                    {"detectorName": detector, "resetDelayMs": AUTO_EXPOSURE_RESET_MS})
    time.sleep(AUTO_EXPOSURE_RESET_MS / 1000 + 0.1)


# --- the measurement and its tests ------------------------------------------------------------

@pytest.fixture(scope="module")
def measurement():
    """Switch to another objective and measure; returns the numbers the tests check."""
    setup = client.setup()
    if not setup["objective"]:
        pytest.skip("the setup has no objective turret")
    status = objective_status()
    slots = [slot for slot, configured in enumerate(status.get("slotConfigured") or [])
             if configured]
    if not status.get("isActive") or not status.get("hasMotor") or len(slots) < 2:
        pytest.skip(f"this rig cannot switch between two objectives: {status}")
    # Lasers and LEDs are listed alike under "lasers"; on these rigs only an LED lights the
    # sensor, and only the name tells them apart.
    leds = [name for name in setup["lasers"]
            if "led" in name.lower() and "laser" not in name.lower()]
    if not leds:
        pytest.skip("no LED among the light sources of the setup")
    led = leds[0]
    # Only an acquisition detector looks through the objective.
    detectors = [name for name, detector in setup["detectors"].items()
                 if detector["forAcquisition"]]
    if not detectors:
        pytest.skip("the setup has no acquisition detector")
    detector = detectors[0]
    camera = client.get_json("/SettingsController/getCameraStatus", {"detectorName": detector})
    if "error" in camera or camera.get("status") == "error":
        pytest.skip(f"{detector}: getCameraStatus failed: {camera.get('error')}")
    if camera.get("isMock") or str(camera.get("model", "")).lower() == "mock":
        pytest.skip(f"{detector}: ImSwitch served a mock camera")

    start = status.get("currentObjective")
    if start not in slots:  # a turret that was never calibrated reports no slot
        start = slots[0]
        switch_objective(start)
    target = next(slot for slot in slots if slot != start)
    result = {"led": led}

    with live_view(detector):
        try:
            # 1. Does the switch change the image? LED on, turret still: measure the noise.
            set_light(led, True)
            auto_expose(detector)
            autofocus()
            before, result["switch_noise"], result["switch_required"] = noise_baseline(detector)
            print(f"\nswitching objective: slot {start} -> {target}")
            result["objective"] = switch_objective(target).get("objectiveName")
            result["switch_change"] = image_difference(before, take_image(detector))

            # 2. Does the LED reach the camera through the new objective?
            autofocus()
            auto_expose(detector)  # before the dark frame: dark and bright share the exposure
            set_light(led, False)
            dark, result["light_noise"], result["light_required"] = noise_baseline(detector)
            set_light(led, True)
            bright = take_image(detector)
            result["light_change"] = image_difference(dark, bright)
            result["brightest_pixel"] = bright.getextrema()[1]
        finally:
            set_light(led, False)
            # A move that was interrupted still runs, and ImSwitch refuses a second one meanwhile.
            client.wait_until(lambda: not objective_status().get("isMovingObjective"),
                              MOVE_TIMEOUT)
            if objective_status().get("currentObjective") != start:
                switch_objective(start)
    print(f"\nobjective switch {start} -> {target}: {result}")
    return result


def test_objective_switch_is_visible(measurement):
    m = measurement
    assert m["switch_change"] >= m["switch_required"], (
        f"switching to {m['objective']} changed the image only by {m['switch_change']:.2f}; "
        f"camera noise floor is {m['switch_noise']:.2f}, need >= {m['switch_required']:.2f} - "
        f"the turret may not have moved")


def test_light_is_visible_after_objective_switch(measurement):
    m = measurement
    assert m["brightest_pixel"] > 0, (
        f"{m['led']} is on but every pixel is still 0: no light through {m['objective']}")
    assert m["light_change"] >= m["light_required"], (
        f"{m['led']}: image changed only by {m['light_change']:.2f} through {m['objective']}; "
        f"camera noise floor is {m['light_noise']:.2f}, need >= {m['light_required']:.2f}")
