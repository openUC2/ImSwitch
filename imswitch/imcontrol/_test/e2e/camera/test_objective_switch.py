"""Switch the objective, see it in the camera, then re-run the light test.

MOVES REAL HARDWARE: turns the objective turret on axis A, applies the Z offset
the setup stores for the target slot, and runs autofocus, which moves Z again.

The whole sequence runs once per session; both tests read the same measurement.

1. Preconditions: two configured slots, an objective motor, a real acquisition
   camera, at least one LED. Anything missing skips.
2. LED on, one auto exposure pass, autofocus on the current objective.
3. Noise baseline and the frame just before the switch.
4. Switch to the other slot, wait for the move to report done.
5. Frame right after the switch -> test_objective_switch_is_visible. Taken
   before the second autofocus, so the change belongs to the switch alone.
6. Autofocus again, then a second auto exposure pass: the new objective
   focuses in a different plane and passes a different amount of light.
7. LED off, dark baseline, LED on, bright frame
   -> test_light_is_visible_after_objective_switch.
8. Teardown: LED off, back to the slot the rig started on - also after a
   failure, Ctrl+C, SIGTERM or a dropped ssh session (SIGHUP).
"""

import os
import signal
import time

import pytest
import requests


BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")
DETECTOR = os.environ.get("IMSWITCH_DETECTOR")
LASER_VALUE = int(os.environ.get("UC2_LASER_VALUE", "1000"))

SETTLE = float(os.environ.get("OBJECTIVE_SETTLE_MS", "500")) / 1000

MOVE_TIMEOUT = float(os.environ.get("OBJECTIVE_MOVE_TIMEOUT", "120"))
AUTOFOCUS_TIMEOUT = float(os.environ.get("OBJECTIVE_AUTOFOCUS_TIMEOUT", "180"))

AUTOFOCUS_RANGE = int(os.environ.get("OBJECTIVE_AUTOFOCUS_RANGE", "100"))
AUTOFOCUS_STEP = int(os.environ.get("OBJECTIVE_AUTOFOCUS_STEP", "10"))

# Shared with the photon tests, so one value tunes both. Must outlast the timer
# that restores manual exposure, or the next frame is still taken in 'once'
# mode.
AUTO_EXPOSURE_RESET_MS = int(os.environ.get("AUTO_EXPOSURE_RESET_MS", "1500"))

# False keeps the Z offset the setup stores per slot, which is the switch as it
# happens in normal use. Set it to skip Z if the rig has no safe Z travel.
SKIP_Z = os.environ.get("OBJECTIVE_SKIP_Z", "").lower() in ("1", "true", "yes")

POLL = 0.1

# Signals that abort the sequence; the teardown ignores them until it is done.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

# The sequence moves hardware, so it runs once and both tests read the result.
_MEASUREMENT = {}


def api(controller, method, **params):
    """Call an ImSwitch endpoint and return its JSON."""
    response = requests.get(
        f"{BASE_URL}/api/{controller}/{method}",
        params=params,
        timeout=60,
    )

    assert response.status_code == 200, response.text

    return response.json()


def is_observation_camera(detector):
    """True for the observation camera, which has no acquisition path."""
    return "observ" in detector.lower()


def objective_status():
    """Full ObjectiveController status."""
    return api("ObjectiveController", "getstatus")


def leds():
    """Names of the configured LEDs, lasers excluded.

    A laser on this rig is mounted so its beam never reaches the sensor, so it
    cannot serve as the light for this test.
    """
    try:
        sources = api(
            "AcceptanceTestController",
            "getAvailableLightSources",
        ).get("light_sources", [])
    except requests.RequestException:
        return []

    return [
        source["name"]
        for source in sources
        if "led" in source["name"].lower() and "laser" not in source["name"].lower()
    ]


def set_light(name, on):
    """Switch one light source on at LASER_VALUE, or off."""
    api(
        "LaserController",
        "setLaserValue",
        laserName=name,
        value=LASER_VALUE if on else 0,
    )
    api(
        "LaserController",
        "setLaserActive",
        laserName=name,
        active=on,
    )
    time.sleep(SETTLE)


def wait_for_objective(request_id):
    """Block until the objective move finishes, and return the final status.

    moveToObjective only starts a thread and answers immediately, so the frame
    after the switch would otherwise be taken while the turret is still moving.
    """
    deadline = time.time() + MOVE_TIMEOUT

    while time.time() < deadline:
        status = objective_status()

        finished = (
            not status.get("isMovingObjective")
            and status.get("lastCompletedObjectiveRequestId") == request_id
        )

        if finished or status.get("lastObjectiveMoveError"):
            return status

        time.sleep(POLL)

    pytest.fail(f"objective move {request_id} did not finish in {MOVE_TIMEOUT}s")


def switch_objective(slot):
    """Move to one objective slot and wait for it, returning the final status."""
    started = api("ObjectiveController", "moveToObjective", slot=slot, skipZ=SKIP_Z)

    # A refusal is answered with 200 and accepted=False, so the body decides.
    if not started.get("accepted"):
        reason = started.get("reason")

        if reason in ("slot_not_configured", "busy"):
            pytest.skip(f"objective slot {slot} unavailable: {started}")

        pytest.fail(f"objective slot {slot} refused: {started}")

    status = wait_for_objective(started["requestId"])

    assert not status.get("lastObjectiveMoveError"), (
        f"objective move to slot {slot} failed: "
        f"{status.get('lastObjectiveMoveError')}"
    )

    assert status.get("currentObjective") == slot, (
        f"objective reports slot {status.get('currentObjective')}, "
        f"expected {slot}"
    )

    time.sleep(SETTLE)

    return status


def restore_objective(slot):
    """Back to slot, first letting an interrupted move run out.

    Moving while the turret still turns would be refused as busy.
    """
    deadline = time.time() + MOVE_TIMEOUT

    while objective_status().get("isMovingObjective") and time.time() < deadline:
        time.sleep(POLL)

    if objective_status().get("currentObjective") != slot:
        switch_objective(slot)


def interrupt(signum, frame):
    """Turn SIGTERM/SIGHUP into KeyboardInterrupt, so the teardown runs.

    Their default action kills Python without running any finally.
    """
    raise KeyboardInterrupt(signal.Signals(signum).name)


def exposure_ms(camera):
    """Current detector exposure in ms, or None if it cannot be read."""
    parameters = api(
        "SettingsController",
        "getCameraStatus",
        detectorName=camera,
    ).get("parameters") or {}

    return (parameters.get("exposure") or {}).get("value")


def auto_exposure(camera, label):
    """Expose once for the scene as it stands right now.

    Deliberately not the session-scoped fixture from conftest: that one runs a
    single pass per pytest session, so in a full run the photon tests would
    have used it up and both objectives would be measured at their exposure.

    The endpoint answers 200 even when it fails, so the exposure before and
    after is the only evidence that it did anything.
    """
    before = exposure_ms(camera)

    api(
        "SettingsController",
        "setDetectorExposureOnce",
        detectorName=camera,
        resetDelayMs=AUTO_EXPOSURE_RESET_MS,
    )

    # Outlast the timer that restores manual mode, plus a margin for the new
    # exposure to reach the running stream.
    time.sleep(AUTO_EXPOSURE_RESET_MS / 1000 + 0.1)

    after = exposure_ms(camera)

    print(f"\nauto exposure ({label}): {before} ms -> {after} ms")

    return before, after


def autofocus(label):
    """Run autofocus and wait for it, skipping when it cannot do its job.

    A rig whose autofocus refuses to start or ends in error cannot produce a
    meaningful light measurement, but that is a property of the rig rather than
    of the objective under test - so it skips instead of failing.
    """
    started = api(
        "AutofocusController",
        "autoFocus",
        rangez=AUTOFOCUS_RANGE,
        resolutionz=AUTOFOCUS_STEP,
    )

    if started.get("status") == "error":
        pytest.skip(f"autofocus ({label}) did not start: {started.get('message')}")

    deadline = time.time() + AUTOFOCUS_TIMEOUT

    while time.time() < deadline:
        status = api("AutofocusController", "getAutofocusStatus")
        state = status.get("state")

        if state in ("error", "aborted"):
            pytest.skip(f"autofocus ({label}) ended as {state!r}")

        if not status.get("isRunning") and state in ("finished", "idle"):
            print(
                f"\nautofocus ({label}): z={status.get('currentZ')}, "
                f"quality={status.get('lastQuality')}"
            )
            time.sleep(SETTLE)
            return status

        time.sleep(POLL)

    api("AutofocusController", "stopAutofocus")
    pytest.fail(f"autofocus ({label}) did not finish in {AUTOFOCUS_TIMEOUT}s")


@pytest.fixture(scope="module")
def acquisition_camera():
    """The acquisition detector, skipping on a mock or an absent one."""
    detectors = api("SettingsController", "getDetectorNames")

    if not detectors:
        pytest.skip("active setup has no detectors")

    if DETECTOR:
        if DETECTOR not in detectors:
            pytest.skip(f"detector {DETECTOR!r} not found (available: {detectors})")

        camera = DETECTOR
    else:
        # The observation camera looks through no objective, so it cannot show
        # the switch the way the acquisition camera does.
        candidates = [
            name for name in detectors if not is_observation_camera(name)
        ]

        if not candidates:
            pytest.skip("setup has no acquisition detector")

        camera = candidates[0]

    status = api("SettingsController", "getCameraStatus", detectorName=camera)

    if "error" in status:
        pytest.skip(f"{camera}: getCameraStatus failed: {status['error']}")

    if status.get("isMock") or str(status.get("model", "")).lower() == "mock":
        pytest.skip(
            f"{camera}: ImSwitch served a mock camera "
            f"(model={status.get('model')!r}), no hardware attached"
        )

    return camera


@pytest.fixture(scope="module")
def two_objectives():
    """Skip unless this rig really can switch between two objectives.

    Counting objectiveNames is not enough: a setup without an objective block
    still reports two default names. Only slotConfigured (a name plus a
    magnification above zero), isActive and hasMotor separate a real turret
    from that fallback, and without a motor the controller changes the reported
    slot without moving anything.
    """
    try:
        status = objective_status()
    except requests.RequestException as exc:
        pytest.skip(f"ImSwitch not reachable at {BASE_URL}: {exc}")

    if not status.get("isActive"):
        pytest.skip("objective support is not active in this setup")

    configured = [
        slot for slot, ok in enumerate(status.get("slotConfigured") or []) if ok
    ]

    if len(configured) < 2:
        pytest.skip(
            f"setup has fewer than two configured objectives: "
            f"names={status.get('availableObjectivesNames')}, "
            f"slotConfigured={status.get('slotConfigured')}"
        )

    if not status.get("hasMotor"):
        pytest.skip("no objective motor: the slot would change without moving")

    return status


@pytest.fixture(scope="module", autouse=True)
def camera_acquisition(two_objectives, acquisition_camera):
    """Keep the acquisition camera streaming for the duration of this module."""
    response = requests.post(
        f"{BASE_URL}/api/LiveViewController/startLiveView",
        params={"detectorName": acquisition_camera},
        timeout=30,
    )

    assert response.status_code == 200, response.text
    status = response.json().get("status")

    # startLiveView answers 200 even when it declines, so the body decides.
    if status == "long_exposure":
        pytest.skip(f"{acquisition_camera}: exposure too long for live view")

    # "already_running" is fine: an active stream is all this module needs.
    assert status in ("success", "already_running"), (
        f"{acquisition_camera}: startLiveView did not start a stream: {status}"
    )

    time.sleep(0.4)

    yield

    # Only hand back what was taken: a stream that was already running belongs
    # to whoever started it.
    if status == "success":
        api("LiveViewController", "stopLiveView", detectorName=acquisition_camera)


def run_sequence(camera, light, start_slot, target_slot,
                 take_image, image_difference, measure_dark_baseline):
    """Switch the objective and measure, leaving the rig as it was found."""
    result = {
        "camera": camera,
        "light": light,
        "start_slot": start_slot,
        "target_slot": target_slot,
    }

    original = {sig: signal.getsignal(sig) for sig in STOP_SIGNALS}
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)

    try:
        set_light(light, True)

        # Expose before the baseline, never between the two frames: the frame
        # before and the frame after the switch have to share one exposure, or
        # the measured change is partly the exposure changing.
        auto_exposure(camera, f"slot {start_slot}")
        autofocus(f"slot {start_slot}")

        # The baseline is measured with the light on and the turret still, so
        # its noise floor describes this scene rather than a dark sensor. It
        # hands back the last frame, which is the one before the switch.
        before, noise_floor, required_change = measure_dark_baseline(camera)

        print(f"\nswitching objective: slot {start_slot} -> {target_slot}")
        switched = switch_objective(target_slot)

        after = take_image(camera)

        result.update(
            objective_name=switched.get("objectiveName"),
            switch_change=image_difference(before, after),
            switch_noise_floor=noise_floor,
            switch_required=required_change,
        )

        # Only now refocus: the frame above had to show the switch alone.
        autofocus(f"slot {target_slot}")

        # The new objective passes a different amount of light, so expose for
        # it before the light test - and before the dark frame, so that dark
        # and bright share one exposure too.
        auto_exposure(camera, f"slot {target_slot}")

        set_light(light, False)
        dark, light_noise, light_required = measure_dark_baseline(camera)

        set_light(light, True)
        bright = take_image(camera)

        result.update(
            light_change=image_difference(dark, bright),
            light_noise_floor=light_noise,
            light_required=light_required,
            bright_max=bright.getextrema()[1],
        )

    finally:
        # A second Ctrl+C must not cut the way back short.
        for sig in STOP_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)

        try:
            try:
                set_light(light, False)
            finally:
                # Leave the rig on the objective it started on, whatever happened.
                restore_objective(start_slot)
        finally:
            for sig, handler in original.items():
                signal.signal(sig, handler)

    return result


@pytest.fixture
def objective_switch(
    two_objectives,
    acquisition_camera,
    take_image,
    image_difference,
    measure_dark_baseline,
):
    """Run the switch sequence once and hand every test the same result."""
    if "result" not in _MEASUREMENT:
        lights = leds()

        if not lights:
            pytest.skip("no LED among the configured light sources")

        start_slot = two_objectives.get("currentObjective")
        configured = [
            slot
            for slot, ok in enumerate(two_objectives.get("slotConfigured") or [])
            if ok
        ]

        # An uncalibrated rig reports no current slot; start from the first
        # configured one so the switch still has somewhere to come from.
        if start_slot not in configured:
            start_slot = configured[0]
            switch_objective(start_slot)

        target_slot = next(slot for slot in configured if slot != start_slot)

        _MEASUREMENT["result"] = run_sequence(
            acquisition_camera,
            lights[0],
            start_slot,
            target_slot,
            take_image,
            image_difference,
            measure_dark_baseline,
        )

    return _MEASUREMENT["result"]


@pytest.mark.hardware
def test_objective_switch_is_visible(objective_switch):
    """The camera image must change when the turret moves to the other slot."""
    change = objective_switch["switch_change"]
    noise_floor = objective_switch["switch_noise_floor"]
    required = objective_switch["switch_required"]

    print(
        f"\nobjective switch {objective_switch['start_slot']} -> "
        f"{objective_switch['target_slot']} "
        f"({objective_switch['objective_name']}): "
        f"pixel_change={change:.2f} "
        f"noise_floor={noise_floor:.2f} "
        f"required={required:.2f}"
    )

    assert change >= required, (
        f"objective switch changed the image only by {change:.2f}; "
        f"camera noise floor is {noise_floor:.2f}, need >= {required:.2f} - "
        f"the turret may not have moved"
    )


@pytest.mark.hardware
def test_light_is_visible_after_objective_switch(objective_switch):
    """The LED must still reach the sensor through the new objective."""
    change = objective_switch["light_change"]
    noise_floor = objective_switch["light_noise_floor"]
    required = objective_switch["light_required"]

    print(
        f"\n{objective_switch['light']} on "
        f"{objective_switch['objective_name']}: "
        f"pixel_change={change:.2f} "
        f"noise_floor={noise_floor:.2f} "
        f"required={required:.2f}"
    )

    assert objective_switch["bright_max"] > 0, (
        f"{objective_switch['light']} is on but every pixel is still 0; "
        f"no light reached the sensor through "
        f"{objective_switch['objective_name']}"
    )

    assert change >= required, (
        f"{objective_switch['light']}: image changed only by {change:.2f} "
        f"through {objective_switch['objective_name']}; "
        f"camera noise floor is {noise_floor:.2f}, need >= {required:.2f}"
    )
