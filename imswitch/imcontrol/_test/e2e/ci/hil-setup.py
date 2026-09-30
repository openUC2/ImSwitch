#!/usr/bin/env python3
"""Swap one ImSwitch image into the rig's deployment, and swap it back again.

The rig side of hil-run.sh, which calls it; that script keeps the arguments,
the lock and the suite itself. Runs ON the Pi, against the local Docker
daemon, with nothing but the standard library -- the host has no requests.

    hil-setup.py swap-in --image sha-7d3adda   # check, swap it in, wait until ready
    hil-setup.py restore [--keep-image]        # put the original image back

swap-in leaves the rig on the test image until restore runs. hil-run.sh runs
restore on every exit path; by hand, you have to.

Why swap instead of installing: forklift owns the deployment, and the image it
pins is the one the device is supposed to run. The swap is a temporary
override that restore removes again, so a finished run -- or a failed one, or
an interrupted one -- leaves the same container the user had.

swap-in writes what restore needs into a state file before it changes
anything. A state file that is still there when swap-in starts means an
earlier run was never restored, and swap-in refuses to swap on top of it.

Exit codes: 0 done, 2 the step could not be performed.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


CONTAINER = os.environ.get("IMSWITCH_CONTAINER", "imswitch-server-1")

# ImSwitch as seen from the Pi itself: :8001 is not published on the host, so
# every host-side request goes through caddy.
HOST_URL = os.environ.get("HIL_HOST_URL", "http://localhost:8000/imswitch")

# An ImSwitch image is ~6.3 GB, and the card has been at 74% before. Refuse to
# pull rather than fill the root filesystem of a machine someone else uses.
MIN_FREE_GB = int(os.environ.get("HIL_MIN_FREE_GB", "15"))

# How long ImSwitch may take to answer after a container swap. It loads the
# setup file, opens the camera SDK and talks to the ESP32 before serving.
READY_TIMEOUT = int(os.environ.get("HIL_READY_TIMEOUT", "180"))

# How long one detector may take to deliver its first frame, counted per
# detector. Deliberately separate from READY_TIMEOUT: /api/version answers as
# soon as the web server is up, while a camera SDK can still be enumerating
# behind it. The Hikrobot camera has needed minutes there, and snapping it in
# that window answers 500 -- which would look like a broken image rather than
# a rig that is not warm yet.
DETECTOR_TIMEOUT = int(os.environ.get("HIL_DETECTOR_TIMEOUT", "300"))

# The override lives in /tmp, never in the forklift stage directory: that
# directory belongs to forklift, and this file must not survive us. It only
# sets an image, so it needs no path resolution of its own.
OVERRIDE_FILE = os.environ.get("HIL_OVERRIDE_FILE", "/tmp/hil-override.compose.yml")

# What restore needs to know about the deployment swap-in changed.
STATE_FILE = os.environ.get("HIL_STATE_FILE", "/tmp/hil-state.json")

DEFAULT_REGISTRY = "ghcr.io/openuc2/imswitch"

POLL = 5

EXIT_OK = 0
EXIT_UNAVAILABLE = 2


class StepFailed(Exception):
    """swap-in or restore cannot go on; the message says why."""


def log(message):
    print(f"[hil-setup] {message}", flush=True)


def warn(message):
    print(f"[hil-setup] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# HTTP and docker.
# ---------------------------------------------------------------------------

def imswitch_request(path, method="GET", timeout=15, **params):
    """(status, body) of one ImSwitch call through caddy; status None if no answer."""
    url = f"{HOST_URL}/api/{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = b"" if method == "POST" else None

    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=data, method=method), timeout=timeout
        ) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (OSError, ValueError):
        return None, b""


def imswitch_json(path, timeout=15):
    """The JSON of a 200 answer, or None for anything else."""
    status, body = imswitch_request(path, timeout=timeout)
    if status != 200:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def imswitch_version():
    status, body = imswitch_request("version", timeout=10)
    return body.decode(errors="replace").strip() if status == 200 else ""


def wait_for_imswitch():
    waited = 0
    while waited < READY_TIMEOUT:
        if imswitch_version():
            return True
        time.sleep(POLL)
        waited += POLL
    return False


def run_logged(*command):
    """Run a command, indenting its output into our log; True if it succeeded."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(f"[hil-setup]   {line.rstrip()}", flush=True)
    return process.wait() == 0


def inspect_container():
    """docker inspect of the container, or None if it does not exist."""
    result = subprocess.run(
        ["docker", "inspect", CONTAINER], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)[0]


def container_image():
    """The image the container runs right now."""
    info = inspect_container()
    return info["Config"]["Image"] if info else ""


def compose_up_command(state, with_override=False):
    """docker compose up for the deployment, with or without our override."""
    command = ["docker", "compose", *state["compose_args"]]
    if with_override:
        command += ["-f", OVERRIDE_FILE]
    return [*command, "up", "-d", "--no-build"]


# ---------------------------------------------------------------------------
# Preconditions. Each one is something that would otherwise fail halfway
# through, with the rig already swapped.
# ---------------------------------------------------------------------------

def check_preconditions():
    # hil-run.sh runs restore after this refusal too, which restores that
    # run; by hand, run restore yourself.
    if os.path.exists(STATE_FILE):
        raise StepFailed(
            f"{STATE_FILE} is left over from a run that was never restored "
            "-- not swapping on top of it"
        )

    if not shutil.which("docker"):
        raise StepFailed("docker not found -- run this on the Pi")

    if inspect_container() is None:
        raise StepFailed(f"container {CONTAINER} not found -- is ImSwitch deployed?")

    free_gb = shutil.disk_usage("/").free // 2**30
    if free_gb < MIN_FREE_GB:
        raise StepFailed(f"only {free_gb}G free on / , need {MIN_FREE_GB}G for the image")

    if not imswitch_version():
        raise StepFailed(f"ImSwitch does not answer at {HOST_URL} -- not swapping anything")

    status, body = imswitch_request("UC2ConfigController/uc2_board_is_connected")
    if status != 200 or body.strip() != b"true":
        answer = body.decode(errors="replace").strip() or "none"
        raise StepFailed(
            f"UC2 board not connected (answer: {answer}) -- the tests would all skip"
        )


# ---------------------------------------------------------------------------
# Where the deployment lives. Read from the container rather than hardcoded:
# forklift moves the package to a new stage directory on every apply, and the
# feature set decides which compose files take part.
# ---------------------------------------------------------------------------

def read_deployment():
    info = inspect_container()
    labels = info["Config"].get("Labels") or {}

    project = labels.get("com.docker.compose.project", "")
    pkg_dir = labels.get("com.docker.compose.project.working_dir", "")
    config_files = labels.get("com.docker.compose.project.config_files", "")

    if not (project and os.path.isdir(pkg_dir) and config_files):
        raise StepFailed(f"cannot read the compose setup off {CONTAINER}")

    # Rebuild the exact -f list the deployment uses, in order: the later files
    # carry the device rules and the caddy labels, and dropping one would start
    # a container that cannot see the hardware or cannot be reached through
    # caddy.
    compose_args = ["--project-name", project, "--project-directory", pkg_dir]
    for file in config_files.split(","):
        # forklift writes absolute paths into the label; a relative one is
        # meant against the package directory.
        path = os.path.join(pkg_dir, file)

        # Our own override can still be listed here: compose records the -f
        # list a container was started with, and a restore that did not have
        # to recreate the container leaves that entry in the label. It is not
        # part of the deployment -- the swap adds it again when it is wanted --
        # and the file is usually gone, so skip it instead of dying on it.
        if path == OVERRIDE_FILE:
            continue

        if not os.path.isfile(path):
            raise StepFailed(f"compose file missing: {path}")
        compose_args += ["-f", path]

    return {
        "project": project,
        "pkg_dir": pkg_dir,
        "compose_args": compose_args,
        "original_image": info["Config"]["Image"],
    }


# ---------------------------------------------------------------------------
# Swap.
# ---------------------------------------------------------------------------

def start_test_image(state):
    """Pull the test image and restart the container on it via the override."""
    test_image = state["test_image"]

    log(f"pulling {test_image}")
    if not run_logged("docker", "pull", test_image):
        raise StepFailed("pull failed -- wrong tag, or not logged in to the registry")

    with open(OVERRIDE_FILE, "w") as override:
        override.write(
            "# Written by hil-setup.py, removed again by its restore step. If you\n"
            "# find this file on a rig, a run was killed hard -- check which image runs.\n"
            "services:\n"
            "  server:\n"
            f"    image: {test_image}\n"
        )

    log("starting the container on the test image")
    if not run_logged(*compose_up_command(state, with_override=True)):
        raise StepFailed("compose up failed")

    log(f"waiting for ImSwitch (up to {READY_TIMEOUT}s)")
    if not wait_for_imswitch():
        raise StepFailed(f"ImSwitch did not answer within {READY_TIMEOUT}s")

    now_running = container_image()
    log(f"now running  {now_running}")
    log(f"reports      {imswitch_version()}")

    if now_running != test_image:
        raise StepFailed(f"the container runs {now_running}, not the test image")


# ---------------------------------------------------------------------------
# Detector readiness. An answering API is not a ready microscope: the camera
# tests snap a frame, and a camera still starting up answers 500. Waiting here
# rather than in the tests keeps the tests honest -- a retry loop inside them
# would hide a camera that genuinely broke.
# ---------------------------------------------------------------------------

def is_observation_camera(detector):
    """Same distinction as conftest.py: the observation camera is not an
    acquisition detector and has to be read through the overview endpoint."""
    return "observ" in detector.lower()


def start_live_view(detector):
    """Start the acquisition loop for one camera, once.

    snapNumpyToFastAPI does not start anything: it reads the newest frame out
    of a ring buffer that only fills while an acquisition loop runs, waits a
    second when that buffer is empty, and then answers 500. On the Hikrobot
    camera the loop is started by the live view, so after a swap the snap keeps
    answering 500 until something starts one -- which is what the readiness
    check below would otherwise wait out in full.

    startLiveView answers 200 even when it declines, so the body decides.
    "already_running" is as good as "success": the stream is what matters, not
    who started it. The stream is left running afterwards -- the suite starts
    its own where it needs one, and the camera is no worse off for it.
    """
    code, body = imswitch_request(
        "LiveViewController/startLiveView", method="POST", timeout=60, detectorName=detector
    )
    if code != 200:
        raise StepFailed(
            f"startLiveView for {detector} answered HTTP {code or 'nothing'} "
            "-- the camera cannot deliver frames"
        )

    try:
        status = json.loads(body).get("status")
    except (ValueError, AttributeError):
        status = None

    if status not in ("success", "already_running"):
        raise StepFailed(
            f"startLiveView for {detector} reported {status or 'no status'} "
            "-- the camera would never deliver a frame"
        )
    log(f"detector    {detector} live view {status}")


def snap_status_code(detector):
    """HTTP status of one snap. Asking the wrong endpoint would wait out the
    full timeout on a camera that was ready all along."""
    if is_observation_camera(detector):
        code, _ = imswitch_request(
            "ExperimentController/snapOverviewImage", method="POST", timeout=30,
            slot_id=1, camera_name="hil_ready_check",
        )
    else:
        code, _ = imswitch_request(
            "RecordingController/snapNumpyToFastAPI", timeout=30,
            detectorName=detector, resizeFactor=0.1,
        )
    return code


def wait_for_detector(detector):
    started = time.monotonic()
    code = None

    while time.monotonic() - started < DETECTOR_TIMEOUT:
        code = snap_status_code(detector)

        if code == 200:
            log(f"detector    {detector} ready after {time.monotonic() - started:.0f}s")
            return True

        # 400 is the overview endpoint saying no overview camera is bound. No
        # amount of waiting changes that, and the suite skips those tests
        # rather than failing them, so it must not hold up the run either.
        if code == 400:
            log(f"detector    {detector}: no overview camera bound (400), not waiting")
            return True

        time.sleep(POLL)

    warn(f"detector    {detector} still answers {code or 'nothing'} after {DETECTOR_TIMEOUT}s")
    return False


def wait_for_detectors():
    log(f"waiting for every detector to deliver a frame (up to {DETECTOR_TIMEOUT}s each)")

    # Read the detectors from the setup instead of naming them here: which
    # cameras exist depends on the setup file the test image loads.
    detectors = imswitch_json("SettingsController/getDetectorNames") or []
    if not detectors:
        raise StepFailed("ImSwitch reports no detectors -- the camera tests would all skip")

    # Every detector is waited out even after one fails, so the log shows how
    # long each of them really took and whether the timeout is set anywhere
    # near right.
    not_ready = []
    for detector in detectors:
        # An acquisition camera has to be streaming before it can be snapped
        # at all; the observation camera is read straight off the device and
        # needs nothing started.
        if not is_observation_camera(detector):
            start_live_view(detector)

        if not wait_for_detector(detector):
            not_ready.append(detector)

    # Exit 2, not a test failure: a camera that never woke up says nothing
    # about the image, and CI must not page anyone about a red test that never
    # ran.
    if not_ready:
        raise StepFailed(
            f"detector(s) not ready within {DETECTOR_TIMEOUT}s: {' '.join(not_ready)} "
            "-- not running the suite"
        )


# ---------------------------------------------------------------------------
# The two commands.
# ---------------------------------------------------------------------------

def swap_in_test_image(image):
    """swap-in: check the rig, put the test image in, wait until it is ready."""
    # A bare tag is the common case; a full ref stays untouched, so a fork's
    # own registry works without a flag.
    test_image = image if "/" in image else f"{DEFAULT_REGISTRY}:{image}"

    check_preconditions()
    state = read_deployment()
    state["test_image"] = test_image

    log(f"container   {CONTAINER} (project {state['project']})")
    log(f"package     {state['pkg_dir']}")
    log(f"running     {state['original_image']}")
    log(f"testing     {test_image}")

    # Written before the first change, so restore can undo whatever part of
    # the swap happened, whenever it is interrupted.
    with open(STATE_FILE, "w") as file:
        json.dump(state, file)

    start_test_image(state)
    wait_for_detectors()


def restore_original_image(keep_image):
    """restore: put the original image back; a no-op if nothing was swapped."""
    try:
        with open(STATE_FILE) as file:
            state = json.load(file)
    except FileNotFoundError:
        log("nothing was swapped, nothing to restore")
        return

    original_image = state["original_image"]
    test_image = state["test_image"]

    log(f"restoring {original_image}")

    if os.path.exists(OVERRIDE_FILE):
        os.remove(OVERRIDE_FILE)

    command = compose_up_command(state)
    if not run_logged(*command):
        # The state file stays, so restore can simply be run again once the
        # cause is fixed.
        warn(f"RESTORE FAILED -- the rig may still run {test_image}")
        warn(f"fix by hand: {shlex.join(command)}")
        raise StepFailed(f"or retry:   {sys.argv[0]} restore")

    if wait_for_imswitch():
        log("restored, ImSwitch answers again")
    else:
        warn("restored the container, but ImSwitch does not answer yet")

    # Only ever remove what this run pulled, and never the image the rig runs
    # on: the card holds the rollback image too, and that one must stay.
    if not keep_image and test_image != original_image:
        removed = subprocess.run(
            ["docker", "rmi", test_image], capture_output=True
        ).returncode == 0
        log(f"removed {test_image}" if removed else f"kept {test_image} (still in use)")

    os.remove(STATE_FILE)


def main():
    parser = argparse.ArgumentParser(description="Swap an ImSwitch image in and out.")
    commands = parser.add_subparsers(dest="command", required=True)

    swap_in = commands.add_parser(
        "swap-in", help="check the rig, swap the test image in, wait until ready"
    )
    swap_in.add_argument("--image", required=True, help="bare tag or full image ref")

    restore = commands.add_parser("restore", help="put the original image back")
    restore.add_argument("--keep-image", action="store_true", help="keep the pulled image")

    args = parser.parse_args()

    try:
        if args.command == "swap-in":
            swap_in_test_image(args.image)
        else:
            restore_original_image(args.keep_image)
    except StepFailed as error:
        warn(str(error))
        return EXIT_UNAVAILABLE
    except KeyboardInterrupt:
        warn(f"{args.command} interrupted")
        return EXIT_UNAVAILABLE

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
