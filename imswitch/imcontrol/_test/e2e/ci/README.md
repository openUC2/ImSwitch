# Hardware-in-the-loop

`hil-run.sh` answers one question: **does this ImSwitch image work on the rig?**
It swaps the image into the running deployment, runs the suite, and always swaps
back. It runs **on the Pi**; `../run_all.sh` tests whatever image already runs.

## Run

The Pi keeps a copy of the suite in `~/hil-e2e`, which is not a git checkout:

```bash
rsync -av --exclude __pycache__ --exclude .DS_Store --exclude ci/reports \
  imswitch/imcontrol/_test/e2e/ pi@<pi>:~/hil-e2e/
ssh -t pi@<pi> '~/hil-e2e/ci/hil-run.sh --image sha-7d3adda --yes'
ssh -t pi@<pi> '~/hil-e2e/ci/hil-run.sh --image sha-7d3adda --yes --tests "board firmware camera"'
```

`-t` matters: without a tty, Ctrl+C only kills your ssh client while the Pi
keeps running on the swapped image.

| Flag | |
|---|---|
| `--image` | bare tag (resolves against `ghcr.io/openuc2/imswitch`) or full ref |
| `--tests` | folders to run; default all |
| `--out` | report directory; default `ci/reports/` (`junit-<tag>-<time>.xml`) |
| `--keep-image` | keep the pulled image |
| `--yes` | required — the suite moves the stage and switches light |

| Exit | |
|---|---|
| `0` | suite passed |
| `1` | a test failed — a finding about the image |
| `2` | the run could not happen — a problem with the rig or the run |

## What it does

1. **Checks** disk space, ImSwitch, the UC2 board, and a lock (one run per rig).
2. **Reads the deployment off the container** — compose project, directory and
   file list. Absolute and relative paths both work; a stale entry for its own
   override from an earlier run is skipped.
3. **Pulls** and **swaps** the image in via an override that sets only the image.
4. **Waits** for the API, checks the container runs the image under test,
   **starts the live view** of every acquisition camera, then waits until each
   detector delivers a frame (observation camera via `snapOverviewImage`; a
   `400` there means none is bound and is not waited on).
5. **Ships** the suite into the container and runs pytest with `--junitxml`.
6. **Restores** on every exit path, Ctrl+C and SIGTERM included, and removes
   the pulled image unless `--keep-image`.

## Knobs

| Variable | Default | |
|---|---|---|
| `IMSWITCH_CONTAINER` | `imswitch-server-1` | the container |
| `HIL_HOST_URL` | `http://localhost:8000/imswitch` | ImSwitch from the Pi |
| `HIL_CONTAINER_URL` | `http://localhost:8001` | ImSwitch from the tests |
| `HIL_MIN_FREE_GB` | `15` | refuse to pull below this |
| `HIL_READY_TIMEOUT` | `180` | seconds for the API after the swap |
| `HIL_DETECTOR_TIMEOUT` | `300` | seconds **per detector** for a first frame |
| `HIL_LOCK_FILE` | `/tmp/hil-run.lock` | the lock |
| `HIL_OVERRIDE_FILE` | `/tmp/hil-override.compose.yml` | the override |

Test knobs (`PHOTON_*`, …) are read from the environment as usual.

## Worth knowing

- A failed restore is loud and prints what to run by hand.
- It never deletes the image the rig runs on — that is also the rollback image.
- `forklift-apply.service` reapplies the pinned image at boot; a swap lasts one run.
- Worst case the detector wait takes `HIL_DETECTOR_TIMEOUT × detectors`, and each
  readiness check of the observation camera saves a snapshot PNG.
