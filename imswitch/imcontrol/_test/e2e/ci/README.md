# Hardware-in-the-loop

`hil-run.sh` answers one question: **does this ImSwitch image, or this os-rpi
pallet, work on the rig?** It swaps it into the running deployment, runs the
suite, and always swaps back. It runs **on the Pi**; `../run_all.sh` tests
whatever already runs.

- `--image` swaps only that ImSwitch image in, on the pallet the rig runs.
- `--pallet` applies every Docker deployment of an os-rpi commit (ImSwitch, the
  firmware server, caddy, …) with forklift. OS files under `/etc` and `/usr`
  only change at the next boot, so they are not part of the test.

Two files, one job:

- `hil-setup.py` — the rig side, standard library only (the Pi host has no
  `requests`): `swap-in` checks the rig, swaps the image or pallet in and waits
  until every detector delivers a frame; `restore` puts the rig back.
- `hil-run.sh` — the rest: arguments, the lock, `swap-in`, shipping the suite,
  pytest, and `restore` on every exit path.

## Run

The Pi keeps a copy of the suite in `~/hil-e2e`, which is not a git checkout:

```bash
rsync -av --exclude __pycache__ --exclude .DS_Store --exclude ci/reports \
  imswitch/imcontrol/_test/e2e/ pi@<pi>:~/hil-e2e/
ssh -t pi@<pi> '~/hil-e2e/ci/hil-run.sh --image sha-7d3adda --yes'
ssh -t pi@<pi> '~/hil-e2e/ci/hil-run.sh --image sha-7d3adda --yes --tests "board firmware camera"'
ssh -t pi@<pi> '~/hil-e2e/ci/hil-run.sh --pallet github.com/openUC2/os-rpi@<commit> --yes'
```

`-t` matters: without a tty, Ctrl+C only kills your ssh client while the Pi
keeps running on the swapped image.

| Flag | |
|---|---|
| `--image` | bare tag (resolves against `ghcr.io/openuc2/imswitch`) or full ref |
| `--pallet` | `<path>@<version>` of an os-rpi pallet, e.g. a commit of a fork; instead of `--image` |
| `--tests` | folders to run; default all |
| `--out` | report directory; default `ci/reports/` (`junit-<tag or commit>-<time>.xml`) |
| `--keep-image` | keep the images the swap pulled |
| `--yes` | required — the suite moves the stage and switches light |

| Exit | |
|---|---|
| `0` | suite passed |
| `1` | a test failed — a finding about the image |
| `2` | the run could not happen, or the restore failed — a problem with the rig or the run |

## What it does

1. **Locks** the rig (`hil-run.sh`, one run per rig).
2. **Checks** disk space, ImSwitch and the UC2 board (`swap-in`).
3. **Reads the deployment off the container** — compose project, directory and
   file list. Absolute and relative paths both work; a stale entry for its own
   override from an earlier run is skipped. Written to the state file before
   anything changes, so `restore` knows what to undo.
4. **Pulls** and **swaps** the image in via an override that sets only the image.

   With `--pallet`, steps 3 and 4 are instead:
   refuses a local pallet with uncommitted or unpushed changes, and a stage that
   is staged but not applied; writes the applied stage, the local pallet's
   source and commit and its upgrade query to the state file; `forklift plt
   switch` to the test pallet (without remembering it as the upgrade query and
   without forklift's image caching); pulls the images the card does not have
   yet; `sudo -E forklift stage apply`.
5. **Waits** for the API, checks the container runs the image under test,
   **starts the live view** of every acquisition camera, then waits until each
   detector delivers a frame (observation camera via `snapOverviewImage`; a
   `400` there means none is bound and is not waited on).
6. **Ships** the suite into the container (`hil-run.sh`).
7. **Syncs the firmware**: every board to the firmware server's version, the
   master first (`firmware/sync_firmware.py --yes`, driven by the image under
   test). A failed sync only warns; `FIRMWARE_UPDATE=off` skips it.
8. **Runs** pytest with `--junitxml` (`hil-run.sh`).
9. **Restores** on every exit path, Ctrl+C and SIGTERM included (`restore`).
   `--image`: removes the override and the pulled image (unless `--keep-image`).
   The synced firmware stays: it is the firmware server's, which the swap does
   not touch.
   `--pallet`: applies the stage that ran before, then clones the local pallet
   back, sets its upgrade query again and removes the images the swap pulled
   (unless `--keep-image`). The boards keep the firmware the test pallet's
   server synced them to, until the next sync.

By hand, for debugging on a swapped rig — nothing puts it back until `restore`:

```bash
~/hil-e2e/ci/hil-setup.py swap-in --image sha-7d3adda
~/hil-e2e/ci/hil-setup.py swap-in --pallet github.com/openUC2/os-rpi@<commit>
~/hil-e2e/ci/hil-setup.py restore
```

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
| `HIL_STATE_FILE` | `/tmp/hil-state.json` | what `restore` needs to undo `swap-in` |
| `FIRMWARE_UPDATE` | `on` | `off` skips the firmware sync |
| `FORKLIFT_WORKSPACE` | `$HOME` | the forklift workspace `--pallet` swaps the local pallet of |

Test knobs (`PHOTON_*`, …) are read from the environment as usual.

## Worth knowing

- A failed restore is loud, exits `2`, keeps the state file and prints the
  compose command to run by hand; `hil-setup.py restore` retries it.
- A state file left over when `swap-in` starts means an earlier run was never
  restored: `swap-in` refuses to swap, and `hil-run.sh`'s `restore` restores
  that run.
- It never deletes the image the rig runs on — that is also the rollback image.
- `forklift-apply.service` reapplies the pinned image at boot; a swap lasts one run.
- With `--pallet`, a run killed before `restore` leaves the test stage as the
  next one: a reboot would apply it again. `hil-setup.py restore` puts it back.
- Why forklift's image caching is off in `--pallet`: it would also pull every
  image of the stage it falls back to, and a private image it cannot pull again
  (flim-imager) stops it, even when it is on the card.
- The test pallet's stage stays in forklift's stage store as its rollback;
  `forklift stage prune-bun` clears old stages.
- Worst case the detector wait takes `HIL_DETECTOR_TIMEOUT × detectors`, and each
  readiness check of the observation camera saves a snapshot PNG.
