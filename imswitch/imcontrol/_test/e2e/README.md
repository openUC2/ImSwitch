# End-to-End Tests

Tests against a running ImSwitch on the real rig, over its HTTP API only.
Sits alongside `../unit/` (no server) and `../api/` (headless server).

| Folder | What | Hardware |
|---|---|---|
| [`board/`](board/) | UC2 board connected, right master firmware | reads |
| [`camera/`](camera/) | every detector delivers a frame; objective switch | reads · objective test **moves** |
| [`firmware/`](firmware/) | firmware server; every board runs its version | reads · sync **flashes** |
| [`lightsource/`](lightsource/) | lasers/LEDs switch; LEDs reach the camera | **light** |
| [`ledmatrix/`](ledmatrix/) | LED matrix reaches the camera | **light** |
| [`motor/`](motor/) | axes move; camera sees it | **moves** |
| [`ci/`](ci/) | swap an image in, run the suite, swap back | **moves, light** |

## Run

```bash
./run_all.sh                    # everything
./run_all.sh camera             # one folder
./run_all.sh lightsource ledmatrix    # several
```

`run_all.sh` ships this folder into the container on the Pi, **syncs every
board to the firmware server's version** (`firmware/sync_firmware.py`, master
first; `FIRMWARE_UPDATE=off` skips it),
**parks the stage** on objective slot 0
(`motor/move_to_transport.py`), starts the camera stream
(`camera/start_live_view.py`) and runs pytest. Each folder also has its own
runner, see its README.

Runners take `PI_HOST` (default `pi@192.168.178.124`), `IMSWITCH_CONTAINER`
(default `imswitch-server-1`) and forward every test knob you set — any
variable starting with `PHOTON_`, `LEDMATRIX_`, `MOTION_CAMERA_`, `TRANSPORT_`,
`OBJECTIVE_`, `FIRMWARE_`, `AUTO_EXPOSURE_`, plus `IMSWITCH_DETECTOR` and
`UC2_LASER_VALUE`. Unset knobs keep the test's default.

```bash
PHOTON_MIN_DELTA=3 LEDMATRIX_INTENSITY=40 ./run_all.sh ledmatrix
```

## Two URLs

| From | `IMSWITCH_URL` |
|---|---|
| inside the container (where the runners put pytest) | `http://localhost:8001` |
| your machine | `http://<pi>:8000/imswitch` (through caddy) |

`localhost:8000` works in neither place. Straight pytest from your machine:

```bash
IMSWITCH_URL=http://<pi>:8000/imswitch python3 -m pytest imswitch/imcontrol/_test/e2e -v
```

## Rules

- Missing hardware or an unreachable ImSwitch **skips**, never fails.
- The photon tests are the exception: once they run, no light is a **failure**.
- Everything goes through HTTP, so nothing competes with ImSwitch for the serial port.
- Nothing waits or retries inside a test — that would hide real regressions.

## Worth knowing

- After a container restart ImSwitch needs ~30 s to listen. The runners do not
  wait for it; started earlier, everything skips.
- `snapNumpyToFastAPI` answers 500 until a live view runs, and only ever snaps
  the *current* detector. `start_live_view.py` handles the first; the
  observation camera goes through `snapOverviewImage` instead.
- `colors.sh` is sourced by the runners and only decides whether pytest gets
  `--color=yes` (neither `ssh` nor `docker exec` gives it a tty).
