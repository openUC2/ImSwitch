# Camera

Every detector delivers a frame of the right size; the objective turret switches.

| Test | Checks | Hardware |
|---|---|---|
| `test_camera_capture.py::test_camera_returns_image` | a valid image per detector | reads |
| `test_camera_capture.py::test_camera_matches_sensor_resolution` | frame size = `resizeFactor ×` reported size | reads |
| `test_objective_switch.py::test_objective_switch_is_visible` | the image changes when the turret moves | **moves** |
| `test_objective_switch.py::test_light_is_visible_after_objective_switch` | an LED still reaches the sensor afterwards | **moves, light** |

`start_live_view.py` is a helper, not a test: the runners call it before pytest.

## Run

```bash
./run_camera_snap.sh            # capture tests
./run_camera_snap.sh --curl     # quick check from your machine -> /tmp/snap.png
./run_objective_switch.sh       # objective test
OBJECTIVE_SKIP_Z=1 ./run_objective_switch.sh   # turret only, no Z offset
```

| Knob | Default | |
|---|---|---|
| `IMSWITCH_DETECTOR` | all / first acquisition | detector to test |
| `IMSWITCH_EXTERNAL_URL` | `http://<PI_HOST>:8000/imswitch` | URL for `--curl` |
| `OBJECTIVE_SETTLE_MS` | `500` | wait after each move |
| `OBJECTIVE_MOVE_TIMEOUT` | `120` | seconds for a turret move |
| `OBJECTIVE_AUTOFOCUS_TIMEOUT` / `_RANGE` / `_STEP` | `180` / `100` / `10` | autofocus |
| `OBJECTIVE_SKIP_Z` | off | skip the per-slot Z offset |
| `UC2_LASER_VALUE`, `AUTO_EXPOSURE_RESET_MS`, `PHOTON_*` | see [`../laser/`](../laser/) | light test |

## Worth knowing

- `snapNumpyToFastAPI` answers 500 until a live view runs, and snaps only the
  *current* detector — hence `start_live_view.py`, and the observation camera
  (name contains `observ`) goes through `snapOverviewImage`, which saves a PNG
  on the Pi per call.
- Mocks are skipped via `model == "mock"` / `isMock`. `isConnected` is useless:
  it reads `false` on a working Hikrobot camera.
- The expected size comes from `getCameraStatus`, not the setup file — that
  says 1000×1000 where the camera delivers 3072×2048.
- The objective test skips unless the setup has two configured slots
  (`slotConfigured`), is `isActive` and has a motor. Counting `objectiveNames`
  is not enough: a setup without objectives still reports two defaults.
- `--curl` defaults to detector `RPiCam`; set `IMSWITCH_DETECTOR` on other rigs.
