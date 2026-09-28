# Motor

Every axis moves, and the observation camera sees it. **Moves the stage.**

| Test | Checks | Result |
|---|---|---|
| `test_motor_axis_move.py::test_axis_moves_by_step` | each axis reports `+STEP` (10), then moves back | position readback only |
| `test_motor_motion_camera.py::test_axis_motion_is_visible` | A/X/Y/Z: grey value change above noise | fail if not visible |
| `test_motor_motion_camera.py::test_axis_motion_direction` | X → right, Y → down in the image | fail only on the wrong direction; no clear shift skips |
| `test_motor_motion_camera.py::test_z_motion_changes_scale` | Z changes the apparent image scale | skips while motion is visible; fails only if neither is |

The camera tests first park the stage at the transport position with the
turret on the first objective (`move_to_transport.py`, also runnable alone)
and light the sample with the LED matrix at full brightness.

## Run

```bash
./run_motor_test.sh                     # all motor tests
./run_motor_test.sh '/tmp/motor_tests/test_motor_axis_move.py'
./run_motor_motion_camera.sh            # camera tests only
./run_move_to_transport.sh              # just park the stage
```

The `run_motor_test.sh` argument is a pytest target inside the container;
quote it, a `[X]` parametrise id is a glob.

| Knob | Default | |
|---|---|---|
| `MOTION_CAMERA_DISTANCE_UM` | `3000` | travel for A/X/Y |
| `MOTION_CAMERA_Z_DISTANCE_UM` / `_Z_DIRECTION` | `5000` / `-1` | travel for Z |
| `MOTION_CAMERA_SPEED` / `_SETTLE_MS` | `15000` / `500` | move speed, settle time |
| `MOTION_CAMERA_MIN_RATIO` / `_MIN_RATIO_AZ` | `1.8` / `1.25` | visibility bar for X/Y and A/Z |
| `MOTION_CAMERA_ROI_X1` / `X2` / `Y1` / `Y2` | `150` / `340` / `150` / `355` | X/Y region, 640×360 image |
| `MOTION_CAMERA_TOP_CROP_PERCENT` | `25` | A/Z ignore the image top |
| `MOTION_CAMERA_Z_SCALE_MIN` / `MAX` / `STEP` | `0.90` / `1.10` / `0.002` | Z scale search |
| `MOTION_CAMERA_Z_MIN_SCALE_CHANGE` | `0.001` | Z pass bar: minimum scale change |
| `MOTION_CAMERA_Z_SCALE_NOISE_RATIO` | `1.4` | Z pass bar: multiple of stationary noise |
| `LEDMATRIX_INTENSITY` | `255` | illumination here (the LED matrix test defaults to 500) |
| `TRANSPORT_TIMEOUT` / `TRANSPORT_SPEED` | `120` / `20000` | transport move |

## Worth knowing

- Z moves negative first: the transport move switches the Z hard limits off,
  and `+` runs towards the endstop.
- Every move is undone in a `finally`, so a failed measurement leaves the stage
  where it started.
- The Z scale search runs on a `0.002` grid, so a small real change can land on
  the same step as stationary noise — which is why an unclear scale skips
  while the motion itself is visible.
- `FRAME.json` stores the transport position as `0`, behind the endstops.
