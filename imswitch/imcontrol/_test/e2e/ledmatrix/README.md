# LED Matrix

The LED matrix accepts commands and its light reaches the camera. Separate from
[`../lightsource/`](../lightsource/) because `LEDMatrixController` is not in the light list.

| Test | Checks | Proves |
|---|---|---|
| `test_ledmatrix_smoke.py::test_all_leds_off_is_accepted` | `setAllLEDOff` answers 200 | the API path |
| `test_ledmatrix_photon.py::test_led_matrix_is_visible_to_camera` | matrix on vs dark, per camera | photons hit the sensor |

## Run

```bash
./run_ledmatrix_test.sh
./run_ledmatrix_test.sh --measure   # print the numbers, PHOTON_MIN_DELTA=0
```

| Knob | Default | |
|---|---|---|
| `LEDMATRIX_INTENSITY` | `500` | per-channel value, white |
| `PHOTON_*`, `AUTO_EXPOSURE_RESET_MS`, `IMSWITCH_DETECTOR` | see [`../lightsource/`](../lightsource/) | same criterion |

## Worth knowing

- The controller has setters only; nothing reads back whether the matrix is lit.
- Skips without `LEDMatrixController` in the setup.
- The dark frame can be all zeros in a closed enclosure, so only the bright
  frame is required to be non-black.
