# Light sources

Lasers and LEDs switch through ImSwitch; the LEDs reach the camera.

| Test | Checks | Proves |
|---|---|---|
| `test_lightsource_switching.py::test_lightsource_reports_active` | every light: on → read back active/value → off | the API path, not light |
| `test_lightsource_photon.py::test_light_source_is_visible_to_camera` | every LED, per camera: lit frame vs dark | photons hit the sensor |

Lights come from `AcceptanceTestController/getAvailableLightSources`. The photon
test keeps only LEDs: the 488 laser on this rig never reaches the sensor. It
runs once for the acquisition camera and once for the observation camera.

## Run

```bash
./run_lightsource_test.sh
./run_lightsource_test.sh --measure   # print the numbers, PHOTON_MIN_DELTA=0
```

| Knob | Default | |
|---|---|---|
| `UC2_LASER_VALUE` | `1000` | value a light is switched on at |
| `PHOTON_MIN_DELTA` | `1.5` | minimum pixel change |
| `PHOTON_NOISE_FACTOR` | `1.5` | required change as a multiple of the noise floor |
| `PHOTON_NOISE_SAMPLES` / `_DELAY` | `6` / `0.5` | frames and spacing for the noise floor |
| `AUTO_EXPOSURE_RESET_MS` | `1500` | one-shot auto exposure, once per session |
| `IMSWITCH_DETECTOR` | first acquisition | camera to measure with |

## Worth knowing

- A pass needs `change >= max(PHOTON_MIN_DELTA, noise_floor × PHOTON_NOISE_FACTOR)`.
  `--measure` only removes the first part; the noise-relative bar stays.
- Everything is switched off first, the LED matrix included — it is not in the
  light list, and a lit matrix would raise the dark frame.
- ImSwitch reads the setup only at startup. `channel_index` is the LASERid
  (an LED on GPIO2 is `3`) and must be an integer.
