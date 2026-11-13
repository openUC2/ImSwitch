# Hardware-in-the-loop (HIL) tests

Tests of a running ImSwitch on a real microscope, through its HTTP API only. They sit next to
`../unit/` (no server) and `../api/` (headless server), but never import ImSwitch: they need only
`pytest`, `requests`, `pillow` and `numpy`.

**They move the stage and switch light.** Each test file says in its first line what it does.

| Test | Checks | Hardware |
|---|---|---|
| `board_link` | the UC2 board is connected and runs the CANopen master firmware | reads |
| `firmware_server` | the firmware server has images; every board runs its version | reads |
| `camera_capture` | every camera returns a frame at its resolution | reads |
| `objective_switch` | switching the objective shows in the image, and light still gets through | **moves** turret and Z, **light** |
| `lightsource` | lasers/LEDs switch; every LED reaches the cameras | **light** |
| `ledmatrix` | the LED matrix reaches the cameras | **light** |
| `ledmatrix_single` | one LED lights alone (`setLEDs`) and is dimmer than every LED | **light** |
| `frame_calibration` | the one-button FRAME calibration runs through; needs the calibration target in the holder | **moves** (homes, turret), **light**, **writes the setup** (backup: `setup_before_frame_calibration.json`) |
| `motor_axis_move` | each axis steps 10 µm and back | **moves** |
| `motor_motion_camera` | the observation camera sees X/Y/Z/A move | **moves** (parks at the transport position), **light** |

Hardware a setup does not have skips its tests. Hardware that is there but does not work fails.

## Run

```bash
python -m imswitch.imcontrol._test.hil REPORT_DIR [TEST ...] [--sync-firmware]
```

```bash
python -m imswitch.imcontrol._test.hil out                            # everything
python -m imswitch.imcontrol._test.hil out board_link camera_capture  # only these tests
python -m imswitch.imcontrol._test.hil out --sync-firmware            # FLASHES every board first
```

`TEST` is the name from the table: `board_link` runs `test_board_link.py`.

It waits until ImSwitch answers, the board is connected and every camera gives a frame, then
runs the tests and writes `junit.xml`, `report.html` (results, output and photos in one file)
and `summary.md` to `REPORT_DIR`. A test may leave further files there; it finds the folder in
the environment variable `HIL_REPORT_DIR`. Both reports name the setup file the rig ran with. Exit code: `0` all passed, `1` a test failed, `2` the tests
could not run (not ready, firmware sync failed, unknown test, everything skipped).

The test rig (os-rpi, `.github/hil/`) runs exactly this command inside the ImSwitch container.
That one line is all the two repositories share, so the tests always belong to the ImSwitch
version under test.

From a laptop against a rig, without ImSwitch installed:

```bash
IMSWITCH_URL=http://<pi>/imswitch python imswitch/imcontrol/_test/hil out board_link
```

While writing a test, plain pytest works too and shows more:

```bash
cd imswitch/imcontrol/_test/hil
IMSWITCH_URL=http://<pi>/imswitch pytest test_camera_capture.py -s
```

## Files

Every `test_*.py` is one test and stands on its own: its settings are at its top, its helper
functions below them. No test imports another one. Deleting a test is deleting its file.

What the tests share has nothing to do with what they check:

| File | Holds |
|---|---|
| `hil_client.py` | HTTP (`get_json`, `post_json`, `wait_until`), `setup()`, `parametrize_from_rig` |
| `hil_report.py` | `attach(label, image)` and the HTML/Markdown report |
| `conftest.py` | marks the tests `hardware`; skips everything when ImSwitch is not reachable |
| `__main__.py` | the command above |
| `firmware_sync.py` | `--sync-firmware` |

## Which hardware a test uses

There is no list of the rig's hardware here. The tests read it from the setup file ImSwitch
runs with: `ImSwitchConfig/imcontrol_setups/<name>.json`, where `<name>` is the `setupFileName`
in `ImSwitchConfig/config/imcontrol_options.json`. `client.setup_file()` returns the name,
`client.setup()` the content as a dict. ImSwitch parses the file before it answers, so a field
the file leaves out comes back with ImSwitch's default (e.g. `"forAcquisition": false`).

| Setup | Used for |
|---|---|
| `detectors`, `forAcquisition` | the cameras; `true` is an acquisition detector, `false` the observation camera |
| `lasers`, `valueRangeMax` | the lasers and LEDs, and the value they are switched on with |
| `LEDMatrixs` | whether there is an LED matrix |
| `positioners`, `axes` | the stage axes |
| `objective` | whether there is an objective turret |

So a rig with other hardware needs another setup file in ImSwitch, not other tests.

## Writing a test

Copy the test that is closest to the new one, rename it `test_<name>.py`, and change it. It
runs with the others from then on; nothing else has to be registered. A test file has:

1. a first line saying what it checks and what it does to the rig;
2. its settings, as constants;
3. its helper functions and fixtures;
4. the tests, whose steps read top to bottom.

- Call ImSwitch with `client.get_json("/Controller/method", {"param": value})`.
- Take what the rig has from `client.setup()`, and thresholds, distances and timeouts from the
  constants at the top of the file.
- Add a photo to the report with `attach("label", image)`.
- Whatever a test switches on or moves, it puts back, also when it fails (`try/finally`, or a
  fixture with `yield`).
- Code that a second test needs is copied, not shared: a test must stay readable and removable
  by itself.
