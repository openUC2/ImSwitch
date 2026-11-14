# Hardware-in-the-loop (HIL) tests

Tests of a running ImSwitch on a real microscope, through its HTTP API only. They sit next to
`../unit/` (no server) and `../api/` (headless server), but never import ImSwitch: they need only
`pytest`, `requests`, `pillow`, `numpy`, `reportlab` and `pypdf`.

**They move the stage and switch light.** Each test file says in its first line what it does.

| Test | Checks | Hardware |
|---|---|---|
| `board_link` | the UC2 board is connected and runs the CANopen master firmware | reads |
| `firmware_server` | the firmware server has images; every board runs its version | reads |
| `camera_capture` | every camera returns a frame at its resolution | reads |
| `lightsource` | lasers/LEDs switch; every LED reaches the cameras | **light** |
| `ledmatrix` | the LED matrix reaches the cameras | **light** |
| `ledmatrix_single` | one LED lights alone (`setLEDs`) and is dimmer than every LED | **light** |
| `frame_calibration` | the one-button FRAME calibration runs through; needs the calibration target in the holder | **moves** (homes, turret), **light**, **writes the setup** (backup: `setup_before_frame_calibration.json`) |

Hardware a setup does not have skips its tests. Hardware that is there but does not work fails.

They run in the order of the table: the boards, the cameras, the lights, then the calibration.
The order is the list `ORDER` in `conftest.py`.

The calibration is the test of the motors. It homes Z, X, Y and the turret, reads the position
back after every move, follows X and Y in the observation camera, focuses with Z and switches
the objective. It is attempted even when an earlier test failed; its own preflight and busy checks
decide whether it can proceed. Earlier failures still fail the overall run. A step the calibration skips (e.g. the second objective on a rig with
one) is listed in the test's output as not tested.

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

It waits until ImSwitch answers and its setup is readable. Camera startup errors are recorded
without preventing the tests or calibration. It then runs the tests and writes `junit.xml`, `report.html` (results, output and photos in one file)
and `summary.md` to `REPORT_DIR`. `hil-report.pdf` contains the test summary, diagnostics and
photos followed by the original PDF of this run's calibration (`calibration-report.pdf`).
A failed or unstarted calibration is explicitly identified; an older report is never substituted. A test may leave further files there; it finds the folder in
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
| `conftest.py` | the order of the tests (`ORDER`); runs calibration last even after failures; marks the tests `hardware`; skips everything when ImSwitch is not reachable |
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
runs with the others from then on, after the tests listed in `ORDER` (`conftest.py`); add its
name there to give it another place. A test file has:

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

## Combined report and retained calibration

The suite waits for the calibration thread to finish cleanup before exporting its PDF through
`getFrameCalibrationReportPdf`. It checks the run's `startedAt` before and after export.
Calibration results remain in the instrument's setup, even after os-rpi restores the deployment.
A failed calibration may have written partial changes; the setup backup is for manual recovery.
No automatic rollback is performed. Cleanup/export failures are reported as errors.

Optional `HIL_TARGET` and `HIL_RUN_URL` identify the tested image/pallet and Actions run in the
PDF. os-rpi supplies these; laptop runs show that the target was not supplied.
ReportLab renders the test pages; pypdf appends the original calibration PDF. Full captured
output remains in HTML. A PDF generation failure preserves HTML/Markdown and fails the run.
