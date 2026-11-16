# Hardware-in-the-loop (HIL) tests

Tests of a running ImSwitch on a real microscope, through its HTTP API only.
They sit next to `../unit/` (no server) and `../api/` (headless server), but never import ImSwitch: they need only `pytest`, `requests`, `pillow`, `numpy`, `reportlab` and `pypdf`.

**They move the stage and switch light.**
Each test file says at the top of its docstring what it checks and what it does to the rig.

| Test | Checks | Hardware |
|---|---|---|
| `board_link` | ImSwitch talks to the UC2 board, and the board runs the master firmware | reads |
| `firmware_server` | the firmware server has firmware files, and each board runs the server's version | reads |
| `camera_capture` | each camera takes a picture as large as its sensor | reads |
| `motor_axis_move` | each axis moves 10 µm and back; then each axis with homing switched on homes: Z, X, Y, then the turret | **moves** (10 µm, then to the end stops) |
| `lightsource` | each laser and LED switches on and off; the cameras see each LED | **light** |
| `ledmatrix` | the cameras see the LED matrix | **light** |
| `ledmatrix_single` | one LED lights on its own (`setLEDs`) and is dimmer than the whole matrix | **light** |
| `frame_calibration` | the FRAME calibration runs from start to end; needs the calibration target in the holder | **moves** (homes, turret), **light**, **writes the setup** (backup: `setup_before_frame_calibration.json`) |

Hardware a setup does not have skips its tests.
Hardware that is there but does not work fails.

They run in the order of the table: the boards, the cameras, the motors, the lights, then the calibration.
The stage is homed before the lights are tested, so the slide holder is out of the light path.
The order is the list `ORDER` in `conftest.py`.

Some tests do not start after a failure; they are skipped and say which test failed.
The table is `NEEDS` in `conftest.py`:

- `frame_calibration` needs `board_link`, `camera_capture`, `motor_axis_move` and the three light tests.
  It does not need `firmware_server`: a board on another firmware version fails that test, but can still be calibrated.

`motor_axis_move` first steps every axis, then homes them, one case per axis.
When a step failed or an axis did not home, the axes after it are skipped.
The homing cannot see whether an end stop was found: `homeAxis` does not return it.
It fails when ImSwitch answers with an error or not at all.
The calibration homes again and checks every axis, reads the position back after every move, follows X and Y in the observation camera, focuses with Z and switches the objective.
A step the calibration skips (e.g. the second objective on a rig with one) is listed in the test's output as not tested.

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

It waits until ImSwitch answers and its setup is readable.
A camera that does not start is a problem in the report; the tests run anyway.
`REPORT_DIR` receives:

| File | Holds |
|---|---|
| `hil-report.pdf` | the report to read: what was tested, one line per test, then every test with what failed, its notes, its cases and its photos, then the PDFs the tests appended (the calibration's own report) |
| `summary.md` | the setup, the counts, one line per failure and the notes of the tests, for the CI run's page |
| `setup_before_frame_calibration.json` | the setup as it was before the calibration; only when the calibration ran |

Both reports name the setup file the rig ran with.
A test finds the folder in the environment variable `HIL_REPORT_DIR`.
Use a new folder for every run: the two reports are replaced, but a file a test saved there in an older run stays.

While the tests run, the log shows one line per test with the number of tests done (`[12/31]`).
The calibration takes minutes and prints a line when each of its steps starts and ends (`calibration  4/10  First objective: ...`).

Exit code: `0` all passed, `1` a test failed, `2` the tests could not run (not ready, firmware sync failed, unknown test, everything skipped, the PDF could not be written).

The test rig (os-rpi, `.github/hil/`) runs exactly this command inside the ImSwitch container.
That one line is all the two repositories share, so the tests always belong to the ImSwitch version under test.

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

Every `test_*.py` is one test and stands on its own: its settings are at its top, its helper functions below them.
No test imports another one.
Deleting a test is deleting its file.

What the tests share has nothing to do with what they check:

| File | Holds |
|---|---|
| `hil_client.py` | HTTP (`get_json`, `post_json`, `wait_until`), `setup()`, `parametrize_from_rig` |
| `hil_report.py` | `attach(label, image)`, `note(text)`, `append_pdf(title, data)`, the results of the run, `summary.md` |
| `hil_pdf.py` | `hil-report.pdf` |
| `conftest.py` | the order of the tests (`ORDER`); which test needs which (`NEEDS`); `live`; marks the tests `hardware`; skips everything when ImSwitch is not reachable |
| `__main__.py` | the command above |
| `firmware_sync.py` | `--sync-firmware` |

## Which hardware a test uses

There is no list of the rig's hardware here.
The tests read it from the setup file ImSwitch runs with: `ImSwitchConfig/imcontrol_setups/<name>.json`, where `<name>` is the `setupFileName` in `ImSwitchConfig/config/imcontrol_options.json`.
`client.setup_file()` returns the name, `client.setup()` the content as a dict.
ImSwitch parses the file before it answers, so a field the file leaves out comes back with ImSwitch's default (e.g. `"forAcquisition": false`).

| Setup | Used for |
|---|---|
| `detectors`, `forAcquisition` | the cameras; `true` is an acquisition detector, `false` the observation camera |
| `lasers`, `valueRangeMax` | the lasers and LEDs, and the value they are switched on with |
| `LEDMatrixs` | whether there is an LED matrix |
| `positioners`, `axes` | the stage axes |
| `objective` | whether there is an objective turret |

So a rig with other hardware needs another setup file in ImSwitch, not other tests.

## Writing a test

Copy the test that is closest to the new one, rename it `test_<name>.py`, and change it.
It runs with the others from then on, after the tests listed in `ORDER` (`conftest.py`); add its name there to give it another place.
A test file has:

1. a docstring: first a paragraph saying in plain words what it checks, then what it does to the rig (MOVES REAL HARDWARE, SWITCHES LIGHT, Only reads) and anything else worth knowing;
2. its settings, as constants;
3. its helper functions and fixtures;
4. the tests, whose steps read top to bottom.

- Call ImSwitch with `client.get_json("/Controller/method", {"param": value})`.
- Take what the rig has from `client.setup()`, and thresholds, distances and timeouts from the constants at the top of the file.
- Add to the test's part of the report with `attach("label", image)` for a photo and `hil_report.note("...")` for a line of text.
  `hil_report.append_pdf("title", data)` puts a whole PDF behind the pages of the tests.
  The report code knows no test by name.
- The first paragraph of the docstring is what the report shows under "What it checks" (`hil_report.py`).
  Write it as you would say it to a colleague.
- A test that runs for minutes takes the fixture `live` and prints its progress with `live("step 4/10 ...")`; a plain `print` only shows when the test fails.
- Whatever a test switches on or moves, it puts back, also when it fails (`try/finally`, or a fixture with `yield`).
- Code that a second test needs is copied, not shared: a test must stay readable and removable by itself.

## The calibration in the report

The test waits until the calibration's thread has ended, then fetches its PDF with `getFrameCalibrationReportPdf` and checks the run's `startedAt` before and after, so the report of another run is never taken.
The PDF is appended to `hil-report.pdf` as ImSwitch wrote it (`hil_report.append_pdf`).
When it cannot be fetched, the `frame_calibration` part of the report shows that as an error.

The fixtures of `test_frame_calibration.py` put the rig back, each on its own: the calibration is stopped, the light goes off (the calibration leaves it on), the settings go back, and then the report is collected.

The calibration's result stays in the instrument's setup, also after os-rpi has restored the deployment.
A calibration that fails may have written a part of it; `setup_before_frame_calibration.json` is there to go back by hand.
Nothing is rolled back automatically.

os-rpi passes three environment variables that the report shows: `HIL_TARGET` (the image or pallet under test), `HIL_PALLET` (the pallet and every image in it) and `HIL_RUN_URL` (the Actions run).
A run from a laptop has none of them.
