# Firmware

The firmware server, and whether every board runs its version. The tests only
read; `../ci/sync_firmware.py` flashes (see `../ci/README.md`).

| Test | Checks |
|---|---|
| `test_master_firmware_is_reported` | master reports `name`, `version`, `date`, `pindef` |
| `test_firmware_server_is_configured` | an OTA firmware server URL is set |
| `test_firmware_server_lists_binaries` | the server lists `.bin` files with URLs |
| `test_answering_boards_have_firmware` | every board that answers has an image on the server |
| `test_boards_run_server_version` | and runs the server's version |

## Run

```bash
./run_firmware_test.sh              # sync, then the tests
../ci/run_firmware_sync.sh          # sync only: prints what it would flash
../ci/run_firmware_sync.sh --yes    # sync only: flashes
```

| Knob | Default | |
|---|---|---|
| `FIRMWARE_SCAN_TIMEOUT` | `5` | seconds for the CAN scan |
| `FIRMWARE_UPDATE` | `on` | `off` skips the sync in `run_firmware_test.sh`, `run_all.sh` and `hil-run.sh` |

## Worth knowing

- The server's version is the `"version"` in its `version.json`, written by
  the firmware build (youseetoo/uc2-esp32, `tools/write_fw_manifest.py`).
  Every image of that build reports exactly this string, e.g.
  `v2026.0.0-beta.4-9-g3c86acd-t20260930173005-pr133`.
- A server without `version.json` names no version: the version test skips
  and the sync stops with exit `2`.
- ImSwitch without these endpoints (before openUC2/ImSwitch#324) skips the two
  board tests, and the sync stops with exit `2`.
- Unreachable nodes (e.g. laser `20`, gpio `60` on a rig without them) are
  printed, not asserted.
- The CAN scan behind `checkFirmwareUpdates` now and then comes back empty: no
  node, and the master without its `canId`. That would read as "in sync" with
  nothing compared, so the board tests fail on it, and the sync asks up to
  three times before it stops with exit `2`.
