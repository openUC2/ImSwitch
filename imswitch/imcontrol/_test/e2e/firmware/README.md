# Firmware

The firmware server, and whether the boards run the newest firmware it offers.
The tests only read; `update_firmware.py` flashes.

| Test | Checks |
|---|---|
| `test_master_firmware_is_reported` | master reports `name`, `version`, `date`, `pindef` |
| `test_firmware_server_is_configured` | an OTA firmware server URL is set |
| `test_firmware_server_lists_binaries` | the server lists `.bin` files with URLs |
| `test_reachable_bus_devices_have_firmware` | every node that answers the scan has a mapped file |
| `test_boards_run_newest_firmware` | the master and every answering node run the server's version |

## Run

```bash
./run_firmware_test.sh
```

| Knob | Default | |
|---|---|---|
| `FIRMWARE_SCAN_TIMEOUT` | `5` | seconds for the CAN scan |

## Update

`update_firmware.py` runs first in `run_all.sh`: same check as the last test,
then `startCANStreamingOTA` for each outdated node, one by one, and a re-scan.
Exit 1 if a flashed node does not report the newest version afterwards;
`run_all.sh` warns and runs the tests anyway. The USB master is reported, not
flashed. `FIRMWARE_UPDATE=off` skips it.

## Worth knowing

- Newest is the `"version"` in the server's `version.json`, written by the
  firmware build (youseetoo/uc2-esp32, `tools/write_fw_manifest.py`). Every
  image of that build reports exactly this string as `fwVersion` (CANopen OD
  `0x2500`), e.g. `v2026.0.0-beta.4-7-gbd08544-t20260930125929`, so the check
  is plain string equality.
- Servers built before versioned firmware have no `version.json`: the test
  skips and the update flashes nothing. Boards on such firmware report
  `UC2-ESP v2.0`, which a versioned server then flags as outdated.
- The server URL is `host.docker.internal`, reachable only from the container:
  run from your machine, the version test skips.
- Only boards that answer the scan are checked. Unreachable nodes without a
  mapping (e.g. gpio `60`, ptz `61`) are printed by the mapping test; the table
  itself lives in `UC2ConfigController`.
