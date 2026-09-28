# Firmware

The firmware server, and whether the CAN nodes run what it offers. Reads only —
nothing is flashed.

| Test | Checks |
|---|---|
| `test_master_firmware_is_reported` | master reports `name`, `version`, `date`, `pindef` |
| `test_firmware_server_is_configured` | an OTA firmware server URL is set |
| `test_firmware_server_lists_binaries` | the server lists `.bin` files with URLs |
| `test_reachable_bus_devices_have_firmware` | every node that answers the scan has a mapped file |
| `test_reachable_bus_devices_run_current_firmware` | and runs exactly that file's build |

## Run

```bash
./run_firmware_test.sh
```

| Knob | Default | |
|---|---|---|
| `FIRMWARE_SCAN_TIMEOUT` | `5` | seconds for the CAN scan |
| `FIRMWARE_FETCH_TIMEOUT` | `60` | seconds to download one `.bin` |

## Worth knowing

- The currency check compares a node's OD `0x2508` build string with the
  `"UC2-ESP v2.0"\0<time>\0<date>` literals in its `.bin` — the firmware builds
  that string at runtime from exactly those. Not the server's `mod_time`.
- The server URL is `host.docker.internal`, reachable only from the container:
  run from your machine, the currency test skips.
- Only nodes that answer the scan are checked. Unreachable ones without a
  mapping (e.g. gpio `60`, ptz `61`) are printed; the mapping table itself
  lives in `UC2ConfigController`.
