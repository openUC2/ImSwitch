# Board

The UC2 board is connected and runs the right master firmware. Reads only.

| Test | Checks |
|---|---|
| `test_uc2_board_connected` | `UC2ConfigController/is_connected` is `true` |
| `test_correct_esp32_firmware` | `getFirmwareInfo`: `connected`, `isMaster`, `pindef == UC2_canopen_master` |

## Run

```bash
./run_board_test.sh
```

Also `IMSWITCH_URL`, `PYTHON_BIN`, `REMOTE_TEST_DIR`. No test knobs.

## Worth knowing

- `getFirmwareInfo` answering 404 skips: older ImSwitch versions lack it.
