# Firmware versions and updates

Every uc2-esp32 image carries the release it was built from and the image name it is published as.
The firmware server publishes the same release in `version.json`. ImSwitch compares the two for each
connected board. On request it updates the outdated boards in one run and counts a board as done only
when it reports the new version afterwards. Nothing is ever flashed without an explicit user action.

## Identity baked into every build

| Macro | Example | Reported as |
|---|---|---|
| `UC2_FW_VERSION` | `v2026.0.0-beta.4-6-g2108590-t20260930092622` | `/state_get` `identifier_version`, OD `0x2500`, scan `fwVersion` |
| `UC2_FW_IMAGE` | `esp32_UC2_canopen_slave_motor_release_motY.bin` | `/state_get` `identifier_image`, OD `0x2501`, scan `fwImage` |

`UC2_FW_VERSION` forms:

| Build | Example |
|---|---|
| Tagged commit | `v2026.1.0` |
| Untagged commit | `v2026.0.0-beta.4-6-g2108590-t20260930092622` (`git describe` + author time, UTC) |
| Pull request | `…-t20260930092622-pr123` |
| Local build with uncommitted changes | `…-t20260930092622-dirty` |

Where the two macros come from:

- **CI** (`uc2-esp32/.github/workflows/build-frame.yaml`): the `fw-version` job computes the version
  once for the whole run. Each image build gets `UC2_FW_VERSION`, plus `UC2_FW_IMAGE` set to the file
  name it is published under (`esp32_<env>[_mot<axis>].bin`).
- **Local PlatformIO builds**: `tools/fw_version.py` (a `pre:` extra script in every env) derives the
  same formats from git, the env name and `-DCAN_AXIS_ID=CAN_ID_MOT_<X>`. It writes the gitignored
  `main/uc2_fw_version.h` and rewrites it only when a string changes, so only `State.cpp` and
  `CANopenModule.cpp` recompile.

Two boards with the same `UC2_FW_VERSION` run the same build. For untagged builds the `-t` part also
orders builds in time. `identifier_id` stays `"V2.0"` (the API generation), and `identifier_date` /
OD `0x2508` stay the compile timestamp.

Compatibility:
- Firmware built before these changes reports `"UC2-ESP v2.0"` (CAN) or no `identifier_version`
  (serial), and no image name. The version never matches a server version, so these boards show as
  *update available*. Their image is taken from `_get_can_id_firmware_mapping()` (CAN) or the pindef
  (USB).
- OD `0x2500`/`0x2501` are 64 bytes. A master built before that reads at most 39 characters of a
  newer node's strings, so update the master too.

## version.json on the firmware server

The `firmware-image-server` container (Caddy, `/srv` → `<host>/firmware/`) serves it next to the
images. It is written by `tools/write_fw_manifest.py` in the `build-server-ctr-image` job:

```json
{
  "version": "v2026.0.0-beta.4-6-g2108590-t20260930092622",
  "commit": "2108590…",
  "commit_time": "2026-09-30T09:26:22Z",
  "run_url": "https://github.com/youseetoo/uc2-esp32/actions/runs/…",
  "files": {
    "esp32_UC2_canopen_slave_motor_release_motX.bin": {"size": 879632, "sha256": "…"}
  }
}
```

Only images built by that run are listed. A `.bin` copied into the server by hand has no entry, so it
has no known version and its download is not verified.

## ImSwitch

The logic lives in **`imswitch/imcontrol/model/canbus`**, a package without ImSwitch imports. It needs
a uc2rest client and a few callbacks, so it can be split out as a package of its own:

| Module | Contents |
|---|---|
| `network.CanNetwork` | Builds all of the parts below from `client`, `firmware_url`, `cache_dir`, `link`, `UpdateHooks`, and two status callbacks |
| `firmware_server.FirmwareServer` | Listing, `version.json`, sha256-verified downloads (a cached copy whose sha256 matches is reused), which image a board gets |
| `bus.CanBus` | Scan, node-id reassignment, restarts, waiting for a node to report a version |
| `ota.CanOta` | CAN streaming OTA with retries and post-flash verification |
| `usb.UsbFlasher` | esptool flashing, serial bring-up of fresh boards (CAN address, state probe, test action) |
| `updater.FirmwareUpdater` | `check()` / `start()` / `cancel()` of the prompted update |
| `images` | The CAN-ID role table, `update_status()` |
| `guard.SerialPortGuard` | One writer on the master's serial port: a CAN stream or a USB flash |

`UC2ConfigController` exposes it through `controllers/uc2config/can_network_api.py`. Every route stays
`/UC2ConfigController/<name>`. That file adds what only ImSwitch knows: the serial link, busy checks,
lasers off, the homing prompt, and the setup-JSON options.

WiFi OTA was removed on 2026-09-30, along with its endpoints, wizard step and docs. In that mode a
node joined WiFi and pulled its image over ArduinoOTA. Restore it from git history if ever needed.

| Endpoint | What it does |
|---|---|
| `checkFirmwareUpdates(timeout=5, probe_range=False)` | Read-only. Returns `{server_version, updates_available, devices: [{canId, deviceTypeStr, connection, installed_version, filename, image_source, available_version, update_status}]}` |
| `startFirmwareUpdate(can_ids, include_master)` (POST, body = JSON array) | Returns `{"status": "started"}` and runs in the background, or `{"status": "refused", "reasons": [...]}` |
| `getFirmwareUpdateStatus()` | `{state, message, steps: [{canId, connection, from_version, to_version, status, message}], homing_required}` |
| `cancelFirmwareUpdate()` (POST) | Stops after the current board |
| `getFirmwareUpdatePrompt()` | Result of the check after connect, if it found updates |
| `getFirmwareCheckOnConnect()` / `setFirmwareCheckOnConnect(enabled)` | Opt-in for the check after connect; persisted in the setup JSON |
| `listAllFirmwareFiles()` / `listAvailableFirmware()` | Also return `server_version`, plus `version` and `sha256` per file |
| `getFirmwareInfo()` | Also returns `fwVersion` and `fwImage` (needs the current uc2rest) |

`update_status`:

| Value | Meaning |
|---|---|
| `up_to_date` | installed == server |
| `update_available` | differs (includes firmware too old to report a version) |
| `device_newer` | both carry `-t<timestamp>` and the board's is later, e.g. a developer build; not preselected |
| `unknown` | the image is on the server, but there is no `version.json` to compare with |
| `no_firmware` | no image for this board on the server |
| `unreachable` | the node did not answer the scan |

`image_source` is `reported` when the board named its own image, and `mapping` for older firmware.

### What `startFirmwareUpdate` does

1. **Refuses** with the reasons, and nothing starts, when:
   - an experiment, recording, workflow or timelapse runs;
   - a stage is moving or frame-homing (`/motor_get isbusy`), or the objective turret moves;
   - another CAN OTA or USB flash runs;
   - the firmware server is not reachable;
   - a requested node is off the bus or has no image.

   It updates what you pass: by default every node whose status is `update_available`. Explicitly
   listed nodes are updated even when `up_to_date` or `device_newer` (re-flash or downgrade).
2. **Switches all lasers off** through LaserController, so the UI stays in sync.
3. **Updates each CAN node, one at a time:**
   - downloads the image and compares size and sha256 with `version.json` (a mismatch deletes the file
     and fails the step);
   - streams it over CAN (`_can_streaming_ota`, up to 3 attempts);
   - re-scans the bus until the rebooted node reports the new version (up to ~45 s).

   Only then is the step `done`.
4. **Flashes the USB master last** with esptool at 0x10000 (app image). ImSwitch reconnects and
   re-reads the master's version, which must match.
5. **Stops at the first failed step**; later steps become `skipped`.
6. **Motor nodes restart and lose their position.** After a motor update `homing_required` is set, and
   PositionerController's homing recommendation is re-armed.

Cancel stops between boards and aborts a CAN transfer in progress; the node then keeps its old
firmware. A USB flash of the master is never interrupted, because a half-written master needs a manual
reflash.

The CAN OTA wizard (`startCANStreamingOTA` / `startMultipleCANStreamingOTA`) uses the same download
check and post-flash verification.

### Setup JSON options (`uc2Config`)

```json
"uc2Config": {"checkFirmwareOnConnect": true, "firmwareServerUrl": "http://host.docker.internal/firmware"}
```

- **`checkFirmwareOnConnect`** (default `false`): 20 s after startup, if the board is connected and
  nothing is busy, runs `checkFirmwareUpdates` once. If boards are outdated, it emits
  `sigFirmwareUpdatesAvailable` and keeps the result for `getFirmwareUpdatePrompt`. It is read-only.
- **`firmwareServerUrl`** (default: the Docker URL above): needed for the check after connect outside
  Docker. `setOTAFirmwareServer` still changes it at runtime, but does not persist it.

## Frontend

One **Update firmware** dialog (`FirmwareUpdateDialog.jsx`) first asks for the method:

- **Over the CAN bus.** Every board on the bus is updated through the master ImSwitch is connected
  to, and the master itself last over its USB link. It needs a connected master.
  - It shows the version table and preselects outdated boards. It also offers the firmware-server
    setting, *Change ID* per node (by MAC) and a deep scan for boards without a CAN route.
  - It drives `startFirmwareUpdate` and follows `getFirmwareUpdateStatus`, with the live transfer
    bar from `sigOTAStatusUpdate` / `sigUSBFlashStatusUpdate`.
  - The state lives in the backend, so reopening the dialog during an update shows it again.
- **Over a USB cable.** One board on its own port: flash any image, assign a CAN address, test it
  (`UsbFlashWizard.jsx`, rendered inside the dialog). This works without a connected master, for
  recovery.

Entry points:
- **System Update → Firmware**: *Update firmware…*, which starts at the method choice.
- `FirmwareVersionsPanel.jsx`: *Update outdated boards…*, which opens the CAN path with its check.
  The panel also has the *check when ImSwitch starts* option.
- `FirmwareUpdatePrompt.jsx` (App level): *Review…*, which opens the CAN path.

The former CAN OTA wizard is gone. It tracked "update running" in the browser (`isUpdating`), and
closing it did not reset that flag. After a close and reopen, the progress step showed "Waiting to
start…" with no Start button, and nothing was sent until a page reload.

A server without `version.json` (older firmware-image-server images) still works:
- Its file listing shows which images exist.
- Boards show as *Server has no version info*; nothing is preselected, so pick boards by hand.
- Results are reported as flashed but not verified.
- An unreachable server refuses the update.

## CAN OTA reliability (why an upload needed about 3 attempts)

**Cause:**
- The master wrote the image size to OD `0x2F01` with the default 250 ms SDO timeout.
- On the slave, that write runs `esp_ota_begin(part, size)`. That call erases the whole image range
  *before* the SDO reply goes out, so the first attempt failed with `size_write_failed`.
- The slave still finished the erase. A retry then erased already-blank flash, which often, but not
  always, fit into 250 ms.

**Fixes:**

| Where | Change |
|---|---|
| Slave, `CanOpenOTA.cpp` | `esp_ota_begin(part, OTA_WITH_SEQUENTIAL_WRITES, …)`: each 4 KB sector is erased in `esp_ota_write` just before it is written. The image size is now checked against the partition explicitly. |
| Master, `OtaBinaryReceive.cpp` | The size write waits up to 20 s (`SIZE_WRITE_TIMEOUT_MS`), so slaves with old firmware get through on the first attempt. The limit stays below uc2rest's 30 s per-chunk ACK wait. `size_write_failed` reports the elapsed `ms`. |
| Master, `CANopenModule.cpp` | The SDO and streaming wait loops feed the loop task's 10 s panicking watchdog. |
| ImSwitch | Intermediate failures are reported as `attempt_failed` / `retrying`, not the terminal `error`. One upload at a time (`busy`). The upload baud comes from the live link. |
| uc2rest | Restores the link at its own baud. Skips the DTR/RTS reset when the master ended the session itself. Ready timeout 15 s. |

**Bench results (2026-09-30; CAN HAT master `UC2_canopen_master` + XIAO motor node 12, 880 KB image):**

| Master / slave firmware | Result | Chunk #2 ACK* | Transfer |
|---|---|---|---|
| old / old (baseline) | `size_write_failed` on attempt 1 | — | — |
| new / old | success, 1st attempt | 1254 ms | 41 s, 20.9 KB/s |
| new / new | success, 1st attempt | 746 ms | 48 s, 17.7 KB/s |

\*The master ACKs chunk #1 before pushing it over CAN. The ACK of chunk #2 therefore includes the CRC
and size writes, a fixed 500 ms pause and one chunk over CAN (~200 ms). The size write thus took
~0.5 s against an old slave, even on mostly blank flash. Against a new slave it is ~0 ms. The erase
now happens during the transfer instead (~+35 ms per 4 KB chunk).

**Full `startFirmwareUpdate` runs through ImSwitch on the same bench:**
- Node 12 (CAN, verified by re-scan), then the master (esptool at 921600, reconnect, verified): ~80 s
  in total, twice, both on the first attempt.
- The check after connect reported both boards 20 s after startup.
- A served image corrupted by one byte was rejected by the sha256 check before anything was flashed.

## Still open

- Serial status lines carry no session id, so a stale `rx_timeout` line could fail the next attempt.
  ACK lines are two separate UART writes.
- Check on hardware whether the bus-power FET (GPIO4) drops the slaves while the master is in reset
  (uc2rest still resets it after a host-side timeout or cancel).
- Only lasers are switched off before an update; LED matrices are not.
