# CAN OTA Update Procedure

## Overview

The CAN OTA (Over-The-Air) update system enables wireless firmware updates for ESP32-based CAN slave devices in the UC2 microscope system. This document describes the complete workflow from initiating an update to final verification.

**Note:** The master CAN HAT (CAN ID 1) is connected via USB and cannot be updated via WiFi OTA. See the [USB Flash Procedure](#usb-flash-procedure-master-can-hat) section for master updates.

## Architecture

### Components

1. **ImSwitch Backend** (`UC2ConfigController`)
   - Orchestrates the OTA update process
   - Manages firmware repository
   - Tracks device status
   - Communicates with master CAN HAT

2. **Master CAN HAT** (CAN ID 1)
   - Connected via USB to host computer
   - Sends OTA commands to slave devices via CAN bus
   - Receives status updates from slaves

3. **CAN Slave Devices** (Motors: 10-15, Illumination: 20-31, Galvo: 40+)
   - Receive OTA commands via CAN
   - Connect to WiFi network
   - Run Arduino OTA server
   - Apply firmware updates

4. **Firmware Server**
   - HTTP server hosting firmware binaries
   - Default URL: `http://localhost/firmware`
   - Provides JSON API for firmware listing

## Firmware Mapping

### Centralized Mapping (`_get_can_id_firmware_mapping()`)

| CAN ID | Device Type | Firmware File |
|--------|-------------|---------------|
| 1 | Master CAN HAT | `esp32_UC2_3_CAN_HAT_Master.bin` |
| 10-15 | Motors (A, X, Y, Z) | `esp32_seeed_xiao_esp32s3_can_slave_motor.bin` |
| 20-22 | Lasers | `esp32_seeed_xiao_esp32s3_can_slave_illumination.bin` |
| 30-31 | LEDs | `esp32_seeed_xiao_esp32s3_can_slave_illumination.bin` |
| 40+ | Galvo Mirrors | `esp32_seeed_xiao_esp32s3_can_slave_galvo.bin` |

### Custom Firmware Override

Custom firmware can be provided with naming pattern: `id_<CANID>_*.bin`
- Example: `id_11_custom_motor.bin` for Motor X with CAN ID 11
- Custom firmware takes precedence over standard firmware

## OTA Update Workflow

### Phase 1: Initialization (Frontend → Backend)

```
User Action: Click "Start Update" for device
    ↓
Frontend calls: /UC2ConfigController/startSingleDeviceOTA
    Parameters:
    - can_id: Device CAN ID
    - ssid: WiFi network name (optional)
    - password: WiFi password (optional)
    - timeout: OTA timeout in ms (default: 300000)
```

### Phase 2: Command Transmission (Backend → Device)

```
Backend (UC2ConfigController.startSingleDeviceOTA):
    1. Initialize status tracking
        └─> Status: "command_sent"
        └─> Emit: sigOTAStatusUpdate (to frontend via WebSocket)
    
    2. Send CAN command via master HAT
        └─> ESP32.canota.start_ota_update(can_id, ssid, password)
        └─> Command transmitted over CAN bus
    
    3. Return to frontend
        └─> Response: {"status": "success", "message": "OTA initiated"}
```

**Frontend State:** "OTA command sent, waiting for WiFi connection..."

### Phase 3: Device WiFi Connection (Device → Backend)

```
CAN Slave Device:
    1. Receives OTA command via CAN
    2. Connects to specified WiFi network
    3. Starts Arduino OTA server on port 3232
    4. Sends status back via CAN:
        - Success: IP address + hostname
        - Failure: Error code + message

Backend Callback (registerOTACallback):
    Triggered automatically when device responds
    
    If WiFi Success (status == 0):
        ├─> Update status: "wifi_connected"
        ├─> Message: "Device connected at <IP>"
        ├─> Emit: sigOTAStatusUpdate
        └─> Launch upload thread: _upload_firmware_to_device(can_id, ip)
    
    If WiFi Failed (status != 0):
        ├─> Update status: "wifi_failed"
        ├─> Message: Error description
        └─> Emit: sigOTAStatusUpdate
```

**Frontend State:** "Device connected, downloading firmware..."

### Phase 4: Firmware Download (Backend → Firmware Server)

```
Backend (_download_firmware_for_device):
    1. Query firmware server
        ├─> GET http://localhost/firmware
        ├─> Headers: {"Accept": "application/json"}
        └─> Response: List of available firmware files
    
    2. Match firmware to device
        ├─> First: Check for custom firmware (id_<CANID>_*.bin)
        └─> Fallback: Use standard mapping from _get_can_id_firmware_mapping()
    
    3. Download firmware
        ├─> GET http://localhost/firmware/<filename>
        ├─> Save to cache: /tmp/uc2_ota_firmware_cache/
        └─> Update status: "downloading"
```

### Phase 5: Firmware Upload (Backend → Device)

```
Backend (_upload_firmware_to_device):
    1. Prepare upload
        ├─> Status: "uploading"
        ├─> Progress: 0%
        └─> Emit: sigOTAStatusUpdate
    
    2. Upload via Arduino OTA protocol
        ├─> espota.upload_ota(esp_ip, firmware_path, port=3232)
        ├─> Progress callbacks every 10%
        │   └─> Emit: sigOTAStatusUpdate with progress
        └─> Timeout: 20 seconds per chunk
    
    3. Monitor progress
        └─> Progress updates: 0% → 10% → 20% → ... → 100%
            └─> Each step emits: sigOTAStatusUpdate
```

**Frontend State:** "Uploading: [=========>] 50%"

### Phase 6: Completion

```
Upload Success (result == 0):
    ├─> Status: "success"
    ├─> Progress: 100%
    ├─> Message: "✅ Firmware uploaded successfully"
    ├─> Timestamp: ISO 8601 format
    └─> Emit: sigOTAStatusUpdate (FINAL)

Upload Failure (result != 0):
    ├─> Status: "failed"
    ├─> Error: Error description
    ├─> Message: "❌ OTA upload failed with code <N>"
    └─> Emit: sigOTAStatusUpdate (FINAL)
```

**Frontend State:** "✅ Update complete!" or "❌ Update failed: <error>"

### Phase 7: Device Reboot

```
CAN Slave Device:
    1. Receives complete firmware
    2. Verifies checksum
    3. Writes to flash memory
    4. Automatic reboot
    5. Runs new firmware
    6. Reconnects to CAN bus
```

## Status Tracking

### Status Object Structure

Each device's OTA status is tracked in `_ota_status[can_id]`:

```python
{
    "can_id": 11,                           # CAN ID of device
    "status": "initiated" | "wifi_connected" | "success" | "failed",
    "upload_status": "command_sent" | "wifi_connected" | "downloading" | 
                     "uploading" | "success" | "failed",
    "upload_progress": 0-100,               # Upload percentage
    "message": "User-friendly status message",
    "timestamp": "2026-01-01T14:30:00",    # ISO 8601 timestamp
    "ip": "192.168.1.100",                  # Device IP (if connected)
    "hostname": "UC2-CAN-11.local",         # mDNS hostname
    "statusMsg": "Device-provided message",
    "upload_error": "Error details",        # Only if failed
    "upload_timestamp": "ISO timestamp",    # Only on success
    "success": true/false                   # WiFi connection success
}
```

### Signal Emission Points

The `sigOTAStatusUpdate` signal is emitted at these points:

1. **Command sent** - Initial API call
2. **WiFi connected** - Device callback (success)
3. **WiFi failed** - Device callback (failure)
4. **Download started** - Status update only
5. **Upload started** - Progress 0%
6. **Upload progress** - Every 10% increment
7. **Upload complete** - Final success/failure

## API Endpoints

### Start Single Device Update

```http
POST /UC2ConfigController/startSingleDeviceOTA
Content-Type: application/json

{
    "can_id": 11,
    "ssid": "MyNetwork",           # Optional
    "password": "MyPassword",      # Optional
    "timeout": 300000              # Optional, default: 5 min
}
```

**Response:**
```json
{
    "status": "success",
    "message": "OTA update initiated for device 11",
    "can_id": 11,
    "command_response": {...}
}
```

### Start Multiple Device Update

```http
POST /UC2ConfigController/startMultipleDeviceOTA
Content-Type: application/json

{
    "can_ids": [10, 11, 12, 13],
    "ssid": "MyNetwork",
    "password": "MyPassword",
    "timeout": 300000,
    "delay_between": 2             # Delay between devices in seconds
}
```

### Get OTA Status

```http
GET /UC2ConfigController/getOTAStatus?can_id=11
```

**Response:**
```json
{
    "can_id": 11,
    "status": "success",
    "upload_status": "success",
    "upload_progress": 100,
    "message": "✅ Firmware uploaded successfully",
    "timestamp": "2026-01-01T14:35:00",
    "ip": "192.168.1.100",
    "upload_timestamp": "2026-01-01T14:35:00"
}
```

### List Available Firmware

```http
GET /UC2ConfigController/listAvailableFirmware
```

**Response:**
```json
{
    "status": "success",
    "firmware_server": "http://localhost/firmware",
    "firmware_count": 4,
    "firmware": {
        "1": {
            "filename": "esp32_UC2_3_CAN_HAT_Master.bin",
            "url": "http://localhost/firmware/esp32_UC2_3_CAN_HAT_Master.bin",
            "can_id": 1,
            "size": 1321136,
            "mod_time": "2025-12-03T16:31:06Z"
        },
        "11": {
            "filename": "esp32_seeed_xiao_esp32s3_can_slave_motor.bin",
            "url": "http://localhost/firmware/esp32_seeed_xiao_esp32s3_can_slave_motor.bin",
            "can_id": 11,
            "size": 850768,
            "mod_time": "2025-12-03T16:31:06Z"
        }
    }
}
```

### Configure WiFi Credentials

```http
POST /UC2ConfigController/setOTAWiFiCredentials
Content-Type: application/json

{
    "ssid": "MyNetwork",
    "password": "MyPassword"
}
```

### Configure Firmware Server

```http
POST /UC2ConfigController/setOTAFirmwareServer
Content-Type: application/json

{
    "server_url": "http://192.168.1.10/firmware"
}
```

### Clear Firmware Cache

```http
POST /UC2ConfigController/clearOTAFirmwareCache
```

## Error Handling

### Common Failure Points

| Error Type | Phase | Cause | Recovery |
|------------|-------|-------|----------|
| WiFi connection failed | Phase 3 | Invalid credentials, out of range | Check SSID/password, move closer |
| Firmware not found | Phase 4 | Missing firmware on server | Upload firmware to server |
| Download timeout | Phase 4 | Server unreachable | Check server URL and network |
| Upload timeout | Phase 5 | Network instability | Retry update |
| Upload failed | Phase 5 | Device busy, low memory | Restart device, retry |
| Verification failed | Phase 6 | Corrupted download | Clear cache, retry |

### Status Codes

**WiFi Connection (from device):**
- `0` - Success
- `1` - WiFi connection failed
- `2` - OTA initialization failed

**Upload Result (from espota):**
- `0` - Success
- `1` - Connection failed
- `2` - Upload timeout
- `3` - Verification failed

## Frontend Integration

### WebSocket Events

Frontend listens to WebSocket events on the `UC2ConfigController` namespace:

```javascript
// Subscribe to OTA status updates
socket.on('sigOTAStatusUpdate', (statusUpdate) => {
    const { can_id, upload_status, upload_progress, message } = statusUpdate;
    
    // Update UI based on status
    switch(upload_status) {
        case 'command_sent':
            showStatus("Sending command...");
            break;
        case 'wifi_connected':
            showStatus("Device connected, preparing upload...");
            break;
        case 'uploading':
            updateProgress(upload_progress);
            break;
        case 'success':
            showSuccess(message);
            enableButton();
            break;
        case 'failed':
            showError(message);
            enableButton();
            break;
    }
});
```

### Expected Status Sequence

```
command_sent (0%)
    ↓
wifi_connected (0%)
    ↓
downloading (0%)
    ↓
uploading (0%)
    ↓
uploading (10%)
    ↓
uploading (20%)
    ↓
    ...
    ↓
uploading (90%)
    ↓
success (100%)
```

## Best Practices

### 1. Network Configuration
- Use stable WiFi network (2.4 GHz preferred)
- Ensure devices are within range
- Avoid network congestion during updates
- Static IP for firmware server recommended

### 2. Firmware Management
- Keep firmware server synchronized with build system
- Use version control for firmware binaries
- Test firmware on single device before batch updates
- Maintain firmware changelog

### 3. Update Strategy
- Update devices one at a time unless necessary
- Use `delay_between` for sequential updates
- Avoid updating critical devices simultaneously
- Test functionality after each update

### 4. Monitoring
- Watch console logs for detailed progress
- Monitor device LED indicators during update
- Check device reconnection to CAN bus
- Verify device functionality post-update

### 5. Troubleshooting
- Use `clearOTAFirmwareCache` if downloads corrupt
- Check `getOTAStatus` for detailed error info
- Verify firmware server accessibility
- Test WiFi credentials separately

## Security Considerations

### Current Implementation
- No encryption on firmware download
- No authentication for Arduino OTA
- WiFi credentials transmitted over CAN bus
- Firmware server requires local network access

### Recommendations
- Use HTTPS for firmware server in production
- Implement OTA password protection
- Encrypt CAN bus communication
- Use VPN for remote firmware server access
- Implement firmware signing and verification

## Performance Metrics

### Typical Update Times
- WiFi connection: 5-10 seconds
- Firmware download: 2-5 seconds (depends on server)
- Firmware upload: 20-60 seconds (depends on size)
- Device reboot: 3-5 seconds
- **Total time per device: 30-80 seconds**

### Firmware Sizes
- Motor firmware: ~850 KB
- Illumination firmware: ~797 KB
- Galvo firmware: ~796 KB
- Master HAT firmware: ~1.3 MB (USB update only)

## USB Flash Procedure (Master CAN HAT)

The master CAN HAT (CAN ID 1) is connected via USB and cannot use WiFi OTA. Instead, it uses `esptool` for firmware flashing.

### USB Flash Workflow

```
Phase 1: Initialization
├─> User opens USB Flash Wizard
├─> List available serial ports
├─> Load firmware information from server
└─> Select port (auto-detect or manual)

Phase 2: Preparation
├─> Disconnect ImSwitch from ESP32 (close serial connection)
├─> Download master firmware from server
└─> Resolve target serial port

Phase 3: Flash
├─> (Optional) Erase flash memory
├─> Run esptool write_flash command
└─> Wait for completion

Phase 4: Reconnection
├─> Wait for device reboot (1 second)
├─> Reconnect ImSwitch to ESP32
└─> Verify connection
```

### USB Flash API Endpoints

#### List Serial Ports

```http
GET /UC2ConfigController/listSerialPorts
```

**Response:**
```json
[
  {
    "device": "/dev/ttyACM0",
    "description": "UC2 CAN HAT",
    "manufacturer": "Espressif",
    "product": "ESP32-S3",
    "hwid": "USB VID:PID=303A:1001",
    "vid": 12346,
    "pid": 4097,
    "serial_number": "12345678"
  }
]
```

#### Flash Master Firmware

```http
POST /UC2ConfigController/flashMasterFirmwareUSB
Content-Type: application/json

{
  "port": null,           // null = auto-detect
  "match": "HAT",         // Substring to find port
  "baud": 921600,         // Flash baud rate
  "flash_offset": 0,      // 0x0 for merged images
  "erase_flash": false,   // Erase before write
  "reconnect_after": true // Reconnect after flash
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Master firmware flashed via USB",
  "port": "/dev/ttyACM0",
  "firmware": "/tmp/uc2_ota_firmware_cache/esp32_UC2_3_CAN_HAT_Master.bin",
  "baud": 921600,
  "flash_offset": 0,
  "erase_flash": false,
  "reconnect_after": true
}
```

#### Get USB Flash Status

```http
GET /UC2ConfigController/getUSBFlashStatus
```

**Response:**
```json
{
  "status": "flashing",
  "progress": 60,
  "message": "Writing firmware to device...",
  "details": null,
  "timestamp": "2026-01-01T15:30:00"
}
```

### USB Flash Status Updates

The backend emits `sigUSBFlashStatusUpdate` signals at each stage:

| Progress | Status | Message |
|----------|--------|---------|
| 5% | disconnecting | Disconnecting from ESP32... |
| 10% | downloading | Downloading firmware from server... |
| 20% | downloading | Firmware downloaded: esp32_UC2_3_CAN_HAT_Master.bin |
| 25% | flashing | Using port: /dev/ttyACM0 |
| 30% | erasing | Erasing flash memory... (optional) |
| 40% | flashing | Flash erased successfully (optional) |
| 45% | flashing | Writing firmware to device... |
| 85% | flashing | Firmware written successfully! |
| 90% | reconnecting | Reconnecting to device... |
| 100% | success | ✅ Firmware flashed and reconnected! |

### USB Flash Frontend Integration

The frontend uses the `UsbFlashWizard` component with the `usbFlashSlice` Redux store:

```javascript
// Listen for USB flash status updates via WebSocket
socket.on('sigUSBFlashStatusUpdate', (status) => {
  dispatch(usbFlashSlice.updateFlashProgress(status));
});
```

### USB Flash Wizard Steps

1. **Port Selection** - Choose auto-detect or manual port selection
2. **Firmware Info** - Review firmware and configure flash options
3. **Flash Progress** - Monitor flashing progress
4. **Complete** - View results and close wizard

### USB Flash Requirements

- **esptool**: Python package for ESP32 flashing (`pip install esptool`)
- **USB Connection**: Device must be connected via USB (not WiFi)
- **Serial Port Access**: User must have permissions to access serial ports
- **Firmware Server**: Master firmware must be available on configured server

### USB Flash Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| No serial ports found | USB not connected | Connect device via USB |
| Port access denied | Permission issue | Add user to `dialout` group (Linux) |
| esptool not available | Missing package | `pip install esptool` |
| Firmware not found | Server misconfigured | Check firmware server URL |
| Write failed | Connection issue | Check USB cable, retry |
| Reconnect failed | Device not responding | Manual power cycle required |

## Future Enhancements

1. **Batch Progress Tracking** - Overall progress for multiple devices
2. **Rollback Mechanism** - Restore previous firmware on failure
3. **Automatic Retry** - Retry failed updates automatically
4. **Device Grouping** - Update by device type
5. **Scheduled Updates** - Automatic updates at specific times
6. **Version Management** - Track firmware versions per device
7. **Differential Updates** - Only update changed portions
8. **Health Monitoring** - Pre-update device health check

## References

- Arduino OTA Documentation: https://arduino-esp8266.readthedocs.io/en/latest/ota_updates/readme.html
- ESP32 OTA Update: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/ota.html
- esptool Documentation: https://docs.espressif.com/projects/esptool/en/latest/esp32/
- UC2-REST CAN Protocol: See `DOCUMENTATION/CAN_OTA_Documentation.md` in UC2-REST repository
