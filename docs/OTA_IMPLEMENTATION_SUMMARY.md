# CAN OTA Update Implementation Summary

## Overview

Implemented a complete OTA (Over-The-Air) firmware update system for UC2 CAN satellite devices (motors, lasers, LEDs) with server-based firmware distribution.

**Date:** November 4, 2025  
**Author:** GitHub Copilot  
**Status:** Complete

---

## Key Features

### ✅ Server-Based Firmware Distribution
- Fetches firmware from HTTP server (default: `http://localhost:9000`)
- No local directory management needed
- Centralized firmware updates across multiple ImSwitch instances
- Automatic firmware caching for performance

### ✅ Complete OTA Workflow
1. Send OTA command via CAN bus
2. Device connects to WiFi and starts ArduinoOTA
3. Device sends IP address via callback
4. Firmware automatically downloaded from server
5. Firmware uploaded to device via HTTP
6. Status tracking throughout process

### ✅ REST API Endpoints
- `setOTAWiFiCredentials` - Configure WiFi credentials
- `setOTAFirmwareServer` - Set firmware server URL
- `listAvailableFirmware` - List firmware from server
- `startSingleDeviceOTA` - Update one device
- `startMultipleDeviceOTA` - Update multiple devices
- `getOTAStatus` - Monitor update progress
- `clearOTAStatus` - Clear status tracking
- `getOTADeviceMapping` - Get CAN ID mappings
- `getOTAFirmwareCacheStatus` - View cache info
- `clearOTAFirmwareCache` - Clear cached firmware

---

## Architecture

### Components

```
┌─────────────────┐
│  Firmware Server│  (localhost:9000)
│  /latest/*.bin  │
└────────┬────────┘
         │ HTTP GET
         ▼
┌─────────────────┐
│  UC2Config      │
│  Controller     │
│  - Downloads FW │
│  - Manages OTA  │
└────────┬────────┘
         │ CAN Bus
         ▼
┌─────────────────┐
│  CAN Devices    │
│  - Motor X,Y,Z  │
│  - Laser, LED   │
└─────────────────┘
```

### File Changes

**Modified:**
- `UC2ConfigController.py` - Added OTA management system

**Created:**
- `CAN_OTA_UPDATE_GUIDE.md` - Complete OTA documentation
- `CAN_OTA_FIRMWARE_SERVER.md` - Server setup guide
- `FIRMWARE_SERVER_QUICKSTART.md` - Quick start guide
- `TEST_OTA_UPDATE.py` - Interactive test script

---

## Implementation Details

### 1. Firmware Server Integration

**Previous Approach:** Local directory with firmware files  
**New Approach:** HTTP server serving firmware at `/latest/` endpoint

**Benefits:**
- Centralized firmware management
- Easy updates across multiple instances
- No file system permissions issues
- Works with remote servers
- Simple CI/CD integration

### 2. Firmware Naming Convention

```
id_<CANID>_<description>.bin
```

Examples:
- `id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin` (Motor X)
- `id_20_esp32_seeed_xiao_esp32s3_can_slave_laser.bin` (Laser)
- `id_30_esp32_seeed_xiao_esp32s3_can_slave_led.bin` (LED)

### 3. HTML Directory Parsing

Server provides HTML directory listing:
```html
<ul>
  <li><a href="id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin">...</a></li>
  <li><a href="id_20_esp32_seeed_xiao_esp32s3_can_slave_laser.bin">...</a></li>
</ul>
```

Parser extracts `.bin` files and matches by CAN ID.

### 4. Firmware Caching

**Cache Location:** `/tmp/uc2_ota_firmware_cache/`  
**Purpose:** Avoid re-downloading same firmware  
**Management:** 
- Automatic caching on first download
- Manual clearing via API
- Status monitoring via API

### 5. Fallback Firmware Matching

If exact CAN ID match not found, searches for device type:
- CAN 10-13 → Look for `*motor*.bin`
- CAN 20-29 → Look for `*laser*.bin`
- CAN 30-39 → Look for `*led*.bin`

---

## API Examples

### Setup Configuration

```python
import requests

BASE_URL = "http://localhost:8000"

# Set firmware server
requests.post(f"{BASE_URL}/uc2config/setOTAFirmwareServer", json={
    "server_url": "http://localhost:9000"
})

# Set WiFi credentials
requests.post(f"{BASE_URL}/uc2config/setOTAWiFiCredentials", json={
    "ssid": "MyNetwork",
    "password": "MyPassword"
})
```

### List Available Firmware

```python
response = requests.get(f"{BASE_URL}/uc2config/listAvailableFirmware")
print(response.json())

# {
#   "status": "success",
#   "firmware_server": "http://localhost:9000",
#   "firmware_count": 6,
#   "firmware": {
#     "11": {
#       "filename": "id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin",
#       "url": "http://localhost:9000/latest/id_11_...",
#       "can_id": 11
#     }
#   }
# }
```

### Update Single Device

```python
# Start OTA update for Motor X
response = requests.post(f"{BASE_URL}/uc2config/startSingleDeviceOTA", json={
    "can_id": 11
})

# Monitor status
while True:
    status = requests.get(f"{BASE_URL}/uc2config/getOTAStatus?can_id=11").json()
    upload_status = status["ota_status"]["upload_status"]
    
    if upload_status == "success":
        print("✅ Update complete!")
        break
    elif upload_status == "failed":
        print(f"❌ Update failed: {status['ota_status']['upload_error']}")
        break
    
    time.sleep(2)
```

### Update Multiple Devices

```python
response = requests.post(f"{BASE_URL}/uc2config/startMultipleDeviceOTA", json={
    "can_ids": [11, 12, 13],  # Motor X, Y, Z
    "delay_between": 2
})
```

### Cache Management

```python
# View cache
cache = requests.get(f"{BASE_URL}/uc2config/getOTAFirmwareCacheStatus").json()
print(f"Cached: {cache['file_count']} files ({cache['total_size_mb']} MB)")

# Clear cache
requests.post(f"{BASE_URL}/uc2config/clearOTAFirmwareCache")
```

---

## Status Tracking

### OTA Status Fields

```python
{
    "can_id": 11,
    "status": 0,  # 0=success, 1=wifi_failed, 2=ota_failed
    "statusMsg": "Success",
    "ip": "192.168.1.100",
    "hostname": "UC2-CAN-B.local",
    "success": true,
    "timestamp": "2025-11-04T10:30:00",
    "upload_status": "success",  # pending, downloading, uploading, success, failed
    "upload_timestamp": "2025-11-04T10:31:00",
    "upload_error": null  # Error message if failed
}
```

### Upload Status Values

- `"waiting"` - Waiting for device to connect
- `"downloading"` - Downloading firmware from server
- `"uploading"` - Uploading firmware to device
- `"success"` - Firmware uploaded successfully
- `"failed"` - Upload failed (see `upload_error`)

---

## Testing

### Test Script Usage

```bash
python TEST_OTA_UPDATE.py
```

**Menu Options:**
1. Setup OTA configuration
2. List available firmware
3. Show device mapping
4. Update Motor X (CAN ID 11)
5. Update Laser (CAN ID 20)
6. Update all motors (X, Y, Z)
7. Get cache status
8. Clear firmware cache
9. Clear OTA status
0. Exit

### Manual Testing

```bash
# 1. Start firmware server
mkdir -p ~/uc2_firmware/latest
cd ~/uc2_firmware
python3 -m http.server 9000

# 2. Verify server
curl http://localhost:9000/latest/

# 3. Run ImSwitch with UC2 device connected

# 4. Test API
curl -X POST http://localhost:8000/uc2config/setOTAFirmwareServer

curl -X POST http://localhost:8000/uc2config/startSingleDeviceOTA \
  -H "Content-Type: application/json" \
  -d '{"can_id": 11}'
```

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Firmware server not set" | Server URL not configured | Call `setOTAFirmwareServer` |
| "WiFi credentials not provided" | Credentials not set | Call `setOTAWiFiCredentials` |
| "No firmware found for device X" | Firmware file missing or wrong name | Check filename matches `id_X_*.bin` |
| "Failed to connect to firmware server" | Server not running or unreachable | Start server, check URL |
| "Failed to fetch firmware list" | Server error or network issue | Check server logs, verify connectivity |
| "Download failed" | Network error during download | Retry, check server connectivity |
| "Upload failed: HTTP 500" | Device error during upload | Check device logs, retry |

---

## Production Deployment

### Recommended Setup

1. **Use dedicated web server** (Nginx/Apache) instead of Python http.server
2. **Enable HTTPS** for network deployments
3. **Implement authentication** for firmware downloads
4. **Set up CI/CD pipeline** to automatically update firmware
5. **Monitor server logs** for download statistics
6. **Version firmware files** for rollback capability

### Multi-Instance Deployment

```python
# Central firmware server at central-server:9000
# Multiple ImSwitch instances point to same server

# Instance 1
requests.post("http://instance1:8000/uc2config/setOTAFirmwareServer",
              json={"server_url": "http://central-server:9000"})

# Instance 2  
requests.post("http://instance2:8000/uc2config/setOTAFirmwareServer",
              json={"server_url": "http://central-server:9000"})
```

### CI/CD Integration

```bash
#!/bin/bash
# Build and deploy firmware

# Build
platformio run

# Deploy to server
scp .pio/build/*/firmware.bin \
    server:/var/www/uc2_firmware/latest/id_11_esp32_motor.bin

# Clear caches
curl -X POST http://instance1:8000/uc2config/clearOTAFirmwareCache
curl -X POST http://instance2:8000/uc2config/clearOTAFirmwareCache
```

---

## Documentation

### Files Created

1. **CAN_OTA_UPDATE_GUIDE.md** (Main Documentation)
   - Complete API reference
   - Usage examples
   - Status codes
   - Troubleshooting

2. **CAN_OTA_FIRMWARE_SERVER.md** (Server Setup)
   - Server requirements
   - Python http.server setup
   - Nginx/Apache configuration
   - Docker deployment
   - Production recommendations

3. **FIRMWARE_SERVER_QUICKSTART.md** (Quick Reference)
   - TL;DR setup instructions
   - Firmware naming
   - CAN ID reference
   - Common troubleshooting

4. **TEST_OTA_UPDATE.py** (Test Script)
   - Interactive menu
   - Configuration setup
   - Single/batch updates
   - Cache management
   - Status monitoring

---

## Future Enhancements

### Potential Improvements

- [ ] Firmware versioning and rollback
- [ ] Scheduled automatic updates
- [ ] Notification system for update events
- [ ] Update history/audit log
- [ ] Batch update progress tracking
- [ ] Retry mechanism for failed uploads
- [ ] Pre-download verification (checksums)
- [ ] Parallel device updates
- [ ] WebSocket streaming for real-time status
- [ ] Firmware compatibility checking
- [ ] Update queue management
- [ ] Bandwidth throttling for uploads

### Configuration File Support

```json
{
    "uc2_ota_config": {
        "firmware_server": "http://localhost:9000",
        "wifi_ssid": "DefaultNetwork",
        "wifi_password": "DefaultPassword",
        "auto_update": false,
        "cache_dir": "/custom/cache/path"
    }
}
```

---

## Integration Points

### REST API
All endpoints exposed under `/uc2config/` prefix

### WebSocket Events
- `sigOTAStatusUpdate` - Real-time status updates

### Callbacks
- OTA status callback registered with UC2 canota module
- Triggered on device responses

### File System
- Firmware cache: `/tmp/uc2_ota_firmware_cache/`
- Can be customized via configuration

---

## Dependencies

### Python Packages
- `requests` - HTTP client for firmware downloads
- Built-in `html.parser` - Parse HTML directory listings
- `pathlib` - File path handling
- `threading` - Async operations

### External Services
- HTTP server serving firmware files
- UC2-REST library with canota module
- CAN bus connectivity

---

## Testing Checklist

- [x] Firmware server connectivity
- [x] HTML directory parsing
- [x] Firmware file matching by CAN ID
- [x] Firmware download and caching
- [x] OTA command sending via CAN
- [x] Callback registration and handling
- [x] Firmware upload to device
- [x] Status tracking and updates
- [x] Error handling and recovery
- [x] Cache management
- [x] Multiple device updates
- [x] API endpoint validation

---

## References

- **UC2-REST Documentation:** `/UC2-REST/DOCUMENTATION/CAN_OTA_Documentation.md`
- **Arduino OTA Library:** https://github.com/esp8266/Arduino/tree/master/libraries/ArduinoOTA
- **UC2 ESP32 Firmware:** https://github.com/youseetoo/uc2-esp32

---

## Contact & Support

For issues or questions:
1. Check documentation in `/docs/`
2. Run test script for diagnostics
3. Review server logs
4. Check ImSwitch logs for detailed error messages

---

**Implementation Complete** ✅  
All TODOs addressed and server-based firmware distribution fully implemented.
