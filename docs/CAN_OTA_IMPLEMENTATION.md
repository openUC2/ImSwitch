# CAN OTA Update Implementation Summary

## Overview

This implementation adds comprehensive OTA (Over-The-Air) firmware update capabilities for UC2 CAN satellite devices to the ImSwitch UC2ConfigController. The system allows wireless firmware updates for motors, lasers, and LED controllers through WiFi connections.

## Key Features

### 1. **WiFi-Based OTA Updates**
- Devices connect to WiFi network using provided credentials
- ArduinoOTA server starts automatically on the device
- Firmware uploaded via HTTP POST (no PlatformIO required)

### 2. **Automatic Firmware Management**
- Smart firmware file matching based on CAN ID
- Fallback to generic firmware by device type
- Support for organized firmware directory structure

### 3. **Status Tracking**
- Real-time OTA progress monitoring
- Per-device status with timestamps
- WebSocket signal emission for live updates

### 4. **Batch Operations**
- Update single device or multiple devices
- Sequential updates with configurable delays
- Comprehensive batch status reporting

### 5. **REST API Integration**
- Full REST API with APIExport decorators
- Easy integration with external tools
- Standard JSON request/response format

## Implementation Details

### Modified Files

#### 1. `UC2ConfigController.py`
**Additions:**
- New signal: `sigOTAStatusUpdate` - Emits OTA status updates
- Instance variables for OTA tracking:
  - `_ota_status` - Dictionary tracking OTA state per device
  - `_ota_lock` - Thread-safe access to status
  - `_firmware_dir` - Path to firmware files
  - `_ota_wifi_ssid`, `_ota_wifi_password` - WiFi credentials

**New Methods:**

1. **Callback Registration:**
   - `registerOTACallback()` - Registers callback for OTA status updates from CAN devices
   
2. **Firmware Management:**
   - `_find_firmware_for_device(can_id)` - Finds appropriate firmware file for device
   - `_upload_firmware_to_device(can_id, ip_address)` - Uploads firmware via HTTP POST

3. **API Endpoints:**
   - `setOTAWiFiCredentials(ssid, password)` - Configure WiFi for OTA
   - `setOTAFirmwareDirectory(directory)` - Set firmware storage location
   - `listAvailableFirmware()` - List all available firmware files
   - `startSingleDeviceOTA(can_id, ...)` - Update single device
   - `startMultipleDeviceOTA(can_ids, ...)` - Update multiple devices
   - `getOTAStatus(can_id)` - Get OTA status for device(s)
   - `clearOTAStatus(can_id)` - Clear OTA status tracking
   - `getOTADeviceMapping()` - Get CAN ID to device type mapping

### New Files

#### 1. `docs/CAN_OTA_UPDATE_GUIDE.md`
Comprehensive user guide covering:
- System architecture and update flow
- Setup instructions
- Complete API reference
- Usage examples (single device, batch, WebSocket)
- Firmware naming conventions
- Status codes and troubleshooting
- Security considerations
- Advanced usage patterns

#### 2. `imswitch/imcontrol/controller/controllers/TEST_OTA_UPDATE.py`
Interactive test script providing:
- Configuration setup
- Firmware listing
- Device mapping display
- Single and batch update testing
- Menu-driven interface
- Real-time status monitoring

## Architecture

### Update Flow

```
┌─────────────┐
│   User/API  │
└──────┬──────┘
       │ 1. startSingleDeviceOTA(can_id)
       ▼
┌──────────────────────┐
│ UC2ConfigController  │
│ - Validates params   │
│ - Initializes status │
└──────┬───────────────┘
       │ 2. Send OTA command via CAN
       ▼
┌──────────────────┐
│  CAN Device      │
│ - Connect WiFi   │
│ - Start OTA srv  │
└──────┬───────────┘
       │ 3. Send callback with IP
       ▼
┌──────────────────────┐
│ OTA Callback Handler │
│ - Update status      │
│ - Trigger upload     │
└──────┬───────────────┘
       │ 4. HTTP POST firmware
       ▼
┌──────────────────┐
│  CAN Device      │
│ - Flash firmware │
│ - Reboot         │
└──────┬───────────┘
       │ 5. Status: success
       ▼
┌──────────────────────┐
│ Status Update        │
│ - Emit signal        │
│ - Update tracking    │
└──────────────────────┘
```

### Status Tracking

Each device's OTA status includes:
```python
{
    "status": 0,  # 0=success, 1=wifi_failed, 2=ota_failed
    "statusMsg": "Success",
    "ip": "192.168.1.100",
    "hostname": "UC2-CAN-B.local",
    "success": True,
    "timestamp": "2025-11-03T10:35:00",
    "upload_status": "success",  # pending, uploading, success, failed
    "upload_timestamp": "2025-11-03T10:36:00",
    "upload_error": None  # Error message if failed
}
```

## Firmware File Organization

### Naming Convention
```
id_<CAN_ID>_esp32_<board>_<device_type>[_variant].bin
```

### Examples
```
firmware/
├── id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin  (Motor A)
├── id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin  (Motor X)
├── id_12_esp32_seeed_xiao_esp32s3_can_slave_motor.bin  (Motor Y)
├── id_13_esp32_seeed_xiao_esp32s3_can_slave_motor.bin  (Motor Z)
├── id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin
└── id_30_esp32_seeed_xiao_esp32s3_can_slave_led_debug.bin
```

### Matching Logic
1. **Exact match:** Look for `id_<CAN_ID>_*.bin`
2. **Type fallback:** Look for `id_*_<device_type>*.bin`
   - CAN IDs 10-13 → `*_motor*.bin`
   - CAN IDs 20-29 → `*_laser*.bin`
   - CAN IDs 30-39 → `*_led*.bin`

## API Usage Examples

### 1. Configuration
```python
# Set WiFi credentials
POST /uc2config/setOTAWiFiCredentials
{"ssid": "MyNetwork", "password": "MyPassword"}

# Set firmware directory
POST /uc2config/setOTAFirmwareDirectory
{"directory": "/home/user/firmware"}
```

### 2. Single Device Update
```python
# Update Motor X
POST /uc2config/startSingleDeviceOTA
{"can_id": 11, "timeout": 300000}

# Check status
GET /uc2config/getOTAStatus?can_id=11
```

### 3. Batch Update
```python
# Update all motors
POST /uc2config/startMultipleDeviceOTA
{"can_ids": [11, 12, 13], "delay_between": 2}

# Get all statuses
GET /uc2config/getOTAStatus
```

## Integration Points

### 1. UC2-REST Library
Uses the existing `canota` module from UC2-REST:
```python
self._master.UC2ConfigManager.ESP32.canota.register_callback(0, ota_callback)
self._master.UC2ConfigManager.ESP32.canota.start_ota_update(can_id, ssid, password)
```

### 2. ImSwitch Signals
Emits `sigOTAStatusUpdate` for WebSocket integration:
```python
self.sigOTAStatusUpdate.emit(ota_response)
```

### 3. REST API
All methods decorated with `@APIExport`:
```python
@APIExport(runOnUIThread=False)
def startSingleDeviceOTA(self, can_id, ssid=None, password=None, timeout=300000):
    ...
```

## Thread Safety

- Uses `threading.Lock()` for status dictionary access
- Firmware upload runs in daemon thread
- Non-blocking OTA command execution
- API calls run on appropriate threads (UI vs background)

## Error Handling

### Device Level
- WiFi connection failures → Status code 1
- OTA server start failures → Status code 2
- Upload failures → Tracked in `upload_error` field

### Controller Level
- Missing WiFi credentials → Error response
- Firmware file not found → Status update with error
- HTTP upload failures → Logged and tracked
- Exception handling with detailed logging

## Testing

### Test Script Features
- Interactive menu system
- Configuration validation
- Single device testing
- Batch update testing
- Status monitoring
- Error reporting

### Usage
```bash
python TEST_OTA_UPDATE.py
```

## Dependencies

### Required
- `uc2rest` library with `canota` module
- `requests` library for HTTP uploads
- `threading` for concurrent operations
- `pathlib` for file operations

### Optional
- WebSocket client for real-time monitoring
- ImSwitch REST API enabled

## Configuration Requirements

### ImSwitch Setup
1. UC2ConfigManager with ESP32 client
2. Serial connection to UC2 master device
3. CAN bus properly configured

### Network Setup
1. WiFi network accessible to CAN devices
2. 2.4GHz WiFi (ESP32 requirement)
3. Network allows HTTP traffic on port 80/3232

### Firmware Files
1. Organized in dedicated directory
2. Follow naming convention
3. Valid ESP32 binary files

## Security Considerations

### Current Implementation
- WiFi credentials stored in memory only
- HTTP upload (unencrypted)
- No authentication on device OTA endpoint

### Recommended Enhancements
1. Store credentials encrypted
2. Use HTTPS for firmware upload (if device supports)
3. Add authentication to OTA endpoints
4. Use dedicated OTA network (VLAN)
5. Implement firmware signing/verification

## Future Enhancements

### Potential Features
1. **Automatic firmware updates:** Periodic checks for new firmware
2. **Rollback capability:** Restore previous firmware if update fails
3. **Firmware versioning:** Track and display firmware versions
4. **Update scheduling:** Schedule updates for off-peak times
5. **Progress reporting:** Detailed upload progress (percentage)
6. **Batch retry:** Automatic retry for failed devices
7. **Configuration backup:** Save/restore device configurations
8. **Update profiles:** Predefined update scenarios

### Integration Ideas
1. **GUI integration:** Add OTA panel to UC2ConfigWidget
2. **Notification system:** Email/SMS alerts on update completion
3. **Logging:** Detailed update history with timestamps
4. **Analytics:** Track update success rates
5. **Remote triggering:** Cloud-based update initiation

## Troubleshooting

### Common Issues

**1. No callback received**
- Check CAN bus connection
- Verify ESP32 serial communication
- Confirm callback registration

**2. WiFi connection fails**
- Verify SSID/password
- Check 2.4GHz network availability
- Ensure signal strength

**3. Firmware upload fails**
- Check network connectivity
- Verify firmware file validity
- Ensure sufficient device flash space

**4. Device not found**
- Confirm CAN ID is correct
- Check device is powered
- Verify CAN bus termination

## Conclusion

This implementation provides a robust, scalable OTA update system for UC2 CAN devices with:
- ✅ Complete REST API integration
- ✅ Automatic firmware management
- ✅ Real-time status tracking
- ✅ Batch update support
- ✅ Comprehensive documentation
- ✅ Test utilities

The system is production-ready with proper error handling, thread safety, and extensibility for future enhancements.
