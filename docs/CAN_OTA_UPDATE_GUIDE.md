# CAN OTA Update System for UC2 Satellites

## Overview

The CAN OTA (Over-The-Air) update system enables wireless firmware updates for UC2 CAN satellite devices (motors, lasers, LEDs) through WiFi connections. The system is integrated into the UC2ConfigController and provides REST API endpoints for triggering and monitoring OTA updates.

## Architecture

### Components

1. **UC2ConfigController** - Main controller with OTA API endpoints
2. **OTA Callback System** - Receives status updates from CAN devices
3. **Firmware Manager** - Finds and serves appropriate firmware files
4. **Status Tracker** - Monitors OTA progress for each device

### Update Flow

```
1. API Call → startSingleDeviceOTA(can_id)
2. Controller → Send OTA command via CAN bus
3. Device → Connect to WiFi, start ArduinoOTA server
4. Device → Send callback with IP address
5. Controller → Upload firmware via HTTP POST
6. Device → Flash new firmware, reboot
7. Controller → Update status, emit signal
```

## Setup

### 1. Configure WiFi Credentials

```python
# Set WiFi credentials for OTA updates
POST /uc2config/setOTAWiFiCredentials
{
    "ssid": "YourWiFiNetwork",
    "password": "YourWiFiPassword"
}
```

### 2. Set Firmware Directory

Organize your firmware files with the naming convention:
```
firmware/
├── id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
├── id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
├── id_12_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
├── id_13_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
├── id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin
└── id_21_esp32_seeed_xiao_esp32s3_can_slave_led_debug.bin
```

```python
# Set firmware directory
POST /uc2config/setOTAFirmwareDirectory
{
    "directory": "/path/to/firmware"
}

# Response includes list of found firmware files
{
    "status": "success",
    "firmware_files": [...],
    "count": 6
}
```

## API Endpoints

### Device Identification

#### Get Device Mapping
```python
GET /uc2config/getOTADeviceMapping

Response:
{
    "status": "success",
    "mapping": {
        "motors": {
            "A": 10,
            "X": 11,
            "Y": 12,
            "Z": 13
        },
        "laser": {
            "laser_0": 20,
            "laser_1": 21
        },
        "led": {
            "led_0": 30,
            "led_1": 31
        }
    }
}
```

### Firmware Management

#### List Available Firmware
```python
GET /uc2config/listAvailableFirmware

Response:
{
    "status": "success",
    "firmware_count": 6,
    "firmware": {
        "11": {
            "filename": "id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin",
            "path": "/path/to/firmware/id_11_...",
            "size": 1234567,
            "modified": "2025-11-03T10:30:00"
        },
        ...
    }
}
```

### OTA Operations

#### Update Single Device
```python
POST /uc2config/startSingleDeviceOTA
{
    "can_id": 11,  // Motor X
    "ssid": "OptionalSSID",  // Optional, uses configured if not provided
    "password": "OptionalPassword",  // Optional
    "timeout": 300000  // 5 minutes in milliseconds
}

Response:
{
    "status": "success",
    "message": "OTA update initiated for device 11",
    "can_id": 11
}
```

#### Update Multiple Devices
```python
POST /uc2config/startMultipleDeviceOTA
{
    "can_ids": [11, 12, 13],  // Motor X, Y, Z
    "delay_between": 2  // Seconds between each device
}

Response:
{
    "status": "success",
    "message": "OTA update initiated for 3 devices",
    "results": [
        {"can_id": 11, "result": {...}},
        {"can_id": 12, "result": {...}},
        {"can_id": 13, "result": {...}}
    ]
}
```

#### Get OTA Status
```python
# Get status for specific device
GET /uc2config/getOTAStatus?can_id=11

Response:
{
    "status": "success",
    "can_id": 11,
    "ota_status": {
        "status": 0,  // 0=success, 1=wifi_failed, 2=ota_failed
        "statusMsg": "Success",
        "ip": "192.168.1.100",
        "hostname": "UC2-CAN-B.local",
        "success": true,
        "timestamp": "2025-11-03T10:35:00",
        "upload_status": "success",  // pending, uploading, success, failed
        "upload_timestamp": "2025-11-03T10:36:00"
    }
}

# Get status for all devices
GET /uc2config/getOTAStatus

Response:
{
    "status": "success",
    "device_count": 3,
    "devices": {
        "11": {...},
        "12": {...},
        "13": {...}
    }
}
```

#### Clear OTA Status
```python
# Clear specific device
POST /uc2config/clearOTAStatus
{"can_id": 11}

# Clear all devices
POST /uc2config/clearOTAStatus
{}
```

## Usage Examples

### Example 1: Update Motor X

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Set WiFi credentials
requests.post(f"{BASE_URL}/uc2config/setOTAWiFiCredentials", json={
    "ssid": "MyNetwork",
    "password": "MyPassword"
})

# 2. Set firmware directory
requests.post(f"{BASE_URL}/uc2config/setOTAFirmwareDirectory", json={
    "directory": "/home/user/uc2_firmware"
})

# 3. Start OTA update for Motor X (CAN ID 11)
response = requests.post(f"{BASE_URL}/uc2config/startSingleDeviceOTA", json={
    "can_id": 11,
    "timeout": 300000
})

# 4. Monitor status
import time
while True:
    status = requests.get(f"{BASE_URL}/uc2config/getOTAStatus?can_id=11").json()
    
    if status["ota_status"]["upload_status"] == "success":
        print("✅ Firmware update completed successfully!")
        break
    elif status["ota_status"]["upload_status"] == "failed":
        print("❌ Firmware update failed!")
        break
    
    print(f"Status: {status['ota_status']['upload_status']}")
    time.sleep(2)
```

### Example 2: Update All Motors

```python
# Update all motor axes (X, Y, Z)
response = requests.post(f"{BASE_URL}/uc2config/startMultipleDeviceOTA", json={
    "can_ids": [11, 12, 13],  # Motor X, Y, Z
    "delay_between": 3
})

# Monitor overall progress
time.sleep(5)  # Give devices time to connect
status = requests.get(f"{BASE_URL}/uc2config/getOTAStatus").json()

for can_id, device_status in status["devices"].items():
    print(f"Device {can_id}: {device_status['upload_status']}")
```

### Example 3: WebSocket Status Monitoring

```python
# Subscribe to OTA status updates via WebSocket
import socketio

sio = socketio.Client()

@sio.on('ota_status_update')
def on_ota_update(data):
    print(f"Device {data['canId']}: {data['statusMsg']}")
    if data['success']:
        print(f"  IP: {data['ip']}")
        print(f"  Ready for upload!")

sio.connect('http://localhost:8000')

# Start OTA update
requests.post(f"{BASE_URL}/uc2config/startSingleDeviceOTA", json={
    "can_id": 20  # Laser
})

# Status updates will be received via WebSocket
sio.wait()
```

## Firmware File Naming Convention

The system uses a specific naming pattern to match firmware files to devices:

```
id_<CAN_ID>_esp32_<board>_<device_type>[_variant].bin
```

**Examples:**
- `id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin` - Motor X (CAN ID 11)
- `id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin` - Laser (CAN ID 20)
- `id_30_esp32_seeed_xiao_esp32s3_can_slave_led.bin` - LED (CAN ID 30)

**Fallback Matching:**
If an exact CAN ID match is not found, the system tries to find a generic firmware file based on device type:
- CAN IDs 10-13: Look for `*_motor*.bin`
- CAN IDs 20-29: Look for `*_laser*.bin`
- CAN IDs 30-39: Look for `*_led*.bin`

## Status Codes

### OTA Setup Status
- `0` - Success (WiFi connected, OTA server started)
- `1` - WiFi connection failed
- `2` - OTA start failed

### Upload Status
- `"pending"` - Waiting for device to be ready
- `"uploading"` - Firmware upload in progress
- `"success"` - Firmware uploaded and flashed successfully
- `"failed"` - Upload or flashing failed

## Troubleshooting

### WiFi Connection Failed
**Problem:** Device status shows `status: 1` (wifi_failed)

**Solutions:**
1. Verify WiFi credentials are correct
2. Check that WiFi network is reachable from device location
3. Ensure WiFi network uses 2.4GHz (ESP32 doesn't support 5GHz)

### OTA Start Failed
**Problem:** Device status shows `status: 2` (ota_failed)

**Solutions:**
1. Device may be busy - wait and try again
2. Verify device is responding to CAN commands
3. Check device has sufficient free memory for OTA

### No Firmware Found
**Problem:** Upload status shows "No firmware file found"

**Solutions:**
1. Verify firmware directory is set correctly
2. Check firmware files follow naming convention
3. Ensure firmware file for the CAN ID exists

### Upload Failed
**Problem:** Upload status shows "failed"

**Solutions:**
1. Check network connectivity between ImSwitch and device
2. Verify device IP address is reachable
3. Ensure firmware file is not corrupted
4. Check device has sufficient flash space

### Timeout
**Problem:** No response from device after OTA command

**Solutions:**
1. Increase timeout parameter (default: 5 minutes)
2. Verify CAN bus connection is stable
3. Check device is powered and functioning

## Integration with Existing Systems

### REST API Integration
All OTA endpoints are exposed via the standard ImSwitch REST API under the `/uc2config/` prefix.

### WebSocket Events
OTA status updates are emitted via the `sigOTAStatusUpdate` signal, which can be subscribed to via WebSocket for real-time monitoring.

### Configuration File
WiFi credentials and firmware directory can be pre-configured in the ImSwitch configuration file:

```json
{
    "uc2_ota_config": {
        "wifi_ssid": "DefaultNetwork",
        "wifi_password": "DefaultPassword",
        "firmware_directory": "/opt/uc2_firmware"
    }
}
```

## Security Considerations

1. **WiFi Credentials**: Stored in memory only, not persisted to disk by default
2. **HTTP Upload**: Firmware upload uses plain HTTP (ArduinoOTA standard)
3. **Access Control**: Use ImSwitch authentication for production deployments
4. **Network Isolation**: Consider using a dedicated network for OTA updates

## Advanced Usage

### Custom Upload Logic
For advanced use cases, you can disable automatic firmware upload:

```python
# Don't set firmware directory - manual control
controller._firmware_dir = None

# Register custom OTA callback
def custom_ota_handler(ota_response):
    if ota_response["success"]:
        ip = ota_response["ip"]
        can_id = ota_response["canId"]
        # Custom upload logic here
        my_custom_upload(ip, can_id)

controller._master.UC2ConfigManager.ESP32.canota.register_callback(1, custom_ota_handler)
```

### Batch Updates with Progress Tracking
```python
devices = [11, 12, 13, 20, 30]  # All devices
results = {}

for can_id in devices:
    response = requests.post(f"{BASE_URL}/uc2config/startSingleDeviceOTA", 
                            json={"can_id": can_id})
    results[can_id] = "initiated"
    time.sleep(2)

# Poll until all complete
while True:
    status = requests.get(f"{BASE_URL}/uc2config/getOTAStatus").json()
    
    all_done = True
    for can_id in devices:
        if str(can_id) in status["devices"]:
            upload_status = status["devices"][str(can_id)]["upload_status"]
            results[can_id] = upload_status
            if upload_status not in ["success", "failed"]:
                all_done = False
    
    print(f"Progress: {results}")
    
    if all_done:
        break
    
    time.sleep(3)

print(f"Final results: {results}")
```

## References

- UC2-REST CAN OTA Documentation: `/UC2-REST/DOCUMENTATION/CAN_OTA_Documentation.md`
- ArduinoOTA Library: https://github.com/esp8266/Arduino/tree/master/libraries/ArduinoOTA
- UC2 ESP32 Firmware: https://github.com/youseetoo/uc2-esp32
