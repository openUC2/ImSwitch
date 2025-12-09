# CAN OTA Quick Reference

## Setup (One-time)

```python
# 1. Set WiFi credentials
POST /uc2config/setOTAWiFiCredentials
{"ssid": "YourNetwork", "password": "YourPassword"}

# 2. Set firmware directory
POST /uc2config/setOTAFirmwareDirectory
{"directory": "/path/to/firmware"}
```

## Device CAN IDs

| Device | CAN ID |
|--------|--------|
| Motor A | 10 |
| Motor X | 11 |
| Motor Y | 12 |
| Motor Z | 13 |
| Laser 0 | 20 |
| Laser 1 | 21 |
| LED 0 | 30 |
| LED 1 | 31 |

## Common Operations

### Update Single Device
```python
POST /uc2config/startSingleDeviceOTA
{"can_id": 11}  # Motor X
```

### Update Multiple Devices
```python
POST /uc2config/startMultipleDeviceOTA
{"can_ids": [11, 12, 13]}  # All motors
```

### Check Status
```python
# Specific device
GET /uc2config/getOTAStatus?can_id=11

# All devices
GET /uc2config/getOTAStatus
```

### List Firmware
```python
GET /uc2config/listAvailableFirmware
```

### Clear Status
```python
POST /uc2config/clearOTAStatus
{"can_id": 11}  # Specific device

POST /uc2config/clearOTAStatus
{}  # All devices
```

## Status Values

### OTA Setup Status
- `0` = Success (WiFi connected, OTA ready)
- `1` = WiFi connection failed
- `2` = OTA start failed

### Upload Status
- `"pending"` = Waiting for device
- `"uploading"` = Firmware upload in progress
- `"success"` = Update completed
- `"failed"` = Update failed

## Firmware Naming

```
id_<CAN_ID>_esp32_<board>_<device_type>[_variant].bin

Examples:
- id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
- id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin
- id_30_esp32_seeed_xiao_esp32s3_can_slave_led.bin
```

## Python Example

```python
import requests
import time

BASE = "http://localhost:8000"

# Setup
requests.post(f"{BASE}/uc2config/setOTAWiFiCredentials",
              json={"ssid": "WiFi", "password": "pass"})
requests.post(f"{BASE}/uc2config/setOTAFirmwareDirectory",
              json={"directory": "/firmware"})

# Update Motor X
requests.post(f"{BASE}/uc2config/startSingleDeviceOTA",
              json={"can_id": 11})

# Monitor
while True:
    r = requests.get(f"{BASE}/uc2config/getOTAStatus?can_id=11")
    status = r.json()["ota_status"]["upload_status"]
    print(status)
    if status in ["success", "failed"]:
        break
    time.sleep(2)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| WiFi failed | Check SSID/password, ensure 2.4GHz |
| No firmware found | Check filename matches `id_<CAN_ID>_*.bin` |
| Upload timeout | Increase timeout parameter |
| No callback | Verify CAN connection, check serial |
| HTTP error | Check network connectivity to device |

## Test Script

```bash
cd /path/to/ImSwitch
python imswitch/imcontrol/controller/controllers/TEST_OTA_UPDATE.py
```

## Documentation

- Full Guide: `docs/CAN_OTA_UPDATE_GUIDE.md`
- Implementation: `docs/CAN_OTA_IMPLEMENTATION.md`
- UC2-REST: `UC2-REST/DOCUMENTATION/CAN_OTA_Documentation.md`
