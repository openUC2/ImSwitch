# OTA Update API Quick Reference

## Setup (Do Once)

```python
import requests
BASE = "http://localhost:8000"

# 1. Set firmware server
requests.post(f"{BASE}/uc2config/setOTAFirmwareServer", json={})

# 2. Set WiFi credentials  
requests.post(f"{BASE}/uc2config/setOTAWiFiCredentials", json={
    "ssid": "YourWiFi", "password": "YourPassword"
})
```

## Update Device

```python
# Single device
requests.post(f"{BASE}/uc2config/startSingleDeviceOTA", json={"can_id": 11})

# Multiple devices
requests.post(f"{BASE}/uc2config/startMultipleDeviceOTA", json={
    "can_ids": [11, 12, 13]
})
```

## Monitor Status

```python
# Get status
status = requests.get(f"{BASE}/uc2config/getOTAStatus?can_id=11").json()
print(status["ota_status"]["upload_status"])  # pending, uploading, success, failed
```

## CAN IDs

| Device | ID | Device | ID |
|--------|----:|--------|----:|
| Motor A | 10 | Laser 0 | 20 |
| Motor X | 11 | Laser 1 | 21 |
| Motor Y | 12 | LED 0 | 30 |
| Motor Z | 13 | LED 1 | 31 |

## Firmware Server

```bash
# Start server
cd ~/uc2_firmware
python3 -m http.server 9000

# Add firmware (must be named id_<CANID>_*.bin)
cp firmware.bin latest/id_11_esp32_motor.bin
```

## Common Commands

```python
# List available firmware
requests.get(f"{BASE}/uc2config/listAvailableFirmware")

# Get device mapping
requests.get(f"{BASE}/uc2config/getOTADeviceMapping")

# Clear cache
requests.post(f"{BASE}/uc2config/clearOTAFirmwareCache")

# Clear status
requests.post(f"{BASE}/uc2config/clearOTAStatus")
```

## Upload Status Flow

```
initiated → downloading → uploading → success
                                   └→ failed
```

## Error Messages

| Message | Fix |
|---------|-----|
| "Firmware server not set" | Call `setOTAFirmwareServer` |
| "WiFi credentials not provided" | Call `setOTAWiFiCredentials` |
| "No firmware found for device X" | Check filename: `id_X_*.bin` |
| "Failed to connect to server" | Start server: `python3 -m http.server 9000` |

## Full Docs

- Setup: `FIRMWARE_SERVER_QUICKSTART.md`
- Server: `CAN_OTA_FIRMWARE_SERVER.md`  
- Complete: `CAN_OTA_UPDATE_GUIDE.md`
