# Quick Start: Firmware Server Setup

## TL;DR

```bash
# 1. Create firmware directory
mkdir -p ~/uc2_firmware/latest

# 2. Add your firmware files with proper naming (id_<CANID>_*.bin)
cp your_motor_firmware.bin ~/uc2_firmware/latest/id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin

# 3. Start server
cd ~/uc2_firmware
python3 -m http.server 9000

# 4. Test it works
curl http://localhost:9000/latest/
```

## Firmware File Naming

Files **must** be named: `id_<CANID>_<description>.bin`

Examples:
```
id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin    # Motor A
id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin    # Motor X  
id_12_esp32_seeed_xiao_esp32s3_can_slave_motor.bin    # Motor Y
id_13_esp32_seeed_xiao_esp32s3_can_slave_motor.bin    # Motor Z
id_20_esp32_seeed_xiao_esp32s3_can_slave_laser.bin    # Laser
id_30_esp32_seeed_xiao_esp32s3_can_slave_led.bin      # LED
```

## CAN ID Reference

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

## Usage in ImSwitch

```python
import requests

# Set firmware server
requests.post("http://localhost:8000/uc2config/setOTAFirmwareServer", json={})

# Set WiFi credentials
requests.post("http://localhost:8000/uc2config/setOTAWiFiCredentials", json={
    "ssid": "YourWiFi",
    "password": "YourPassword"
})

# Update Motor X
requests.post("http://localhost:8000/uc2config/startSingleDeviceOTA", json={
    "can_id": 11
})
```

## Running the Test Script

```bash
# Edit configuration
nano TEST_OTA_UPDATE.py
# Update: WIFI_SSID, WIFI_PASSWORD, FIRMWARE_SERVER_URL

# Run
python TEST_OTA_UPDATE.py
```

## Troubleshooting

**Can't connect to server?**
```bash
# Check if server is running
curl http://localhost:9000/latest/

# Should show HTML directory listing
```

**No firmware found?**
```bash
# Check filenames match pattern
ls ~/uc2_firmware/latest/id_*.bin

# Verify server is serving files
curl http://localhost:9000/latest/ | grep ".bin"
```

**Want to use different port?**
```bash
# Start server on different port
python3 -m http.server 8080

# Update in ImSwitch
requests.post("http://localhost:8000/uc2config/setOTAFirmwareServer", json={
    "server_url": "http://localhost:8080"
})
```

## Full Documentation

See:
- `CAN_OTA_FIRMWARE_SERVER.md` - Complete server setup guide
- `CAN_OTA_UPDATE_GUIDE.md` - Full OTA update documentation
