# OTA Update Test Script

This script provides an interactive interface for testing the CAN OTA update functionality in ImSwitch.

## Prerequisites

1. **ImSwitch Running**
   - ImSwitch must be running with REST API enabled
   - Default URL: `http://localhost:8000`
   - UC2 ESP32 device connected via serial

2. **Firmware Files**
   - Firmware files organized in a directory
   - Files must follow naming convention: `id_<CAN_ID>_*.bin`
   - Example: `id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin`

3. **WiFi Network**
   - 2.4GHz WiFi network accessible to CAN devices
   - SSID and password known

## Configuration

Edit the configuration variables at the top of `TEST_OTA_UPDATE.py`:

```python
IMSWITCH_API_URL = "http://localhost:8000"  # ImSwitch REST API URL
WIFI_SSID = "YourWiFiNetwork"              # WiFi network name
WIFI_PASSWORD = "YourWiFiPassword"          # WiFi password
FIRMWARE_DIR = "/path/to/firmware"          # Firmware directory
```

## Usage

### Run the Script

```bash
cd /path/to/ImSwitch
python imswitch/imcontrol/controller/controllers/TEST_OTA_UPDATE.py
```

### Menu Options

```
1. Setup OTA configuration
   - Sets WiFi credentials
   - Sets firmware directory
   - Lists found firmware files

2. List available firmware
   - Shows all firmware files in configured directory
   - Displays file sizes and modification dates
   - Organized by CAN ID

3. Show device mapping
   - Displays CAN ID assignments
   - Shows motors, lasers, LEDs

4. Update Motor X (CAN ID 11)
   - Updates single motor axis
   - Monitors progress in real-time
   - Shows success/failure status

5. Update Laser (CAN ID 20)
   - Updates laser controller
   - Real-time status monitoring

6. Update all motors (X, Y, Z)
   - Batch update for motors
   - Sequential processing with delays
   - Progress summary for all devices

7. Clear OTA status
   - Clears tracking information
   - Fresh start for new updates

0. Exit
```

## Example Session

```
========================================================
UC2 CAN OTA Update Test Script
========================================================

This script will test the OTA update functionality.
ImSwitch API: http://localhost:8000
WiFi SSID: MyNetwork
Firmware Directory: /home/user/firmware

========================================================
Menu
========================================================
1. Setup OTA configuration
2. List available firmware
3. Show device mapping
4. Update Motor X (CAN ID 11)
5. Update Laser (CAN ID 20)
6. Update all motors (X, Y, Z)
7. Clear OTA status
0. Exit

Enter your choice: 1

========================================================
Setting up OTA configuration...
========================================================

1. Setting WiFi credentials (SSID: MyNetwork)...
   Response: {'status': 'success', 'message': 'WiFi credentials set for SSID: MyNetwork'}

2. Setting firmware directory: /home/user/firmware...
   Response: {'status': 'success', 'firmware_files': ['id_11_...', 'id_20_...'], 'count': 6}

   Found 6 firmware files:
     - id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
     - id_12_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
     - id_13_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
     - id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin
     - id_30_esp32_seeed_xiao_esp32s3_can_slave_led_debug.bin
     - id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin

✅ Configuration completed!
```

## Monitoring Updates

When you start an update, the script will show real-time progress:

```
========================================================
Starting OTA Update: Motor X (CAN ID 11)
========================================================

1. Sending OTA command to device 11...
   Response: {'status': 'success', 'message': 'OTA update initiated for device 11'}

2. Monitoring OTA progress...
   Status: pending
   Status: uploading
   Status: uploading
   Status: success

✅ Firmware update completed successfully!
   Device IP: 192.168.1.100
   Hostname: UC2-CAN-B.local
```

## Batch Updates

For updating multiple devices:

```
========================================================
Starting Batch OTA Update (3 devices)
========================================================

Sending OTA commands to 3 devices...
Response: OTA update initiated for 3 devices

Monitoring update progress...

------------------------------------------------------------
⏳ Motor X (ID 11): uploading
⏳ Motor Y (ID 12): pending
⏳ Motor Z (ID 13): pending
------------------------------------------------------------
✅ Motor X (ID 11): success
⏳ Motor Y (ID 12): uploading
⏳ Motor Z (ID 13): pending
------------------------------------------------------------
✅ Motor X (ID 11): success
✅ Motor Y (ID 12): success
✅ Motor Z (ID 13): success

========================================================
Batch Update Summary
========================================================

✅ Success: 3
❌ Failed: 0
⏳ Pending: 0
```

## Troubleshooting

### Script Can't Connect to ImSwitch
**Error:** `Connection refused` or timeout

**Solutions:**
1. Verify ImSwitch is running
2. Check `IMSWITCH_API_URL` is correct
3. Ensure REST API is enabled in ImSwitch config

### No Firmware Files Found
**Error:** `firmware_count: 0`

**Solutions:**
1. Check `FIRMWARE_DIR` path is correct
2. Verify firmware files exist
3. Ensure files follow naming convention: `id_<CAN_ID>_*.bin`

### Device Not Responding
**Error:** Timeout waiting for OTA completion

**Solutions:**
1. Check CAN bus connection
2. Verify device is powered
3. Check WiFi credentials are correct
4. Ensure device can reach WiFi network

### Update Failed
**Error:** `upload_status: failed`

**Solutions:**
1. Check network connectivity
2. Verify firmware file is valid
3. Ensure device has enough flash space
4. Check device IP is reachable

## API Direct Usage

If you prefer to use the API directly instead of the script:

```python
import requests

BASE_URL = "http://localhost:8000"

# Setup
requests.post(f"{BASE_URL}/uc2config/setOTAWiFiCredentials",
              json={"ssid": "WiFi", "password": "pass"})

requests.post(f"{BASE_URL}/uc2config/setOTAFirmwareDirectory",
              json={"directory": "/firmware"})

# Update device
requests.post(f"{BASE_URL}/uc2config/startSingleDeviceOTA",
              json={"can_id": 11})

# Check status
response = requests.get(f"{BASE_URL}/uc2config/getOTAStatus?can_id=11")
print(response.json())
```

## Advanced Usage

### Custom Timeout
```python
# Increase timeout to 10 minutes
requests.post(f"{BASE_URL}/uc2config/startSingleDeviceOTA",
              json={"can_id": 11, "timeout": 600000})
```

### Override WiFi per Update
```python
# Use different WiFi for specific update
requests.post(f"{BASE_URL}/uc2config/startSingleDeviceOTA",
              json={
                  "can_id": 11,
                  "ssid": "DifferentNetwork",
                  "password": "DifferentPassword"
              })
```

### Custom Delay for Batch Updates
```python
# 5 second delay between devices
requests.post(f"{BASE_URL}/uc2config/startMultipleDeviceOTA",
              json={
                  "can_ids": [11, 12, 13],
                  "delay_between": 5
              })
```

## Exit Codes

- `0` - Normal exit
- `1` - Error during execution (exception raised)

## See Also

- [CAN OTA Update Guide](../docs/CAN_OTA_UPDATE_GUIDE.md) - Complete documentation
- [Implementation Details](../docs/CAN_OTA_IMPLEMENTATION.md) - Technical overview
- [Quick Reference](../docs/CAN_OTA_QUICK_REFERENCE.md) - API quick reference
