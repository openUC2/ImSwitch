# CAN OTA Firmware Server Setup

## Overview

The UC2 CAN OTA update system fetches firmware files from a local HTTP server instead of a directory. This allows for centralized firmware management and easier updates across multiple ImSwitch instances.

## Server Requirements

The firmware server should:
- Serve files via HTTP (default: `http://localhost:9000`)
- Provide a `/latest/` endpoint with directory listing
- Host firmware files with naming pattern: `id_<CANID>_*.bin`

## Quick Setup with Python HTTP Server

### 1. Organize Firmware Files

Create a directory structure:
```bash
mkdir -p ~/uc2_firmware/latest
cd ~/uc2_firmware/latest
```

Place your firmware files with proper naming:
```
latest/
├── id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
├── id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
├── id_12_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
├── id_13_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
├── id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin
└── id_21_esp32_seeed_xiao_esp32s3_can_slave_led_debug.bin
```

### 2. Start the HTTP Server

```bash
# Navigate to the parent directory
cd ~/uc2_firmware

# Start Python HTTP server on port 9000
python3 -m http.server 9000
```

The server will be accessible at `http://localhost:9000/latest/`

### 3. Verify Server is Running

```bash
# Test the server
curl http://localhost:9000/latest/

# Should return HTML with directory listing
```

## Firmware Naming Convention

Firmware files **must** follow this naming pattern:

```
id_<CANID>_<description>.bin
```

Where:
- `<CANID>` is the CAN ID of the device (10-13 for motors, 20-29 for lasers, 30-39 for LEDs)
- `<description>` is any descriptive text (board type, device type, variant, etc.)

### Examples

```
id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin          # Motor A (CAN ID 10)
id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin          # Motor X (CAN ID 11)
id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin    # Laser (CAN ID 20)
id_30_esp32_seeed_xiao_esp32s3_can_slave_led.bin            # LED (CAN ID 30)
```

## ImSwitch Configuration

### Set Firmware Server URL

```python
import requests

# Use default server (localhost:9000)
response = requests.post(
    "http://localhost:8000/uc2config/setOTAFirmwareServer",
    json={}
)

# Or specify custom server
response = requests.post(
    "http://localhost:8000/uc2config/setOTAFirmwareServer",
    json={
        "server_url": "http://192.168.1.100:9000"
    }
)
```

### List Available Firmware

```python
response = requests.get("http://localhost:8000/uc2config/listAvailableFirmware")
print(response.json())

# Output:
# {
#     "status": "success",
#     "firmware_server": "http://localhost:9000",
#     "firmware_count": 6,
#     "firmware": {
#         "10": {
#             "filename": "id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin",
#             "url": "http://localhost:9000/latest/id_10_...",
#             "can_id": 10
#         },
#         ...
#     }
# }
```

## Advanced Server Setup

### Using Nginx

Create nginx configuration:

```nginx
server {
    listen 9000;
    server_name localhost;
    
    location /latest/ {
        alias /path/to/uc2_firmware/latest/;
        autoindex on;
        autoindex_exact_size off;
        autoindex_localtime on;
    }
}
```

### Using Apache

Create Apache configuration:

```apache
Listen 9000

<VirtualHost *:9000>
    DocumentRoot "/path/to/uc2_firmware"
    
    <Directory "/path/to/uc2_firmware/latest">
        Options +Indexes
        Require all granted
    </Directory>
</VirtualHost>
```

### Docker Container

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /firmware
COPY latest/ /firmware/latest/

EXPOSE 9000

CMD ["python", "-m", "http.server", "9000"]
```

Build and run:

```bash
docker build -t uc2-firmware-server .
docker run -d -p 9000:9000 --name uc2-firmware uc2-firmware-server
```

## Firmware Cache

ImSwitch caches downloaded firmware files locally to improve performance.

### View Cache Status

```python
response = requests.get("http://localhost:8000/uc2config/getOTAFirmwareCacheStatus")
print(response.json())

# Output:
# {
#     "status": "success",
#     "cache_directory": "/tmp/uc2_ota_firmware_cache",
#     "exists": true,
#     "file_count": 3,
#     "total_size": 4567890,
#     "total_size_mb": 4.36,
#     "cached_files": [
#         {
#             "filename": "id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin",
#             "size": 1523000,
#             "modified": "2025-11-04T10:30:00"
#         }
#     ]
# }
```

### Clear Cache

```python
# Force fresh downloads on next OTA update
response = requests.post("http://localhost:8000/uc2config/clearOTAFirmwareCache")
print(response.json())

# Output:
# {
#     "status": "success",
#     "message": "Cleared 3 cached firmware files",
#     "cache_directory": "/tmp/uc2_ota_firmware_cache"
# }
```

## Workflow Example

### Complete OTA Update Workflow

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Set up firmware server
print("Setting up firmware server...")
response = requests.post(f"{BASE_URL}/uc2config/setOTAFirmwareServer", json={})
print(response.json())

# 2. List available firmware
print("\nAvailable firmware:")
response = requests.get(f"{BASE_URL}/uc2config/listAvailableFirmware")
firmware = response.json()
for can_id, info in firmware["firmware"].items():
    print(f"  CAN ID {can_id}: {info['filename']}")

# 3. Set WiFi credentials
print("\nSetting WiFi credentials...")
requests.post(f"{BASE_URL}/uc2config/setOTAWiFiCredentials", json={
    "ssid": "YourWiFi",
    "password": "YourPassword"
})

# 4. Start OTA update for Motor X
print("\nStarting OTA update for Motor X (CAN ID 11)...")
response = requests.post(f"{BASE_URL}/uc2config/startSingleDeviceOTA", json={
    "can_id": 11
})
print(response.json())

# 5. Monitor progress
print("\nMonitoring OTA progress...")
for i in range(30):  # Wait up to 60 seconds
    time.sleep(2)
    
    response = requests.get(f"{BASE_URL}/uc2config/getOTAStatus?can_id=11")
    if response.status_code == 200:
        status = response.json()
        
        if "ota_status" in status:
            upload_status = status["ota_status"].get("upload_status", "unknown")
            print(f"  Status: {upload_status}")
            
            if upload_status == "success":
                print("\n✅ Firmware update completed successfully!")
                break
            elif upload_status == "failed":
                error = status["ota_status"].get("upload_error", "Unknown error")
                print(f"\n❌ Firmware update failed: {error}")
                break
    else:
        print("  Waiting for device response...")

# 6. Check cache status
print("\nCache status:")
response = requests.get(f"{BASE_URL}/uc2config/getOTAFirmwareCacheStatus")
cache = response.json()
print(f"  Cached files: {cache['file_count']}")
print(f"  Total size: {cache['total_size_mb']} MB")
```

## Troubleshooting

### Server Not Accessible

**Problem:** Cannot connect to firmware server

**Solutions:**
1. Verify server is running: `curl http://localhost:9000/latest/`
2. Check firewall settings
3. Ensure correct port number
4. Try accessing from the ImSwitch machine

### Firmware Not Found

**Problem:** "No firmware found for device X on server"

**Solutions:**
1. Check firmware file naming matches pattern: `id_<CANID>_*.bin`
2. Verify file exists on server: `curl http://localhost:9000/latest/`
3. Check server directory structure has `/latest/` path
4. Ensure firmware file has correct CAN ID in filename

### Download Failed

**Problem:** Error downloading firmware from server

**Solutions:**
1. Check server logs for errors
2. Verify file permissions on server
3. Ensure sufficient disk space on ImSwitch machine
4. Check network connectivity
5. Clear cache and retry: `clearOTAFirmwareCache`

### Wrong Firmware Downloaded

**Problem:** Generic firmware used instead of specific CAN ID

**Solutions:**
1. Ensure firmware file has exact CAN ID match: `id_11_*.bin` for CAN ID 11
2. Check for typos in filename
3. Clear cache to force re-download
4. Verify server is serving the correct files

## Production Deployment

### Recommendations

1. **Use a proper web server** (Nginx, Apache) instead of Python's http.server
2. **Enable HTTPS** if serving over network
3. **Implement authentication** for firmware downloads
4. **Set up automatic firmware updates** from your build server
5. **Version your firmware** and maintain multiple versions
6. **Monitor server logs** for download statistics
7. **Backup firmware files** regularly

### Multi-Instance Setup

For multiple ImSwitch instances:

```python
# Each instance can point to the same server
# Instance 1
requests.post("http://instance1:8000/uc2config/setOTAFirmwareServer", 
              json={"server_url": "http://central-server:9000"})

# Instance 2
requests.post("http://instance2:8000/uc2config/setOTAFirmwareServer",
              json={"server_url": "http://central-server:9000"})
```

### CI/CD Integration

Automatically update firmware server from your build pipeline:

```bash
#!/bin/bash
# deploy_firmware.sh

# Build firmware
platformio run

# Copy to firmware server
scp .pio/build/*/firmware.bin \
    server:/var/www/uc2_firmware/latest/id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin

# Notify ImSwitch instances
curl -X POST http://instance1:8000/uc2config/clearOTAFirmwareCache
curl -X POST http://instance2:8000/uc2config/clearOTAFirmwareCache
```

## API Reference

See main documentation: `CAN_OTA_UPDATE_GUIDE.md`

### New Endpoints

- `POST /uc2config/setOTAFirmwareServer` - Configure firmware server URL
- `GET /uc2config/getOTAFirmwareCacheStatus` - View cache status
- `POST /uc2config/clearOTAFirmwareCache` - Clear firmware cache
