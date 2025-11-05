#!/usr/bin/env python3
"""
Test script for CAN OTA update functionality in ImSwitch.

This script demonstrates how to use the OTA update API endpoints
to update firmware on UC2 CAN satellite devices.

Usage:
    python TEST_OTA_UPDATE.py

Requirements:
    - ImSwitch running with REST API enabled
    - UC2 ESP32 device connected via serial
    - Firmware files in the configured directory

Copyright 2025 Benedict Diederich, released under LGPL 3.0 or later
"""

import requests
import time
import json
from pathlib import Path

# Configuration
IMSWITCH_API_URL = "http://localhost:8000"  # Adjust to your ImSwitch REST API URL
WIFI_SSID = "YourWiFiNetwork"  # Replace with your WiFi network name
WIFI_PASSWORD = "YourWiFiPassword"  # Replace with your WiFi password
FIRMWARE_SERVER_URL = "http://localhost:9000"  # Firmware server URL

# Device CAN IDs
DEVICE_IDS = {
    "motor_x": 11,
    "motor_y": 12,
    "motor_z": 13,
    "motor_a": 10,
    "laser": 20,
    "led": 30
}


def setup_ota_config():
    """Configure WiFi credentials and firmware server."""
    print("=" * 60)
    print("Setting up OTA configuration...")
    print("=" * 60)
    
    # Set WiFi credentials
    print(f"\n1. Setting WiFi credentials (SSID: {WIFI_SSID})...")
    response = requests.post(
        f"{IMSWITCH_API_URL}/uc2config/setOTAWiFiCredentials",
        json={
            "ssid": WIFI_SSID,
            "password": WIFI_PASSWORD
        }
    )
    print(f"   Response: {response.json()}")
    
    # Set firmware server
    print(f"\n2. Setting firmware server: {FIRMWARE_SERVER_URL}...")
    response = requests.post(
        f"{IMSWITCH_API_URL}/uc2config/setOTAFirmwareServer",
        json={
            "server_url": FIRMWARE_SERVER_URL
        }
    )
    result = response.json()
    print(f"   Response: {result}")
    
    if result.get("status") == "success":
        print(f"\n   Found {result['count']} firmware files:")
        for fw_file in result.get("firmware_files", []):
            print(f"     - {fw_file}")
    
    print("\n✅ Configuration completed!")


def list_available_firmware():
    """List all available firmware files."""
    print("\n" + "=" * 60)
    print("Available Firmware Files")
    print("=" * 60)
    
    response = requests.get(f"{IMSWITCH_API_URL}/uc2config/listAvailableFirmware")
    result = response.json()
    
    if result.get("status") == "success":
        firmware = result.get("firmware", {})
        print(f"\nFound {len(firmware)} firmware files:\n")
        
        for can_id, fw_info in firmware.items():
            print(f"CAN ID {can_id}:")
            print(f"  File: {fw_info['filename']}")
            print(f"  Size: {fw_info['size']:,} bytes")
            print(f"  Modified: {fw_info['modified']}")
            print()
    else:
        print(f"❌ Error: {result.get('message')}")


def get_device_mapping():
    """Get mapping of device types to CAN IDs."""
    print("\n" + "=" * 60)
    print("Device CAN ID Mapping")
    print("=" * 60)
    
    response = requests.get(f"{IMSWITCH_API_URL}/uc2config/getOTADeviceMapping")
    result = response.json()
    
    if result.get("status") == "success":
        mapping = result.get("mapping", {})
        
        print("\nMotors:")
        for axis, can_id in mapping.get("motors", {}).items():
            print(f"  Motor {axis}: CAN ID {can_id}")
        
        print("\nLasers:")
        for laser, can_id in mapping.get("laser", {}).items():
            print(f"  {laser}: CAN ID {can_id}")
        
        print("\nLEDs:")
        for led, can_id in mapping.get("led", {}).items():
            print(f"  {led}: CAN ID {can_id}")
        
        print(f"\nMaster: CAN ID {mapping.get('master')}")


def update_single_device(device_name, can_id, timeout=300000):
    """
    Update a single device.
    
    :param device_name: Human-readable device name (e.g., "Motor X")
    :param can_id: CAN ID of the device
    :param timeout: OTA timeout in milliseconds
    """
    print("\n" + "=" * 60)
    print(f"Starting OTA Update: {device_name} (CAN ID {can_id})")
    print("=" * 60)
    
    # Start OTA update
    print(f"\n1. Sending OTA command to device {can_id}...")
    response = requests.post(
        f"{IMSWITCH_API_URL}/uc2config/startSingleDeviceOTA",
        json={
            "can_id": can_id,
            "timeout": timeout
        }
    )
    result = response.json()
    print(f"   Response: {result}")
    
    if result.get("status") != "success":
        print(f"❌ Failed to start OTA: {result.get('message')}")
        return False
    
    # Monitor status
    print(f"\n2. Monitoring OTA progress...")
    max_wait = 60  # Maximum wait time in seconds
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        response = requests.get(
            f"{IMSWITCH_API_URL}/uc2config/getOTAStatus",
            params={"can_id": can_id}
        )
        
        if response.status_code != 200:
            print(f"   ⏳ Waiting for device response...")
            time.sleep(2)
            continue
        
        status = response.json()
        
        if status.get("status") == "error":
            print(f"   ⏳ Waiting for device response...")
            time.sleep(2)
            continue
        
        ota_status = status.get("ota_status", {})
        upload_status = ota_status.get("upload_status", "unknown")
        
        print(f"   Status: {upload_status}", end="\r")
        
        if upload_status == "success":
            print(f"\n\n✅ Firmware update completed successfully!")
            print(f"   Device IP: {ota_status.get('ip')}")
            print(f"   Hostname: {ota_status.get('hostname')}")
            return True
        
        elif upload_status == "failed":
            print(f"\n\n❌ Firmware update failed!")
            print(f"   Error: {ota_status.get('upload_error', 'Unknown error')}")
            return False
        
        time.sleep(2)
    
    print(f"\n\n⏱️ Timeout waiting for OTA completion")
    return False


def update_multiple_devices(device_list):
    """
    Update multiple devices sequentially.
    
    :param device_list: List of (device_name, can_id) tuples
    """
    print("\n" + "=" * 60)
    print(f"Starting Batch OTA Update ({len(device_list)} devices)")
    print("=" * 60)
    
    can_ids = [can_id for _, can_id in device_list]
    
    # Start OTA for all devices
    print(f"\nSending OTA commands to {len(can_ids)} devices...")
    response = requests.post(
        f"{IMSWITCH_API_URL}/uc2config/startMultipleDeviceOTA",
        json={
            "can_ids": can_ids,
            "delay_between": 2
        }
    )
    result = response.json()
    print(f"Response: {result.get('message')}")
    
    # Monitor progress
    print(f"\nMonitoring update progress...\n")
    max_wait = 120  # Maximum wait time in seconds
    start_time = time.time()
    
    results = {can_id: "pending" for _, can_id in device_list}
    
    while (time.time() - start_time) < max_wait:
        response = requests.get(f"{IMSWITCH_API_URL}/uc2config/getOTAStatus")
        
        if response.status_code != 200:
            time.sleep(3)
            continue
        
        status = response.json()
        devices = status.get("devices", {})
        
        all_done = True
        
        print("\n" + "-" * 60)
        for device_name, can_id in device_list:
            device_status = devices.get(str(can_id), {})
            upload_status = device_status.get("upload_status", "unknown")
            results[can_id] = upload_status
            
            status_icon = "✅" if upload_status == "success" else "❌" if upload_status == "failed" else "⏳"
            print(f"{status_icon} {device_name} (ID {can_id}): {upload_status}")
            
            if upload_status not in ["success", "failed"]:
                all_done = False
        
        if all_done:
            break
        
        time.sleep(3)
    
    # Summary
    print("\n" + "=" * 60)
    print("Batch Update Summary")
    print("=" * 60)
    
    success_count = sum(1 for status in results.values() if status == "success")
    failed_count = sum(1 for status in results.values() if status == "failed")
    
    print(f"\n✅ Success: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⏳ Pending: {len(results) - success_count - failed_count}")


def clear_ota_status():
    """Clear all OTA status information."""
    print("\nClearing OTA status...")
    response = requests.post(
        f"{IMSWITCH_API_URL}/uc2config/clearOTAStatus",
        json={}
    )
    print(f"Response: {response.json()}")


def get_cache_status():
    """Get firmware cache status."""
    print("\n" + "=" * 60)
    print("Firmware Cache Status")
    print("=" * 60)
    
    response = requests.get(f"{IMSWITCH_API_URL}/uc2config/getOTAFirmwareCacheStatus")
    result = response.json()
    
    if result.get("status") == "success":
        print(f"\nCache Directory: {result.get('cache_directory')}")
        print(f"Exists: {result.get('exists')}")
        
        if result.get('exists'):
            print(f"File Count: {result.get('file_count')}")
            print(f"Total Size: {result.get('total_size_mb')} MB")
            
            print("\nCached Files:")
            for file_info in result.get('cached_files', []):
                print(f"  - {file_info['filename']}")
                print(f"    Size: {file_info['size']:,} bytes")
                print(f"    Modified: {file_info['modified']}")
    else:
        print(f"❌ Error: {result.get('message')}")


def clear_cache():
    """Clear firmware cache."""
    print("\n" + "=" * 60)
    print("Clearing Firmware Cache")
    print("=" * 60)
    
    response = requests.post(f"{IMSWITCH_API_URL}/uc2config/clearOTAFirmwareCache")
    result = response.json()
    
    if result.get("status") == "success":
        print(f"\n✅ {result.get('message')}")
        print(f"Cache Directory: {result.get('cache_directory')}")
    else:
        print(f"❌ Error: {result.get('message')}")


def main():
    """Main test script."""
    print("\n" + "=" * 60)
    print("UC2 CAN OTA Update Test Script")
    print("=" * 60)
    
    # Test configuration
    print("\nThis script will test the OTA update functionality.")
    print(f"ImSwitch API: {IMSWITCH_API_URL}")
    print(f"WiFi SSID: {WIFI_SSID}")
    print(f"Firmware Server: {FIRMWARE_SERVER_URL}")
    
    # Menu
    while True:
        print("\n" + "=" * 60)
        print("Menu")
        print("=" * 60)
        print("1. Setup OTA configuration")
        print("2. List available firmware")
        print("3. Show device mapping")
        print("4. Update Motor X (CAN ID 11)")
        print("5. Update Laser (CAN ID 20)")
        print("6. Update all motors (X, Y, Z)")
        print("7. Get cache status")
        print("8. Clear firmware cache")
        print("9. Clear OTA status")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "1":
            setup_ota_config()
        
        elif choice == "2":
            list_available_firmware()
        
        elif choice == "3":
            get_device_mapping()
        
        elif choice == "4":
            update_single_device("Motor X", DEVICE_IDS["motor_x"])
        
        elif choice == "5":
            update_single_device("Laser", DEVICE_IDS["laser"])
        
        elif choice == "6":
            devices = [
                ("Motor X", DEVICE_IDS["motor_x"]),
                ("Motor Y", DEVICE_IDS["motor_y"]),
                ("Motor Z", DEVICE_IDS["motor_z"])
            ]
            update_multiple_devices(devices)
        
        elif choice == "7":
            get_cache_status()
        
        elif choice == "8":
            clear_cache()
        
        elif choice == "9":
            clear_ota_status()
        
        elif choice == "0":
            print("\nExiting...")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
