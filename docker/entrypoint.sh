#!/usr/bin/env bash
# ImSwitch Docker Container Entry Point
#
# This script handles the initialization and startup of ImSwitch in a Docker container.
# It supports both server and terminal modes, configuration management, and external storage detection.
#
# Environment Variables:
#   HEADLESS           - Run in headless mode: "true"/"1" (GUI disabled)
#   SSL                - Enable SSL: "true" (default) or "false"/"0"
#   HTTP_PORT          - HTTP port for the server (default: 8001)
#   CONFIG_PATH        - Custom configuration directory path
#   CONFIG_FILE        - Specific configuration file to use
#   DATA_PATH          - Default data storage path
#   SCAN_EXT_DATA_PATH - Enable external storage scanning: "true"/"1" or "false" (default)
#   EXT_DATA_PATH      - Mount point directory for external drives (e.g., /media, /Volumes)
#
# Interactive Shell:
#   For an interactive shell with ImSwitch environment activated, use:
#   docker run --entrypoint=venv-shell.sh <image-name>
#
# Storage Management:
#   The new storage management system automatically handles:
#   - Detection of external drives when SCAN_EXT_DATA_PATH=true
#   - Automatic fallback to default paths when external storage unavailable
#   - Runtime path switching via REST API
#
# For more information, see: https://github.com/openUC2/ImSwitch

#set -euo pipefail

# For Picamera2 support on Raspberry Pi, run with:
# docker run -it --privileged \
#   -v /run/dbus/system_bus_socket:/run/dbus/system_bus_socket \
#   -v /dev/dma_heap:/dev/dma_heap \
#   --device /dev/video0:/dev/video0 \
#   --device /dev/video10:/dev/video10 \
#   --device /dev/video11:/dev/video11 \
#   --device /dev/video12:/dev/video12 \
#   -e MODE=terminal \
#   ghcr.io/openuc2/imswitch-noqt:sha-5d54391


check_pi_camera() {
        log "Running libcamera-probe"
        if command -v libcamera-probe >/dev/null 2>&1; then
            libcamera-probe --verbose 2>&1 | tee /tmp/libcamera-probe.log
            if grep -q "Found.*imx" /tmp/libcamera-probe.log; then
                log "Pi camera detected via libcamera-probe"
                return 0
            else
                log "libcamera-probe did not detect a Pi camera"
            fi
        else
            log "libcamera-probe not installed"
        fi

        log "Checking dmesg for camera sensors"
        if dmesg | grep -E "imx[0-9]{3}|ov5647|arducam|rpi_cam" >/dev/null 2>&1; then
            log "Camera sensor driver detected in kernel logs"
            return 0
        else
            log "No camera-related kernel messages found"
        fi

        log "Checking for /dev/video* devices"
        if ls /dev/video* >/dev/null 2>&1; then
            log "Video devices present: $(ls /dev/video*)"
            return 0
        else
            log "No /dev/video* devices found"
        fi

        return 1
}

log() { echo "[$(date +'%F %T')] $*"; }

# ============================================================================
# Environment Variable Defaults
# ============================================================================
CONFIG_PATH="${CONFIG_PATH:-}"
SSL=${SSL:-false}
SCAN_EXT_DATA_PATH=${SCAN_EXT_DATA_PATH:-false}

# ============================================================================
# Server Mode - Start ImSwitch
# ============================================================================
log 'Starting ImSwitch container in server mode'


log 'Listing USB devices'
lsusb || log 'lsusb not available'
log 'Checking Raspberry Pi camera availability'


if check_pi_camera; then
    log "Raspberry Pi camera: AVAILABLE"
else
    log "Raspberry Pi camera: NOT AVAILABLE"
fi


log 'Checking external storage mount points'
ls /media 2>/dev/null || log 'No /media directory'



# ============================================================================
# Configuration Path Setup
# ============================================================================
log "Using CONFIG_PATH: $CONFIG_PATH"

# Validate CONFIG_PATH is provided and exists
if [[ -z "$CONFIG_PATH" ]]; then
    log "Error: Configuration path (CONFIG_PATH) not provided."
    exit 1
fi
if [[ ! -d "$CONFIG_PATH" ]]; then
    log "Error: Configuration path '$CONFIG_PATH' does not exist."
    exit 1
fi

# CONFIG_FILE is optional - if not provided, ImSwitch will use imcontrol_options.json
if [[ -n "$CONFIG_FILE" ]]; then
    log "Using CONFIG_FILE: $CONFIG_FILE"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log "Warning: Configuration file '$CONFIG_FILE' does not exist. Will try to use filename from CONFIG_PATH."
    fi
else
    log "CONFIG_FILE not provided - will use default from imcontrol_options.json"
fi

# List available configuration files
log 'Available configuration files:'
ls "${CONFIG_PATH}/imcontrol_setups" 2>/dev/null || log 'Config directory not found'

# Display current options if available
if [[ -f "${CONFIG_PATH}/config/imcontrol_options.json" ]]; then
    log 'Current imcontrol_options.json:'
    cat "${CONFIG_PATH}/config/imcontrol_options.json"
fi

# ============================================================================
# Data Paths Setup
# ============================================================================

# Validate DATA_PATH is provided and exists
if [[ -z "$DATA_PATH" ]]; then
    log "Error: Data path (DATA_PATH) not provided."
    exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
    log "Warning: Data path '$DATA_PATH' does not exist yet. Will be created by Docker volume mount."
fi

# Validate EXT_DATA_PATH if external scanning is enabled
if [[ $SCAN_EXT_DATA_PATH == "1" || $SCAN_EXT_DATA_PATH == "True" || $SCAN_EXT_DATA_PATH == "true" ]]; then
    if [[ -z "$EXT_DATA_PATH" ]]; then
        log "Warning: External data scanning enabled but EXT_DATA_PATH not provided."
    elif [[ ! -d "$EXT_DATA_PATH" ]]; then
        log "Warning: External data path '$EXT_DATA_PATH' does not exist yet."
    fi
fi

# ============================================================================
# Activate Python Environment
# ============================================================================
source /opt/conda/bin/activate imswitch

# ============================================================================
# Build ImSwitch Command Line Arguments
# ============================================================================
# Note: The new storage management system handles path resolution internally.
# Legacy arguments are preserved for backward compatibility.

params=()

# Headless mode
if [[ $HEADLESS == "1" || $HEADLESS == "True" || $HEADLESS == "true" ]]; then
    params+=" --headless"
fi

# SSL configuration
if [[ $SSL == "0" || $SSL == "False" || $SSL == "false" ]]; then
    params+=" --no-ssl"
fi

# Network configuration
params+=" --http-port ${HTTP_PORT:-8001}"

# Path configuration
# The storage manager will handle path resolution and validation
params+=" --config-folder ${CONFIG_PATH}"

# Only pass --config-file if CONFIG_FILE is set
if [[ -n "$CONFIG_FILE" ]]; then
    params+=" --config-file ${CONFIG_FILE}"
fi

params+=" --data-folder ${DATA_PATH}"

# External storage scanning
# When enabled, ImSwitch will automatically detect and use external drives
if [[ $SCAN_EXT_DATA_PATH == "1" || $SCAN_EXT_DATA_PATH == "True" || $SCAN_EXT_DATA_PATH == "true" ]]; then
    params+=" --scan-ext-data-folder"
fi

# External mount point directory
# Typically /media (Linux) or /Volumes (macOS)
params+=" --ext-data-folder ${EXT_DATA_PATH}"

# ============================================================================
# Start ImSwitch
# ============================================================================
log 'Starting ImSwitch with the following parameters:'
log "python3 -m imswitch $params"
# params is a single string, so we use eval to properly pass it as multiple arguments
# example: eval python3 -m imswitch --headless --http-port 8001 --no-ssl --config-folder /Users/bene/ImSwitchConfig --config-file example_virtual_microscope.json --data-folder /Users/bene/ImSwitchConfig  --scan-ext-data-folder --ext-data-folder /Volumes

python3 -m imswitch $params
