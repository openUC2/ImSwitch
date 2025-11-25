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

# Display system information
log 'Listing USB devices'
lsusb || log 'lsusb not available'

log 'Checking external storage mount points'
ls /media 2>/dev/null || log 'No /media directory'



# ============================================================================
# Configuration Path Setup
# ============================================================================
log "Using CONFIG_PATH: $CONFIG_PATH"

# in case the user doesn'T provide the CONFIG_PATH, quit with an error 
if [[ -z "$CONFIG_PATH" ]]; then
    log "Error: Configuration path '$CONFIG_PATH' not provided."
    exit 1
fi
if [[ ! -d "$CONFIG_PATH" ]]; then
    log "Error: Configuration path '$CONFIG_PATH' does not exist."
    exit 1
fi

# in case the user doesn'T provide the CONFIG_PATH, quit with an error 
if [[ -z "$CONFIG_FILE" ]]; then
    log "Error: Configuration file '$CONFIG_FILE' not provided."
    exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    log "Error: Configuration file '$CONFIG_FILE' does not exist."
    exit 1
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

# in case the user doesn'T provide the DATA_PATH, quit with an error 
if [[ -z "$DATA_PATH" ]]; then
    log "Error: Data path '$DATA_PATH' not provided."
    exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
    log "Error: Data path '$DATA_PATH' does not exist."
    exit 1
fi

# in case the user doesn'T provide the EXT_DATA_PATH, quit with an error 
if [[ -z "$EXT_DATA_PATH" ]]; then
    log "Error: External data path '$EXT_DATA_PATH' not provided."
    exit 1
fi
if [[ ! -d "$EXT_DATA_PATH" ]]; then
    log "Error: External data path '$EXT_DATA_PATH' does not exist."
    exit 1
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
params+=" --config-file ${CONFIG_FILE}"
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

python3 -m imswitch $params
