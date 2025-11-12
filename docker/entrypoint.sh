#!/usr/bin/env bash
# ImSwitch Docker Container Entry Point
# 
# This script handles the initialization and startup of ImSwitch in a Docker container.
# It supports both server and terminal modes, configuration management, and external storage detection.
#
# Environment Variables:
#   MODE                - Container mode: "server" (default) or "terminal"
#   HEADLESS           - Run in headless mode: "true"/"1" (GUI disabled)
#   SSL                - Enable SSL: "true" (default) or "false"/"0"
#   HTTP_PORT          - HTTP port for the server (default: 8001)
#   CONFIG_PATH        - Custom configuration directory path
#   CONFIG_FILE        - Specific configuration file to use
#   DATA_PATH          - Default data storage path
#   SCAN_EXT_DATA_PATH - Enable external storage scanning: "true"/"1" or "false" (default)
#   EXT_DATA_PATH      - Mount point directory for external drives (e.g., /media, /Volumes)
#   PIP_PACKAGES       - Additional Python packages to install
#   UPDATE_GIT         - Update ImSwitch from git: "true"/"1" or "false" (default)
#   UPDATE_CONFIG      - Update config repository: "true" or "false" (default)
#   UPDATE_INSTALL_GIT - Update and reinstall ImSwitch: "true"/"1" or "false" (default)
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
MODE="${MODE:-server}"
PIP_PACKAGES="${PIP_PACKAGES:-}"
PERSISTENT_PIP_DIR="${PERSISTENT_PIP_DIR:-/persistent_pip_packages}"
UPDATE_GIT="${UPDATE_GIT:-false}"
UPDATE_CONFIG="${UPDATE_CONFIG:-false}"
CONFIG_PATH="${CONFIG_PATH:-}"
UPDATE_INSTALL_GIT="${UPDATE_INSTALL_GIT:-false}"
SSL=${SSL:-false}
SCAN_EXT_DATA_PATH=${SCAN_EXT_DATA_PATH:-false}

# ============================================================================
# Terminal Mode - Drop into interactive bash
# ============================================================================
if [[ "${MODE}" == "terminal" ]]; then
    source /opt/conda/bin/activate imswitch
    log 'Starting the container in terminal mode'
    exec bash
fi

# ============================================================================
# Server Mode - Start ImSwitch
# ============================================================================
log 'Starting ImSwitch container in server mode'

# Display system information
log 'Listing USB devices'
lsusb || log 'lsusb not available'

log 'Checking external storage mount points'
ls /media 2>/dev/null || log 'No /media directory'
ls /Volumes 2>/dev/null || log 'No /Volumes directory'

# ============================================================================
# Git Patch Management (for development)
# ============================================================================
PATCH_DIR=/tmp/ImSwitch-changes
PATCH_FILE=$PATCH_DIR/diff.patch 

mkdir -p "$PATCH_DIR"
if [[ -f "$PATCH_FILE" ]]; then
    log "Applying stored patch from $PATCH_FILE"
    cd /tmp/ImSwitch
    git apply "$PATCH_FILE" || log 'Warning: Failed to apply patch'
else
    log 'No patch file found, proceeding without applying changes'
fi

# ============================================================================
# Configuration Repository Management
# ============================================================================
if [[ "$UPDATE_CONFIG" = "true" ]]; then
    log 'Updating ImSwitchConfig repository'
    cd /tmp/ImSwitchConfig
    git pull || log 'Warning: Failed to update config repository'
fi

# ============================================================================
# Configuration File Selection
# ============================================================================
if [[ -z "$CONFIG_PATH" ]]; then
    log 'No CONFIG_PATH set, using default config path'
    CONFIG_FILE="${CONFIG_FILE:-/tmp/ImSwitchConfig/imcontrol_setups/example_virtual_microscope.json}"
else
    log "Using custom CONFIG_PATH: $CONFIG_PATH"
    CONFIG_FILE=None
    log 'Available configuration files:'
    ls /config/ImSwitchConfig/imcontrol_setups 2>/dev/null || log 'Config directory not found'
    
    # Display current options if available
    if [[ -f /config/ImSwitchConfig/config/imcontrol_options.json ]]; then
        log 'Current imcontrol_options.json:'
        cat /config/ImSwitchConfig/config/imcontrol_options.json
    fi
fi

# ============================================================================
# Python Package Installation
# ============================================================================
mkdir -p "$PERSISTENT_PIP_DIR"
export PYTHONUSERBASE="$PERSISTENT_PIP_DIR"
export PATH="$PERSISTENT_PIP_DIR/bin:$PATH"

if [[ -n "$PIP_PACKAGES" ]]; then
    log "Installing additional packages: $PIP_PACKAGES"
    for package in $PIP_PACKAGES; do
        /opt/conda/bin/conda run -n imswitch uv pip install --user "$package" || \
            log "Warning: Failed to install package: $package"
    done
fi

# ============================================================================
# ImSwitch Repository Update (for development)
# ============================================================================
if [[ "$UPDATE_GIT" == "true" || "$UPDATE_GIT" == "1" ]]; then
    log 'Updating ImSwitch from git repository'
    cd /tmp/ImSwitch
    
    # Save current changes as patch
    if [[ -f "$PATCH_FILE" ]]; then
        log "Applying stored patch before update"
        git apply "$PATCH_FILE" || { log 'Error: Failed to apply patch, aborting update'; exit 1; }
    fi
    
    git fetch origin
    git diff HEAD origin/master > "$PATCH_FILE"
    
    if [[ -s "$PATCH_FILE" ]]; then
        log "New changes detected, patch saved at: $PATCH_FILE"
    else
        log "No new changes detected"
        rm -f "$PATCH_FILE"
    fi
    
    git merge origin/master || log 'Warning: Failed to merge changes'
fi

if [[ "$UPDATE_INSTALL_GIT" == "true" || "$UPDATE_INSTALL_GIT" == "1" ]]; then
    log 'Updating and reinstalling ImSwitch'
    cd /tmp/ImSwitch
    git pull
    /bin/bash -c 'source /opt/conda/bin/activate imswitch && uv pip install --target /persistent_pip_packages /tmp/ImSwitch' || \
        log 'Warning: Failed to reinstall ImSwitch'
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
params+=" --config-folder ${CONFIG_PATH:-None}"
params+=" --config-file ${CONFIG_FILE:-None}"
params+=" --data-folder ${DATA_PATH:-None}"

# External storage scanning
# When enabled, ImSwitch will automatically detect and use external drives
if [[ $SCAN_EXT_DATA_PATH == "1" || $SCAN_EXT_DATA_PATH == "True" || $SCAN_EXT_DATA_PATH == "true" ]]; then
    params+=" --scan-ext-data-folder"
fi

# External mount point directory
# Typically /media (Linux) or /Volumes (macOS)
params+=" --ext-data-folder ${EXT_DATA_PATH:-None}"

# ============================================================================
# Start ImSwitch
# ============================================================================
log 'Starting ImSwitch with the following parameters:'
log "python3 -m imswitch $params"

python3 -m imswitch $params
