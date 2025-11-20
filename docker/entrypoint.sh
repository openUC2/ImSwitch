#!/usr/bin/env bash

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

log() { echo "[$(date +'%F %T')] $*"; }

# Provide safe default values for variables
MODE="${MODE:-server}"
PIP_PACKAGES="${PIP_PACKAGES:-}"
PERSISTENT_PIP_DIR="${PERSISTENT_PIP_DIR:-/persistent_pip_packages}"
UPDATE_GIT="${UPDATE_GIT:-false}"
UPDATE_CONFIG="${UPDATE_CONFIG:-false}"
CONFIG_PATH="${CONFIG_PATH:-}"
UPDATE_INSTALL_GIT="${UPDATE_INSTALL_GIT:-false}"
SSL=${SSL:-false}
SCAN_EXT_DATA_PATH=${SCAN_EXT_DATA_PATH:-false}

if [[ "${MODE}" != "terminal" ]];
then
    echo 'Starting the container'
    echo 'Listing USB Bus'
    lsusb
    echo 'Listing external storage devices'
    ls /media

    PATCH_DIR=/tmp/ImSwitch-changes
    PATCH_FILE=$PATCH_DIR/diff.patch 

    mkdir -p "$PATCH_DIR"
    if [[ -f "$PATCH_FILE" ]]
    then
        echo "Applying stored patch from $PATCH_FILE"
        cd /tmp/ImSwitch
        git apply "$PATCH_FILE"
    else
        echo 'No patch file found, proceeding without applying changes'
    fi
    
    if [[ "$UPDATE_CONFIG" = "true" ]]
    then
        echo 'Pulling the ImSwitchConfig repository'
        cd /tmp/ImSwitchConfig
        git pull
    fi
    if [[ -z "$CONFIG_PATH" ]]
    then
        echo 'No CONFIG_PATH set, using default config path'
        CONFIG_FILE="${CONFIG_FILE:-/tmp/ImSwitchConfig/imcontrol_setups/example_virtual_microscope.json}"
    else
        echo 'Using custom CONFIG_PATH:' "$CONFIG_PATH" ' which maps to the following files:'
        CONFIG_FILE=None
        echo 'Listing Config Dir'
        ls /config/ImSwitchConfig/imcontrol_setups

        # printing the content of /config/ImSwitchConfig/config/imcontrol_options.json
        cat /config/ImSwitchConfig/config/imcontrol_options.json

    fi
    
    mkdir -p "$PERSISTENT_PIP_DIR"
    export PYTHONUSERBASE="$PERSISTENT_PIP_DIR"
    export PATH="$PERSISTENT_PIP_DIR/bin:$PATH"
    if [[ -n "$PIP_PACKAGES" ]]
    then
        echo "Installing additional packages with UV: $PIP_PACKAGES"
        for package in $PIP_PACKAGES
        do
            /opt/conda/bin/conda run -n imswitch uv pip install --user $package
        done
    fi
    if [[ "$UPDATE_GIT" == true || "$UPDATE_GIT" == "1" ]]
    then
        PATCH_DIR="/tmp/ImSwitch-changes"
        PATCH_FILE="$PATCH_DIR/diff.patch"
        mkdir -p "$PATCH_DIR"
        cd /tmp/ImSwitch
        if [[ -f "$PATCH_FILE" ]]
        then
            echo "Applying stored patch to ImSwitch from: $PATCH_FILE"
            git apply "$PATCH_FILE" || { echo 'Failed to apply patch, aborting fetch'; exit 1; }
        fi
        echo 'Fetching the latest changes from ImSwitch repository'
        git fetch origin
        echo 'Checking for differences between local and remote branch'
        git diff HEAD origin/master > "$PATCH_FILE"
        if [[ -s "$PATCH_FILE" ]]
        then
            echo "New changes detected, patch saved at: $PATCH_FILE"
        else
            echo "No new changes detected in ImSwitch repository, patch not updated"
            rm -f "$PATCH_FILE"
        fi
        echo 'Merging fetched changes from origin/master'
        git merge origin/master
    fi
    if [[ "$UPDATE_INSTALL_GIT" == "true" || "$UPDATE_INSTALL_GIT" == "1" ]]
    then
        echo 'Pulling the ImSwitch repository and installing with UV'
        cd /tmp/ImSwitch
        git pull
        /bin/bash -c 'source /opt/conda/bin/activate imswitch && uv pip install --target /persistent_pip_packages /tmp/ImSwitch'
    fi
    source /opt/conda/bin/activate imswitch
    USB_DEVICE_PATH=${USB_DEVICE_PATH:-/dev/bus/usb}

    params=()
    if [[ $HEADLESS == "1" || $HEADLESS == "True" || $HEADLESS == "true" ]]
    then
        params+=" --headless"
    fi;
    if [[ $SSL == "0" || $SSL == "False" || $SSL == "false" ]]
    then
        params+=" --no-ssl"
    fi;
    params+=" --http-port ${HTTP_PORT:-8001}"
    params+=" --config-folder ${CONFIG_PATH:-None}"
    params+=" --config-file ${CONFIG_FILE:-None}"
    params+=" --ext-data-folder ${DATA_PATH:-None}"
    if [[ $SCAN_EXT_DATA_PATH == "1" || $SCAN_EXT_DATA_PATH == "True" || $SCAN_EXT_DATA_PATH == "true" ]]
    then
        params+=" --scan-ext-data-folder"
    fi;
    params+=" --ext-data-folder ${EXT_DATA_PATH:-None}"
    echo 'Starting Imswitch with the following parameters:'
    echo '/tmp/ImSwitch/main.py' "${params[@]}"
    python3 -m imswitch $params
else
    source /opt/conda/bin/activate imswitch
    echo 'Starting the container in terminal mode'
    exec bash
fi
