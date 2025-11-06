#!/usr/bin/env bash

#set -euo pipefail

log() { echo "[$(date +'%F %T')] $*"; }

# ---- NetworkManager / D-Bus mode selection ----
# WIFI_MODE=host      -> expect host NM; require /run/dbus and /etc/machine-id bind-mounts
# WIFI_MODE=container -> run dbus-daemon + NetworkManager inside the container
WIFI_MODE="${WIFI_MODE:-host}"

# Provide safe default values for variables
MODE="${MODE:-server}"
PIP_PACKAGES="${PIP_PACKAGES:-}"
PERSISTENT_PIP_DIR="${PERSISTENT_PIP_DIR:-/persistent_pip_packages}"
UPDATE_GIT="${UPDATE_GIT:-false}"
UPDATE_CONFIG="${UPDATE_CONFIG:-false}"
CONFIG_PATH="${CONFIG_PATH:-}"
UPDATE_INSTALL_GIT="${UPDATE_INSTALL_GIT:-false}"
SSL=${SSL:-false}
SCAN_EXT_DRIVE_MOUNT=${SCAN_EXT_DRIVE_MOUNT:-false}

start_container_nm() {
  log "WIFI_MODE=container → starting dbus-daemon and NetworkManager in container"
  mkdir -p /run/dbus
  # Ensure machine-id exists (for dbus/NM)
  [[ -s /etc/machine-id ]] || dbus-uuidgen --ensure=/etc/machine-id

  dbus-daemon --system --fork || true
  # Start NM (no systemd)
  /usr/sbin/NetworkManager --no-daemon >/var/log/NetworkManager.log 2>&1 &
  # Wait until NM is ready (max ~5s)
  for i in {1..10}; do
    if nmcli general status >/dev/null 2>&1; then
      log "NetworkManager is up (container)"
      return 0
    fi
    sleep 0.5
  done
  log "WARN: NetworkManager did not report ready; continuing"
}

setup_host_nm() {
  log "WIFI_MODE=host → do not start NM in container"
  if [[ ! -S /run/dbus/system_bus_socket ]]; then
    log "WARN: /run/dbus/system_bus_socket not mounted; host nmcli calls will fail"
  fi
  if [[ ! -s /etc/machine-id ]]; then
    log "WARN: /etc/machine-id not mounted; host nmcli may reject D-Bus"
  fi
}


if [[ "$WIFI_MODE" == "container" ]]; then
  start_container_nm
else
  setup_host_nm
fi



if [[ "${MODE}" != "terminal" ]];
then
    echo 'Starting the container'
    echo 'Listing USB Bus'
    lsusb
    echo 'Listing external storage devices'
    ls /media

    # D-Bus/NetworkManager handling is done above via WIFI_MODE
    echo 'Networking initialized (WIFI_MODE handled)'

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
    params+=" --socket-port ${SOCKET_PORT:-8002}"
    params+=" --config-folder ${CONFIG_PATH:-None}"
    params+=" --config-file ${CONFIG_FILE:-None}"
    params+=" --ext-data-folder ${DATA_PATH:-None}"
    if [[ $SCAN_EXT_DRIVE_MOUNT == "1" || $SCAN_EXT_DRIVE_MOUNT == "True" || $SCAN_EXT_DRIVE_MOUNT == "true" ]]
    then
        params+=" --scan-ext-drive-mount"
    fi;
    params+=" --ext-drive-mount ${EXT_DRIVE_MOUNT:-None}"
    echo 'Starting Imswitch with the following parameters:'
    echo '/tmp/ImSwitch/main.py' "${params[@]}"
    python3 -m imswitch $params
else
    source /opt/conda/bin/activate imswitch
    echo 'Starting the container in terminal mode'
    # Networking is handled via WIFI_MODE at script start
    exec bash
fi
