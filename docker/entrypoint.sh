#!/usr/bin/env bash

if [[ ! ("$MODE" == "terminal") ]];
then
    echo 'Starting the container'
    echo 'Listing USB Bus'
    lsusb
    echo 'Listing external storage devices'
    ls /media
    echo 'Starting SSH Server'
    /usr/sbin/sshd -D
    echo 'Starting SFTP Server'
    /usr/sbin/vsftpd /etc/vsftpd.conf
    echo 'Listing Config Dir'
    ls /tmp/ImSwitchConfig/imcontrol_setups
    
    # Start D-Bus service (required for NetworkManager)
    echo 'Starting D-Bus service'
    service dbus start
    
    # start network manager
    echo 'Starting Network Manager'
    service network-manager start
    echo 'Checking Network Manager status'
    nmcli general status

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
        CONFIG_FILE="${CONFIG_FILE:-/tmp/ImSwitchConfig/imcontrol_setups/example_virtual_microscope.json}"
    else
        CONFIG_FILE=None
    fi
    
    PERSISTENT_PIP_DIR="${PERSISTENT_PIP_DIR:-/persistent_pip_packages}"
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
    if [[ $ssl == "0" || $ssl == "False" || $ssl == "false" ]]
    then
        params+=" --no-ssl"
    fi;
    params+=" --http-port ${HTTP_PORT:-8001}"
    params+=" --socket-port ${SOCKET_PORT:-8002}"
    params+=" --config-folder ${CONFIG_PATH:-None}"
    params+=" --config-file ${CONFIG_FILE:-None}"
    params+=" --ext-data-folder ${DATA_PATH:-None}"
    if [[ $scan-ext-drive-mount == "1" || $scan-ext-drive-mount == "True" || $scan-ext-drive-mount == "true" ]]
    then
        params+=" --scan-ext-drive-mount"
    fi;
    params+=" --ext-drive-mount ${EXT_DRIVE_MOUNT:-None}"
    echo 'Starting Imswitch with the following parameters:'
    echo "${params[@]}"
    python3 /tmp/ImSwitch/main.py $params
else
    source /opt/conda/bin/activate imswitch
    echo 'Starting the container in terminal mode'
    exec bash
fi
