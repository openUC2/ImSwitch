#!/usr/bin/env bash
# Sync the boards to the firmware server's version: sync_firmware.py on the Pi,
# inside the imswitch container. One SSH connection, so one password prompt.
#
#   ./run_firmware_sync.sh          # only print what would be flashed
#   ./run_firmware_sync.sh --yes    # flash: the master first, then the CAN nodes
#
# FLASHES FIRMWARE with --yes. Arguments go to sync_firmware.py, and its exit
# code is this script's (0 in sync, 1 a board is not, 2 could not run).
#
# Override with PI_HOST / IMSWITCH_CONTAINER / IMSWITCH_URL.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

PI="${PI_HOST:-pi@192.168.178.124}"
CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"
URL="${IMSWITCH_URL:-http://localhost:8001}"

# Unpacked straight into the container. ustar carries no pax headers, so GNU
# tar on the Pi does not warn about macOS SCHILY.fflags. python3 -u, because
# docker exec gives no tty and the progress lines should arrive as they happen.
tar --no-xattrs --format=ustar -czf - -C "$DIR" sync_firmware.py |
ssh "$PI" "docker exec -i $CONTAINER sh -c 'rm -rf /tmp/firmware_sync \
        && mkdir -p /tmp/firmware_sync \
        && tar xzf - -C /tmp/firmware_sync' \
    && docker exec -e IMSWITCH_URL=$URL $CONTAINER \
        python3 -u /tmp/firmware_sync/sync_firmware.py $*"
