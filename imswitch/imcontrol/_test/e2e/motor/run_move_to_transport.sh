#!/usr/bin/env bash

# Drive the stage on the remote Pi to its stored transport position.
# One SSH connection, so one password prompt.
#
#   ./run_move_to_transport.sh
#
# MOVES REAL HARDWARE. See move_to_transport.py for where the target comes from
# and why the Z hard limits are off during the move.
#
# Override with PI_HOST / IMSWITCH_CONTAINER / IMSWITCH_URL /
# TRANSPORT_TIMEOUT / TRANSPORT_SPEED.

set -euo pipefail

PI="${PI_HOST:-pi@192.168.178.124}"
CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"
IMSWITCH_URL="${IMSWITCH_URL:-http://localhost:8001}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

ENVS="-e IMSWITCH_URL=$IMSWITCH_URL"

# Forwarded only when set, so an unset one keeps the script's own default.
[ -n "${TRANSPORT_TIMEOUT:-}" ] && ENVS="$ENVS -e TRANSPORT_TIMEOUT=$TRANSPORT_TIMEOUT"
[ -n "${TRANSPORT_SPEED:-}" ] && ENVS="$ENVS -e TRANSPORT_SPEED=$TRANSPORT_SPEED"

# A single file needs no tar/docker cp round trip: it is piped through ssh into
# python's stdin, and docker exec -i keeps that stdin open.
ssh "$PI" "docker exec -i $ENVS '$CONTAINER' '$PYTHON_BIN' -" \
    < "$LOCAL_DIR/move_to_transport.py"
