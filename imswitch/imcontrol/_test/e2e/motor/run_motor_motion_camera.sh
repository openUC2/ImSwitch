#!/usr/bin/env bash

# Run the camera-witnessed motion test on the remote Pi.
#
#   ./run_motor_motion_camera.sh
#   MOTION_CAMERA_DISTANCE_UM=500 ./run_motor_motion_camera.sh
#
# MOVES REAL HARDWARE: first to the transport position, then every axis out and
# back.
#
# Everything goes through run_motor_test.sh, so its overrides and the
# MOTION_CAMERA_* knobs work here too.

set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_TEST_DIR="${REMOTE_TEST_DIR:-/tmp/motor_tests}"

REMOTE_TEST_DIR="$REMOTE_TEST_DIR" \
    exec "$LOCAL_DIR/run_motor_test.sh" "$REMOTE_TEST_DIR/test_motor_motion_camera.py"
