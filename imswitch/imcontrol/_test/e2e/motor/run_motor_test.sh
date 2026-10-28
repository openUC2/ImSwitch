#!/usr/bin/env bash

# Run the motor tests on the remote Pi inside the ImSwitch container.
#
#   ./run_motor_test.sh                 # every motor test
#   ./run_motor_test.sh '/tmp/motor_tests/test_motor_axis_move.py'
#   ./run_motor_test.sh '/tmp/motor_tests/test_motor_axis_move.py::test_axis_moves_by_step[ESP32Stage-X]'
#
# The argument is a pytest target as seen inside the container, so it starts
# with REMOTE_TEST_DIR. Quote it: the [X] of a parametrised id is a glob.
#
# test_motor_motion_camera.py moves real hardware and is NOT gated: a plain run
# parks the stage at the transport position and moves every axis.

set -euo pipefail

PI="${PI_HOST:-pi@192.168.178.124}"
CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"
IMSWITCH_URL="${IMSWITCH_URL:-http://localhost:8001}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REMOTE_TEST_DIR="${REMOTE_TEST_DIR:-/tmp/motor_tests}"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

. "$LOCAL_DIR/../colors.sh"

ENVS="-e IMSWITCH_URL=$IMSWITCH_URL"

# ENVS must stay a SINGLE LINE: it is interpolated into the ssh command below.
# Forward every test knob that is set; an unset one keeps the test's default.
for knob in $(compgen -v | grep -E '^(MOTION_CAMERA_|TRANSPORT_|LEDMATRIX_)'); do
    [ -z "${!knob:-}" ] || ENVS="$ENVS -e $knob=${!knob}"
done

# Optional pytest targets, defaulting to the whole folder. Each is single
# quoted for the remote shell, so a [Y] parametrised id is not globbed.
TARGETS=""
for target in "${@:-$REMOTE_TEST_DIR}"; do
    TARGETS="$TARGETS '$target'"
done

# ustar carries no pax headers, so GNU tar on the Pi does not warn about the
# SCHILY.fflags macOS bsdtar would write.
#
# Upload and run share one ssh master connection, so one password prompt. The
# run needs its own call: the upload's stdin is the tar stream.
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="/tmp/motor-ssh-$$" -o ControlPersist=60)
trap 'ssh -o ControlPath="/tmp/motor-ssh-$$" -O exit "$PI" 2>/dev/null || true' EXIT

tar --no-xattrs --format=ustar -czf - \
    -C "$LOCAL_DIR/.." conftest.py \
    -C "$LOCAL_DIR" . |
ssh "${SSH_OPTS[@]}" "$PI" "
    set -e

    cat > /tmp/motor_tests.tgz

    docker cp \
        /tmp/motor_tests.tgz \
        '$CONTAINER:/tmp/motor_tests.tgz' \
        >/dev/null

    docker exec '$CONTAINER' sh -c \
        'rm -rf \"$REMOTE_TEST_DIR\" &&
         mkdir -p \"$REMOTE_TEST_DIR\" &&
         tar xzf /tmp/motor_tests.tgz -C \"$REMOTE_TEST_DIR\"'
"

# -t and -it give pytest a terminal, so Ctrl+C reaches it inside the container
# instead of only killing the local ssh client while the stage keeps moving.
# Only on a terminal: docker exec -it refuses to start without one.
SSH_TTY_FLAG="" DOCKER_TTY_FLAG=""
if [ -t 0 ]; then SSH_TTY_FLAG="-t" DOCKER_TTY_FLAG="-it"; fi

ssh $SSH_TTY_FLAG "${SSH_OPTS[@]}" "$PI" "
    docker exec \
        $DOCKER_TTY_FLAG \
        $ENVS \
        '$CONTAINER' \
        '$PYTHON_BIN' -m pytest \
        $TARGETS \
        -s \
        -v \
        -ra \
        --tb=line \
        $COLOR \
        -m hardware \
        -p no:arkitekt_next \
        -p no:cacheprovider \
        -o 'markers=hardware: tests requiring real microscope hardware'
"
