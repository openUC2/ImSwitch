#!/usr/bin/env bash

# Run the laser tests on the remote Pi inside the ImSwitch container.
#
# Ships the whole folder, so both files run: test_laser_switching.py drives the
# API and reads state back, test_laser_photon.py measures with the camera
# whether light arrived. Each makes one test case per reported laser/LED.
#
#   ./run_laser_test.sh
#   ./run_laser_test.sh --measure    # print brightness, drop the threshold
#
# --measure is for recalibrating after moving a light or the optics. A light
# with no signal at all still fails: that is a result, not a calibration
# question.
#
# Override with PI_HOST / IMSWITCH_CONTAINER / IMSWITCH_URL / PYTHON_BIN /
# REMOTE_TEST_DIR, plus any test knob (see README.md):
#
#   PI_HOST=pi@192.168.1.20 ./run_laser_test.sh

set -euo pipefail

PI="${PI_HOST:-pi@192.168.178.124}"

CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"

# ImSwitch URL from the perspective of the container.
IMSWITCH_URL="${IMSWITCH_URL:-http://localhost:8001}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

REMOTE_TEST_DIR="${REMOTE_TEST_DIR:-/tmp/laser_tests}"

# The whole directory is shipped, so adding a test file here is enough.
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

. "$LOCAL_DIR/../colors.sh"

# Overwrite the knob before forwarding rather than appending a second -e, so
# nothing depends on how docker resolves the same name given twice.
EXTRA=""
if [ "${1:-}" = "--measure" ]; then
    PHOTON_MIN_DELTA=0
    EXTRA="-s"
fi

ENVS="-e IMSWITCH_URL=$IMSWITCH_URL"

# Forward every test knob that is set; an unset one keeps the test's default.
for knob in $(compgen -v | grep -E '^(IMSWITCH_DETECTOR|UC2_LASER_VALUE|PHOTON_|AUTO_EXPOSURE_)'); do
    [ -z "${!knob:-}" ] || ENVS="$ENVS -e $knob=${!knob}"
done

# Ship the folder plus the shared conftest.py from one level up into a clean
# temporary directory in the container, which keeps the runner independent of
# the local repository path. ustar carries no pax headers, so GNU tar on the Pi
# does not warn about macOS SCHILY.fflags.
tar --no-xattrs --format=ustar -czf - -C "$LOCAL_DIR/.." conftest.py -C "$LOCAL_DIR" . |
ssh "$PI" "
    set -e

    cat > /tmp/laser_tests.tgz

    docker cp \
        /tmp/laser_tests.tgz \
        '$CONTAINER:/tmp/laser_tests.tgz' \
        >/dev/null

    docker exec '$CONTAINER' sh -c \
        'rm -rf \"$REMOTE_TEST_DIR\" &&
         mkdir -p \"$REMOTE_TEST_DIR\" &&
         tar xzf /tmp/laser_tests.tgz -C \"$REMOTE_TEST_DIR\"'

    docker exec \
        $ENVS \
        '$CONTAINER' \
        '$PYTHON_BIN' -m pytest \
        '$REMOTE_TEST_DIR' \
        -v \
        -ra \
        --tb=line \
        $COLOR \
        $EXTRA \
        -m hardware \
        -p no:arkitekt_next \
        -p no:cacheprovider \
        -o 'markers=hardware: tests requiring real microscope hardware'
"
