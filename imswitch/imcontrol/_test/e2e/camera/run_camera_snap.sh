#!/usr/bin/env bash
# Run the camera snap test on the Pi, inside the imswitch container.
#
#   ./run_camera_snap.sh              # pytest run
#   ./run_camera_snap.sh --curl       # quick check, saves the PNG locally
#
# Two URLs, because the two modes run in two places: IMSWITCH_URL is ImSwitch
# as pytest sees it from inside the container, IMSWITCH_EXTERNAL_URL as --curl
# sees it from this machine (through caddy, derived from PI_HOST).
#
# Override with PI_HOST / IMSWITCH_CONTAINER / IMSWITCH_URL /
# IMSWITCH_EXTERNAL_URL / IMSWITCH_DETECTOR.
set -euo pipefail

# conftest.py and colors.sh live one level up, in the suite root.
DIR="$(cd "$(dirname "$0")" && pwd)"

. "$DIR/../colors.sh"

PI="${PI_HOST:-pi@192.168.178.124}"

CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"

ENVS="-e IMSWITCH_URL=${IMSWITCH_URL:-http://localhost:8001}"

# Forwarded only when set, so the test keeps its own default of snapping from
# the first detector the setup reports.
[ -n "${IMSWITCH_DETECTOR:-}" ] && ENVS="$ENVS -e IMSWITCH_DETECTOR=$IMSWITCH_DETECTOR"

# ${PI#*@} drops the ssh user, leaving the host the rig is on.
EXTERNAL="${IMSWITCH_EXTERNAL_URL:-http://${PI#*@}:8000/imswitch}"

# --curl needs an explicit detectorName in the URL, so it cannot fall back to
# "first reported detector" and needs a concrete default of its own.
CURL_DETECTOR="${IMSWITCH_DETECTOR:-RPiCam}"

if [ "${1:-}" = "--curl" ]; then
    curl -sS -o /tmp/snap.png -w 'HTTP %{http_code}  %{content_type}  %{size_download} bytes\n' \
        "$EXTERNAL/api/RecordingController/snapNumpyToFastAPI?detectorName=$CURL_DETECTOR&resizeFactor=0.1"
    echo "saved to /tmp/snap.png"
    exit 0
fi

# Ship the shared conftest.py alongside the test: pytest reads it from the same
# directory, so both land in one temporary folder. ustar carries no pax
# headers, so GNU tar on the Pi does not warn about macOS SCHILY.fflags.
tar --no-xattrs --format=ustar -czf - -C "$DIR/.." conftest.py -C "$DIR" test_camera_capture.py start_live_view.py |
ssh "$PI" "cat > /tmp/camera_tests.tgz \
    && docker cp /tmp/camera_tests.tgz $CONTAINER:/tmp/ >/dev/null \
    && docker exec $CONTAINER sh -c 'rm -rf /tmp/camera_tests \
        && mkdir -p /tmp/camera_tests \
        && tar xzf /tmp/camera_tests.tgz -C /tmp/camera_tests' \
    && (docker exec $ENVS $CONTAINER python3 /tmp/camera_tests/start_live_view.py ||
        echo 'run_camera_snap: live view not started, the tests may fail' >&2) \
    && docker exec $ENVS $CONTAINER python3 -m pytest /tmp/camera_tests \
        -v --tb=line $COLOR -p no:arkitekt_next -o markers=hardware"
