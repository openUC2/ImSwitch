#!/usr/bin/env bash
# Sync the boards' firmware, then run the firmware tests, on the Pi inside the
# imswitch container. One SSH connection, so one password prompt.
#
#   ./run_firmware_test.sh
#
# FLASHES FIRMWARE: ci/sync_firmware.py --yes first brings every board that
# answers to the firmware server's version, the master before the CAN nodes
# (FIRMWARE_UPDATE=off skips it). A failed sync is reported and the tests run
# anyway; they only read.
#
# Override with PI_HOST / IMSWITCH_CONTAINER / IMSWITCH_URL, plus any test
# knob (see README.md).
set -euo pipefail

# conftest.py and colors.sh live one level up, in the suite root.
DIR="$(cd "$(dirname "$0")" && pwd)"

. "$DIR/../colors.sh"

PI="${PI_HOST:-pi@192.168.178.124}"
CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"

ENVS="-e IMSWITCH_URL=${IMSWITCH_URL:-http://localhost:8001}"

# Forward every test knob that is set; an unset one keeps the test's default.
for knob in $(compgen -v | grep -E '^(FIRMWARE_)'); do
    [ -z "${!knob:-}" ] || ENVS="$ENVS -e $knob=${!knob}"
done

# python3 -u: docker exec gives no tty, and the sync's progress lines should
# arrive as they happen.
SYNC="docker exec $ENVS $CONTAINER python3 -u /tmp/firmware_tests/sync_firmware.py --yes \
        || echo 'run_firmware_test: firmware sync failed, running the tests anyway' >&2"
if [ "${FIRMWARE_UPDATE:-on}" = off ]; then
    SYNC="echo 'run_firmware_test: firmware sync skipped (FIRMWARE_UPDATE=off)'"
fi

# Ship the shared conftest.py alongside the test: pytest reads it from the same
# directory, so everything lands in one temporary folder, the sync from ci/
# included. ustar carries no pax headers, so GNU tar on the Pi does not warn
# about macOS SCHILY.fflags.
tar --no-xattrs --format=ustar -czf - -C "$DIR/.." conftest.py \
    -C "$DIR" test_firmware_server.py -C "$DIR/../ci" sync_firmware.py |
ssh "$PI" "cat > /tmp/firmware_tests.tgz \
    && docker cp /tmp/firmware_tests.tgz $CONTAINER:/tmp/ >/dev/null \
    && docker exec $CONTAINER sh -c 'rm -rf /tmp/firmware_tests \
        && mkdir -p /tmp/firmware_tests \
        && tar xzf /tmp/firmware_tests.tgz -C /tmp/firmware_tests' \
    && { $SYNC; } \
    && docker exec $ENVS $CONTAINER python3 -m pytest /tmp/firmware_tests/test_firmware_server.py \
        -v -s -ra --tb=line $COLOR -p no:arkitekt_next -p no:cacheprovider -o markers=hardware"
