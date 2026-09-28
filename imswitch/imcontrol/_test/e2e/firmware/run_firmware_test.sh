#!/usr/bin/env bash
# Run the firmware server tests on the Pi, inside the imswitch container.
# One SSH connection, so one password prompt.
#
#   ./run_firmware_test.sh
#
# Read-only: this checks the firmware server and the CAN-id mapping, it never
# flashes anything.
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

# Ship the shared conftest.py alongside the test: pytest reads it from the same
# directory, so both land in one temporary folder. ustar carries no pax
# headers, so GNU tar on the Pi does not warn about macOS SCHILY.fflags.
tar --no-xattrs --format=ustar -czf - -C "$DIR/.." conftest.py -C "$DIR" test_firmware_server.py |
ssh "$PI" "cat > /tmp/firmware_tests.tgz \
    && docker cp /tmp/firmware_tests.tgz $CONTAINER:/tmp/ >/dev/null \
    && docker exec $CONTAINER sh -c 'rm -rf /tmp/firmware_tests \
        && mkdir -p /tmp/firmware_tests \
        && tar xzf /tmp/firmware_tests.tgz -C /tmp/firmware_tests' \
    && docker exec $ENVS $CONTAINER python3 -m pytest /tmp/firmware_tests \
        -v -s -ra --tb=line $COLOR -p no:arkitekt_next -p no:cacheprovider -o markers=hardware"
