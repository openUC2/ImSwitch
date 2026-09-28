#!/usr/bin/env bash
# Run the LED matrix tests on the Pi, inside the imswitch container.
#
#   ./run_ledmatrix_test.sh              # pytest run
#   ./run_ledmatrix_test.sh --measure    # print dark/bright/change, no assert
#
# Override with PI_HOST / IMSWITCH_CONTAINER / IMSWITCH_URL, plus any test
# knob (see README.md).
set -euo pipefail

# conftest.py and colors.sh live one level up, in the suite root.
DIR="$(cd "$(dirname "$0")" && pwd)"

. "$DIR/../colors.sh"

PI="${PI_HOST:-pi@192.168.178.124}"
CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"

# Overwrite the knob before forwarding rather than appending a second -e, so
# nothing depends on how docker resolves the same name given twice.
if [ "${1:-}" = "--measure" ]; then
    PHOTON_MIN_DELTA=0
    EXTRA="-v -s"
else
    EXTRA="-v"
fi

ENVS="-e IMSWITCH_URL=${IMSWITCH_URL:-http://localhost:8001}"

# Forward every test knob that is set; an unset one keeps the test's default.
for knob in $(compgen -v | grep -E '^(IMSWITCH_DETECTOR|LEDMATRIX_|PHOTON_|AUTO_EXPOSURE_)'); do
    [ -z "${!knob:-}" ] || ENVS="$ENVS -e $knob=${!knob}"
done

# Ship the shared conftest.py alongside the tests: pytest reads it from the
# same directory, so both land in one temporary folder. ustar carries no pax
# headers, so GNU tar on the Pi does not warn about macOS SCHILY.fflags.
tar --no-xattrs --format=ustar -czf - -C "$DIR/.." conftest.py -C "$DIR" test_ledmatrix_photon.py test_ledmatrix_smoke.py |
ssh "$PI" "cat > /tmp/ledmatrix_tests.tgz \
    && docker cp /tmp/ledmatrix_tests.tgz $CONTAINER:/tmp/ >/dev/null \
    && docker exec $CONTAINER sh -c 'rm -rf /tmp/ledmatrix_tests \
        && mkdir -p /tmp/ledmatrix_tests \
        && tar xzf /tmp/ledmatrix_tests.tgz -C /tmp/ledmatrix_tests' \
    && docker exec $ENVS $CONTAINER python3 -m pytest \
        /tmp/ledmatrix_tests $EXTRA -ra --tb=line $COLOR \
        -p no:arkitekt_next -p no:cacheprovider -o markers=hardware"
