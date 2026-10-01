#!/usr/bin/env bash
#
# hil-run.sh -- test one ImSwitch image on the real rig, then put the rig back.
#
#   hil-setup.py swap-in  ->  ship the suite  ->  firmware sync  ->  pytest
#                                       ... always hil-setup.py restore
#
# Runs ON the Pi, not from your machine: it talks to the local Docker daemon.
# From your machine use run_all.sh, which tests whatever image already runs.
#
# hil-setup.py does the rig side: the checks, the swap, the detector wait and
# the restore. This script keeps the rest: the arguments, the lock, and the
# suite itself.
#
# Usage:
#   ci/hil-run.sh --image sha-7d3adda --yes
#   ci/hil-run.sh --image ghcr.io/openuc2/imswitch:sha-7d3adda --yes \
#                 --tests "board firmware" --out reports
#
# MOVES THE STAGE AND SWITCHES LIGHT ON: --yes is mandatory, so no scheduler
# and no stray call can actuate the rig by accident.
#
# Exit codes (CI depends on them):
#   0  the suite passed
#   1  a test failed
#   2  the run could not be performed (bad arguments, no disk, no ImSwitch,
#      no board, a detector that never became ready, another run in progress,
#      the swap or the restore failed, the run was interrupted). A failed
#      restore wins over the test result: the rig needs a human either way.
set -uo pipefail

CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"

# ImSwitch as seen from inside the container, where the tests run: :8001
# directly, without the caddy prefix hil-setup.py goes through.
CONTAINER_URL="${HIL_CONTAINER_URL:-http://localhost:8001}"

# One rig, one run. A second run would fight the first over the container.
LOCK_FILE="${HIL_LOCK_FILE:-/tmp/hil-run.lock}"

DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE_DIR="$(cd "$DIR/.." && pwd)"
SETUP="$DIR/hil-setup.py"

IMAGE=""
TESTS=""
OUT_DIR="$DIR/reports"
CONFIRMED=0
RESTORE_ARGS=""

EXIT_OK=0
EXIT_FAILED=1
EXIT_UNAVAILABLE=2

log()  { printf '[hil-run] %s\n' "$*"; }
warn() { printf '[hil-run] %s\n' "$*" >&2; }
die()  { warn "$*"; exit "$EXIT_UNAVAILABLE"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --image)      IMAGE="${2:-}"; shift 2 ;;
        --tests)      TESTS="${2:-}"; shift 2 ;;
        --out)        OUT_DIR="${2:-}"; shift 2 ;;
        --yes)        CONFIRMED=1; shift ;;
        --keep-image) RESTORE_ARGS="--keep-image"; shift ;;
        -h|--help)    sed -n '2,29p' "$0"; exit "$EXIT_OK" ;;
        *)            die "unknown argument: $1" ;;
    esac
done

[ -n "$IMAGE" ] || die "usage: $0 --image <tag or ref> --yes [--tests \"motor camera\"]"
[ "$CONFIRMED" = "1" ] || die "refusing to actuate the rig without --yes"

command -v python3 >/dev/null || die "python3 not found -- run this on the Pi"

exec 9>"$LOCK_FILE" || die "cannot open lock file $LOCK_FILE"
flock -n 9 || die "another hil-run holds $LOCK_FILE -- one run per rig"


# ---------------------------------------------------------------------------
# Swap in, and back out on every exit path. The restore is installed before
# swap-in starts, so an interrupted run still gives the microscope back on the
# image it came with. restore knows what swap-in changed; if swap-in changed
# nothing, restore does nothing.
# ---------------------------------------------------------------------------

trap 'python3 "$SETUP" restore $RESTORE_ARGS || exit "$EXIT_UNAVAILABLE"' EXIT
trap 'exit "$EXIT_UNAVAILABLE"' INT TERM

python3 "$SETUP" swap-in --image "$IMAGE" || exit "$EXIT_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Ship the suite and run it. The image carries no e2e folder, and a swapped
# container starts with an empty /tmp, so this happens after the swap.
# ---------------------------------------------------------------------------

log "shipping the suite into the container"
tar --format=ustar -czf - -C "$SUITE_DIR" --exclude=__pycache__ --exclude=ci . |
    docker exec -i "$CONTAINER" sh -c \
        'rm -rf /tmp/e2e && mkdir -p /tmp/e2e && tar xzf - -C /tmp/e2e' ||
    die "could not ship the suite"


# ---------------------------------------------------------------------------
# Firmware sync: every board to the firmware server's version before the
# tests, as run_all.sh does (firmware/sync_firmware.py, the master first).
# FIRMWARE_UPDATE=off skips it. A failed sync only warns: the firmware tests
# report the boards themselves, and an image without ImSwitch's update
# endpoints cannot sync at all.
# ---------------------------------------------------------------------------

if [ "${FIRMWARE_UPDATE:-on}" = off ]; then
    log "firmware sync skipped (FIRMWARE_UPDATE=off)"
else
    log "syncing the firmware"
    docker exec -e "IMSWITCH_URL=$CONTAINER_URL" "$CONTAINER" \
        python3 -u /tmp/e2e/firmware/sync_firmware.py --yes 2>&1 |
        sed 's/^/[hil-run]   /' ||
        warn "firmware sync failed, running the tests anyway"
fi

TARGETS=""
for folder in $TESTS; do TARGETS="$TARGETS /tmp/e2e/$folder"; done
[ -z "$TARGETS" ] && TARGETS="/tmp/e2e"

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/junit-${IMAGE##*:}-$(date +%Y-%m-%d_%H-%M-%S).xml"

log "running pytest on:${TARGETS}"

# No --tb=line here: CI reads the report, and a human reading a failed nightly
# wants the assertion, not one line of it.
docker exec -e "IMSWITCH_URL=$CONTAINER_URL" "$CONTAINER" \
    python3 -m pytest $TARGETS \
        -v -ra --tb=short \
        --junitxml=/tmp/e2e-report.xml \
        -p no:arkitekt_next -p no:cacheprovider -o markers=hardware
STATUS=$?

docker cp "$CONTAINER:/tmp/e2e-report.xml" "$REPORT" >/dev/null 2>&1 &&
    log "report $REPORT" ||
    warn "no report written -- pytest died before it could write one"

if [ "$STATUS" -eq 0 ]; then
    log "PASS  $IMAGE"
    exit "$EXIT_OK"
fi

warn "FAIL  $IMAGE (pytest exit $STATUS)"
exit "$EXIT_FAILED"
