#!/usr/bin/env bash
# Run the whole e2e suite on the Pi, inside the imswitch container.
#
# FLASHES FIRMWARE: every board that answers is synced to the firmware server's
# version first, the master before the CAN nodes (FIRMWARE_UPDATE=off skips it).
# MOVES THE STAGE: parks it at the transport position once before the tests,
# so every run starts from the same place. Runs for a folder filter too.
#
#   ./run_all.sh                 # everything
#   ./run_all.sh camera          # only camera/
#   ./run_all.sh lightsource ledmatrix # several folders
#
# Override with PI_HOST / IMSWITCH_CONTAINER, plus any test knob (see README.md).
set -euo pipefail
PI="${PI_HOST:-pi@192.168.178.53}"
CONTAINER="${IMSWITCH_CONTAINER:-imswitch-server-1}"
DIR="$(cd "$(dirname "$0")" && pwd)"

. "$DIR/colors.sh"

# Inside the container ImSwitch runs on :8001 without the caddy prefix.
ENVS="-e IMSWITCH_URL=http://localhost:8001"

# Forward every test knob that is set; an unset one keeps the test's default.
for knob in $(compgen -v | grep -E '^(IMSWITCH_DETECTOR|UC2_LASER_VALUE|PHOTON_|AUTO_EXPOSURE_|LEDMATRIX_|MOTION_CAMERA_|TRANSPORT_|OBJECTIVE_|FIRMWARE_)'); do
    [ -z "${!knob:-}" ] || ENVS="$ENVS -e $knob=${!knob}"
done

# Optional folder filter: ./run_all.sh camera lightsource
TARGETS=""
for arg in "$@"; do TARGETS="$TARGETS /tmp/e2e/$arg"; done
[ -z "$TARGETS" ] && TARGETS="/tmp/e2e"

# ustar carries no pax headers, so GNU tar on the Pi does not warn about the
# SCHILY.fflags macOS bsdtar would write.
#
# Upload and run share one ssh master connection, so one password prompt. The
# run needs its own call: the upload's stdin is the tar stream.
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="/tmp/e2e-ssh-$$" -o ControlPersist=60)
trap 'ssh -o ControlPath="/tmp/e2e-ssh-$$" -O exit "$PI" 2>/dev/null || true' EXIT

tar --no-xattrs --format=ustar -czf - -C "$DIR" . | ssh "${SSH_OPTS[@]}" "$PI" "
    cat > /tmp/e2e.tgz &&
    docker cp /tmp/e2e.tgz $CONTAINER:/tmp/e2e.tgz >/dev/null &&
    docker exec $CONTAINER sh -c 'rm -rf /tmp/e2e && mkdir -p /tmp/e2e && tar xzf /tmp/e2e.tgz -C /tmp/e2e'
"

# -t and -it give pytest a terminal, so Ctrl+C reaches it inside the container
# instead of only killing the local ssh client. Only on a terminal: docker
# exec -it refuses to start without one.
SSH_TTY_FLAG="" DOCKER_TTY_FLAG=""
if [ -t 0 ]; then SSH_TTY_FLAG="-t" DOCKER_TTY_FLAG="-it"; fi

SYNC="docker exec $ENVS $CONTAINER python3 -u /tmp/e2e/ci/sync_firmware.py --yes ||
        echo 'run_all: firmware sync failed, running the tests anyway' >&2"
if [ "${FIRMWARE_UPDATE:-on}" = off ]; then
    SYNC="echo 'run_all: firmware sync skipped (FIRMWARE_UPDATE=off)'"
fi

# Sync firmware, park the stage and start the camera before pytest. None is
# allowed to cost us the run: a setup without a positioner cannot park, and a camera that
# refuses to stream is something the tests themselves report better.
#
# The live view is what makes the camera deliver frames at all -- the snap
# endpoint only reads a buffer that an acquisition loop fills.
ssh $SSH_TTY_FLAG "${SSH_OPTS[@]}" "$PI" "
    $SYNC

    docker exec $ENVS $CONTAINER python3 /tmp/e2e/motor/move_to_transport.py ||
        echo 'run_all: transport move failed, running the tests anyway' >&2

    docker exec $ENVS $CONTAINER python3 /tmp/e2e/camera/start_live_view.py ||
        echo 'run_all: live view not started, the camera tests may fail' >&2

    docker exec $DOCKER_TTY_FLAG $ENVS $CONTAINER python3 -m pytest $TARGETS \
        -v -ra --tb=line $COLOR -p no:arkitekt_next -p no:cacheprovider -o markers=hardware
"
