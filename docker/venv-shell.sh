#!/usr/bin/env bash
# ImSwitch Docker Container Interactive Shell
#
# This script provides an interactive shell with the ImSwitch UV virtual environment activated.
# Usage: docker run --entrypoint=venv-shell.sh <image-name>

log() { echo "[$(date +'%F %T')] $*"; }

# Activate the ImSwitch UV virtual environment
export PATH="/root/.local/bin:$PATH"
source /opt/imswitch/.venv/bin/activate

log 'Starting the container in terminal mode'
log 'ImSwitch UV environment activated'

# Drop into interactive bash
exec bash
