#!/usr/bin/env bash
# ImSwitch Docker Container Interactive Shell
#
# This script provides an interactive shell with the ImSwitch conda environment activated.
# Usage: docker run --entrypoint=venv-shell.sh <image-name>

log() { echo "[$(date +'%F %T')] $*"; }

# Activate the ImSwitch conda environment
source /opt/conda/bin/activate imswitch

log 'Starting the container in terminal mode'
log 'ImSwitch environment activated'

# Drop into interactive bash
exec bash
