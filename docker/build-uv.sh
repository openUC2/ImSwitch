#!/usr/bin/env -S bash -eux

# Install necessary dependencies and prepare the environment

apt-get update
apt-get install -y \
  wget \
  gnupg \
  python3.11 \
  python3.11-venv \
  python3-pip

# Add Raspberry Pi repository for proper picamera2 dependencies
echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list \
  && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E

apt-get update && apt -y upgrade

# Install UV package manager (fast Rust-based pip replacement)
wget --quiet -O /tmp/uv-installer.sh https://astral.sh/uv/install.sh
bash /tmp/uv-installer.sh

# Source UV environment to add to PATH (installs to /root/.local/bin)
source /root/.local/bin/env

# Create UV virtual environment for ImSwitch
mkdir -p /opt/imswitch
cd /opt/imswitch
uv venv --python 3.11 .venv

# Clean up build-only tools

apt-get remove -y \
  wget

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /root/.cache/uv
rm -rf /root/.cache/pip
rm -rf /tmp/*
