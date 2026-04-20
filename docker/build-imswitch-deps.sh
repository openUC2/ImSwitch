#!/usr/bin/env -S bash -eux

# Note(ethanjli): this build script should only include things which don't change much/often, so
# that the container image layer corresponding to this build script won't need to be re-pulled so
# often; anything which changes often should instead go in the build-imswitch.sh script!

apt-get update
apt-get install -y \
  build-essential \
  mesa-utils \
  libhdf5-dev \
  usbutils \
  libglib2.0-0 \
  git
# TODO(ethanjli): find a way to not rely on git inside a container image

# Install system picamera2 which pulls in compatible libcamera dependencies
# Note: python3-picamera2 will automatically install the correct libcamera version
apt-get update && apt install -y --no-install-recommends \
         python3-picamera2 \
     && apt-get clean \
     && apt-get autoremove \
     && rm -rf /var/cache/apt/archives/* \
     && rm -rf /var/lib/apt/lists/*

# Activate UV environment and install dependencies
export PATH="/root/.local/bin:$PATH"

# Set up the project directory with build metadata so uv sync can pre-install all
# slow-changing dependencies without installing ImSwitch itself yet.
# This creates a separately-cached Docker image layer from the rapidly-changing source.
mkdir -p /opt/imswitch/imswitch
cat >/opt/imswitch/imswitch/__init__.py <<EOF
# temporary placeholder to be overwritten by build-imswitch.sh
__version__ = "0.0.0"
EOF
cp /mnt/ImSwitch/pyproject.toml /opt/imswitch/pyproject.toml
cp /mnt/ImSwitch/uv.lock /opt/imswitch/uv.lock
cd /opt/imswitch

# Install all deps from the lockfile without installing ImSwitch itself.
# Using --frozen ensures the lockfile is used as-is (no resolver conflicts).
uv sync --no-install-project --frozen --python 3.11

# Reinstall psygnal from source to work around binary-wheel ABI issues
uv pip install psygnal --no-binary :all:

# Clean up build-only tools

apt-get remove -y \
  build-essential \
  libhdf5-dev

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /root/.cache/uv
rm -rf /root/.cache/pip
rm -rf /tmp/*
