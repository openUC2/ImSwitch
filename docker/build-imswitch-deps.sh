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
source /opt/imswitch/.venv/bin/activate

# Install psygnal without binaries
uv pip install psygnal --no-binary :all:

# Fix the version of OME-ZARR
uv pip install zarr==2.11.3

# Install deps listed in pyproject.toml, but don't install ImSwitch yet:
mkdir -p /tmp/ImSwitch/imswitch
cat >/tmp/ImSwitch/imswitch/__init__.py <<EOF
# temporary placeholder to be overwritten
__version__ = "0.0.0"
EOF
cp /mnt/ImSwitch/pyproject.toml /tmp/ImSwitch/pyproject.toml
cd /tmp/ImSwitch
uv pip install /tmp/ImSwitch

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
