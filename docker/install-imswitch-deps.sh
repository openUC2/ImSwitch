#!/usr/bin/env -S bash -eux

# Note(ethanjli): this build script should only include things which don't change much/often, so
# that the container image layer corresponding to this build script won't need to be re-pulled so
# often; anything which changes often should instead go in the build-imswitch.sh script!

apt-get update
apt-get install -y \
  git \
  build-essential \
  mesa-utils \
  libhdf5-dev \
  usbutils \
  libglib2.0-0

# Only install dependencies; this creates a separately-cached Docker image layer from the rapidly-changing source.

uv sync --locked --no-install-project

# Reinstall psygnal from source to work around binary-wheel ABI issues
uv pip install psygnal --no-binary :all:

# Clean up build-only tools

apt-get remove -y \
  build-essential \
  git \
  libhdf5-dev
apt -y autoremove
