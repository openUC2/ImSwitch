#!/usr/bin/env -S bash -eux

# Install necessary dependencies and prepare the environment

apt-get update
apt-get install -y \
  wget \
  gnupg

# Add Raspberry Pi repository for proper picamera2 dependencies
echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list \
  && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E

apt-get update && apt -y upgrade

# Install Miniforge based on architecture
case "$TARGETPLATFORM" in
"linux/arm64")
  cpu=aarch64
  ;;
"linux/amd64")
  cpu=x86_64
  ;;
*)
  echo "Unknown target platform $TARGETPLATFORM!"
  exit 1
  ;;
esac
wget --quiet \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$cpu.sh" \
  -O /tmp/miniforge.sh
/bin/bash /tmp/miniforge.sh -b -p /opt/conda

# Create conda environment and install packages
/opt/conda/bin/conda create -y --name imswitch python=3.11

# Clean up build-only tools

apt-get remove -y \
  wget

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
