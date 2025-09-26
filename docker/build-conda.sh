#!/usr/bin/env -S bash -eux

# Install necessary dependencies and prepare the environment

apt-get update
apt-get install -y \
  wget

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
rm /tmp/miniforge.sh

# Create conda environment and install packages
/opt/conda/bin/conda create -y --name imswitch python=3.11
# Don't include Python 3.12 in the image, since we rely on Python 3.11:
/opt/conda/bin/conda remove python3.12

# Clean up build-only tools

apt-get remove -y \
  wget

# Clean up all the package managers at the end

apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
