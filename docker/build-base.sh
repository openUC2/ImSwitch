#!/usr/bin/env -S bash -eux

# Install necessary dependencies and prepare the environment

apt-get update \
  -o Acquire::AllowInsecureRepositories=true \
  -o Acquire::AllowDowngradeToInsecureRepositories=true \
  -o Acquire::AllowUnsignedRepositories=true
apt-get install -y --allow-unauthenticated \
  wget \
  unzip \
  python3 \
  python3-pip \
  build-essential \
  git \
  mesa-utils \
  libhdf5-dev \
  nano \
  usbutils \
  sudo \
  libglib2.0-0

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
/opt/conda/bin/conda install -n imswitch -y -c \
  conda-forge \
  h5py \
  numcodecs
export PATH=/opt/conda/bin:$PATH # note: this is only temporary; the container must update PATH, too

# Clean up all the package managers at the end

apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
