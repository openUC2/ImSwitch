#!/usr/bin/env -S bash -eux

# TODO(ethanjli): find a way to not rely on git inside a container image
apt-get update
apt-get install -y \
  git \
  build-essential \
  mesa-utils \
  libhdf5-dev \
  usbutils \
  libglib2.0-0

/opt/conda/bin/conda install -n imswitch -y -c conda-forge \
  h5py
/opt/conda/bin/conda install \
  numcodecs=0.15.0 \
  numpy=2.1.2
/bin/bash -c "source /opt/conda/bin/activate imswitch && \
  conda install -c conda-forge scikit-image=0.19.3"

# Install nmcli
# TODO(ethanjli): can we interact with the host's NetworkManager API without installing and running
# nmcli in the container?
apt-get install -y \
  network-manager \
  dbus \
  systemd \
  sudo

# TODO(ethanjli): can we delete this since we're repeating it later?
# Install UC2-REST first - as it will be installed via ImSwitch again
git clone https://github.com/openUC2/UC2-REST /tmp/UC2-REST
cd /tmp/UC2-REST
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/UC2-REST"

# Question(ethanjli): what does the following note mean? It sounds suspicious...
# first install all the dependencies not not to install them again in a potential "breaking update"
# Clone the repository and install dependencies
git clone https://github.com/openUC2/imSwitch /tmp/ImSwitch
cd /tmp/ImSwitch
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch"

# Clone the config folder
git clone https://github.com/openUC2/ImSwitchConfig /tmp/ImSwitchConfig

# we want psygnal to be installed without binaries - so first remove it
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip uninstall psygnal -y"
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install psygnal --no-binary :all:"

# fix the version of OME-ZARR
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install zarr==2.11.3"

# Clone the config folder
cd /tmp/ImSwitchConfig
# Note(ethanjli): I feel like we should use the local files without of running `git pull` inside the
# Dockerfile, as that will desynchronize the container images from the state of the repository...
git pull

# Copy current local ImSwitch code instead of pulling from git
cd /tmp/ImSwitch-local
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch-local"

# Install UC2-REST
cd /tmp/UC2-REST
git pull
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/UC2-REST"

# install arkitekt
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install https://github.com/openUC2/imswitch-arkitekt-next/archive/refs/heads/master.zip"

# Clean up build-only tools

apt-get remove -y \
  git \
  build-essential \
  libhdf5-dev

# Clean up all the package managers at the end

apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
