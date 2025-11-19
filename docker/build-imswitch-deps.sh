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

/opt/conda/bin/conda install -n imswitch -y -c conda-forge \
  h5py
/opt/conda/bin/conda install \
  numcodecs=0.15.0 \
  numpy=2.1.2
/bin/bash -c "source /opt/conda/bin/activate imswitch && \
  conda install -c conda-forge scikit-image=0.19.3"

# we want psygnal to be installed without binaries - so first remove it
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip uninstall psygnal -y"
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install psygnal --no-binary :all:"

# fix the version of OME-ZARR
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install zarr==2.11.3"

# install deps listed in pyproject.toml, but don't install ImSwitch yet:
mkdir -p /tmp/ImSwitch/imswitch
cat >/tmp/ImSwitch/imswitch/__init__.py <<EOF
# temporary placeholder to be overwritten
__version__ = "0.0.0"
EOF
cp /mnt/ImSwitch/pyproject.toml /tmp/ImSwitch/pyproject.toml
cd /tmp/ImSwitch
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch"

# Clean up build-only tools

apt-get remove -y \
  build-essential \
  libhdf5-dev

# Install raspberry pi camera dependencies (only on ARM platforms)
# https://github.com/hyzhak/pi-camera-in-docker/blob/main/Dockerfile
# Note: Requires Debian Bookworm for Python 3.11 compatibility
if [[ "$TARGETPLATFORM" == "linux/arm64" ]] || [[ "$TARGETPLATFORM" == "linux/arm/v7" ]]; then
  echo "Installing picamera2 dependencies for ARM platform..."
  apt update && apt install -y --no-install-recommends gnupg
  echo "deb http://archive.raspberrypi.org/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E
  apt update
  # Install picamera2 and handle configuration errors from GTK/systemd dependencies
  # Temporarily disable exit-on-error for dpkg configuration issues
  set +e
  apt install -y --no-install-recommends python3-picamera2
  # Try to fix any partial configuration issues
  dpkg --configure -a
  # Verify picamera2 was installed successfully
  if dpkg -l | grep -q python3-picamera2; then
    echo "python3-picamera2 installed successfully"
    set -e
  else
    echo "ERROR: python3-picamera2 installation failed"
    set -e
    exit 1
  fi
  # needs:
  #    volumes:
  #      - /run/udev:/run/udev:ro
fi

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
