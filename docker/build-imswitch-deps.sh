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
  libglib2.0-0

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

# install deps listed in pyproject.toml, but don't install ImSwitch yet (because we've only mounted
# pyproject.toml, not any ImSwitch source files, into the container when we run this build script)
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch"

# Clean up build-only tools

apt-get remove -y \
  build-essential \
  libhdf5-dev

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
