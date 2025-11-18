#!/usr/bin/env -S bash -eux

# Install pip and picamera2 for Raspberry Pi support
# Note: picamera2 requires system libcamera libraries
apt update && apt install -y --no-install-recommends \
         python3-pip \
         python3-picamera2 \
         python3-libcamera \
         libcamera0

# Create symlinks to make system picamera2 and libcamera accessible from conda environment
# This is necessary because picamera2 relies on system-installed libcamera binaries
CONDA_SITE_PACKAGES="/opt/conda/envs/imswitch/lib/python3.11/site-packages"
SYSTEM_SITE_PACKAGES="/usr/lib/python3/dist-packages"

mkdir -p "$CONDA_SITE_PACKAGES"

# Symlink picamera2 module
if [[ -d "$SYSTEM_SITE_PACKAGES/picamera2" ]]; then
    ln -sf "$SYSTEM_SITE_PACKAGES/picamera2" "$CONDA_SITE_PACKAGES/picamera2"
fi

# Symlink libcamera module
if [[ -d "$SYSTEM_SITE_PACKAGES/libcamera" ]]; then
    ln -sf "$SYSTEM_SITE_PACKAGES/libcamera" "$CONDA_SITE_PACKAGES/libcamera"
fi

# Symlink _libcamera (the C extension)
if [[ -f "$SYSTEM_SITE_PACKAGES/_libcamera.so" ]]; then
    ln -sf "$SYSTEM_SITE_PACKAGES/_libcamera.so" "$CONDA_SITE_PACKAGES/_libcamera.so"
fi

# Also symlink the .so files if they exist in different locations
for libcam_so in "$SYSTEM_SITE_PACKAGES"/libcamera*.so; do
    if [[ -f "$libcam_so" ]]; then
        ln -sf "$libcam_so" "$CONDA_SITE_PACKAGES/$(basename "$libcam_so")"
    fi
done

# Install ImSwitch from our local copy of the repo
cp -r /mnt/ImSwitch /tmp/ImSwitch
cd /tmp/ImSwitch
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch"
# Note(ethanjli): we delete DLL files because they take up a significant amount of space, and
# they should be useless in Linux anyways (as they're Windows-specific)
shopt -s globstar
ls /opt/conda/envs/imswitch/lib/*/*/imswitch/**/*.dll
rm -rf /opt/conda/envs/imswitch/lib/*/*/imswitch/**/*.dll

# Install UC2-REST
git clone https://github.com/openUC2/UC2-REST /tmp/UC2-REST
cd /tmp/UC2-REST
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/UC2-REST"

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
