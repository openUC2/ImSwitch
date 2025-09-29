#!/usr/bin/env -S bash -eux

# Install the Hik driver

apt-get update
apt-get install -y \
  wget \
  unzip

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
cd /tmp
wget "https://github.com/openUC2/ImSwitchDockerInstall/releases/download/imswitch-master/MVS-3.0.1_${cpu}_20241128.deb"
dpkg -i "MVS-3.0.1_${cpu}_20241128.deb"
rm -f "MVS-3.0.1_${cpu}_20241128.deb"
mkdir -p /opt/MVS/bin/fonts
rm -rf /opt/MVS/doc

# Install the Daheng camera driver

cd /tmp
wget https://dahengimaging.com/downloads/Galaxy_Linux_Python_2.0.2106.9041.tar_1.gz
tar -zxvf Galaxy_Linux_Python_2.0.2106.9041.tar_1.gz
case "$TARGETPLATFORM" in
"linux/arm64")
  # note: the different CPU architectures also have different build numbers, for whatever reason!
  archive="Galaxy_Linux-armhf_Gige-U3_32bits-64bits_1.5.2303.9202"
  # presumably Daheng only builds 32-bit ARM binaries (armhf), but those do work on 64-bit ARM
  ;;
"linux/amd64")
  archive="Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221"
  ;;
*)
  echo "Unknown target platform $TARGETPLATFORM!"
  exit 1
  ;;
esac
wget "https://dahengimaging.com/downloads/$archive.zip"
unzip "$archive.zip"
cd "/tmp/$archive"
chmod +x Galaxy_camera.run
cd /tmp/Galaxy_Linux_Python_2.0.2106.9041/api
/bin/bash -c "source /opt/conda/bin/activate imswitch && python3 setup.py build"
/bin/bash -c "source /opt/conda/bin/activate imswitch && python3 setup.py install"

# Create the udev rules directory
mkdir -p /etc/udev/rules.d

# Run the installer script using expect to automate Enter key presses
echo "Y En Y" | "/tmp/$archive/Galaxy_camera.run"

# Install VimbaX only for ARM64

if [ "$TARGETPLATFORM" = "linux/arm64" ]; then
  echo "Installing VimbaX SDK for ARM64..."
  if ! wget --no-check-certificate \
    https://downloads.alliedvision.com/VimbaX/VimbaX_Setup-2025-1-Linux_ARM64.tar.gz -O VimbaX_Setup-2025-1-Linux_ARM64.tar.gz; then
    echo "VimbaX SDK download failed. Please ensure the file is present in the build context."
    exit 1
  fi
  tar -xzf VimbaX_Setup-2025-1-Linux_ARM64.tar.gz -C /opt
  mv /opt/VimbaX_2025-1 /opt/VimbaX
  rm VimbaX_Setup-2025-1-Linux_ARM64.tar.gz
  cd /opt/VimbaX/cti
  ./Install_GenTL_Path.sh
  /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install https://github.com/alliedvision/VmbPy/releases/download/1.1.0/vmbpy-1.1.0-py3-none-linux_aarch64.whl"
  export GENICAM_GENTL64_PATH="/opt/VimbaX/cti"
fi
rm -rf /opt/VimbaX/doc

# Clean up build-only tools

apt-get remove -y \
  wget \
  unzip \
  g++ \
  g++-11 \
  gcc \
  gcc-11

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
