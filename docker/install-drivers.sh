#!/usr/bin/env -S bash -eux

apt-get update
apt-get install -y \
  wget \
  unzip \
  gnupg

# Set up virtual environment

uv venv
source /opt/imswitch/.venv/bin/activate

# Install the Hik driver

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
uv pip install setuptools wheel # for building Python bindings
python3 setup.py build
python3 setup.py install

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
  # Install VmbPy using UV pip
  uv pip install https://github.com/alliedvision/VmbPy/releases/download/1.1.0/vmbpy-1.1.0-py3-none-linux_aarch64.whl
  export GENICAM_GENTL64_PATH="/opt/VimbaX/cti"
fi
rm -rf /opt/VimbaX/doc

# Install picamera2

tmpdir="$(mktemp -d)"
gpg --homedir "$tmpdir" --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E
gpg --homedir "$tmpdir" --export 82B129927FA3303E >/etc/apt/keyrings/raspi.gpg
echo "deb [signed-by=/etc/apt/keyrings/raspi.gpg] http://archive.raspberrypi.org/debian/ trixie main" >/etc/apt/sources.list.d/raspi.list
apt-get update
apt install -y --no-install-recommends python3-picamera2
# Install simplejpeg in UV environment to avoid NumPy ABI compatibility issues
# The system python3-simplejpeg is compiled against system NumPy, but we need it for UV venv NumPy
uv pip install --no-cache-dir simplejpeg --force-reinstall --python /opt/imswitch/.venv/bin/python
# Link system picamera2 and required modules to UV venv; this ensures that the UV venv Python uses the system's picamera2 with proper libcamera bindings
UV_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
echo '/usr/lib/python3/dist-packages' >"$UV_SITE_PACKAGES"/system-packages.pth
python -c 'import sys; print("Python paths:"); [print(p) for p in sys.path]'

# Clean up build-only tools

apt-get remove -y \
  wget \
  unzip \
  g++ \
  g++-11
apt -y autoremove

# Clean up /tmp
rm -rf /tmp/*
