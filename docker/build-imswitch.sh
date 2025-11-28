#!/usr/bin/env -S bash -eux

# Activate UV environment
export PATH="/root/.local/bin:$PATH"
source /opt/imswitch/.venv/bin/activate

# Install ImSwitch from our local copy of the repo
cp -r /mnt/ImSwitch /tmp/ImSwitch
cd /tmp/ImSwitch
uv pip install /tmp/ImSwitch
# Note(ethanjli): we delete DLL files because they take up a significant amount of space, and
# they should be useless in Linux anyways (as they're Windows-specific)
shopt -s globstar
ls /opt/imswitch/.venv/lib/*/site-packages/imswitch/**/*.dll 2>/dev/null || true
rm -rf /opt/imswitch/.venv/lib/*/site-packages/imswitch/**/*.dll 2>/dev/null || true

# Install UC2-REST
git clone https://github.com/openUC2/UC2-REST /tmp/UC2-REST
cd /tmp/UC2-REST
uv pip install /tmp/UC2-REST

# Link system picamera2 and required modules to UV venv
# This ensures that the UV venv Python uses the system's picamera2 with proper libcamera bindings
UV_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
echo '/usr/lib/python3/dist-packages' > $UV_SITE_PACKAGES/system-packages.pth
python -c 'import sys; print("Python paths:"); [print(p) for p in sys.path]'

# Install simplejpeg in UV environment to avoid NumPy ABI compatibility issues
# The system python3-simplejpeg is compiled against system NumPy, but we need it for UV venv NumPy
uv pip install --no-cache-dir simplejpeg --force-reinstall

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /root/.cache/uv
rm -rf /root/.cache/pip
rm -rf /tmp/*
