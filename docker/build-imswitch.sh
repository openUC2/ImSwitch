#!/usr/bin/env -S bash -eux

# Activate UV environment
export PATH="/root/.local/bin:$PATH"

# Copy the full ImSwitch source into the project directory and install with uv sync.
# We use the permanent /opt/imswitch location so that:
#   1. The editable install works (source is referenced in-place)
#   2. uv run can find the project at runtime (entrypoint.sh)
# uv.lock is already present from build-imswitch-deps.sh; overwrite it to stay consistent.
cp -r /mnt/ImSwitch/. /opt/imswitch/
cd /opt/imswitch

# Use uv sync with --frozen so the pre-solved lockfile is used directly —
# no resolver runs, no pip conflicts. UC2-REST and all other deps come from the lockfile.
uv sync --frozen

# Note(ethanjli): we delete DLL files because they take up a significant amount of space, and
# they should be useless in Linux anyways (as they're Windows-specific)
shopt -s globstar
ls /opt/imswitch/.venv/lib/*/site-packages/imswitch/**/*.dll 2>/dev/null || true
rm -rf /opt/imswitch/.venv/lib/*/site-packages/imswitch/**/*.dll 2>/dev/null || true

# UC2-REST is installed from PyPI via the lockfile by uv sync above — no git clone needed.

# Link system picamera2 and required modules to UV venv
# This ensures that the UV venv Python uses the system's picamera2 with proper libcamera bindings
UV_SITE_PACKAGES=$(/opt/imswitch/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
echo '/usr/lib/python3/dist-packages' > $UV_SITE_PACKAGES/system-packages.pth
/opt/imswitch/.venv/bin/python -c 'import sys; print("Python paths:"); [print(p) for p in sys.path]'

# Install simplejpeg in UV environment to avoid NumPy ABI compatibility issues
# The system python3-simplejpeg is compiled against system NumPy, but we need it for UV venv NumPy
uv pip install --no-cache-dir simplejpeg --force-reinstall --python /opt/imswitch/.venv/bin/python

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /root/.cache/uv
rm -rf /root/.cache/pip
rm -rf /tmp/*
