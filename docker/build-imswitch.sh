#!/usr/bin/env -S bash -eux

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

# Link system picamera2 and required modules to conda environment
# This ensures that the conda Python uses the system's picamera2 with proper libcamera bindings
/bin/bash -c "source /opt/conda/bin/activate imswitch && \
    CONDA_SITE_PACKAGES=\$(python -c 'import site; print(site.getsitepackages()[0])') && \
    echo '/usr/lib/python3/dist-packages' > \$CONDA_SITE_PACKAGES/system-packages.pth && \
    python -c 'import sys; print(\"Python paths:\"); [print(p) for p in sys.path]'"

# Install simplejpeg in conda environment to avoid NumPy ABI compatibility issues
# The system python3-simplejpeg is compiled against system NumPy, but we need it for conda NumPy
/bin/bash -c "source /opt/conda/bin/activate imswitch && \
    pip install --no-cache-dir simplejpeg --force-reinstall"

# Clean up all the package managers at the end

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
