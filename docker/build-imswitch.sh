#!/usr/bin/env -S bash -eux

# TODO(ethanjli): find a way to not rely on git inside a container image
apt-get update
apt-get install -y \
  git

# Install ImSwitch from our local copy of the repo
cp -r /mnt/ImSwitch /tmp/ImSwitch
cd /tmp/ImSwitch
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch"

# Delete DLL files
# Note(ethanjli): we do this because they take up a significant amount of space, and they should be
# useless in Linux anyways (as they're Windows-specific)
rm -rf /opt/conda/envs/imswitch/lib/**/*.dll

# Install UC2-REST
git clone https://github.com/openUC2/UC2-REST /tmp/UC2-REST
cd /tmp/UC2-REST
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/UC2-REST"

# install arkitekt
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install https://github.com/openUC2/imswitch-arkitekt-next/archive/refs/heads/master.zip"

# Clean up build-only tools

apt-get remove -y \
  git

# Clean up all the package managers at the end

apt autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
