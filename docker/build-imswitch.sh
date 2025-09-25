#!/usr/bin/env -S bash -eux

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

# Clean up all the package managers at the end

apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
