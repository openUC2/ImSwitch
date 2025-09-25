#!/usr/bin/env -S bash -eux

# Question(ethanjli): could we just merge this into build-base.sh?
/opt/conda/bin/conda install numcodecs=0.15.0 numpy=2.1.2
/bin/bash -c "source /opt/conda/bin/activate imswitch && \
    conda install scikit-image=0.19.3 -c conda-forge"

# Install nmcli
apt-get install -y --allow-unauthenticated \
    network-manager \
    dbus \
    systemd

# Install UC2-REST first - as it will be installed via ImSwitch again
git clone https://github.com/openUC2/UC2-REST /tmp/UC2-REST
cd /tmp/UC2-REST
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/UC2-REST"

# Question(ethanjli): what does the following note mean? It sounds suspicious...
# first install all the dependencies not not to install them again in a potential "breaking update"
# Clone the repository and install dependencies
git clone https://github.com/openUC2/imSwitch /tmp/ImSwitch
cd /tmp/ImSwitch
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch"

# Clone the config folder
git clone https://github.com/openUC2/ImSwitchConfig /tmp/ImSwitchConfig

# we want psygnal to be installed without binaries - so first remove it
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip uninstall psygnal -y"
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install psygnal --no-binary :all:"

# fix the version of OME-ZARR
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install zarr==2.11.3"

# Clean up all the package managers at the end

apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
