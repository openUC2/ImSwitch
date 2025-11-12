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

apt -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*
/opt/conda/bin/conda clean --all -f -y
rm -rf /opt/conda/pkgs/*
pip3 cache purge || true
rm -rf /root/.cache/pip
rm -rf /tmp/*
