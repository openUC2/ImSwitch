#!/usr/bin/env -S bash -eux
# Only arm64 needs the prebuilt adapters; on amd64 just run `mmcore install`.
[ "$TARGETPLATFORM" = "linux/arm64" ] || { echo "skip: not arm64"; exit 0; }

MM_REF="${MM_REF:-mm-v1}"   # the release tag the adapters were published under

# Install tools needed for downloading/unzipping adapters and building pymmcore.
apt-get update
apt-get install -y --no-install-recommends \
	build-essential \
	swig \
	python3-dev \
	libboost-all-dev \
	wget \
	unzip

cd /tmp
wget -q "https://github.com/openUC2/ImSwitchDockerInstall/releases/download/imswitch-master/micro-manager-arm64.zip"
mkdir -p /opt
unzip micro-manager-arm64.zip -C /opt    # -> /opt/micro-manager/lib/micro-manager/*.so
cat /opt/DEVICE_INTERFACE_VERSION.txt || true
rm -f micro-manager-arm64.zip

# pymmcore has no aarch64 wheel on PyPI — build it from source so its embedded
# MMCore matches the adapters' device interface version.
source /opt/imswitch/.venv/bin/activate
uv pip install pymmcore --no-binary pymmcore
uv pip install "pymmcore-plus[cli]>=0.10"

# sanity check: embedded device API should match DEVICE_INTERFACE_VERSION.txt
python3 -c "import pymmcore; print('pymmcore API:', pymmcore.CMMCore().getAPIVersionInfo())"

apt-get remove -y build-essential swig
apt -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*