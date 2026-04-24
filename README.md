# ImSwitch

[![DOI](https://joss.theoj.org/papers/10.21105/joss.03394/status.svg)](https://doi.org/10.21105/joss.03394)

ImSwitch is a Python program which aims at generalizing microscope control. Here is an intro video from Jacopo (developer of the original ImSwitch project) about ImSwitch: <https://www.youtube.com/watch?v=B54QCt5OQPI>

The openUC2/ImSwitch repo is a hard fork of the upstream project at [ImSwitch/ImSwitch](https://github.com/ImSwitch/ImSwitch), which is no longer maintained; openUC2/ImSwitch runs ImSwitch as a server in headless mode with an HTTP API which used by a React single-page-app browser GUI (also served from the ImSwitch server). This fork does not maintain the upstream's Qt-based desktop GUI.

## Development

These instructions are for people developing and maintaining ImSwitch.

### Setup

Run:

```bash
# Install uv
cd ~
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone the ImSwitch repository
git clone https://github.com/openUC2/ImSwitch
cd ImSwitch

# Create a virtual environment and install ImSwitch with UV
uv venv --system-site-packages
uv sync

# then start it in headless mode with the API server:
uv run python main.py --headless --http-port 8001
```

# Alternative installation with uv pip (not recommended, may cause dependency issues):

```bash
source .venv/bin/activate
# Yes, we need to start using `uv sync` instead for reproducible installs with our lockfile...but we're still ignoring the lockfile right now:
uv pip install -e .[dev]

# Run a workaround for some brokenness:
uv pip uninstall psygnal
uv pip install psygnal --no-binary :all:

# Install UC2-REST from source
cd ~
git clone https://github.com/openuc2/UC2-REST
cd UC2-REST
uv pip install . -e

# Download ImSwitch configurations (yes, this is a janky way of getting configurations)
cd ~
git clone https://github.com/openUC2/ImSwitchConfig

# Set up permissions for the serial driver, if needed:
newgrp dialout
sudo usermod -a -G dialout $USER

# eventually change ownership of the ImSwitchConfig Folder
sudo chown  -R pi:pi /home/pi/ImSwitchConfig

# Then reboot your computer, or at least fully log out of your user account and then log back in
```

eventually install libraries for aiortc

```
sudo apt install \
    libpcap-dev \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    build-essential \
    python3-dev
```

### Optional For PiCamera in UV 

```
# then install picamera2
cd ~/ImSwitch
source .venv/bin/activate
uv venv --system-site-packages -y
sudo apt-get update && sudo apt install -y --no-install-recommends \
         python3-picamera2 -y \
     && sudo apt-get clean \
     && sudo apt-get autoremove \
     && sudo rm -rf /var/cache/apt/archives/* \
     && sudo rm -rf /var/lib/apt/lists/*
```

Next, follow the setup instructions for the development environment for the frontend in
[./frontend/README.md](./frontend/README.md#setup).

For detailed UV usage instructions, refer to [docs/uv-guide.md](docs/uv-guide.md).

### Special Case for MAC (ARM64) running HIKRobotics Cameras (x86_64 drivers)

The HikRobot MVS SDK 2.0.0 for macOS is x86_64 only. On Apple Silicon (M1/M2/M3) it must
run under Rosetta 2. The libusb-based USB3 Vision transport layer initializer can hang
inside `dlopen` on macOS 14+ under Rosetta. The most reliable solution is a dedicated
i386/x86_64 conda/mamba environment.

#### Option A: mamba with osx-64 (recommended — works like old conda setup)

```bash
# Create an x86_64 (osx-64) Python 3.11 environment
CONDA_SUBDIR=osx-64 mamba create -n intel_env python=3.11 -y
conda activate intel_env
conda config --env --set subdir osx-64

# Verify it's running as x86_64 under Rosetta
python -c "import platform; print(platform.machine())"  # x86_64

# Install ImSwitch
cd ~/ImSwitch   # adjust path as needed
pip install -e .

# Run (Rosetta translates automatically when subdir=osx-64)
python main.py --headless --http-port 8001
```

#### Option B: uv venv with python.org universal2 installer

```bash
# Download the universal2 installer (contains both x86_64 and arm64)
# https://www.python.org/ftp/python/3.11.11/python-3.11.11-macos11.pkg
# Install it, then:

cd ~/ImSwitch
uv venv --python /usr/local/bin/python3.11 .venv_x86_311

# Verify architecture
arch -x86_64 .venv_x86_311/bin/python3.11 -c "import platform; print(platform.machine())"
uv run main.py --venv .venv_x86_311 --headless --http-port 8001
```
# → x86_64

source .venv_x86_311/bin/activate
uv pip install -e .

# Always launch under x86_64 so Rosetta uses the x86_64 dylibs
source .venv_x86_311/bin/activate
arch -x86_64 python main.py --headless --http-port 8001
```

### Running

To run ImSwitch, first activate the virtual environment if you haven't already done so:

```bash
source .venv/bin/activate
```

and then make an up-to-date build of the frontend if you haven't already done so:

```bash
npm --prefix ./frontend run build
```

and then launch the ImSwitch server:

```bash
# Yes, we should use `uv run` instead of `python`...but first we need to make `uv sync` work:
python main.py --headless --http-port 8001
```

Then, in your web browser, you can open <https://localhost:8001/imswitch/ui/index.html>

## Deployment

This version of ImSwitch is meant to be deployed as part of
[rpi-imswitch-os](https://github.com/openUC2/rpi-imswitch-os).
rpi-imswitch-os is the only officially-supported method of deploying ImSwitch for testing purposes
and for operation in production.

## Related repositories

ImSwitch is a core component of the UC2 (You.See.Too) ecosystem, designed to be integrated with various UC2 hardware and software components:

### Hardware Integration
- **[UC2-ESP Firmware](https://github.com/youseetoo/uc2-esp32)**: Low-level firmware for ESP32-based controllers that manage UC2 hardware components like motors, LEDs, and lasers. ImSwitch communicates with these devices through serial protocols.
- **[UC2-REST Interface](https://github.com/openUC2/UC2-REST/)**: A REST API middleware that provides standardized HTTP communication between ImSwitch and UC2 devices. This enables remote control and web-based interfaces.

### Docker Support
ImSwitch provides comprehensive Docker support for containerized deployments, enabling easy installation across different platforms and cloud environments. The Docker implementation supports both headless and GUI modes, making it suitable for both automated systems and interactive use. See the [Docker documentation](docker/README.md) for detailed setup instructions and configuration options.

### End-to-End Operating System
- **[rpi-imswitch-os](https://github.com/openUC2/rpi-imswitch-os)**: A complete Raspberry Pi-based operating system image with ImSwitch and all UC2 components pre-installed and configured. This is the officially-supported way to deploy and use ImSwitch.

## Documentation

Documentation for the upstream project is at
[imswitch.readthedocs.io](https://imswitch.readthedocs.io/).
For documentation about openUC2/ImSwitch, please refer to
[openuc2.github.io](https://openuc2.github.io)
and search for ImSwitch-related topics.
