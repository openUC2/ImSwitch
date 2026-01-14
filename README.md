# ImSwitch

[![DOI](https://joss.theoj.org/papers/10.21105/joss.03394/status.svg)](https://doi.org/10.21105/joss.03394)

It is a fork - that has been detached from the original ImSwitch fork: https://github.com/ImSwitch/ImSwitch

``ImSwitch`` is a software solution in Python that aims at generalizing microscope control by using an architecture based on the model-view-presenter (MVP) to provide a solution for flexible control of multiple microscope modalities.

## Statement of need

The constant development of novel microscopy methods with an increased number of dedicated
hardware devices poses significant challenges to software development.
ImSwitch is designed to be compatible with many different microscope modalities and customizable to the
specific design of individual custom-built microscopes, all while using the same software. We
would like to involve the community in further developing ImSwitch in this direction, believing
that it is possible to integrate current state-of-the-art solutions into one unified software.

## UC2 Ecosystem Integration

ImSwitch is a core component of the UC2 (You.See.Too) ecosystem, designed to work seamlessly with various UC2 hardware and software components:

### Hardware Integration
- **[UC2-ESP Firmware](https://github.com/youseetoo/uc2-esp32)**: Low-level firmware for ESP32-based controllers that manage UC2 hardware components like motors, LEDs, and lasers. ImSwitch communicates with these devices through serial protocols.
- **[UC2-REST Interface](https://github.com/openUC2/UC2-REST/)**: A REST API middleware that provides standardized HTTP communication between ImSwitch and UC2 devices. This enables remote control and web-based interfaces.

### Web Interface
- **[ImSwitch React Frontend](https://github.com/openUC2/ImSwitch/tree/master/frontend)**: A modern web-based user interface built with React that provides remote access to ImSwitch functionality through web browsers. This allows for mobile control and remote microscopy operations.

### Complete Operating System
- **[ImSwitch-OS](https://github.com/openUC2/rpi-imswitch-os/)**: A complete Raspberry Pi-based operating system image with ImSwitch and all UC2 components pre-installed and configured. This provides a plug-and-play solution for UC2 microscopy systems.

### Docker Support
ImSwitch provides comprehensive Docker support for containerized deployments, enabling easy installation across different platforms and cloud environments. The Docker implementation supports both headless and GUI modes, making it suitable for both automated systems and interactive use. See the [Docker documentation](docker/README.md) for detailed setup instructions and configuration options.

## Installation


### Install using UV (Recommended)

ImSwitch can be installed using UV, a fast Python package installer written in Rust that's significantly faster than pip. Python 3.9 or later is required. Additionally, certain components (the image reconstruction module and support for TIS cameras) require the software to be running on Windows, but most of the functionality is available on other operating systems as well.

**Why UV?**
- **~10-100x faster** package installation and dependency resolution
- **Better dependency resolution** - finds compatible package versions more reliably
- **Lock file support** - ensures reproducible builds across environments
- **Drop-in replacement** for pip with improved caching and parallelization

First, install UV:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install ImSwitch:

```bash
uv pip install ImSwitchUC2
```

#### For developers working from source:

```bash
# Clone the repository
git clone https://github.com/openUC2/ImSwitch/
cd ImSwitch

# Create a virtual environment and install with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .[PyQt5,dev]

# UV automatically creates a uv.lock file for reproducible builds
```

For detailed UV usage instructions, see [docs/uv-guide.md](docs/uv-guide.md).

You will then be able to start ImSwitch with this command:

```
imswitch
```


### Install from Github (UC2 version)

This is an outdated version how you can install it through conda.

**Installation**
```
cd ~/Documents
git clone https://github.com/openUC2/ImSwitch/
cd ImSwitch
# alternatively download this repo, unzip the .zip-file and open the command prompt in this directory
conda create -n imswitch python=3.9 -y
conda activate imswitch

# Install UV (recommended for faster package management)
curl -LsSf https://astral.sh/uv/install.sh | sh  # On macOS/Linux
# OR on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies with UV (faster and more reliable)
uv pip install -r requirements.txt
uv pip install -e .
uv pip install git+https://gitlab.com/bionanoimaging/nanoimagingpack

cd ~/Documents/
# if there is a folder called ImSwitchConfig => rename it!
git clone https://github.com/beniroquai/ImSwitchConfig
# Alternatively download the repository as a zip, unzip the file into the folder Documents/ImSwitchConfig
```

## Misc things

### Installation of drivers:

For this, please reffer to the bash files in this repo: https://github.com/openUC2/ImSwitchDockerInstall


### Permissions for the serial driver

```
sudo usermod -a -G dialout $USER
newgrp dialout
ls -l /dev/ttyUSB0
```

## Documentation

Legacy documentation is available at [imswitch.readthedocs.io](https://imswitch.readthedocs.io). For everything up-to-date with this openUC2 Fork, please refer to https://openuc2.github.io/ and search for `ImSwitch` related topics

## Testing

ImSwitch has automated testing through GitHub Actions, including UI and unit tests. It is also possible to manually inspect and test the software without any device since it contains mockers that are automatically initialized if the instrumentation specified in the config file is not detected.

## Contributing

Read the [contributing section](https://imswitch.readthedocs.io/en/latest/contributing.html) in the documentation if you want to help us improve and further develop ImSwitch!
