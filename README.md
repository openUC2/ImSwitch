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
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the ImSwitch repository
git clone https://github.com/openUC2/ImSwitch
cd ImSwitch

# Create a virtual environment and install ImSwitch with UV
uv venv
source .venv/bin/activate
# Yes, we need to start using `uv sync` instead for reproducible installs with our lockfile...but we're still ignoring the lockfile right now:
uv pip install -e .[dev]

# Run a workaround for some brokenness:
uv pip uninstall psygnal
uv pip install psygnal --no-binary :all:

# Download ImSwitch configurations (yes, this is a janky way of getting configurations)
cd ~
git clone https://github.com/openUC2/ImSwitchConfig

# Set up permissions for the serial driver, if needed:
newgrp dialout
sudo usermod -a -G dialout $USER
# Then reboot your computer, or at least fully log out of your user account and then log back in
```

Next, follow the setup instructions for the development environment for the frontend in
[./frontend/README.md](./frontend/README.md#setup).

For detailed UV usage instructions, refer to [docs/uv-guide.md](docs/uv-guide.md).

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
