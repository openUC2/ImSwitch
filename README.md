# ImSwitch

[![DOI](https://joss.theoj.org/papers/10.21105/joss.03394/status.svg)](https://doi.org/10.21105/joss.03394)

``ImSwitch`` is a software solution in Python that aims at generalizing microscope control by using an architecture based on the model-view-presenter (MVP) to provide a solution for flexible control of multiple microscope modalities. Here is an intro video from Jacopo about ImSwitch: <https://www.youtube.com/watch?v=B54QCt5OQPI>

The openUC2/ImSwitch repo is a hard fork of the original ImSwitch (which is no longer maintained) which provides a web browser GUI frontend to ImSwitch.

## Development

These instructions are for people developing and maintaining ImSwitch.

### Setup

This version of ImSwitch is installed using UV, a fast Python package installer written in Rust which, in comparison to pip, is:

- much faster for package installation and dependency resolution
- more reliable in finding compatible package versions
- more reproducible for software builds

To set up your development environment, run:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the ImSwitch repository
git clone https://github.com/openUC2/ImSwitch/
cd ImSwitch

# Create a virtual environment and install ImSwitch with UV
uv venv
source .venv/bin/activate
uv pip install -e .

# Run a workaround for some brokenness:
uv pip uninstall psygnal
uv pip install psygnal --no-binary :all:

# Download ImSwitch configurations (yes, this is a janky way of getting configurations)
cd ~
git clone https://github.com/openUC2/ImSwitchConfig
```

For detailed UV usage instructions, see [docs/uv-guide.md](docs/uv-guide.md).

### Running

To run ImSwitch, first activate the virtual environment:

```bash
source .venv/bin/activate
```

and then run:

```bash
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
