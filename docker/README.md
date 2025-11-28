## ImSwitch in Docker

ImSwitch provides comprehensive Docker support for easy deployment and cross-platform compatibility. The Docker implementation uses UV (fast Rust-based package manager) for improved build times and dependency management.

### Features
- **Headless Operation**: Run ImSwitch without GUI for automated systems and remote control
- **Web Interface**: Access ImSwitch through web browsers via the integrated REST API
- **Hardware Access**: Direct USB and serial device access for camera and stage control
- **Persistent Storage**: Configurable data and configuration persistence
- **Multi-architecture Support**: Compatible with ARM64 (Raspberry Pi, Jetson) and x86_64 systems
- **Fast Package Management**: Uses UV for 10-100x faster package installation compared to pip

### Key Capabilities
- HTTP/HTTPS REST API server for remote control
- WebSocket communication for real-time data streaming  
- Configurable port mapping and SSL support
- External drive mounting for data storage
- Pip package installation for custom extensions
- Git repository updates for development workflows

### Quick Start
```bash
# Pull and run the latest ImSwitch Docker image
docker pull openuc2/imswitch:latest
docker run -p 8001:8001 openuc2/imswitch:latest
```

For detailed Docker documentation and configuration options, visit: https://openuc2.github.io/docs/ImSwitch/ImSwitchDocker

