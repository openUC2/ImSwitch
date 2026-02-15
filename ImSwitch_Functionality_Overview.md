# ImSwitch Functionality Overview

## Core Hardware Control

### Stage & Positioning
- **Multi-axis motorized stage control** (X, Y, Z, A axes)
  - Position reading and setting for each axis
  - Movement speed and acceleration control
  - Homing and calibration routines
  - Virtual stage support for testing/simulation
  - Hard limit detection and safety mechanisms
  - Support for multiple positioner types (ESP32, virtual, etc.)
  - constant speed movement mode (moveForever) for smooth scanning/pumps
  - Sending xy positions to microcontroller and run trigger series (indepedent from Python and serial => fast)

### Objective Control
- **Motorized objective turret/revolver**
  - Multiple objective positions/states
  - Automatic objective switching
  - Pixel size calibration per objective
  - NA (Numerical Aperture) configuration
  - Magnification settings
  - Z-offset compensation per objective
  - Home position calibration (e.g. zero of objective)

### Illumination Control

#### Lasers
- **Laser on/off control**
- **Laser intensity/power adjustment**
  - Power range configuration (min/max values)
  - Binary (on/off) or analog (variable power) modes
  - Wavelength configuration
  - Multiple laser support
  - Various laser types supported:
    - ESP32-based LED lasers
    - Cobolt lasers
    - Virtual lasers
    - REST API-controlled lasers
    - SQUID lasers
    - CoolLED systems

#### LEDs
- **LED matrix control**
  - Individual LED on/off
  - Pattern generation
  - Intensity control
  - ESP32-based LED matrix support

### Camera/Detector Control
- **Live image acquisition and streaming**
- **Image capture and storage**
- **Multiple camera support**
- **Camera parameters:**
  - Exposure time
  - Gain
  - Binning
  - ROI (Region of Interest)
  - Pixel format
- **Supported camera types (..that we use):**
  - daheng
  - HIK
  - Tucam
  - Raspberry Pi Camera
  - Virtual cameras for testing

### Additional Hardware
- **Galvo scanners** (for laser scanning)
- **Temperature control**
- **Flow control** (microfluidics)

##  Autofocus 
- usable in other controllers
- **Software-based Autofocus**
  - Image-based focus metrics (contrast/sharpness)
  - Z-stack acquisition for focus finding (coarse => fine)
  - Continuous focus measurement for debugging
- **Hardware-based Autofocus**
  - Laser-reflection-based autofocus
  - measure deflection of a laser beam to determine focus position by measureing the shift of a reflected laser spot on a camera
  - use laser autofocus result to adjust the Z position of the stage to maintain focus during long-term experiments
  - one-shot: compute look up table while moving z and measuring the laser spot position, then use this look up table for fast autofocus during experiments; move back to z at autofocus value
  - continuous: continuously measure the laser spot position and adjust z in real-time to maintain focus (e.g. for long-term timelapse experiments) with PID control loop (unstable)

## Advanced Imaging Modes

### Microscopy Techniques
- **Lightsheet microscopy**
  - Continuous scanning with Zarr storage (continouos sweep vs step-acquire)
  - Z-stack acquisition
  - visualization of 3D data in-browser with VizarrViewer
  - tile-based 3d volume acquisition 
- **Holography**
  - Off-axis holography
  - Inline holography
  - HoliSheet (combined holography + lightsheet)
- **SIM** (Structured Illumination Microscopy)
- **DPC** (Differential Phase Contrast)
- **ISM** (Image Scanning Microscopy)
- **STORM reconstruction**
- **Composite imaging** (multi-modal combinations)
  - ExperimentController where we provide a protocol for multi dimensional acquisitions (xyz, multi channel, autofocus )

### Computational Imaging
- **DPC processing**
- **Phase retrieval**
- **Flatfield correction**

## Data Acquisition & Storage

### Recording Capabilities
- **Snap single images**
- **Video/time-lapse recording**
- **Multi-dimensional acquisition (through experimentcontroller):**
  - Time series
  - Z-stacks
  - Multi-position
  - Multi-channel
- **Storage formats:**
  - TIFF
  - Zarr (for large datasets)
  - MP4/AVI (video)
- **Metadata embedding**

### Scan Types
- **ROI scanning** (Region of Interest, Click/Select and go back to that position/timelapse)
- **Tile scanning** (large area mosaics)
- **Stage scan acquisition**
- **Well plate scanning** (keep annotations e.g. A1, B2, etc.)

## Automation & Workflows

### Experiment Management
- **Timelapse experiments**
  - Configurable intervals
  - Multi-position support
- **Workflow automation**
  - Sequential operations
  - Conditional logic
  - "Smart Microscopy"
- **Autofocus**
  - Hardware and software autofocus
  - Continuous focus maintenance
- **FOV (Field of View) locking**
  - Position drift correction

### Calibration
- **Pixel calibration**
- **Affine transformation calibration**
- **Stage center calibration**
- **Objective calibration**
- **AprilTag grid calibration** for acceptance test (does the stage move in the right direciton)

## User Interface & Control

### Frontend
- **Web-based React frontend**
- **Live image streaming to browser**
- **WebRTC support** for low-latency streaming
- **MJPEG streaming**
- **Interactive controls for all hardware**
- **Real-time parameter adjustment**

### API & Integration
- **REST HTTP API** for remote control
- **Python scripting interface**
- **Hypha integration** (distributed computing)
- **Arkitekt integration** (workflow management)

### Configuration
- **JSON-based setup files**
- **Hardware configuration per setup**
- **User settings persistence**
- **Multiple configuration profiles**

## Communication & Connectivity

### Hardware Communication
- **RS232/Serial communication**
- **USB device support**
- **Network/TCP communication** (i.e. Socket.io for async updates in the frontend)
- **REST API clients**
- **UC2-REST integration** for UC2 hardware ecosystem

### Remote Operation
- **Headless mode** (no local GUI)
- **Web browser access**
- **API-based remote control**
- **WebRTC streaming**

## Development & Testing

### Debugging Tools
- **Virtual/mock hardware** for testing
- **Logging system**
- **Stress testing framework**
- **Acceptance testing**

### Extensibility
- **Plugin architecture** for new devices (not really used at the moment - but would be nice to have in order to add e.g. new camera types more easily)
- **Manager-based design pattern**
- **Easy addition of:**
  - New camera types
  - New laser/LED controllers
  - New positioners
  - New imaging modes

## Storage & Organization

### Data Management
- **Automatic file naming**
- **Folder organization**
- **Metadata tracking**
- **Instrument metadata management**
- **Storage location configuration**

## UC2 Ecosystem Integration

### UC2-Specific Features
- **UC2 configuration manager**
- **ESP32 firmware integration**
- **LED matrix patterns**
- **Modular microscope support**
- **UC2-REST device control**

## Docker & Deployment

- **Containerized deployment**
- **Multi-platform support** (Linux, macOS, Windows)
- **rpi-imswitch-os** integration
- **Cloud deployment ready**


## Summary

### What is Established ✅
- Comprehensive hardware control (stages, lasers, cameras, objectives)
- Advanced imaging modes (holography, lightsheet, SIM, DPC)
- Data acquisition and storage with multiple formats
- Web-based frontend with live streaming
- REST API for remote control
- Workflow automation and experiment management
- Extensive calibration tools
- UC2 hardware ecosystem integration
- Docker deployment support

### What Needs Development 
- Enhanced multi-user support
- Cloud-native data storage integration
- Advanced AI/ML integration for analysis
- Mobile app interface
- Enhanced collaborative features
- Improved documentation and tutorials
- More standardized data formats (OME-TIFF, etc.)
- Performance optimization for large datasets
- Better error handling and recovery
- Extended hardware compatibility testing

### Key Strengths 
- Modular and extensible architecture
- Wide hardware compatibility
- Multiple imaging modalities
- Web-based interface (headless operation)
- Active development and UC2 ecosystem integration
- Open-source and community-driven
