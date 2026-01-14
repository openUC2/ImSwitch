# LepmonController React Component Update

## Overview

This document summarizes the updates made to the LepmonController React component to match the enhanced Python backend controller capabilities.

## Key Features Implemented

### 1. Light Control System
- **Individual LED Controls**: Toggle switches for each available LED (gelb/yellow, blau/blue, rot/red)
- **Specialized LED Controls**: Dedicated UV LED and Visible LED controls
- **Master Controls**: "All Lights On/Off" functionality
- **Real-time State Updates**: WebSocket integration for light state changes

### 2. Hardware Status Monitoring
- **GPIO Availability**: Real-time monitoring of GPIO hardware availability
- **OLED Display Status**: Status of OLED display hardware
- **I2C Sensor Status**: I2C bus and sensor availability monitoring
- **Simulation Mode Indicator**: Shows when running in simulation mode vs hardware mode

### 3. Image Display & Capture
- **Latest Image Display**: Shows the most recent image captured from the LepmonCamera
- **Base64 Image Rendering**: Displays images streamed via WebSocket in real-time
- **Manual Image Capture**: Button to trigger single image capture
- **Image Format Support**: Supports JPEG, TIFF, and PNG formats

### 4. OLED Display Integration
- **4-Line Display Simulator**: Visual representation of the OLED display content
- **Real-time Updates**: Display content updates via WebSocket signals
- **Manual Display Updates**: Button to send custom display content

### 5. Timing Configuration
- **Acquisition Interval**: Configure time between image acquisitions
- **Stabilization Time**: Pre-experiment stabilization period
- **Pre/Post Acquisition Delays**: Fine-grained timing control
- **Real-time Updates**: Configuration changes applied immediately

### 6. LepmonOS Integration
- **Startup/Shutdown Controls**: Trigger LepmonOS system sequences
- **Live Sensor Updates**: Fetch real-time sensor data
- **Display Management**: Update OLED display remotely

## API Endpoints Used

### Core Experiment Control
- `GET /LepmonController/getStatus` - Get system status
- `GET /LepmonController/getInitialParams` - Get initial parameters
- `POST /LepmonController/startExperiment` - Start experiment
- `POST /LepmonController/stopLepmonExperiment` - Stop experiment
- `POST /LepmonController/focusMode` - Start focus mode
- `POST /LepmonController/reboot` - System reboot

### Hardware Control
- `GET /LepmonController/getHardwareStatus` - Get hardware status
- `POST /LepmonController/setLightState` - Control individual LEDs
- `POST /LepmonController/setAllLightsState` - Control all LEDs
- `POST /LepmonController/updateLCDDisplay` - Update OLED display

### Configuration
- `GET /LepmonController/getTimingConfig` - Get timing configuration
- `POST /LepmonController/setTimingConfig` - Update timing configuration

### Image & Data
- `POST /LepmonController/lepmonSnapImage` - Capture single image
- `GET /LepmonController/getSensorDataLive` - Get live sensor data

### LepmonOS Specific
- `POST /LepmonController/lepmonStartup` - LepmonOS startup sequence
- `POST /LepmonController/lepmonShutdown` - LepmonOS shutdown sequence
- `POST /LepmonController/lepmonUVLed` - Control UV LED specifically
- `POST /LepmonController/lepmonVisibleLed` - Control Visible LED specifically

## WebSocket Signals Handled

### Existing Signals (Enhanced)
- `sigImagesTaken` - Image count updates
- `temperatureUpdate` - Temperature and humidity data
- `sigFocusSharpness` - Focus sharpness values during focus mode
- `freeSpaceUpdate` - Storage space updates

### New Signals Added
- `sigUpdateImage` - Real-time image updates (filtered for LepmonCamera)
- `sigLightStateChanged` - Light state change notifications
- `sigLCDDisplayUpdate` - OLED display content updates
- `sigButtonPressed` - Button press events

## Redux State Enhancements

### New State Properties
```javascript
// Light Controls
lightStates: {}           // Current on/off state of each LED
availableLights: []       // List of available LED names

// Hardware Status  
hardwareStatus: {         // Hardware availability status
  gpio_available: false,
  oled_available: false,
  i2c_available: false,
  simulation_mode: true
}

// Images
latestImage: null         // Base64 encoded latest image
imageFormat: "jpeg"       // Current image format

// Display
lcdDisplay: {             // OLED display content
  line1: "", line2: "", line3: "", line4: ""
}

// Timing Configuration
timingConfig: {           // Acquisition timing parameters
  acquisitionInterval: 60,
  stabilizationTime: 5,
  preAcquisitionDelay: 2,
  postAcquisitionDelay: 1
}
```

## User Interface Improvements

### Organized Layout
- **Collapsible Sections**: Accordion-style organization for different feature groups
- **Status Dashboard**: Hardware status with color-coded chips
- **Visual Feedback**: Real-time updates and status indicators

### Professional Controls
- **Grouped Functionality**: Related controls grouped logically
- **Clear Labeling**: Descriptive labels and status indicators
- **Responsive Design**: Adapts to different screen sizes

## Missing Functionality (Future Enhancements)

### 1. Button State Monitoring
- Real-time button state display
- Button simulation interface
- Button event history

### 2. Advanced Sensor Data Display
- Live sensor data dashboard beyond temperature/humidity
- Historical sensor data charts
- Sensor calibration interface

### 3. Configuration Management
- Load/save configuration files
- Configuration templates
- Backup/restore functionality

### 4. LepmonOS Advanced Features
- HMI menu integration
- Advanced capturing modes
- LoRa communication status

## Technical Notes

### Endpoint Naming Consistency
- **Fixed**: Changed `/stopExperiment` to `/stopLepmonExperiment` to match Python backend
- **Consistent**: All endpoint names now match Python controller method names

### Error Handling
- Comprehensive try-catch blocks for all API calls
- User-friendly error messages
- Graceful degradation when hardware unavailable

### Performance Considerations
- Efficient WebSocket message handling
- Minimal re-renders through proper Redux usage
- Lazy loading of images and data

## Testing Recommendations

1. **Hardware Integration Testing**: Test with actual LepMon hardware
2. **Simulation Mode Testing**: Verify all functions work in simulation mode
3. **WebSocket Reliability**: Test real-time updates under various network conditions
4. **UI Responsiveness**: Test on different screen sizes and devices
5. **Error Scenarios**: Test behavior when backend services are unavailable

## Migration Guide

For users upgrading from the previous version:

1. **Redux Store**: The lepmon slice has been significantly expanded
2. **WebSocket Handling**: New signal types are automatically handled
3. **API Compatibility**: All existing functionality remains compatible
4. **UI Changes**: New features are in collapsible sections, existing controls unchanged

## Future Development

### Planned Enhancements
1. Real-time charts for sensor data
2. Image gallery with thumbnail navigation
3. Advanced timing profiles
4. Remote configuration management
5. Multi-device support

### Extension Points
- Plugin architecture for custom sensors
- Configurable dashboard layouts
- Custom LED control sequences
- Automated experiment workflows

This comprehensive update brings the React frontend in line with the enhanced Python LepmonController, providing a professional interface for all hardware control and monitoring capabilities.