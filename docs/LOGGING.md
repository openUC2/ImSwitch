# Logging System Documentation

## Overview

ImSwitch now has an enhanced logging system that supports:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File logging with timestamp-based filenames
- WebSocket-based log streaming to frontend (in headless mode)
- REST API for log file management and download

## Configuration

### Log Level Configuration

You can configure the log level in the `ImSwitchConfig`:

```python
from imswitch.config import get_config, update_config

# Update log level
config = update_config(log_level="DEBUG")

# Or set it when starting ImSwitch
from imswitch.__main__ import main
main(is_headless=True, log_level="DEBUG")
```

Available log levels:
- `DEBUG`: Detailed information, typically of interest only when diagnosing problems
- `INFO`: Confirmation that things are working as expected (default)
- `WARNING`: An indication that something unexpected happened
- `ERROR`: A more serious problem
- `CRITICAL`: A very serious error

### File Logging Configuration

File logging is enabled by default. Log files are created in the `logs` subdirectory of your ImSwitch config folder with timestamps:

```
ImSwitchConfig/
  └── logs/
      ├── imswitch_2025-10-10_06-39-12.log
      ├── imswitch_2025-10-10_07-15-23.log
      └── ...
```

You can configure file logging:

```python
config = update_config(
    log_level="INFO",
    log_to_file=True,  # Enable/disable file logging
    log_folder="/custom/path/to/logs"  # Custom log folder path
)
```

## WebSocket Log Streaming

In headless mode, log messages are automatically streamed via WebSocket to connected clients. This allows frontend applications to display real-time log messages.

The log messages are emitted via the `sigLog` signal with the following structure:

```json
{
  "signal": "sigLog",
  "args": {
    "timestamp": 1696934352.123,
    "level": "INFO",
    "name": "imswitch.component",
    "message": "Log message text",
    "formatted": "2025-10-10 06:39:12 INFO [imswitch.component] Log message text"
  }
}
```

## REST API

### List Log Files

**Endpoint:** `GET /LogController/listLogFiles`

Returns a list of available log files in alphanumerical order.

**Response:**
```json
{
  "log_files": [
    {
      "filename": "imswitch_2025-10-10_06-39-12.log",
      "size": 12345,
      "modified": "2025-10-10T06:45:30",
      "path": "/path/to/logs/imswitch_2025-10-10_06-39-12.log"
    }
  ]
}
```

### Download Log File

**Endpoint:** `GET /LogController/downloadLogFile?filename=<filename>`

Downloads a specific log file.

**Parameters:**
- `filename`: The name of the log file to download (e.g., `imswitch_2025-10-10_06-39-12.log`)

**Response:** The log file content as `text/plain`

## Usage Examples

### Basic Usage

```python
from imswitch.imcommon.model import initLogger

# Create a logger for your component
logger = initLogger('MyComponent')

# Log messages at different levels
logger.debug("Detailed debug information")
logger.info("Normal operation")
logger.warning("Something unexpected happened")
logger.error("An error occurred")
```

### Programmatic Setup

```python
from imswitch.imcommon.model import setup_logging

# Setup logging with custom configuration
setup_logging(
    log_level="DEBUG",
    log_to_file=True,
    log_folder="/var/log/imswitch",
    config_folder=None
)
```

### Enable Signal Emission

```python
from imswitch.imcommon.model import enable_signal_emission
from imswitch.imcommon.framework.noqt import sigLog

# Enable signal emission for log messages
enable_signal_emission(sigLog)

# Now all log messages will be emitted via the sigLog signal
```

## Frontend Integration

Frontend applications can listen for log messages via WebSocket:

```javascript
const socket = new WebSocket("wss://your-server:8002/ws");

socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.signal === "sigLog") {
        const logData = data.args;
        console.log(`[${logData.level}] ${logData.message}`);
        
        // Display in UI
        displayLogMessage(logData.level, logData.formatted);
    }
};
```

You can also fetch and download log files via REST API:

```javascript
// List available log files
fetch('/LogController/listLogFiles')
    .then(response => response.json())
    .then(data => {
        console.log('Available log files:', data.log_files);
    });

// Download a specific log file
const filename = 'imswitch_2025-10-10_06-39-12.log';
window.location.href = `/LogController/downloadLogFile?filename=${filename}`;
```

## Architecture

The logging system consists of several components:

1. **setup_logging()**: Initializes the logging system with file handlers and console handlers
2. **SignalEmittingHandler**: Custom logging handler that emits log messages via signals
3. **sigLog**: Global signal for log messages (in noqt.py)
4. **REST API endpoints**: Provide access to log files via HTTP

## Notes

- Log files are created with timestamps to avoid conflicts
- Old log files are not automatically deleted - you may want to implement log rotation
- Signal emission is only enabled in headless mode by default
- Log file downloads use security measures to prevent directory traversal attacks
