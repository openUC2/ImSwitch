# CAN OTA Update Frontend Implementation Summary

## Overview
This implementation provides a complete frontend wizard for Over-The-Air (OTA) firmware updates for CAN-connected devices in the microscope-app React application. The system follows the established patterns in the ImSwitch-ReactAPP architecture with Redux state management, WebSocket communication, and modular API functions.

## Architecture Components

### 1. Redux State Management (`src/state/slices/canOtaSlice.js`)
**Purpose:** Centralized state management for the entire OTA workflow

**Key State Properties:**
- **Wizard Navigation:** `currentStep`, `isWizardOpen` (6-step wizard flow)
- **WiFi Configuration:** `wifiSsid`, `wifiPassword`, default credentials
- **Firmware Server:** `firmwareServerUrl`, `availableFirmware` list
- **Device Management:** `scannedDevices`, `selectedDeviceIds`
- **Update Progress:** `updateProgress` (per-device status), progress counters
- **Loading States:** Individual flags for async operations

**Registered in:** `src/state/store.js` as `canOtaState`

### 2. Backend API Functions (`src/backendapi/`)
All API calls follow the existing pattern using `createAxiosInstance()`:

- `apiUC2ConfigControllerGetOTAWiFiCredentials.js` - Fetch current WiFi credentials
- `apiUC2ConfigControllerSetOTAWiFiCredentials.js` - Set WiFi SSID/password
- `apiUC2ConfigControllerGetOTAFirmwareServer.js` - Get firmware server URL
- `apiUC2ConfigControllerSetOTAFirmwareServer.js` - Configure firmware server
- `apiUC2ConfigControllerListAvailableFirmware.js` - List firmware files from server
- `apiUC2ConfigControllerScanCanbus.js` - Scan for CAN devices
- `apiUC2ConfigControllerStartSingleDeviceOTA.js` - Start OTA for one device
- `apiUC2ConfigControllerStartMultipleDeviceOTA.js` - Start OTA for multiple devices
- `apiUC2ConfigControllerGetOTAStatus.js` - Query OTA status
- `apiUC2ConfigControllerClearOTAStatus.js` - Clear status tracking
- `apiUC2ConfigControllerGetOTADeviceMapping.js` - Get CAN ID to device type mapping

### 3. Main Wizard Component (`src/components/CanOtaWizard.jsx`)
**Purpose:** Multi-step wizard dialog for guiding users through the OTA process

**Wizard Steps:**
1. **WiFi Setup (Step 0)**
   - Load and display default WiFi credentials from backend
   - Allow user to modify SSID/password
   - Save credentials before proceeding

2. **Firmware Server (Step 1)**
   - Configure firmware server URL
   - Validate server connectivity
   - Display available firmware files from server
   - Fail early if server is unreachable

3. **Scan Devices (Step 2)**
   - Trigger CAN bus scan (5-second timeout)
   - Display discovered devices with CAN ID, type, and status
   - Show device type mapping (e.g., "MOTOR_X (CAN ID: 11)")

4. **Select Devices (Step 3)**
   - Display scanned devices with checkboxes
   - Show firmware availability status per device
   - "Select All" / "Deselect All" options
   - Validate at least one device is selected

5. **Update Progress (Step 4)**
   - Real-time progress tracking via WebSocket
   - Per-device status (initiated, in_progress, completed, failed)
   - Progress bars and status chips
   - Overall completion counter

6. **Completion (Step 5)**
   - Summary of successful/failed updates
   - Option to start another OTA cycle
   - Finish and close wizard

**Key Features:**
- Material-UI components for consistent styling
- Loading states for all async operations
- Error handling with user-friendly alerts
- Device type name resolution (master, motor, laser, led)

### 4. WebSocket Integration (`src/middleware/WebSocketHandler.js`)
**Purpose:** Real-time OTA status updates from Python backend

**Added Handler:**
```javascript
else if (dataJson.name === "sigOTAStatusUpdate") {
  // Process OTA status: { canId, status, statusMsg, ip, hostname, success }
  // Update Redux state with progress
  // Auto-detect completion when all devices are done
}
```

**Status Mapping:**
- `status: 0` → "completed" (100% progress)
- `status: 1` → "wifi_failed" (0% progress)
- `status: 2` → "ota_failed" (50% progress)

**Integration:** Dispatches `canOtaSlice.setUpdateProgress()` to update Redux state in real-time

### 5. Integration with UC2ConfigurationController (`src/components/UC2ConfigurationController.jsx`)
**Added:**
- Import `CanOtaWizard` component
- State: `showCanOtaWizard` (boolean)
- New card in "Configuration Setup" tab:
  - Title: "CAN Device Firmware Update"
  - Description of OTA functionality
  - "Launch CAN OTA Wizard" button
  - Disabled when UC2 not connected
  - Warning alert when disconnected

**Location:** First tab, after the main configuration setup card

## Backend Enhancements (Python)

### Modified Methods in `UC2ConfigController.py`

#### `startSingleDeviceOTA()`
**Enhancement:** Download firmware BEFORE initiating OTA (fail early pattern)
```python
# 1. Download firmware first
firmware_path = self._download_firmware_for_device(can_id)
if firmware_path is None:
    return error ("No firmware available")

# 2. Then start OTA command
response = ESP32.canota.start_ota_update(...)
```

**Benefits:**
- Fails immediately if firmware is missing
- Prevents devices from connecting to WiFi unnecessarily
- Better user experience with earlier error detection

#### `startMultipleDeviceOTA()`
**Enhancement:** Pre-download ALL firmware before starting ANY OTA
```python
# 1. Download all firmware files first
for can_id in can_ids:
    firmware_path = _download_firmware_for_device(can_id)
    if firmware_path is None:
        missing_firmware.append(can_id)

# 2. Fail if ANY firmware is missing
if missing_firmware:
    return error with list of missing devices

# 3. Start all OTA updates sequentially
for can_id in can_ids:
    startSingleDeviceOTA(can_id, ...)
```

**Benefits:**
- Atomic operation: either all devices can be updated or none
- Clear error reporting of which devices lack firmware
- No partial update failures due to missing firmware

## User Flow

1. **User opens UC2ConfigurationController**
2. **Clicks "Launch CAN OTA Wizard"** (only enabled when UC2 connected)
3. **Step 1: WiFi Setup**
   - Default credentials pre-loaded from backend
   - User confirms or modifies SSID/password
   - Credentials saved to backend
4. **Step 2: Firmware Server**
   - Default server URL loaded (e.g., http://localhost:9000)
   - User confirms or changes URL
   - System validates server and lists available firmware
5. **Step 3: Scan Devices**
   - User clicks "Scan CAN Bus"
   - System discovers devices (e.g., Motor X=11, Motor Y=12, Laser=20)
   - Displays device types and status
6. **Step 4: Select Devices**
   - User selects which devices to update
   - System shows which devices have matching firmware
   - "Select All" option available
7. **Step 5: Update Progress**
   - System downloads ALL firmware files first (fail early)
   - If any firmware missing: immediate error, no devices started
   - If all firmware ready: sequential OTA updates begin
   - WebSocket provides real-time status updates
   - Progress bars show per-device status
8. **Step 6: Completion**
   - Summary of results (success/failure counts)
   - Option to update more devices or finish

## Error Handling

### Frontend
- WiFi credentials validation (both fields required)
- Firmware server connectivity check
- At least one device must be selected
- User-friendly error alerts at each step
- Loading spinners during async operations

### Backend
- Early firmware download validation
- Missing firmware detection before starting OTA
- Detailed error messages with device IDs
- Status tracking for debugging

## Testing Checklist

- [ ] Redux store properly configured with `canOtaState`
- [ ] WiFi credentials load from backend
- [ ] WiFi credentials save to backend
- [ ] Firmware server validation works
- [ ] Firmware list displays correctly
- [ ] CAN bus scan discovers devices
- [ ] Device selection updates Redux state
- [ ] Multiple device OTA pre-downloads all firmware
- [ ] WebSocket receives `sigOTAStatusUpdate` events
- [ ] Progress bars update in real-time
- [ ] Completion step shows accurate results
- [ ] "Update More Devices" button resets wizard
- [ ] UC2 connection check enables/disables wizard button

## File Manifest

### Frontend Files Created/Modified
```
src/state/slices/canOtaSlice.js                                    [NEW]
src/state/store.js                                                 [MODIFIED]
src/backendapi/apiUC2ConfigControllerGetOTAWiFiCredentials.js      [NEW]
src/backendapi/apiUC2ConfigControllerSetOTAWiFiCredentials.js      [NEW]
src/backendapi/apiUC2ConfigControllerGetOTAFirmwareServer.js       [NEW]
src/backendapi/apiUC2ConfigControllerSetOTAFirmwareServer.js       [NEW]
src/backendapi/apiUC2ConfigControllerListAvailableFirmware.js      [NEW]
src/backendapi/apiUC2ConfigControllerScanCanbus.js                 [NEW]
src/backendapi/apiUC2ConfigControllerStartSingleDeviceOTA.js       [NEW]
src/backendapi/apiUC2ConfigControllerStartMultipleDeviceOTA.js     [NEW]
src/backendapi/apiUC2ConfigControllerGetOTAStatus.js               [NEW]
src/backendapi/apiUC2ConfigControllerClearOTAStatus.js             [NEW]
src/backendapi/apiUC2ConfigControllerGetOTADeviceMapping.js        [NEW]
src/components/CanOtaWizard.jsx                                    [NEW]
src/components/UC2ConfigurationController.jsx                      [MODIFIED]
src/middleware/WebSocketHandler.js                                 [MODIFIED]
```

### Backend Files Modified
```
ImSwitch/imswitch/imcontrol/controller/controllers/UC2ConfigController.py  [MODIFIED]
  - startSingleDeviceOTA() - Added early firmware download
  - startMultipleDeviceOTA() - Added bulk firmware pre-download
```

## Design Patterns Used

1. **Redux Slice Pattern:** All state in one dedicated slice with clear actions
2. **API Module Pattern:** One file per endpoint following existing conventions
3. **Wizard Pattern:** Multi-step form with validation and progress tracking
4. **WebSocket Event Pattern:** Real-time updates via socket signals
5. **Fail Early Pattern:** Validate all requirements before starting long operations
6. **Atomic Operations:** All-or-nothing approach for multi-device updates

## Next Steps / Future Enhancements

1. **Retry Logic:** Automatic retry for failed devices
2. **Firmware Version Display:** Show current vs. available firmware versions
3. **Batch Scheduling:** Schedule OTA updates for off-peak hours
4. **Rollback Support:** Keep previous firmware for emergency rollback
5. **Progress Persistence:** Save OTA status to survive page refresh
6. **Detailed Logging:** Download OTA logs for troubleshooting
7. **Device Grouping:** Update all motors, all lasers, etc. with one click

## Dependencies

- React with Hooks
- Redux Toolkit (`@reduxjs/toolkit`)
- Material-UI (`@mui/material`, `@mui/icons-material`)
- Socket.IO Client (`socket.io-client`)
- Axios (via `createAxiosInstance`)
- MessagePack (`@msgpack/msgpack`) for WebSocket communication

## Notes

- All components follow English comments as per project guidelines
- Material-UI theming consistent with existing components
- WebSocket handler integrated into existing `WebSocketHandler.js`
- Backend Python methods maintain existing API export patterns
- No breaking changes to existing functionality
