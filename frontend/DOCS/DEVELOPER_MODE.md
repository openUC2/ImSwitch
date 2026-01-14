# ImSwitch Developer Mode 🔧

## Overview

Developer Mode is a hidden backdoor that enables access to all ImSwitch backend-dependent settings and configuration pages, even when the backend is offline or unreachable. This is designed for troubleshooting, development, and emergency configuration scenarios.

## Activation Methods

### Method 1: Konami Code (Classic)

Press the following key sequence anywhere in the application:

```
↑ ↑ ↓ ↓ ← → ← → B A
```

(Arrow keys: Up, Up, Down, Down, Left, Right, Left, Right, then B, A)

### Method 2: Quick Developer Access

Hold `Ctrl + Shift` and press `D E V` in sequence.

## Visual Indicators

When Developer Mode is active, you'll see:

1. **Settings Icon Glow**: The settings gear icon in the top toolbar will have an orange glow
2. **Badge Color**: Settings badge changes to orange/warning color
3. **Tooltip Update**: Hover tooltip shows "🔧 Developer Mode"
4. **Menu Item**: A "Developer Mode" option appears in the settings menu to deactivate
5. **Console Message**: Developer activation logged to browser console

## Available Features

With Developer Mode active, the following normally-disabled features become accessible:

### Settings Menu Access

- **System Settings**: Hardware configuration interface
- **ImSwitch Backend Settings**: Microscope configuration
- **WiFi Configuration**: Network management interface
- **System Updates**: Firmware and system update tools

### Menu Item Indicators

Backend-dependent menu items show "(Developer Mode)" in their secondary text when accessed without backend connection.

## Security & Limitations

### Security

- **Hidden by Design**: No UI hints exist for activation methods
- **Session-Only**: Developer mode resets on page reload
- **Local Storage**: Persisted only in browser localStorage, not server-side
- **Development Logging**: Activation instructions only shown in development builds

### Limitations

- **UI Access Only**: Provides interface access, but API calls may still fail
- **No Backend Simulation**: Does not mock or simulate backend responses
- **Error Handling**: Individual components must handle offline/error states gracefully
- **Feature Dependent**: Some features may be non-functional without actual backend

## Developer Instructions

### Activation Check

```javascript
import { useDeveloperMode } from "../utils/useDeveloperMode";

const { isDeveloperMode, activateDeveloperMode, deactivateDeveloperMode } =
  useDeveloperMode();
```

### Conditional Access Logic

```javascript
// Get backend connection status
const isBackendConnected = useSelector((state) => state.uc2.backendConnected);

// Developer override: Allow access when backend offline
const allowAccess = isBackendConnected || isDeveloperMode;

// Use in component logic
<MenuItem disabled={!allowAccess}>
  <ListItemText
    primary="System Settings"
    secondary={
      allowAccess
        ? isDeveloperMode && !isBackendConnected
          ? "Hardware configuration (Developer Mode)"
          : "Hardware configuration"
        : "Requires backend connection"
    }
  />
</MenuItem>;
```

### Console Instructions (Development Only)

In development mode, console will show activation instructions:

```
🔧 ImSwitch Developer Backdoor Available:
  • Konami Code: ↑↑↓↓←→←→BA
  • Quick Access: Ctrl+Shift+D+E+V
  • Enables settings access when backend offline
```

## Use Cases

### 1. Backend Connection Issues

When backend is unreachable but you need to check/modify connection settings or troubleshoot configuration.

### 2. Development & Testing

Test UI components and flows without requiring a running backend server.

### 3. Emergency Configuration

Access critical settings during system recovery or when backend services are down.

### 4. Demo & Training

Show interface elements and navigation flows without full system setup.

## Implementation Details

### Hook Location

`src/utils/useDeveloperMode.js`

### Key Components

- **Keyboard Event Listeners**: Global window event listeners for key sequences
- **localStorage Persistence**: Maintains state across page navigation (until reload)
- **Memory Management**: Limits sequence history to prevent memory leaks
- **State Management**: React hooks with proper cleanup

### Integration Points

- **SettingsMenu.jsx**: Primary integration point with visual indicators
- **Connection Components**: Override access checks with developer mode
- **Individual Settings Pages**: Should handle offline states gracefully

### Testing

Comprehensive test suite in `src/__tests__/useDeveloperMode.test.js` covering:

- Keyboard sequence detection
- State management
- localStorage integration
- Memory leak prevention
- Development vs production behavior

## Troubleshooting

### Developer Mode Not Activating

1. Ensure no modifier keys (Ctrl, Alt, Shift, Cmd) are pressed during Konami code
2. For Ctrl+Shift+DEV, hold modifiers throughout entire sequence
3. Check browser console for error messages
4. Verify key event propagation isn't blocked by other components

### Visual Indicators Not Showing

1. Check localStorage: `localStorage.getItem('imswitch_developer_mode')`
2. Refresh page to reset state
3. Verify SettingsMenu component is rendered and receiving hook state

### Features Still Disabled

1. Developer mode provides UI access only
2. Individual components may have additional offline handling
3. Check component implementation for `allowAccess` pattern usage
4. API calls will still fail without actual backend

---

**⚠️ Important**: This is a developer-only feature. Do not document activation methods in user-facing documentation or provide hints in production UI.
