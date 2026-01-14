# Histogram Display Implementation

This document describes the implementation of histogram display functionality in the ImSwitch React application.

## Overview

The histogram feature allows users to visualize the intensity distribution of the camera stream in real-time. When enabled, it displays as an overlay on the live view image.

## Implementation Details

### 1. State Management (LiveStreamSlice.js)

Added new state fields to manage histogram data:
- `histogramX`: Array of intensity units/bins
- `histogramY`: Array of histogram values  
- `showHistogram`: Boolean to control histogram visibility

New actions:
- `setHistogramData({ x, y })`: Updates histogram data from backend
- `setShowHistogram(boolean)`: Controls histogram display

### 2. WebSocket Signal Handling (WebSocketHandler.js)

Added processing for `sigHistogramComputed` signal:
```javascript
if (dataJson.name == "sigHistogramComputed") {
  if (dataJson.args && dataJson.args.p0 && dataJson.args.p1) {
    dispatch(liveStreamSlice.setHistogramData({
      x: dataJson.args.p0, // units
      y: dataJson.args.p1  // hist
    }));
  }
}
```

Expected signal format from backend:
```javascript
{
  name: "sigHistogramComputed",
  args: {
    p0: [0, 25, 50, ...], // intensity units/bins
    p1: [10, 15, 20, ...] // histogram counts
  }
}
```

### 3. Display Component (LiveViewComponent.js)

Enhanced the live view component with histogram overlay:
- Uses Chart.js Bar component for histogram visualization
- Positioned as absolute overlay (top-left corner)
- Semi-transparent background for better visibility
- Only displays when `showHistogram` is true and data is available

### 4. User Controls (LiveView.js)

Added checkbox control for histogram display:
- Appears only when histogram functionality is available (`histogramActive`)
- Toggles `showHistogram` state in Redux store
- Integrates with existing live view controls

## Usage

1. **Backend Setup**: Ensure backend emits `sigHistogramComputed` signals with histogram data
2. **Enable Histogram**: Check "Show Histogram Overlay" checkbox in live view
3. **View Histogram**: Real-time histogram appears as overlay on camera stream

## Technical Notes

- Uses existing Chart.js infrastructure for consistent styling
- Minimal changes to existing codebase - reuses components where possible
- Follows Redux patterns for state management
- Handles edge cases (empty data, missing signals)

## Testing

Created comprehensive tests in `histogram.test.js`:
- Redux state management
- Action creators
- Signal data format validation
- State reset functionality

Run tests with:
```bash
npx react-scripts test --testPathPattern=histogram.test.js --watchAll=false
```