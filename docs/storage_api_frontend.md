# Storage Management API - Frontend Integration Guide

This document provides a quick reference for frontend developers integrating the storage management API.

## Quick Start

The storage management API provides 5 endpoints for managing data storage paths:

```javascript
// Base URL
const API_BASE = 'http://localhost:8001';

// Get current storage status
const status = await fetch(`${API_BASE}/api/storage/status`).then(r => r.json());

// List external drives  
const drives = await fetch(`${API_BASE}/api/storage/external-drives`).then(r => r.json());

// Switch to external drive
await fetch(`${API_BASE}/api/storage/set-active-path`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ path: '/media/usb-drive', persist: true })
});
```

## API Reference

### GET /api/storage/status

Get comprehensive storage information.

**Response**:
```typescript
interface StorageStatus {
  active_path: string;
  fallback_path: string | null;
  available_external_drives: ExternalDrive[];
  scan_enabled: boolean;
  mount_paths: string[];
  free_space_gb: number;
  total_space_gb: number;
  percent_used: number;
}
```

### GET /api/storage/external-drives

List all detected external storage devices.

**Response**:
```typescript
interface ExternalDrivesResponse {
  drives: ExternalDrive[];
}

interface ExternalDrive {
  path: string;
  label: string;
  writable: boolean;
  free_space_gb: number;
  total_space_gb: number;
  filesystem: string;
  is_active: boolean;
}
```

### POST /api/storage/set-active-path

Set the active storage path.

**Request**:
```typescript
interface SetActivePathRequest {
  path: string;
  persist: boolean;  // Save preference for next session
}
```

**Response**:
```typescript
interface SetActivePathResponse {
  success: boolean;
  active_path: string;
  persisted: boolean;
  message: string;
}
```

**Error Response (400)**:
```typescript
interface ErrorResponse {
  detail: string;  // Human-readable error message
}
```

### GET /api/storage/config-paths

Get all configuration-related paths.

**Response**:
```typescript
interface ConfigPaths {
  config_path: string;
  data_path: string;
  active_data_path: string;
}
```

### POST /api/storage/update-config-path

Update configuration paths (advanced usage).

**Request**:
```typescript
interface UpdateConfigPathRequest {
  config_path?: string;  // Optional
  data_path?: string;    // Optional
  persist: boolean;
}
```

**Response**:
```typescript
interface UpdateConfigPathResponse {
  success: boolean;
  message: string;
  config_path: string;
  data_path: string;
  active_data_path: string;
}
```

## Example Implementation

### React Component (TypeScript)

```typescript
import React, { useState, useEffect } from 'react';

interface ExternalDrive {
  path: string;
  label: string;
  writable: boolean;
  free_space_gb: number;
  total_space_gb: number;
  is_active: boolean;
}

function StorageManager() {
  const [drives, setDrives] = useState<ExternalDrive[]>([]);
  const [activePath, setActivePath] = useState<string>('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadDrives();
    loadStatus();
  }, []);

  const loadDrives = async () => {
    const response = await fetch('/api/storage/external-drives');
    const data = await response.json();
    setDrives(data.drives);
  };

  const loadStatus = async () => {
    const response = await fetch('/api/storage/status');
    const data = await response.json();
    setActivePath(data.active_path);
  };

  const switchDrive = async (path: string) => {
    setLoading(true);
    try {
      const response = await fetch('/api/storage/set-active-path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, persist: true })
      });
      
      const data = await response.json();
      if (data.success) {
        setActivePath(data.active_path);
        await loadDrives(); // Refresh drive list
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Storage Management</h2>
      <p>Active Path: {activePath}</p>
      
      <h3>Available Drives</h3>
      {drives.map(drive => (
        <div key={drive.path}>
          <strong>{drive.label}</strong>
          <p>
            {drive.free_space_gb.toFixed(1)} GB free of {drive.total_space_gb.toFixed(1)} GB
          </p>
          {drive.is_active ? (
            <span>✓ Active</span>
          ) : (
            <button 
              onClick={() => switchDrive(drive.path)}
              disabled={loading || !drive.writable}
            >
              Switch to this drive
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

export default StorageManager;
```

### Redux Integration

```typescript
// storageSlice.ts
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

interface StorageState {
  drives: ExternalDrive[];
  activePath: string;
  status: 'idle' | 'loading' | 'succeeded' | 'failed';
}

export const fetchDrives = createAsyncThunk(
  'storage/fetchDrives',
  async () => {
    const response = await fetch('/api/storage/external-drives');
    return response.json();
  }
);

export const setActivePath = createAsyncThunk(
  'storage/setActivePath',
  async ({ path, persist }: { path: string; persist: boolean }) => {
    const response = await fetch('/api/storage/set-active-path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, persist })
    });
    return response.json();
  }
);

const storageSlice = createSlice({
  name: 'storage',
  initialState: {
    drives: [],
    activePath: '',
    status: 'idle'
  } as StorageState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchDrives.fulfilled, (state, action) => {
        state.drives = action.payload.drives;
        state.status = 'succeeded';
      })
      .addCase(setActivePath.fulfilled, (state, action) => {
        state.activePath = action.payload.active_path;
      });
  }
});

export default storageSlice.reducer;
```

## User Workflow

### Recommended UX Flow

1. **Initial State**: Show current active storage path
2. **Scan for Drives**: Call `/api/storage/external-drives` on mount
3. **Display Options**: Show list of available drives with capacity
4. **User Selection**: Allow user to click on a drive to switch
5. **Confirmation**: Show success/error message after switching
6. **Persist**: Always use `persist: true` to remember user preference

### UI Components to Consider

1. **Storage Status Badge**: Show current drive and free space
2. **Drive List**: Table/list of available drives
3. **Switch Button**: Per-drive action to switch storage
4. **Capacity Bar**: Visual indicator of used/free space
5. **Notification Toast**: Success/error feedback

### Edge Cases to Handle

1. **No External Drives**: Show message "No external drives detected"
2. **Non-Writable Drive**: Disable switch button, show warning
3. **Drive Removed**: Handle gracefully, system auto-falls back
4. **Insufficient Space**: Show warning if drive is almost full
5. **API Errors**: Display user-friendly error messages

## Polling Strategy

For real-time updates, poll the status endpoint:

```typescript
useEffect(() => {
  const interval = setInterval(() => {
    fetch('/api/storage/status')
      .then(r => r.json())
      .then(updateStatus);
  }, 10000); // Poll every 10 seconds
  
  return () => clearInterval(interval);
}, []);
```

**Note**: Future versions may support WebSocket for real-time updates.

## Error Handling

```typescript
async function switchDrive(path: string) {
  try {
    const response = await fetch('/api/storage/set-active-path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, persist: true })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to switch drive');
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Storage switch failed:', error);
    // Show error toast to user
    showErrorToast(error.message);
    throw error;
  }
}
```

## Testing

### Mock API Responses

```typescript
// For testing without backend
const mockDrives = {
  drives: [
    {
      path: '/media/usb-drive-1',
      label: 'USB_DRIVE',
      writable: true,
      free_space_gb: 128.5,
      total_space_gb: 256.0,
      filesystem: 'ext4',
      is_active: true
    }
  ]
};

// Use in tests
jest.mock('fetch');
global.fetch = jest.fn(() =>
  Promise.resolve({
    json: () => Promise.resolve(mockDrives)
  })
);
```

## Browser Compatibility

All endpoints use standard Fetch API, compatible with:
- Chrome/Edge 42+
- Firefox 39+
- Safari 10.1+
- All modern browsers

For older browsers, include a fetch polyfill.

## Security Considerations

1. **Path Validation**: All paths are validated server-side
2. **CORS**: Configured to allow requests from trusted origins
3. **Authentication**: Use existing ImSwitch authentication
4. **Input Sanitization**: Handled by FastAPI/Pydantic

## Support

For issues or questions:
- Check `/docs` endpoint for OpenAPI documentation
- See `docs/storage_management.md` for detailed documentation
- Contact backend team for API changes

## Future Features

Features planned for future releases:

1. **WebSocket Support**: Real-time storage change notifications
2. **Multiple Drives**: Support for switching between multiple drives
3. **Auto-Switch**: Automatically use external drive when available
4. **Space Warnings**: Notifications when storage is low
5. **Drive Health**: SMART status and health indicators

---

Last Updated: 2024-11-12
Version: 1.0
