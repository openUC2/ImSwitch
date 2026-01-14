/**
 * Camera Status API Integration Example for microscope-app
 * 
 * This file demonstrates how to integrate the new getCameraStatus endpoint
 * into the React frontend for displaying comprehensive camera information.
 */

import { useState, useEffect } from 'react';

/**
 * Hook to fetch and monitor camera status
 * @param {string} detectorName - Optional detector name, uses current if not provided
 * @param {number} refreshInterval - How often to refresh status in ms (0 = no auto-refresh)
 */
export function useCameraStatus(detectorName = null, refreshInterval = 0) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const url = detectorName 
        ? `/api/SettingsController/getCameraStatus?detectorName=${detectorName}`
        : '/api/SettingsController/getCameraStatus';
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    
    if (refreshInterval > 0) {
      const interval = setInterval(fetchStatus, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [detectorName, refreshInterval]);

  return { status, loading, error, refresh: fetchStatus };
}

/**
 * Component to display camera status information
 */
export function CameraStatusDisplay({ detectorName = null }) {
  const { status, loading, error, refresh } = useCameraStatus(detectorName, 5000); // Refresh every 5s

  if (loading && !status) {
    return <div>Loading camera status...</div>;
  }

  if (error) {
    return (
      <div className="error">
        <h3>Error loading camera status</h3>
        <p>{error}</p>
        <button onClick={refresh}>Retry</button>
      </div>
    );
  }

  if (!status) {
    return <div>No camera status available</div>;
  }

  return (
    <div className="camera-status">
      <h2>Camera Status</h2>
      
      {/* Basic Information */}
      <section className="status-section">
        <h3>Hardware</h3>
        <dl>
          <dt>Model:</dt>
          <dd>{status.model}</dd>
          
          <dt>Type:</dt>
          <dd>{status.cameraType || 'Unknown'}</dd>
          
          <dt>Sensor Size:</dt>
          <dd>{status.sensorWidth} × {status.sensorHeight} pixels</dd>
          
          <dt>Pixel Size:</dt>
          <dd>
            {Array.isArray(status.pixelSizeUm) 
              ? `${status.pixelSizeUm[status.pixelSizeUm.length - 1]} µm`
              : `${status.pixelSizeUm} µm`}
          </dd>
        </dl>
      </section>

      {/* Status Information */}
      <section className="status-section">
        <h3>Status</h3>
        <dl>
          <dt>Connected:</dt>
          <dd className={status.isConnected ? 'status-ok' : 'status-error'}>
            {status.isConnected ? '✓ Connected' : '✗ Disconnected'}
          </dd>
          
          <dt>Acquiring:</dt>
          <dd className={status.isAcquiring ? 'status-active' : 'status-idle'}>
            {status.isAcquiring ? 'Active' : 'Idle'}
          </dd>
          
          <dt>Mock Camera:</dt>
          <dd>{status.isMock ? 'Yes' : 'No'}</dd>
          
          <dt>Color:</dt>
          <dd>{status.isRGB ? 'RGB' : 'Monochrome'}</dd>
        </dl>
      </section>

      {/* Current Settings */}
      <section className="status-section">
        <h3>Current Frame</h3>
        <dl>
          <dt>Size:</dt>
          <dd>{status.currentWidth} × {status.currentHeight} pixels</dd>
          
          <dt>ROI Position:</dt>
          <dd>
            {Array.isArray(status.frameStart)
              ? `(${status.frameStart[0]}, ${status.frameStart[1]})`
              : 'N/A'}
          </dd>
          
          <dt>Binning:</dt>
          <dd>{status.binning}×{status.binning}</dd>
        </dl>
      </section>

      {/* Parameters */}
      <section className="status-section">
        <h3>Camera Parameters</h3>
        {status.parameters && Object.keys(status.parameters).length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Units</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(status.parameters)
                .filter(([name]) => ['exposure', 'gain', 'blacklevel', 'frame_rate'].includes(name))
                .map(([name, param]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{param.value}</td>
                    <td>{param.units || '-'}</td>
                    <td>{param.type}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        ) : (
          <p>No parameters available</p>
        )}
      </section>

      {/* Trigger Information */}
      {status.currentTriggerSource && (
        <section className="status-section">
          <h3>Trigger</h3>
          <dl>
            <dt>Current Source:</dt>
            <dd>{status.currentTriggerSource}</dd>
            
            {status.availableTriggerTypes && (
              <>
                <dt>Available Types:</dt>
                <dd>{status.availableTriggerTypes.join(', ')}</dd>
              </>
            )}
          </dl>
        </section>
      )}

      {/* Refresh Button */}
      <button onClick={refresh} className="refresh-button">
        Refresh Status
      </button>
    </div>
  );
}

/**
 * Compact camera status badge for toolbar/header
 */
export function CameraStatusBadge({ detectorName = null }) {
  const { status, loading } = useCameraStatus(detectorName, 2000); // Refresh every 2s

  if (loading || !status) {
    return <span className="camera-badge loading">⟳</span>;
  }

  const statusClass = status.isConnected 
    ? (status.isAcquiring ? 'acquiring' : 'connected')
    : 'disconnected';

  return (
    <span className={`camera-badge ${statusClass}`} title={`${status.model} - ${status.isConnected ? 'Connected' : 'Disconnected'}`}>
      <span className="indicator">●</span>
      <span className="model">{status.model}</span>
      {status.isAcquiring && <span className="acquiring-indicator">📹</span>}
    </span>
  );
}

/**
 * Example API call function
 */
export async function getCameraStatus(detectorName = null) {
  const url = detectorName 
    ? `/api/SettingsController/getCameraStatus?detectorName=${detectorName}`
    : '/api/SettingsController/getCameraStatus';
  
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch camera status: ${response.status}`);
  }
  
  const data = await response.json();
  
  if (data.error) {
    throw new Error(data.error);
  }
  
  return data;
}

/**
 * Example CSS styles (add to your stylesheet)
 */
const exampleStyles = `
.camera-status {
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
}

.status-section {
  margin: 20px 0;
  padding: 15px;
  background: white;
  border-radius: 4px;
}

.status-section h3 {
  margin-top: 0;
  color: #333;
  border-bottom: 2px solid #007bff;
  padding-bottom: 8px;
}

.status-section dl {
  display: grid;
  grid-template-columns: 150px 1fr;
  gap: 10px;
}

.status-section dt {
  font-weight: 600;
  color: #666;
}

.status-section dd {
  margin: 0;
}

.status-ok {
  color: #28a745;
  font-weight: bold;
}

.status-error {
  color: #dc3545;
  font-weight: bold;
}

.status-active {
  color: #007bff;
  font-weight: bold;
}

.camera-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 12px;
  font-weight: 500;
}

.camera-badge.connected {
  background: #d4edda;
  color: #155724;
}

.camera-badge.acquiring {
  background: #cce5ff;
  color: #004085;
}

.camera-badge.disconnected {
  background: #f8d7da;
  color: #721c24;
}

.camera-badge .indicator {
  font-size: 8px;
}

.camera-badge.connected .indicator {
  color: #28a745;
}

.camera-badge.acquiring .indicator {
  color: #007bff;
  animation: pulse 2s infinite;
}

.camera-badge.disconnected .indicator {
  color: #dc3545;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.refresh-button {
  margin-top: 15px;
  padding: 8px 16px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.refresh-button:hover {
  background: #0056b3;
}
`;

/**
 * Usage Example in a React component
 */
const ExampleUsage = () => {
  return (
    <div>
      {/* Full status display */}
      <CameraStatusDisplay />
      
      {/* Compact badge for toolbar */}
      <header>
        <h1>Microscope Control</h1>
        <CameraStatusBadge />
      </header>
      
      {/* Manual API call */}
      <button onClick={async () => {
        const status = await getCameraStatus();
        console.log('Camera status:', status);
        alert(`Camera: ${status.model}\nConnected: ${status.isConnected}`);
      }}>
        Check Camera
      </button>
    </div>
  );
};
