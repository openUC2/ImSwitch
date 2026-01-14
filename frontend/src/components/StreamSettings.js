import React, { useState, useEffect, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Switch,
  FormControlLabel,
  Box,
  Alert,
  Divider,
  Button
} from '@mui/material';
import apiLiveViewControllerGetStreamParameters from '../backendapi/apiLiveViewControllerGetStreamParameters';
import apiLiveViewControllerSetStreamParameters from '../backendapi/apiLiveViewControllerSetStreamParameters';
import apiSettingsControllerSetJpegQuality from '../backendapi/apiSettingsControllerSetJpegQuality';
import * as liveStreamSlice from '../state/slices/LiveStreamSlice.js';

/**
 * Stream Settings Panel - Runtime configuration for binary streaming
 * GET/POST /api/settings/getStreamParams and /api/settings/setStreamParams
 */
const StreamSettings = ({ onOpen }) => {
  const dispatch = useDispatch();
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  
  // Use persistent settings from Redux, with local draft for editing
  const [draftSettings, setDraftSettings] = useState(liveStreamState.streamSettings);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [updatePending, setUpdatePending] = useState(false);
  const [isLegacyBackend, setIsLegacyBackend] = useState(liveStreamState.isLegacyBackend);
  
  // Auto-retry timeout errors
  const [retryCount, setRetryCount] = useState(0);
  const maxRetries = 3;

  // Submit settings to backend and update Redux
  const handleSubmit = useCallback(async () => {
    try {
      setUpdatePending(true);
      setError(null);
      
      // Determine which protocol is enabled
      const isJpegMode = draftSettings.jpeg.enabled;
      const protocol = isJpegMode ? 'jpeg' : 'binary';
      
      // Only try to update binary settings if not in legacy mode and binary is enabled
      if (!isLegacyBackend && draftSettings.binary.enabled) {
        await apiLiveViewControllerSetStreamParameters('binary', {
          compression_algorithm: draftSettings.binary.compression.algorithm,
          compression_level: draftSettings.binary.compression.level,
          subsampling_factor: draftSettings.binary.subsampling.factor,
          throttle_ms: draftSettings.binary.throttle_ms
        });
      }
      
      // Update JPEG quality if JPEG is enabled
      if (draftSettings.jpeg.enabled && draftSettings.jpeg.quality !== undefined) {
        try {
          await apiLiveViewControllerSetStreamParameters('jpeg', {
            jpeg_quality: draftSettings.jpeg.quality,
            subsampling_factor: 1,
            throttle_ms: 100
          });
        } catch (err) {
          console.warn('Failed to set JPEG parameters:', err.message);
        }
      }
      
      // Update Redux with persistent settings
      dispatch(liveStreamSlice.setStreamSettings(draftSettings));
      
      // Update format in Redux
      const currentFormat = draftSettings.binary.enabled ? 'binary' : 'jpeg';
      dispatch(liveStreamSlice.setImageFormat(currentFormat));
      
      console.log('Stream settings submitted successfully');
    } catch (err) {
      setError(`Failed to submit settings: ${err.message}`);
    } finally {
      setUpdatePending(false);
    }
  }, [draftSettings, isLegacyBackend, dispatch]);
  
  // Reset to defaults
  const handleReset = useCallback(() => {
    const defaultSettings = {
      current_compression_algorithm: "jpeg",
      binary: {
        enabled: false,
        compression: { algorithm: "lz4", level: 0 },
        subsampling: { factor: 4 },
        throttle_ms: 100,
        bitdepth_in: 12,
        pixfmt: "GRAY16"
      },
      jpeg: {
        enabled: true,
        quality: 85
      }
    };
    setDraftSettings(defaultSettings);
  }, []);
  
  // Load settings from backend with auto-retry
  const loadSettings = useCallback(async (isRetry = false) => {
    try {
      setLoading(true);
      setError(null);
      
      // Get stream parameters from new LiveViewController endpoint
      const allParams = await apiLiveViewControllerGetStreamParameters();
      console.log('Stream parameters from LiveViewController:', allParams);
      
      // Transform backend response to frontend format
      const mergedSettings = {
        current_compression_algorithm: allParams.binary ? 'binary' : 'jpeg',
        binary: {
          enabled: allParams.binary ? true : false,
          compression: {
            algorithm: allParams.binary?.compression_algorithm || 'lz4',
            level: allParams.binary?.compression_level || 0
          },
          subsampling: { factor: allParams.binary?.subsampling_factor || 4 },
          throttle_ms: allParams.binary?.throttle_ms || 100,
          bitdepth_in: 12,
          pixfmt: "GRAY16"
        },
        jpeg: {
          enabled: allParams.jpeg ? true : false,
          quality: allParams.jpeg?.jpeg_quality || 85
        }
      };
      
      setDraftSettings(mergedSettings);
      dispatch(liveStreamSlice.setStreamSettings(mergedSettings));
      
      // Update format in Redux based on detected or configured stream type
      const currentFormat = mergedSettings.binary.enabled ? 'binary' : 'jpeg';
      dispatch(liveStreamSlice.setImageFormat(currentFormat));
      
      console.log('Stream settings loaded successfully');
      
      // Reset retry count on success
      setRetryCount(0);
      
      // Dispatch backend capabilities
      dispatch(liveStreamSlice.setIsLegacyBackend(false));
      dispatch(liveStreamSlice.setBackendCapabilities({
        binaryStreaming: true,
        webglSupported: true
      }));
      
    } catch (err) {
      console.warn('Failed to load stream settings:', err.message);
      
      // Handle timeout with auto-retry
      if (err.message.includes('timeout') && retryCount < maxRetries && !isRetry) {
        setRetryCount(prev => prev + 1);
        setTimeout(() => loadSettings(true), 2000 * retryCount); // Exponential backoff
        setError(`Connection timeout. Retrying... (${retryCount + 1}/${maxRetries})`);
        return;
      }
      
      // Detect legacy backend
      if (err.message.includes('404') || err.message.includes('Not Found')) {
        console.log('Legacy backend detected');
        setIsLegacyBackend(true);
        dispatch(liveStreamSlice.setIsLegacyBackend(true));
        
        // Use Redux settings or defaults for legacy
        const legacySettings = {
          ...liveStreamState.streamSettings,
          binary: { ...liveStreamState.streamSettings.binary, enabled: false },
          jpeg: { ...liveStreamState.streamSettings.jpeg, enabled: true }
        };
        setDraftSettings(legacySettings);
        
        setError('Legacy backend detected - binary streaming not available');
      } else {
        setError(`Failed to load settings: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  }, [retryCount, liveStreamState.streamSettings, dispatch]);

  // Load initial settings
  useEffect(() => {
    loadSettings();
  }, []);
  

  // Auto-retry on menu open for timeout errors
  useEffect(() => {
    if (onOpen && error && error.includes('timeout')) {
      const handleMenuOpen = () => {
        if (retryCount < maxRetries) {
          loadSettings();
        }
      };
      onOpen(handleMenuOpen);
    }
  }, [onOpen, error, retryCount, loadSettings]);

  // Handle setting changes in draft mode (no auto-submit)
  const handleSettingChange = useCallback((path, value) => {
    // Create a proper deep copy to avoid frozen object issues
    const newDraftSettings = JSON.parse(JSON.stringify(draftSettings));
    
    // Navigate to the nested property and update it
    const keys = path.split('.');
    let current = newDraftSettings;
    for (let i = 0; i < keys.length - 1; i++) {
      current = current[keys[i]];
    }
    current[keys[keys.length - 1]] = value;
    
  }, [draftSettings]);

  if (loading) {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography>Loading stream settings...</Typography>
      </Paper>
    );
  }

  // Handle retry for timeout errors
  const handleRetry = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = await apiLiveViewControllerGetStreamParameters();
      
      // Transform to frontend format
      const transformedSettings = {
        current_compression_algorithm: params.binary ? 'binary' : 'jpeg',
        binary: {
          enabled: params.binary ? true : false,
          compression: {
            algorithm: params.binary?.compression_algorithm || 'lz4',
            level: params.binary?.compression_level || 0
          },
          subsampling: { factor: params.binary?.subsampling_factor || 4 },
          throttle_ms: params.binary?.throttle_ms || 100,
          bitdepth_in: 12,
          pixfmt: "GRAY16"
        },
        jpeg: {
          enabled: params.jpeg ? true : false,
          quality: params.jpeg?.jpeg_quality || 85
        }
      };
      
      setDraftSettings(transformedSettings);
      const initialFormat = transformedSettings.binary?.enabled ? 'binary' : 'jpeg';
      dispatch(liveStreamSlice.setImageFormat(initialFormat));
    } catch (err) {
      setError(`Failed to load settings: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (error && error.includes('timeout')) {
    return (
      <Paper sx={{ p: 2 }}>
        <Alert severity="error" action={
          <Button color="inherit" size="small" onClick={handleRetry}>
            Retry
          </Button>
        }>
          {error}
        </Alert>
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper sx={{ p: 2 }}>
        <Alert severity="error">{error}</Alert>
      </Paper>
    );
  }

  return (
    <Paper sx={{ 
      p: 2, 
      minWidth: 300,
      maxWidth: 400,
      maxHeight: '80vh',
      overflowY: 'auto',
      overflowX: 'hidden',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <Typography variant="h6" gutterBottom>
        Stream Settings
        {updatePending && (
          <Typography component="span" sx={{ ml: 1, fontSize: '0.8em', color: 'text.secondary' }}>
            (submitting...)
          </Typography>
        )}
        
      </Typography>
      
      {/* Submit/Reset Buttons */}
      <Box sx={{ mb: 2, display: 'flex', gap: 1 }}>
        <Button 
          variant="contained" 
          onClick={handleSubmit}
          size="small"
        >
          Submit Settings
        </Button>
        <Button 
          variant="outlined" 
          onClick={handleReset}
          size="small"
        >
          Reset to Defaults
        </Button>
      </Box>
      
      {/* Current Status */}
      <Box sx={{ mb: 2, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
        <Typography variant="body2" fontWeight="bold">
          Current: Format={liveStreamState.imageFormat?.toUpperCase() || 'UNKNOWN'}, Range: {liveStreamState.minVal}–{liveStreamState.maxVal}
        </Typography>
      </Box>
      
      {/* Legacy Backend Warning */}
      {isLegacyBackend && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="body2" fontWeight="bold">Legacy Backend Detected</Typography>
          <Typography variant="body2">
            Your backend doesn't support binary streaming. Only JPEG streaming is available.
            Consider updating your backend for better performance.
          </Typography>
        </Alert>
      )}
      
      {/* Stream Format Selector */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Stream Format
        </Typography>
        
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Stream Type</InputLabel>
          <Select
            value={draftSettings.binary.enabled ? 'binary' : 'jpeg'}
            label="Stream Type"
            onChange={(e) => {
              const isBinary = e.target.value === 'binary';
              handleSettingChange('binary.enabled', isBinary);
              handleSettingChange('jpeg.enabled', !isBinary);
              
              // Update compression algorithm in draft
              const newAlgorithm = isBinary ? 'binary' : 'jpeg';
              handleSettingChange('current_compression_algorithm', newAlgorithm);
            }}
            disabled={isLegacyBackend}
          >
            <MenuItem value="binary">Binary (16-bit) - High Quality</MenuItem>
            <MenuItem value="jpeg">JPEG (8-bit) - Legacy</MenuItem>
          </Select>
        </FormControl>
      </Box>
      
      {/* Binary Stream Settings */}
      {draftSettings.binary.enabled && (
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Binary Stream Settings
        </Typography>
        
          <Box sx={{ ml: 2 }}>
            {/* Compression Settings */}
            <Box sx={{ mb: 2 }}>
              <FormControl fullWidth sx={{ mb: 1 }}>
                <InputLabel>Compression Algorithm</InputLabel>
                <Select
                  value={draftSettings.binary.compression.algorithm}
                  label="Compression Algorithm"
                  onChange={(e) => handleSettingChange('binary.compression.algorithm', e.target.value)}
                >
                  <MenuItem value="none">None</MenuItem>
                  <MenuItem value="lz4">LZ4</MenuItem>
                  <MenuItem value="zstd">Zstandard</MenuItem>
                </Select>
              </FormControl>
              
              {draftSettings.binary.compression.algorithm !== 'none' && (
                <TextField
                  fullWidth
                  type="number"
                  label="Compression Level"
                  value={draftSettings.binary.compression.level}
                  onChange={(e) => handleSettingChange('binary.compression.level', parseInt(e.target.value) || 0)}
                  inputProps={{ min: 0, max: 9 }}
                  helperText="0 = fastest, 9 = best compression"
                />
              )}
            </Box>
            
            {/* Subsampling Settings */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>Subsampling</Typography>
              
              <TextField
                fullWidth
                type="number"
                label="Subsampling Factor"
                value={draftSettings.binary.subsampling.factor}
                onChange={(e) => handleSettingChange('binary.subsampling.factor', parseInt(e.target.value) || 1)}
                inputProps={{ min: 1, max: 4 }}
                helperText="1 = full resolution, 2 = half, etc."
                sx={{ mb: 1 }}
              />
            </Box>
            
            {/* Throttle Settings */}
            <TextField
              fullWidth
              type="number"
              label="Throttle (ms)"
              value={draftSettings.binary.throttle_ms}
              onChange={(e) => handleSettingChange('binary.throttle_ms', parseInt(e.target.value) || 50)}
              inputProps={{ min: 1, max: 1000 }}
              helperText="Minimum time between frames"
              sx={{ mb: 2 }}
            />
            
            {/* Read-only info */}
            <Box sx={{  p: 1, borderRadius: 1 }}>
              <Typography variant="body2">
                Bit Depth: {draftSettings.binary.bitdepth_in}-bit
              </Typography>
              <Typography variant="body2">
                Pixel Format: {draftSettings.binary.pixfmt}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic', color: 'text.secondary' }}>
                Note: RGB binary streaming requires backend support
              </Typography>
            </Box>
          </Box>
      </Box>
      )}
      
      {/* JPEG Stream Settings */}
      {draftSettings.jpeg.enabled && (
      <Box sx={{ mb: 2 }}>
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="subtitle1" gutterBottom>
          JPEG Stream Settings
        </Typography>
        
        <Alert severity="info" sx={{ mb: 2 }}>
          JPEG streaming provides 8-bit images. For scientific imaging with better dynamic range, 
          consider using binary streaming if your backend supports it.
        </Alert>
        
        <Box sx={{ ml: 2 }}>
          <TextField
            fullWidth
            type="number"
            label="Compression Quality"
            value={draftSettings.jpeg.quality}
            onChange={(e) => handleSettingChange('jpeg.quality', parseInt(e.target.value) || 85)}
            inputProps={{ min: 1, max: 100 }}
            helperText="1 = lowest quality/size, 100 = highest quality/size"
            sx={{ mb: 1 }}
          />
        </Box>
      </Box>
      )}
      
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Paper>
  );
};

// Simple debounce utility
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

export default StreamSettings;