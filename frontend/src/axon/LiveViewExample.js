import React from 'react';
import { Provider } from 'react-redux';
import { Box, Typography } from '@mui/material';
import store from '../state/store';
import LiveViewControlWrapper from './LiveViewControlWrapper';

/**
 * Example component demonstrating the unified LiveViewControlWrapper
 * 
 * This component shows how to use the enhanced LiveViewControlWrapper that includes:
 * - Frontend intensity scaling (0-255 range)
 * - Scale bar overlay
 * - Position control overlay
 * - Redux state management for image data and scaling parameters
 */
const LiveViewExample = () => {
  return (
    <Provider store={store}>
      <Box sx={{ width: '100%', height: '100vh', p: 2 }}>
        <Typography variant="h4" gutterBottom>
          Unified Live View Component Example
        </Typography>
        
        <Typography variant="body1" sx={{ mb: 2 }}>
          This demonstrates the unified LiveViewControlWrapper component with:
        </Typography>
        
        <Box component="ul" sx={{ mb: 3 }}>
          <li>Frontend intensity scaling using Canvas API (0-255 range)</li>
          <li>Overlay intensity sliders on the right side</li>
          <li>Scale bar showing actual measurements (when pixel size is available)</li>
          <li>Position control overlay for stage movement</li>
          <li>Redux state management for all parameters</li>
        </Box>

        <Typography variant="h6" gutterBottom>
          Live View Component:
        </Typography>
        
        <Box sx={{ width: '800px', height: '600px', border: '1px solid #ccc', borderRadius: 1 }}>
          <LiveViewControlWrapper />
        </Box>

        <Typography variant="body2" sx={{ mt: 2 }}>
          <strong>Note:</strong> The intensity scaling is now performed in the frontend using Canvas API,
          eliminating the need for backend processing. This provides better performance for 8-bit JPEG 
          compressed image streams.
        </Typography>
      </Box>
    </Provider>
  );
};

export default LiveViewExample;