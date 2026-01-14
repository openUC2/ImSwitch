import React from 'react';
import { useSelector } from 'react-redux';
import { getLiveStreamState } from '../state/slices/LiveStreamSlice';
import { Box, Typography } from '@mui/material';

/**
 * StreamStatsOverlay - Display streaming performance statistics
 * 
 * Shows real-time metrics about the video stream including:
 * - Frame latency (current and average)
 * - Frame count
 * - FPS, bandwidth, compression ratio
 * 
 * Usage: Place this component over your video viewer
 * <Box position="relative">
 *   <YourVideoViewer />
 *   <StreamStatsOverlay />
 * </Box>
 */
const StreamStatsOverlay = ({ show = false }) => {
  const liveStreamState = useSelector(getLiveStreamState);
  const { stats } = liveStreamState;

  if (!show) return null;

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 10,
        right: 10,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '8px 12px',
        borderRadius: '4px',
        fontFamily: 'monospace',
        fontSize: '12px',
        zIndex: 1000,
        pointerEvents: 'none', // Don't block mouse events
        minWidth: '200px'
      }}
    >
      <Typography variant="caption" component="div" sx={{ fontWeight: 'bold', mb: 0.5 }}>
        Stream Stats
      </Typography>
      
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '4px 8px' }}>
        <span>Latency:</span>
        <span style={{ 
          color: stats.latency_ms > 200 ? '#ff6b6b' : 
                 stats.latency_ms > 100 ? '#ffd43b' : '#51cf66' 
        }}>
          {stats.latency_ms.toFixed(0)}ms
        </span>
        
        <span>Avg Latency:</span>
        <span>{stats.avg_latency_ms.toFixed(0)}ms</span>
        
        <span>Frames:</span>
        <span>{stats.frameCount}</span>
        
        {stats.fps > 0 && (
          <>
            <span>FPS:</span>
            <span>{stats.fps.toFixed(1)}</span>
          </>
        )}
        
        {stats.bps > 0 && (
          <>
            <span>Bandwidth:</span>
            <span>{(stats.bps / 1000000).toFixed(2)} Mbps</span>
          </>
        )}
        
        {stats.compressionRatio > 0 && (
          <>
            <span>Compression:</span>
            <span>{stats.compressionRatio.toFixed(1)}x</span>
          </>
        )}
      </Box>
    </Box>
  );
};

export default StreamStatsOverlay;
