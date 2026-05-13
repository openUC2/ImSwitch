import React, { useState } from 'react';
import { Paper, Tabs, Tab, Box, Typography } from '@mui/material';
import SetLasersTab from './FRAMESettings/SetLasersTab';
import TestHomingTab from './FRAMESettings/TestHomingTab';
import PixelCalibrationTab from './FRAMESettings/PixelCalibrationTab';
import ManualPixelCalibrationTab from './FRAMESettings/ManualPixelCalibrationTab';
import ObjectiveControllerTab from './FRAMESettings/ObjectiveControllerTab';
import StageOffsetCalibrationTab from './FRAMESettings/StageOffsetCalibrationTab';

/**
 * FRAMESettings - Main component for pixel calibration and frame setup
 *
 * Provides a tabbed interface for:
 * - Automatic Pixel Calibration (stage affine calibration)
 * - Manual Pixel Calibration (two-point pixel-size calibration)
 * - Set Lasers (laser channel configuration)
 * - Test Homing (axis homing verification)
 * - Objective Controller (objective management)
 * - Stage Offset Calibration (raster scan -> stage offset)
 */
const FRAMESettings = () => {
  const [selectedTab, setSelectedTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  return (
    <Paper sx={{ p: 3, maxWidth: 1400, margin: '0 auto' }}>
      <Typography variant="h4" gutterBottom>
        FRAME Settings
      </Typography>
      
      <Tabs 
        value={selectedTab} 
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}
      >
        <Tab label="Automatic Pixel Calibration" />
        <Tab label="Manual Pixel Calibration" />
        <Tab label="Set Lasers" />
        <Tab label="Test Homing" />
        <Tab label="Objective Controller" />
        <Tab label="Stage Offset Calibration" />
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {selectedTab === 0 && <PixelCalibrationTab />}
        {selectedTab === 1 && <ManualPixelCalibrationTab />}
        {selectedTab === 2 && <SetLasersTab />}
        {selectedTab === 3 && <TestHomingTab />}
        {selectedTab === 4 && <ObjectiveControllerTab />}
        {selectedTab === 5 && <StageOffsetCalibrationTab />}
      </Box>
    </Paper>
  );
};

export default FRAMESettings;
