import React, { useState } from 'react';
import { Paper, Tabs, Tab, Box, Typography } from '@mui/material';
import TrackMotionTab from './FRAMESettings/TrackMotionTab';
import SetLasersTab from './FRAMESettings/SetLasersTab';
import TestHomingTab from './FRAMESettings/TestHomingTab';
import PixelCalibrationTab from './FRAMESettings/PixelCalibrationTab';
import ObjectiveControllerTab from './FRAMESettings/ObjectiveControllerTab';

/**
 * FRAMESettings - Main component for pixel calibration and frame setup
 * 
 * Provides a tabbed interface for:
 * - Track Motion (AprilTag grid calibration)
 * - Set Lasers (laser channel configuration)
 * - Test Homing (axis homing verification)
 * - Pixel Calibration (stage affine calibration)
 * - Objective Controller (objective management)
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
        <Tab label="Track Motion" />
        <Tab label="Set Lasers" />
        <Tab label="Test Homing" />
        <Tab label="Pixel Calibration" />
        <Tab label="Objective Controller" />
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {selectedTab === 0 && <TrackMotionTab />}
        {selectedTab === 1 && <SetLasersTab />}
        {selectedTab === 2 && <TestHomingTab />}
        {selectedTab === 3 && <PixelCalibrationTab />}
        {selectedTab === 4 && <ObjectiveControllerTab />}
      </Box>
    </Paper>
  );
};

export default FRAMESettings;
