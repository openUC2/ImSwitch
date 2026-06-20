import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Tabs, Tab, Box } from "@mui/material";
import SetLasersTab from "./FRAMESettings/SetLasersTab";
import TestHomingTab from "./FRAMESettings/TestHomingTab";
import FrameHomingTab from "./FRAMESettings/FrameHomingTab";
import PixelCalibrationTab from "./FRAMESettings/PixelCalibrationTab";
import ManualPixelCalibrationTab from "./FRAMESettings/ManualPixelCalibrationTab";
import VerifyCalibrationTab from "./FRAMESettings/VerifyCalibrationTab";
import ReadNoiseCalibrationTab from "./FRAMESettings/ReadNoiseCalibrationTab";
import ExtendedObjectiveController from "./ObjectiveController";
import StageOffsetCalibrationTab from "./FRAMESettings/StageOffsetCalibrationTab";
import OpticalFlowTab from "./FRAMESettings/OpticalFlowTab";
import { selectHasController } from "../state/slices/BackendCapabilitiesSlice";

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
// localStorage handshake key: another view (e.g. the startup "Homing
// Recommended" dialog) can request a specific tab to be opened on mount by
// setting this key before navigating to the FRAME Settings plugin. Consumed
// once so a later manual visit still opens the default tab.
const INITIAL_TAB_KEY = "frameSettings.initialTab";

const FRAMESettings = () => {
  const [selectedTab, setSelectedTab] = useState(() => {
    const requested = localStorage.getItem(INITIAL_TAB_KEY);
    if (requested) {
      localStorage.removeItem(INITIAL_TAB_KEY);
      return requested;
    }
    return "autoPixel";
  });
  const hasObjectiveController = useSelector(
    selectHasController("ObjectiveController"),
  );

  useEffect(() => {
    if (!hasObjectiveController && selectedTab === "objective") {
      setSelectedTab("autoPixel");
    }
  }, [hasObjectiveController, selectedTab]);

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  return (
    <Box sx={{ width: "100%" }}>
      <Tabs
        value={selectedTab}
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}
      >
        <Tab value="autoPixel" label="Automatic Pixel Calibration" />
        <Tab value="manualPixel" label="Manual Pixel Calibration" />
        <Tab value="verifyCalibration" label="Verify Calibration" />
        <Tab value="readNoise" label="Read Noise Calibration" />
        <Tab value="setLasers" label="Set Lasers" />
        <Tab value="testHoming" label="Test Homing" />
        <Tab value="frameHoming" label="Frame Homing & Transport" />
        {hasObjectiveController && (
          <Tab value="objective" label="Objective Controller" />
        )}
        <Tab value="stageOffset" label="Stage Offset Calibration" />
        <Tab value="opticalFlow" label="Optical Flow Alignment" />
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {selectedTab === "autoPixel" && <PixelCalibrationTab />}
        {selectedTab === "manualPixel" && <ManualPixelCalibrationTab />}
        {selectedTab === "verifyCalibration" && <VerifyCalibrationTab />}
        {selectedTab === "readNoise" && <ReadNoiseCalibrationTab />}
        {selectedTab === "setLasers" && <SetLasersTab />}
        {selectedTab === "testHoming" && <TestHomingTab />}
        {selectedTab === "frameHoming" && <FrameHomingTab />}
        {hasObjectiveController && selectedTab === "objective" && (
          <ExtendedObjectiveController />
        )}
        {selectedTab === "stageOffset" && <StageOffsetCalibrationTab />}
        {selectedTab === "opticalFlow" && <OpticalFlowTab />}
      </Box>
    </Box>
  );
};

export default FRAMESettings;
