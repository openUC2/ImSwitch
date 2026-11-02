import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import { Tabs, Tab, Box } from "@mui/material";
import SetLasersTab from "./FRAMESettings/SetLasersTab";
import FrameHomingTab from "./FRAMESettings/FrameHomingTab";
import ZStageSyncTab from "./FRAMESettings/ZStageSyncTab";
import ReadNoiseCalibrationTab from "./FRAMESettings/ReadNoiseCalibrationTab";
import ExtendedObjectiveController from "./ObjectiveController";
import OpticalFlowTab from "./FRAMESettings/OpticalFlowTab";
import CollisionDetectionTab from "./FRAMESettings/CollisionDetectionTab";
import { selectHasController } from "../state/slices/BackendCapabilitiesSlice";

/**
 * FRAMESettings - Main component for pixel calibration and frame setup
 *
 * Expert tools for the FRAME. Pixel size, camera rotation, stage offset and
 * objective offsets are calibrated by the FRAME Calibration app
 * (FrameCalibrationApp.jsx) with the calibration plate.
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
    return "readNoise";
  });
  const hasObjectiveController = useSelector(
    selectHasController("ObjectiveController"),
  );

  useEffect(() => {
    if (!hasObjectiveController && selectedTab === "objective") {
      setSelectedTab("readNoise");
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
        <Tab value="readNoise" label="Read Noise Calibration" />
        <Tab value="setLasers" label="Set Lasers" />
        <Tab value="frameHoming" label="Frame Homing & Transport" />
        <Tab value="zStageSync" label="Z-Stage Sync" />
        {hasObjectiveController && (
          <Tab value="objective" label="Objective Controller" />
        )}
        <Tab value="opticalFlow" label="Optical Flow Alignment" />
        <Tab value="collision" label="Collision Detection" />
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {selectedTab === "readNoise" && <ReadNoiseCalibrationTab />}
        {selectedTab === "setLasers" && <SetLasersTab />}
        {selectedTab === "frameHoming" && <FrameHomingTab />}
        {selectedTab === "zStageSync" && <ZStageSyncTab />}
        {hasObjectiveController && selectedTab === "objective" && (
          <ExtendedObjectiveController />
        )}
        {selectedTab === "opticalFlow" && <OpticalFlowTab />}
        {selectedTab === "collision" && <CollisionDetectionTab />}
      </Box>
    </Box>
  );
};

export default FRAMESettings;
