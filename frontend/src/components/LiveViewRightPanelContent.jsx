import React, { useState, useEffect } from "react";
import { Box, Tabs, Tab, Typography } from "@mui/material";

import AxisControl from "./AxisControl.jsx";
import JoystickControl from "./JoystickControl.jsx";
import VirtualJoystickControl from "./VirtualJoystickControl.js";
import AutofocusController from "./AutofocusController";
import IlluminationController from "./IlluminationController";
import ObjectiveSwitcher from "./ObjectiveSwitcher";
import DetectorTriggerController from "./DetectorTriggerController";
import ExtendedLEDMatrixController from "./ExtendedLEDMatrixController.jsx";

import apiLEDMatrixControllerGetStatus from "../backendapi/apiLEDMatrixControllerGetStatus";

export default function LiveViewRightPanelContent({
  stageControlTab,
  setStageControlTab,
  hostIP,
  hostPort,
}) {
  const [hasLEDMatrix, setHasLEDMatrix] = useState(false);

  useEffect(() => {
    let isMounted = true;

    const fetchLEDMatrixStatus = async () => {
      try {
        const status = await apiLEDMatrixControllerGetStatus();
        if (isMounted) {
          setHasLEDMatrix(Boolean(status?.available));
        }
      } catch (error) {
        if (isMounted) {
          setHasLEDMatrix(false);
        }
      }
    };

    fetchLEDMatrixStatus();

    return () => {
      isMounted = false;
    };
  }, [hostIP, hostPort]);

  return (
    <>
      <Box mb={3}>
        <Typography variant="h6">Stage Control</Typography>

        <Tabs
          value={stageControlTab}
          onChange={(_, v) => setStageControlTab(v)}
          sx={{ mb: 2 }}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Multiple Axis View" />
          <Tab label="Joystick Control" />
          <Tab label="Virtual Joystick (speed mode)" />
        </Tabs>

        {stageControlTab === 0 && (
          <AxisControl hostIP={hostIP} hostPort={hostPort} />
        )}

        {stageControlTab === 1 && (
          <JoystickControl hostIP={hostIP} hostPort={hostPort} />
        )}

        {stageControlTab === 2 && (
          <VirtualJoystickControl hostIP={hostIP} hostPort={hostPort} />
        )}
      </Box>

      <Box mb={3}>
        <Typography variant="h6">Autofocus</Typography>
        <AutofocusController hostIP={hostIP} hostPort={hostPort} />
      </Box>

      <Box mb={3}>
        <Typography variant="h6">Illumination</Typography>
        <IlluminationController hostIP={hostIP} hostPort={hostPort} />
      </Box>

      <Box mb={3}>
        <Typography variant="h6">Objective</Typography>
        <ObjectiveSwitcher hostIP={hostIP} hostPort={hostPort} />
      </Box>

      {hasLEDMatrix && (
        <Box mb={3}>
          <Typography variant="h6">Extended LED Matrix</Typography>
          <ExtendedLEDMatrixController hostIP={hostIP} hostPort={hostPort} />
        </Box>
      )}

      <Box mb={3}>
        <Typography variant="h6">Detector Trigger</Typography>
        <DetectorTriggerController hostIP={hostIP} hostPort={hostPort} />
      </Box>
    </>
  );
}
