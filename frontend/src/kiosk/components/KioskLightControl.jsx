/**
 * KioskLightControl.jsx
 * 
 * Simplified Light/LED Control for Kiosk Mode
 * - LED Matrix controls
 * - Laser controls
 * - Touch-optimized sliders and buttons
 */

import { Box } from "@mui/material";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../../state/slices/ConnectionSettingsSlice";
import IlluminationController from "../../components/IlluminationController";
import ExtendedLEDMatrixController from "../../components/ExtendedLEDMatrixController";

const KioskLightControl = () => {
  const connectionSettings = useSelector(getConnectionSettingsState);
  const { ip, apiPort } = connectionSettings;

  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        overflow: "auto",
        p: 1,
      }}
    >
      {/* Laser/Light Controls */}
      <IlluminationController hostIP={ip} hostPort={apiPort} />
      
      {/* LED Matrix Controls */}
      <Box sx={{ mt: 2 }}>
        <ExtendedLEDMatrixController />
      </Box>
    </Box>
  );
};

export default KioskLightControl;
