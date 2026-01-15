/**
 * KioskStageControl.jsx
 * 
 * Minimaler Stage Control für Kiosk Mode
 * - AxisControl.jsx direkt mit Connection Settings
 */

import { Box } from "@mui/material";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../../state/slices/ConnectionSettingsSlice";
import AxisControl from "../../components/AxisControl";

const KioskStageControl = () => {
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  return (
    <Box 
      sx={{ 
        height: "100%", 
        width: "100%", 
        p: 0,
        "& .MuiCard-root": {
          boxShadow: 0,
        },
        "& .MuiCardContent-root": {
          padding: "0 !important",
        },
        "& .MuiPaper-root": {
          padding: "6px !important",
        },
        "& .MuiTypography-h6": {
          fontSize: "0.85rem",
        },
        "& .MuiTypography-body2": {
          fontSize: "0.7rem",
        },
        "& .MuiButton-root": {
          fontSize: "0.65rem",
          padding: "3px 6px",
          minHeight: "22px",
        },
        "& .MuiChip-root": {
          height: "18px",
          fontSize: "0.6rem",
        },
        "& .MuiTextField-root input": {
          fontSize: "0.7rem",
          padding: "3px 6px",
        },
        "& .MuiIconButton-root": {
          padding: "3px",
        },
      }}
    >
      <AxisControl hostIP={hostIP} hostPort={hostPort} hidePositions={true} />
    </Box>
  );
};

export default KioskStageControl;
