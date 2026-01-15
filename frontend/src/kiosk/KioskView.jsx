/**
 * KioskView.jsx
 * 
 * Minimaler Kiosk Mode
 * - Links: LiveStream (65% Breite)
 * - Rechts: Multi-axis Stage Control (35% Breite)
 * 
 * Optimized for 800x480 touch display
 */

import { Box, Paper } from "@mui/material";
import KioskLiveStream from "./components/KioskLiveStream";
import KioskStageControl from "./components/KioskStageControl";

const KioskView = () => {
  return (
    <Box
      sx={{
        display: "flex",
        height: "100vh",
        width: "100vw",
        overflow: "hidden",
        bgcolor: "background.default",
        gap: 0,
        p: 0,
      }}
    >
      {/* Livestream - 65% of screen width */}
      <Box
        sx={{
          flex: "0 0 65%",
          height: "100%",
          overflow: "hidden",
          bgcolor: "black",
        }}
      >
        <KioskLiveStream />
      </Box>

      {/* Multi-axis Stage Control - 35% of screen width */}
      <Box
        sx={{
          flex: "0 0 35%",
          height: "100%",
          overflow: "auto",
          bgcolor: "background.default",
        }}
      >
        <KioskStageControl />
      </Box>
    </Box>
  );
};

export default KioskView;
