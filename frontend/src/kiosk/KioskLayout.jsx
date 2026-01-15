/**
 * KioskLayout.jsx
 * 
 * Minimal layout for Raspberry Pi Kiosk Mode
 * No navigation drawer, no top bar - just the content
 * 
 * Optimized for 800x480 touch display
 */

import { Box } from "@mui/material";
import KioskView from "./KioskView";

const KioskLayout = () => {
  return (
    <Box
      sx={{
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        backgroundColor: "background.default",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <KioskView />
    </Box>
  );
};

export default KioskLayout;
