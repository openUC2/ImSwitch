/**
 * KioskLiveStream.jsx
 * 
 * Simplified LiveStream component for Kiosk Mode
 * - Auto-starts stream
 * - Minimal controls overlay
 * - Touch-optimized
 */

import { useEffect } from "react";
import { Box } from "@mui/material";
import LiveViewControlWrapper from "../../axon/LiveViewControlWrapper";

const KioskLiveStream = () => {
  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        position: "relative",
        backgroundColor: "black",
        overflow: "hidden",
        "& > div": {
          width: "100% !important",
          height: "100% !important",
          display: "flex !important",
          alignItems: "center !important",
          justifyContent: "center !important",
        },
        "& canvas": {
          width: "100% !important",
          height: "100% !important",
          objectFit: "cover !important",  // Fill entire space, crop if needed
        },
      }}
    >
      {/* Reuse existing LiveView component */}
      <LiveViewControlWrapper
        kioskMode={true}  // Flag to enable kiosk-specific features
        autoStart={true}  // Auto-start stream
      />
    </Box>
  );
};

export default KioskLiveStream;
