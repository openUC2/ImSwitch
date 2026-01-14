import React from "react";
import { Button, Box, useTheme, useMediaQuery } from "@mui/material";

export default function StageNudgePad({ onMove }) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const buttonStyle = {
    minHeight: isMobile ? 60 : 48,
    minWidth: isMobile ? 60 : 48,
    fontSize: isMobile ? '1.5rem' : '1.2rem',
    padding: isMobile ? '16px' : '12px',
    touchAction: 'manipulation',
    userSelect: 'none',
    WebkitUserSelect: 'none',
  };

  const handleTouch = (direction) => (event) => {
    event.preventDefault();
    onMove(direction);
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: isMobile ? 2 : 1,
        touchAction: 'manipulation',
      }}
    >
      <Button 
        variant="contained" 
        onClick={handleTouch("up")}
        onTouchStart={handleTouch("up")}
        sx={{ ...buttonStyle, mb: 0 }}
      >
        ↑
      </Button>
      <Box sx={{ display: "flex", gap: isMobile ? 2 : 1 }}>
        <Button 
          variant="contained" 
          onClick={handleTouch("left")}
          onTouchStart={handleTouch("left")}
          sx={buttonStyle}
        >
          ←
        </Button>
        <Button 
          variant="contained" 
          onClick={handleTouch("right")}
          onTouchStart={handleTouch("right")}
          sx={buttonStyle}
        >
          →
        </Button>
      </Box>
      <Button 
        variant="contained" 
        onClick={handleTouch("down")}
        onTouchStart={handleTouch("down")}
        sx={{ ...buttonStyle, mt: 0 }}
      >
        ↓
      </Button>
    </Box>
  );
}
