import React, { useState, useRef, useEffect } from "react";
import { Box, IconButton } from "@mui/material";
import { DragHandle as DragHandleIcon } from "@mui/icons-material";

const ResizablePanel = ({ children }) => {
  // Retrieve the stored width from localStorage, or default to 300px if not found
  const storedWidth = localStorage.getItem("panelWidth");
  const initialWidth = storedWidth ? parseInt(storedWidth) : window.innerWidth * 0.5; // Default to 300px if not found

  const [width, setWidth] = useState(initialWidth);
  const panelRef = useRef(null);
  const startX = useRef(0);

  const handleMouseDown = (e) => {
    startX.current = e.clientX;
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleMouseMove = (e) => {
    const delta = e.clientX - startX.current;
    setWidth((prevWidth) => {
      const newWidth = Math.max(prevWidth + delta, 100); // Minimum width of 100px
      // Save the new width to localStorage
      localStorage.setItem("panelWidth", newWidth);
      return newWidth;
    });
    startX.current = e.clientX;
  };

  const handleMouseUp = () => {
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  // Split children into left and right panels
  const [leftPanelContent, rightPanelContent] = children;

  return (
    <Box display="flex" height="100vh" width="100%">
      <Box
        ref={panelRef}
        sx={{
          width: `${width}px`,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {leftPanelContent}
      </Box>

      <IconButton
        sx={{
          cursor: "ew-resize",
          height: "100%",
          width: "8px",
          padding: "8px",
        }}
        onMouseDown={handleMouseDown}
      >
        <DragHandleIcon sx={{transform: "rotate(90deg)"}}/>
      </IconButton>

      <Box
        sx={{
          flexGrow: 1,
          padding: 0,
          overflow: "auto",
        }}
      >
        {rightPanelContent}
      </Box>
    </Box>
  );
};

export default ResizablePanel;
