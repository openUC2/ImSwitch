/**
 * PictureInPicture.js
 *
 * A draggable, resizable floating overlay that shows the live camera stream.
 * Stays visible across all tabs so the user can always monitor the microscope
 * while working in the Well Selector, Parameter editor, etc.
 */

import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  IconButton,
  Tooltip,
  Paper,
  Box,
} from "@mui/material";
import {
  PictureInPictureAlt,
  Close,
  DragIndicator,
  OpenInFull,
  CloseFullscreen,
} from "@mui/icons-material";
import LiveViewControlWrapper from "./LiveViewControlWrapper";

// Size presets for the PiP window
const SIZE_SMALL = { width: 280, height: 210 };
const SIZE_LARGE = { width: 480, height: 360 };

/**
 * Floating PiP overlay for live camera stream.
 *
 * @param {boolean} visible - Whether the PiP window is shown
 * @param {function} onClose - Callback to hide the PiP
 */
const PictureInPicture = ({ visible, onClose }) => {
  const [position, setPosition] = useState({ x: 16, y: 16 });
  const [size, setSize] = useState(SIZE_SMALL);
  const [isDragging, setIsDragging] = useState(false);
  const [isLarge, setIsLarge] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const containerRef = useRef(null);

  // Start drag
  const handleMouseDown = useCallback((e) => {
    // Only drag from the header bar
    if (e.target.closest(".pip-drag-handle")) {
      setIsDragging(true);
      dragOffset.current = {
        x: e.clientX - position.x,
        y: e.clientY - position.y,
      };
      e.preventDefault();
    }
  }, [position]);

  // Drag move
  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e) => {
      setPosition({
        x: e.clientX - dragOffset.current.x,
        y: e.clientY - dragOffset.current.y,
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging]);

  // Toggle size
  const toggleSize = useCallback(() => {
    setIsLarge((prev) => !prev);
    setSize((prev) => (prev === SIZE_SMALL ? SIZE_LARGE : SIZE_SMALL));
  }, []);

  if (!visible) return null;

  return (
    <Paper
      ref={containerRef}
      elevation={8}
      onMouseDown={handleMouseDown}
      sx={{
        position: "fixed",
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
        zIndex: 1300,
        overflow: "hidden",
        borderRadius: "8px",
        border: "2px solid",
        borderColor: "primary.main",
        display: "flex",
        flexDirection: "column",
        cursor: isDragging ? "grabbing" : "default",
        userSelect: isDragging ? "none" : "auto",
      }}
    >
      {/* Header / drag handle */}
      <Box
        className="pip-drag-handle"
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 0.5,
          py: 0.25,
          backgroundColor: "primary.main",
          color: "primary.contrastText",
          cursor: "grab",
          minHeight: 28,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <DragIndicator fontSize="small" />
          <span style={{ fontSize: 11, fontWeight: 600 }}>Live Preview</span>
        </Box>
        <Box>
          <Tooltip title={isLarge ? "Shrink" : "Expand"}>
            <IconButton size="small" onClick={toggleSize} sx={{ color: "inherit", p: 0.25 }}>
              {isLarge ? <CloseFullscreen fontSize="small" /> : <OpenInFull fontSize="small" />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Close PiP">
            <IconButton size="small" onClick={onClose} sx={{ color: "inherit", p: 0.25 }}>
              <Close fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Live view content */}
      <Box sx={{ flex: 1, position: "relative", overflow: "hidden" }}>
        <LiveViewControlWrapper enableStageMovement={false} />
      </Box>
    </Paper>
  );
};

/**
 * Toggle button to show/hide the PiP overlay.
 * Place this anywhere in the toolbar.
 */
export const PiPToggleButton = ({ onClick, active }) => (
  <Tooltip title={active ? "Hide Live Preview" : "Show Live Preview (PiP)"}>
    <IconButton
      onClick={onClick}
      color={active ? "primary" : "default"}
      size="small"
    >
      <PictureInPictureAlt />
    </IconButton>
  </Tooltip>
);

export default PictureInPicture;
