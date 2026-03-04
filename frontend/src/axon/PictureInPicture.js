/**
 * PictureInPicture.js
 *
 * A draggable, freely resizable floating overlay that shows the live camera
 * stream.  Drag the title bar to reposition; drag the bottom-right corner
 * handle to resize.  The live image always scales to fill the content area.
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
} from "@mui/icons-material";
import LiveViewControlWrapper from "./LiveViewControlWrapper";

const DEFAULT_SIZE  = { width: 340, height: 270 };
const MIN_SIZE      = { width: 160, height: 130 };

/**
 * Floating PiP overlay for live camera stream.
 *
 * @param {boolean}  visible  – Whether the PiP window is shown
 * @param {function} onClose  – Callback to hide the PiP
 */
const PictureInPicture = ({ visible, onClose }) => {
  const [position,   setPosition]   = useState({ x: 16, y: 16 });
  const [size,       setSize]       = useState(DEFAULT_SIZE);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);

  // Store mutable values that don't need to trigger re-renders
  const dragOffset  = useRef({ x: 0, y: 0 });  // offset when drag started
  const resizeStart = useRef({ mouseX: 0, mouseY: 0, width: 0, height: 0 });

  // ── Drag (title bar) ────────────────────────────────────────────────────
  const handleDragMouseDown = useCallback((e) => {
    setIsDragging(true);
    dragOffset.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    };
    e.preventDefault();
  }, [position]);

  useEffect(() => {
    if (!isDragging) return;
    const onMove = (e) => setPosition({
      x: e.clientX - dragOffset.current.x,
      y: e.clientY - dragOffset.current.y,
    });
    const onUp = () => setIsDragging(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
    };
  }, [isDragging]);

  // ── Resize (bottom-right handle) ────────────────────────────────────────
  const handleResizeMouseDown = useCallback((e) => {
    setIsResizing(true);
    resizeStart.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      width:  size.width,
      height: size.height,
    };
    e.preventDefault();
    e.stopPropagation(); // don't start a drag too
  }, [size]);

  useEffect(() => {
    if (!isResizing) return;
    const onMove = (e) => {
      const dx = e.clientX - resizeStart.current.mouseX;
      const dy = e.clientY - resizeStart.current.mouseY;
      setSize({
        width:  Math.max(MIN_SIZE.width,  resizeStart.current.width  + dx),
        height: Math.max(MIN_SIZE.height, resizeStart.current.height + dy),
      });
    };
    const onUp = () => setIsResizing(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
    };
  }, [isResizing]);

  if (!visible) return null;

  const cursorStyle = isDragging ? "grabbing" : isResizing ? "nwse-resize" : "default";

  return (
    <Paper
      elevation={8}
      sx={{
        position: "fixed",
        left:     position.x,
        top:      position.y,
        width:    size.width,
        height:   size.height,
        zIndex:   1300,
        overflow: "hidden",
        borderRadius: "8px",
        border:       "2px solid",
        borderColor:  "primary.main",
        display:      "flex",
        flexDirection: "column",
        cursor:   cursorStyle,
        userSelect: (isDragging || isResizing) ? "none" : "auto",
      }}
    >
      {/* ── Title bar / drag handle ─────────────────────────────────── */}
      <Box
        onMouseDown={handleDragMouseDown}
        sx={{
          display:        "flex",
          alignItems:     "center",
          justifyContent: "space-between",
          px: 0.5,
          py: 0.25,
          backgroundColor: "primary.main",
          color:           "primary.contrastText",
          cursor:          "grab",
          minHeight:       28,
          flexShrink:      0,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <DragIndicator fontSize="small" />
          <span style={{ fontSize: 11, fontWeight: 600 }}>Live Preview</span>
        </Box>
        <Tooltip title="Close">
          <IconButton
            size="small"
            onClick={onClose}
            onMouseDown={(e) => e.stopPropagation()} // don't trigger drag
            sx={{ color: "inherit", p: 0.25 }}
          >
            <Close fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      {/* ── Live view content ────────────────────────────────────────── */}
      {/* Use absolute fill so the live view canvas always covers the area
          regardless of its own internal sizing preferences. */}
      <Box sx={{ flex: 1, position: "relative", overflow: "hidden" }}>
        <Box
          sx={{
            position: "absolute",
            top: 0, left: 0,
            width: "100%", height: "100%",
          }}
        >
          <LiveViewControlWrapper enableStageMovement={false} />
        </Box>
      </Box>

      {/* ── Resize handle (bottom-right corner) ─────────────────────── */}
      <Box
        onMouseDown={handleResizeMouseDown}
        sx={{
          position: "absolute",
          right:    0,
          bottom:   0,
          width:    16,
          height:   16,
          cursor:   "nwse-resize",
          // Visual cue: small triangle drawn with borders
          borderTop:   "8px solid transparent",
          borderLeft:  "8px solid transparent",
          borderBottom: `8px solid rgba(255,255,255,0.5)`,
          borderRight:  `8px solid rgba(255,255,255,0.5)`,
          zIndex:   10,
        }}
      />
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

