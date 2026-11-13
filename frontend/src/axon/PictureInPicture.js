/**
 * PictureInPicture.js
 *
 * A draggable, resizable floating window with the live camera stream. Drag the
 * title bar to move it, drag the corner grip to resize it. It opens in the
 * bottom-right corner and, until the user resizes it, takes the camera frame's
 * aspect ratio so the whole image shows without bars or cropping.
 */

import React, { useState, useRef, useCallback } from "react";
import { IconButton, Tooltip, Paper, Box } from "@mui/material";
import { PictureInPictureAlt, Close, DragIndicator } from "@mui/icons-material";
import LiveViewControlWrapper from "./LiveViewControlWrapper";
import DetectorToggle from "./DetectorToggle";

const DEFAULT_SIZE = { width: 480, height: 400 };
const MIN_SIZE = { width: 200, height: 150 };
const MARGIN = 16; // px kept free between the window and the screen edge
const RESIZED_KEY = "pip-resized"; // set once the user has resized: hide the hint

// Keep at least the title bar on screen.
const clampPosition = (x, y, width) => ({
  x: Math.min(Math.max(x, 0), window.innerWidth - Math.min(width, 120)),
  y: Math.min(Math.max(y, 0), window.innerHeight - 40),
});

/**
 * Floating PiP overlay for live camera stream.
 *
 * @param {boolean}  visible  – Whether the PiP window is shown
 * @param {function} onClose  – Callback to hide the PiP
 */
const PictureInPicture = ({ visible, onClose }) => {
  const [size, setSize] = useState(DEFAULT_SIZE);
  const sizeRef = useRef(size);
  sizeRef.current = size;
  const [position, setPosition] = useState(() => ({
    x: Math.max(0, window.innerWidth - DEFAULT_SIZE.width - MARGIN),
    y: Math.max(0, window.innerHeight - DEFAULT_SIZE.height - MARGIN),
  }));
  const [showResizeHint, setShowResizeHint] = useState(() => {
    try {
      return !localStorage.getItem(RESIZED_KEY);
    } catch {
      return true;
    }
  });

  // Pointer events with pointer capture: one code path for mouse and touch,
  // and moves keep arriving when the pointer leaves the element.
  const dragRef = useRef(null); // { dx, dy } while moving
  const resizeRef = useRef(null); // { x, y, width, height } while resizing
  const userResized = useRef(false);
  const contentRef = useRef(null);
  const lastFrameSize = useRef("");

  const startDrag = useCallback(
    (e) => {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      dragRef.current = { dx: e.clientX - position.x, dy: e.clientY - position.y };
    },
    [position],
  );
  const onDragMove = useCallback(
    (e) => {
      if (!dragRef.current) return;
      setPosition(
        clampPosition(
          e.clientX - dragRef.current.dx,
          e.clientY - dragRef.current.dy,
          size.width,
        ),
      );
    },
    [size.width],
  );
  const endDrag = useCallback(() => {
    dragRef.current = null;
  }, []);

  const startResize = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      e.currentTarget.setPointerCapture(e.pointerId);
      resizeRef.current = { x: e.clientX, y: e.clientY, ...size };
    },
    [size],
  );
  const onResizeMove = useCallback(
    (e) => {
      const start = resizeRef.current;
      if (!start) return;
      userResized.current = true;
      setSize({
        width: Math.max(
          MIN_SIZE.width,
          Math.min(start.width + e.clientX - start.x, window.innerWidth - position.x),
        ),
        height: Math.max(
          MIN_SIZE.height,
          Math.min(start.height + e.clientY - start.y, window.innerHeight - position.y),
        ),
      });
    },
    [position],
  );
  const endResize = useCallback(() => {
    if (resizeRef.current && userResized.current && showResizeHint) {
      setShowResizeHint(false);
      try {
        localStorage.setItem(RESIZED_KEY, "1");
      } catch {
        /* the hint just shows again next time */
      }
    }
    resizeRef.current = null;
  }, [showResizeHint]);

  // Until the user resizes, match the image area to the frame's aspect ratio
  // so the whole frame fills the window. Keeps the window on screen.
  const fitToFrame = useCallback((frameWidth, frameHeight) => {
    const box = contentRef.current;
    const key = `${frameWidth}x${frameHeight}`;
    if (userResized.current || !box || !frameWidth || !frameHeight) return;
    if (lastFrameSize.current === key) return;
    lastFrameSize.current = key;
    const delta =
      Math.round((box.clientWidth * frameHeight) / frameWidth) - box.clientHeight;
    if (Math.abs(delta) < 2) return;
    const height = Math.max(MIN_SIZE.height, sizeRef.current.height + delta);
    setSize((s) => ({ ...s, height }));
    setPosition((p) => ({
      ...p,
      y: Math.max(0, Math.min(p.y, window.innerHeight - height - MARGIN)),
    }));
  }, []);

  if (!visible) return null;

  return (
    <Paper
      elevation={8}
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
      }}
    >
      {/* ── Title bar / drag handle ─────────────────────────────────── */}
      <Box
        onPointerDown={startDrag}
        onPointerMove={onDragMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 0.5,
          px: 0.5,
          py: 0.25,
          backgroundColor: "primary.main",
          color: "primary.contrastText",
          cursor: "grab",
          touchAction: "none",
          userSelect: "none",
          minHeight: 28,
          flexShrink: 0,
        }}
      >
        <DragIndicator fontSize="small" />
        <span style={{ fontSize: 11, fontWeight: 600 }}>Live Preview</span>
        {showResizeHint && (
          <span style={{ fontSize: 11, opacity: 0.85, marginLeft: "auto" }}>
            Drag the corner ◢ to resize
          </span>
        )}
        <Tooltip title="Close">
          <IconButton
            size="small"
            onClick={onClose}
            onPointerDown={(e) => e.stopPropagation()} // don't start a drag
            sx={{ color: "inherit", p: 0.25, ml: showResizeHint ? 0.5 : "auto" }}
          >
            <Close fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      {/* ── Camera selector ─────────────────────────────────────────── */}
      {/* Tabs to choose which detector to display (widefield / overview / …).
          Auto-hidden by DetectorToggle when only one camera exists. */}
      <Box sx={{ px: 0.5, pt: 0.5, flexShrink: 0, "& > div": { mb: 0 } }}>
        <DetectorToggle />
      </Box>

      {/* ── Live view content: fills the window, frame fitted inside ──── */}
      <Box ref={contentRef} sx={{ flex: 1, minHeight: 0, position: "relative" }}>
        <Box sx={{ position: "absolute", inset: 0 }}>
          <LiveViewControlWrapper
            enableStageMovement={false}
            onImageLoad={fitToFrame}
            fill
          />
        </Box>
      </Box>

      {/* ── Resize grip (bottom-right corner) ───────────────────────── */}
      <Box
        onPointerDown={startResize}
        onPointerMove={onResizeMove}
        onPointerUp={endResize}
        onPointerCancel={endResize}
        title="Drag to resize"
        aria-label="Resize live preview"
        sx={{
          position: "absolute",
          right: 0,
          bottom: 0,
          width: 22,
          height: 22,
          cursor: "nwse-resize",
          touchAction: "none",
          zIndex: 10,
          color: "primary.contrastText",
          backgroundColor: "primary.main",
          borderTopLeftRadius: 6,
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "flex-end",
        }}
      >
        {/* Classic window grip: three diagonal strokes */}
        <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
          <path
            d="M15 5 L5 15 M15 9 L9 15 M15 13 L13 15"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
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
