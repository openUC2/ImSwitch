import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Box,
  Typography,
  Button,
  ButtonGroup,
  Paper,
  Chip,
  Slider,
} from "@mui/material";
import SpeedIcon from "@mui/icons-material/Speed";
import TouchAppIcon from "@mui/icons-material/TouchApp";
import RotateRightIcon from "@mui/icons-material/RotateRight";

/**
 * VirtualJoystickControl - On-screen joystick for stage control
 * 
 * Features:
 * - Virtual joystick UI (mouse/touch control)
 * - Continuous movement with speed control
 * - Adjustable speed multiplier (1x, 10x, 100x, 1000x)
 * - Sliders for A and Z axes
 * - Rate-limited API calls (200ms)
 * - Visual feedback with position indicator
 * - Simultaneous X/Y control
 */
export default function VirtualJoystickControl({ hostIP, hostPort }) {
  const [speedMultiplier, setSpeedMultiplier] = useState(100);
  const [currentSpeed, setCurrentSpeed] = useState({ x: 0, y: 0, a: 0, z: 0 });
  const [isMoving, setIsMoving] = useState(false);
  const [joystickPosition, setJoystickPosition] = useState({ x: 0, y: 0 });
  const [aAxisSpeed, setAAxisSpeed] = useState(0);
  const [zAxisSpeed, setZAxisSpeed] = useState(0);

  // Refs
  const joystickRef = useRef(null);
  const isDraggingRef = useRef(false);
  const animationFrameRef = useRef(null);
  const lastApiCallRef = useRef({ x: 0, y: 0, a: 0, z: 0 });
  const lastApiCallTimeRef = useRef(0);

  // Configuration
  const JOYSTICK_RADIUS = 100; // Radius of joystick base (reduced for better layout)
  const HANDLE_RADIUS = 30; // Radius of joystick handle
  const API_RATE_LIMIT = 100; // milliseconds between API calls
  const MAX_SPEED = 10000; // Maximum speed value to send to API

  /**
   * Send movement command to API (simultaneous X/Y/A/Z)
   */
  const sendMovementCommand = useCallback(async (speeds, isStop = false) => {
    try {
      const params = new URLSearchParams({
        speedX: Math.round(speeds.x),
        speedY: Math.round(speeds.y),
        speedZ: Math.round(speeds.z),
        speedA: Math.round(speeds.a),
        is_stop: isStop,
      });

      await fetch(
        `${hostIP}:${hostPort}/imswitch/api/PositionerController/movePositionerForeverXYZA?${params}`
      );
    } catch (error) {
      console.error("Error sending movement command:", error);
    }
  }, [hostIP, hostPort]);

  /**
   * Stop movement on all axes
   */
  const stopAllMovement = useCallback(async () => {
    await sendMovementCommand({ x: 0, y: 0, z: 0, a: 0 }, true);
    setIsMoving(false);
    setCurrentSpeed({ x: 0, y: 0, z: 0, a: 0 });
    setJoystickPosition({ x: 0, y: 0 });
    setAAxisSpeed(0);
    setZAxisSpeed(0);
  }, [sendMovementCommand]);

  /**
   * Calculate speed from joystick position
   */
  const calculateSpeed = useCallback((position) => {
    // Normalize to -1 to 1 range
    const normalizedX = position.x / JOYSTICK_RADIUS;
    const normalizedY = position.y / JOYSTICK_RADIUS;

    // Calculate speed values (invert Y for intuitive control)
    const speedX = Math.round(normalizedX * speedMultiplier * 100);
    const speedY = Math.round(-normalizedY * speedMultiplier * 100); // Invert Y

    // Clamp to max speed
    const clampedSpeedX = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, speedX));
    const clampedSpeedY = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, speedY));

    return { x: clampedSpeedX, y: clampedSpeedY };
  }, [speedMultiplier]);

  /**
   * Update movement based on joystick position and sliders
   */
  const updateMovement = useCallback(() => {
    const xySpeed = calculateSpeed(joystickPosition);
    const allSpeeds = {
      x: xySpeed.x,
      y: xySpeed.y,
      z: Math.round(zAxisSpeed * speedMultiplier),
      a: Math.round(aAxisSpeed * speedMultiplier),
    };
    
    setCurrentSpeed(allSpeeds);

    const now = Date.now();
    const timeSinceLastCall = now - lastApiCallTimeRef.current;

    if (timeSinceLastCall >= API_RATE_LIMIT) {
      const hasMovement = 
        allSpeeds.x !== 0 || 
        allSpeeds.y !== 0 || 
        allSpeeds.z !== 0 || 
        allSpeeds.a !== 0;
      
      const speedChanged =
        allSpeeds.x !== lastApiCallRef.current.x ||
        allSpeeds.y !== lastApiCallRef.current.y ||
        allSpeeds.z !== lastApiCallRef.current.z ||
        allSpeeds.a !== lastApiCallRef.current.a;

      if (hasMovement && speedChanged) {
        sendMovementCommand(allSpeeds, false);
        lastApiCallRef.current = allSpeeds;
        lastApiCallTimeRef.current = now;
        setIsMoving(true);
      } else if (!hasMovement && isMoving) {
        // Send explicit stop when all speeds are zero
        sendMovementCommand({ x: 0, y: 0, z: 0, a: 0 }, true);
        lastApiCallRef.current = { x: 0, y: 0, z: 0, a: 0 };
        lastApiCallTimeRef.current = now;
        setIsMoving(false);
      }
    }
  }, [joystickPosition, zAxisSpeed, aAxisSpeed, calculateSpeed, isMoving, speedMultiplier]);

  /**
   * Handle joystick interaction (mouse or touch)
   */
  const handleInteractionStart = (e) => {
    e.preventDefault();
    isDraggingRef.current = true;
    updateJoystickPosition(e);
  };

  const handleInteractionMove = (e) => {
    if (!isDraggingRef.current) return;
    e.preventDefault();
    updateJoystickPosition(e);
  };

  const handleInteractionEnd = (e) => {
    e.preventDefault();
    isDraggingRef.current = false;
    setJoystickPosition({ x: 0, y: 0 });
    // Send stop command for X/Y when joystick returns to center
    const speeds = {
      x: 0,
      y: 0,
      z: Math.round(zAxisSpeed * speedMultiplier),
      a: Math.round(aAxisSpeed * speedMultiplier),
    };
    sendMovementCommand(speeds, zAxisSpeed === 0 && aAxisSpeed === 0);
  };

  /**
   * Update joystick position from mouse/touch event
   */
  const updateJoystickPosition = (e) => {
    if (!joystickRef.current) return;

    const rect = joystickRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    // Get pointer position (works for both mouse and touch)
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    // Calculate distance from center
    let deltaX = clientX - centerX;
    let deltaY = clientY - centerY;

    // Limit to joystick radius
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    if (distance > JOYSTICK_RADIUS) {
      const angle = Math.atan2(deltaY, deltaX);
      deltaX = Math.cos(angle) * JOYSTICK_RADIUS;
      deltaY = Math.sin(angle) * JOYSTICK_RADIUS;
    }

    setJoystickPosition({ x: deltaX, y: deltaY });
  };

  /**
   * Continuous update loop
   */
  useEffect(() => {
    if (!isDraggingRef.current && zAxisSpeed === 0 && aAxisSpeed === 0) return;

    const loop = () => {
      updateMovement();
      animationFrameRef.current = requestAnimationFrame(loop);
    };

    animationFrameRef.current = requestAnimationFrame(loop);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [updateMovement, zAxisSpeed, aAxisSpeed]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      stopAllMovement();
    };
  }, [stopAllMovement]);

  return (
    <Box>
      {/* Emergency Stop - Always visible at top */}
      <Button
        variant="contained"
        color="error"
        fullWidth
        onClick={stopAllMovement}
        sx={{ mb: 2, py: 1.5, fontSize: "1.1rem", fontWeight: "bold" }}
      >
        🛑 EMERGENCY STOP
      </Button>

      {/* Speed Multiplier Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <SpeedIcon />
          <Typography variant="subtitle1">Speed Multiplier</Typography>
        </Box>
        <ButtonGroup fullWidth size="small">
          <Button
            variant={speedMultiplier === 1 ? "contained" : "outlined"}
            onClick={() => setSpeedMultiplier(1)}
          >
            1x
          </Button>
          <Button
            variant={speedMultiplier === 10 ? "contained" : "outlined"}
            onClick={() => setSpeedMultiplier(10)}
          >
            10x
          </Button>
          <Button
            variant={speedMultiplier === 100 ? "contained" : "outlined"}
            onClick={() => setSpeedMultiplier(100)}
          >
            100x
          </Button>
          <Button
            variant={speedMultiplier === 1000 ? "contained" : "outlined"}
            onClick={() => setSpeedMultiplier(1000)}
          >
            1000x
          </Button>
        </ButtonGroup>
      </Paper>

      {/* Virtual Joystick */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <TouchAppIcon />
          <Typography variant="subtitle1">X/Y Control</Typography>
        </Box>
        
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            minHeight: 200,
            userSelect: "none",
            touchAction: "none",
          }}
        >
          {/* Joystick Base */}
          <Box
            ref={joystickRef}
            onMouseDown={handleInteractionStart}
            onMouseMove={handleInteractionMove}
            onMouseUp={handleInteractionEnd}
            onMouseLeave={handleInteractionEnd}
            onTouchStart={handleInteractionStart}
            onTouchMove={handleInteractionMove}
            onTouchEnd={handleInteractionEnd}
            onTouchCancel={handleInteractionEnd}
            sx={{
              position: "relative",
              width: JOYSTICK_RADIUS * 2,
              height: JOYSTICK_RADIUS * 2,
              borderRadius: "50%",
              background: "linear-gradient(145deg, #2a2a2a, #1a1a1a)",
              boxShadow: "inset 0 0 20px rgba(0,0,0,0.5)",
              cursor: "pointer",
              border: "2px solid #444",
            }}
          >
            {/* Center crosshair */}
            <Box
              sx={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: 4,
                height: 4,
                borderRadius: "50%",
                bgcolor: "#666",
              }}
            />
            
            {/* Axis lines */}
            <Box
              sx={{
                position: "absolute",
                top: "50%",
                left: 0,
                width: "100%",
                height: 1,
                bgcolor: "#333",
              }}
            />
            <Box
              sx={{
                position: "absolute",
                left: "50%",
                top: 0,
                height: "100%",
                width: 1,
                bgcolor: "#333",
              }}
            />

            {/* Joystick Handle */}
            <Box
              sx={{
                position: "absolute",
                top: "50%",
                left: "50%",
                width: HANDLE_RADIUS * 2,
                height: HANDLE_RADIUS * 2,
                borderRadius: "50%",
                background: isMoving
                  ? "linear-gradient(145deg, #ff6b6b, #ee5a6f)"
                  : "linear-gradient(145deg, #4a9eff, #357abd)",
                boxShadow: "0 4px 8px rgba(0,0,0,0.3)",
                transform: `translate(calc(-50% + ${joystickPosition.x}px), calc(-50% + ${joystickPosition.y}px))`,
                transition: isDraggingRef.current ? "none" : "transform 0.2s ease-out",
                border: "3px solid rgba(255,255,255,0.3)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "white",
                fontSize: 12,
                fontWeight: "bold",
                pointerEvents: "none",
              }}
            >
              {isMoving ? "■" : "●"}
            </Box>
          </Box>
        </Box>
      </Paper>

      {/* A Axis Control */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <RotateRightIcon />
          <Typography variant="subtitle1">A Axis (Rotation)</Typography>
        </Box>
        <Box sx={{ px: 2 }}>
          <Slider
            value={aAxisSpeed}
            onChange={(_, value) => setAAxisSpeed(value)}
            onChangeCommitted={() => {
              if (aAxisSpeed === 0) {
                // Send stop when slider returns to zero
                const speeds = {
                  x: currentSpeed.x,
                  y: currentSpeed.y,
                  z: currentSpeed.z,
                  a: 0,
                };
                sendMovementCommand(speeds, currentSpeed.x === 0 && currentSpeed.y === 0 && currentSpeed.z === 0);
              }
            }}
            min={-1}
            max={1}
            step={0.01}
            marks={[
              { value: -1, label: "←" },
              { value: 0, label: "0" },
              { value: 1, label: "→" },
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => Math.round(value * speedMultiplier)}
          />
          <Typography variant="body2" color="text.secondary" align="center">
            Speed: {Math.round(aAxisSpeed * speedMultiplier)}
          </Typography>
        </Box>
      </Paper>

      {/* Z Axis Control */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <Typography variant="subtitle1">Z Axis (Focus)</Typography>
        </Box>
        <Box sx={{ px: 2 }}>
          <Slider
            value={zAxisSpeed}
            onChange={(_, value) => setZAxisSpeed(value)}
            onChangeCommitted={() => {
              if (zAxisSpeed === 0) {
                // Send stop when slider returns to zero
                const speeds = {
                  x: currentSpeed.x,
                  y: currentSpeed.y,
                  z: 0,
                  a: currentSpeed.a,
                };
                sendMovementCommand(speeds, currentSpeed.x === 0 && currentSpeed.y === 0 && currentSpeed.a === 0);
              }
            }}
            min={-1}
            max={1}
            step={0.01}
            marks={[
              { value: -1, label: "↓" },
              { value: 0, label: "0" },
              { value: 1, label: "↑" },
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => Math.round(value * speedMultiplier)}
          />
          <Typography variant="body2" color="text.secondary" align="center">
            Speed: {Math.round(zAxisSpeed * speedMultiplier)}
          </Typography>
        </Box>
      </Paper>

      {/* Current Speed Display */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle1" mb={1}>
          Current Speed
        </Typography>
        <Box display="grid" gridTemplateColumns="1fr 1fr" gap={2}>
          <Box>
            <Typography variant="body2" color="text.secondary">X Axis</Typography>
            <Typography variant="h6">{currentSpeed.x}</Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">Y Axis</Typography>
            <Typography variant="h6">{currentSpeed.y}</Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">A Axis</Typography>
            <Typography variant="h6">{currentSpeed.a}</Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">Z Axis</Typography>
            <Typography variant="h6">{currentSpeed.z}</Typography>
          </Box>
        </Box>
        {isMoving && (
          <Chip
            label="Moving"
            color="warning"
            size="small"
            sx={{ mt: 2 }}
          />
        )}
      </Paper>

      {/* Instructions */}
      <Paper sx={{ p: 2, bgcolor: "action.hover" }}>
        <Typography variant="subtitle2" mb={1}>
          Instructions
        </Typography>
        <Typography variant="body2" color="text.secondary">
          • <strong>X/Y:</strong> Click and drag the joystick handle
          <br />
          • <strong>A/Z:</strong> Use the sliders for rotation and focus
          <br />
          • Distance from center = movement speed
          <br />
          • Release joystick to stop X/Y movement
          <br />
          • Return sliders to center (0) to stop A/Z movement
          <br />• Use emergency stop to halt all axes immediately
        </Typography>
      </Paper>
    </Box>
  );
}
