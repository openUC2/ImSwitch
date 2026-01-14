import React, { useState, useEffect, useRef } from "react";
import {
  Box,
  Typography,
  Button,
  ButtonGroup,
  Paper,
  Alert,
  Chip,
} from "@mui/material";
import SportsEsportsIcon from "@mui/icons-material/SportsEsports";
import SpeedIcon from "@mui/icons-material/Speed";

/**
 * GamepadSpeedControl - Control stage with physical gamepad (PS4, Xbox, etc.)
 * 
 * Features:
 * - Read left analog stick for X/Y movement
 * - Continuous movement with speed control
 * - Adjustable speed multiplier (1x, 10x, 100x, 1000x)
 * - Rate-limited API calls (200ms)
 * - Dead zone handling for analog sticks
 */
export default function GamepadSpeedControl({ hostIP, hostPort }) {
  const [gamepadConnected, setGamepadConnected] = useState(false);
  const [gamepadName, setGamepadName] = useState("");
  const [speedMultiplier, setSpeedMultiplier] = useState(100);
  const [currentSpeed, setCurrentSpeed] = useState({ x: 0, y: 0, z: 0 });
  const [isMoving, setIsMoving] = useState(false);

  // Refs for animation loop and rate limiting
  const animationFrameRef = useRef(null);
  const lastApiCallRef = useRef({ x: 0, y: 0, z: 0 });
  const lastApiCallTimeRef = useRef(0);

  // Configuration
  const DEAD_ZONE = 0.15; // Ignore stick values below this threshold
  const API_RATE_LIMIT = 200; // milliseconds between API calls
  const MAX_SPEED = 100000; // Maximum speed value to send to API

  /**
   * Detect gamepad connection/disconnection
   */
  useEffect(() => {
    const handleGamepadConnected = (e) => {
      console.log("Gamepad connected:", e.gamepad);
      setGamepadConnected(true);
      setGamepadName(e.gamepad.id);
    };

    const handleGamepadDisconnected = (e) => {
      console.log("Gamepad disconnected:", e.gamepad);
      setGamepadConnected(false);
      setGamepadName("");
    };

    window.addEventListener("gamepadconnected", handleGamepadConnected);
    window.addEventListener("gamepaddisconnected", handleGamepadDisconnected);

    // Check if gamepad is already connected
    const gamepads = navigator.getGamepads();
    for (let i = 0; i < gamepads.length; i++) {
      if (gamepads[i]) {
        setGamepadConnected(true);
        setGamepadName(gamepads[i].id);
        break;
      }
    }

    return () => {
      window.removeEventListener("gamepadconnected", handleGamepadConnected);
      window.removeEventListener("gamepaddisconnected", handleGamepadDisconnected);
    };
  }, []);

  /**
   * Apply dead zone to analog stick value
   */
  const applyDeadZone = (value) => {
    if (Math.abs(value) < DEAD_ZONE) return 0;
    // Scale the remaining range to 0-1
    const sign = Math.sign(value);
    const scaledValue = (Math.abs(value) - DEAD_ZONE) / (1 - DEAD_ZONE);
    return sign * scaledValue;
  };

  /**
   * Send movement command to API
   */
  const sendMovementCommand = async (axis, speed, isStop = false) => {
    try {
      const params = new URLSearchParams({
        axis: axis.toUpperCase(),
        speed: Math.round(speed),
        is_stop: isStop,
      });

      await fetch(
        `${hostIP}:${hostPort}/imswitch/api/PositionerController/movePositionerForever?${params}`
      );
    } catch (error) {
      console.error(`Error sending movement command for ${axis}:`, error);
    }
  };

  /**
   * Stop movement on all axes
   */
  const stopAllMovement = async () => {
    await Promise.all([
      sendMovementCommand("X", 0, true),
      sendMovementCommand("Y", 0, true),
      sendMovementCommand("Z", 0, true),
    ]);
    setIsMoving(false);
    setCurrentSpeed({ x: 0, y: 0, z: 0 });
  };

  /**
   * Main gamepad polling loop
   */
  useEffect(() => {
    if (!gamepadConnected) {
      // Stop any ongoing movement when gamepad disconnects
      if (isMoving) {
        stopAllMovement();
      }
      return;
    }

    const pollGamepad = () => {
      const gamepads = navigator.getGamepads();
      const gamepad = gamepads[0] || gamepads[1] || gamepads[2] || gamepads[3];

      if (!gamepad) {
        animationFrameRef.current = requestAnimationFrame(pollGamepad);
        return;
      }

      // Read left analog stick (typically axes 0 and 1)
      // X axis: axes[0] (left = -1, right = +1)
      // Y axis: axes[1] (up = -1, down = +1)
      const rawX = gamepad.axes[0] || 0;
      const rawY = gamepad.axes[1] || 0;

      // Apply dead zone
      const normalizedX = applyDeadZone(rawX);
      const normalizedY = applyDeadZone(rawY);

      // Calculate speed values (invert Y for intuitive control)
      const speedX = Math.round(normalizedX * speedMultiplier *1000);
      const speedY = Math.round(-normalizedY * speedMultiplier *1000); // Invert Y

      // Clamp to max speed
      const clampedSpeedX = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, speedX));
      const clampedSpeedY = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, speedY));

      // Update display
      setCurrentSpeed({ x: clampedSpeedX, y: clampedSpeedY, z: 0 });

      // Check if we should send API calls (rate limiting)
      const now = Date.now();
      const timeSinceLastCall = now - lastApiCallTimeRef.current;

      if (timeSinceLastCall >= API_RATE_LIMIT) {
        const hasMovement = clampedSpeedX !== 0 || clampedSpeedY !== 0;
        const speedChanged =
          clampedSpeedX !== lastApiCallRef.current.x ||
          clampedSpeedY !== lastApiCallRef.current.y;

        if (hasMovement && speedChanged) {
          // Send movement commands
          if (clampedSpeedX !== lastApiCallRef.current.x) {
            sendMovementCommand("X", clampedSpeedX, false);
          }
          if (clampedSpeedY !== lastApiCallRef.current.y) {
            sendMovementCommand("Y", clampedSpeedY, false);
          }

          lastApiCallRef.current = {
            x: clampedSpeedX,
            y: clampedSpeedY,
            z: 0,
          };
          lastApiCallTimeRef.current = now;
          setIsMoving(true);
        } else if (!hasMovement && isMoving) {
          // Stop movement if joystick returned to center
          stopAllMovement();
          lastApiCallRef.current = { x: 0, y: 0, z: 0 };
          lastApiCallTimeRef.current = now;
        }
      }

      animationFrameRef.current = requestAnimationFrame(pollGamepad);
    };

    animationFrameRef.current = requestAnimationFrame(pollGamepad);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      stopAllMovement();
    };
  }, [gamepadConnected, speedMultiplier, isMoving, hostIP, hostPort]);

  /**
   * Emergency stop button
   */
  const handleEmergencyStop = () => {
    stopAllMovement();
  };

  return (
    <Box>
      {/* Gamepad Status */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center" gap={1} mb={1}>
          <SportsEsportsIcon />
          <Typography variant="subtitle1">Gamepad Status</Typography>
        </Box>
        {gamepadConnected ? (
          <Box>
            <Chip label="Connected" color="success" size="small" sx={{ mb: 1 }} />
            <Typography variant="body2" color="text.secondary">
              {gamepadName}
            </Typography>
          </Box>
        ) : (
          <Alert severity="info">
            No gamepad detected. Please connect a gamepad (PS4, Xbox, etc.) and press any button.
          </Alert>
        )}
      </Paper>

      {/* Speed Multiplier Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <SpeedIcon />
          <Typography variant="subtitle1">Speed Multiplier</Typography>
        </Box>
        <ButtonGroup fullWidth>
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

      {/* Current Speed Display */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle1" mb={1}>
          Current Speed
        </Typography>
        <Box display="flex" gap={2}>
          <Box flex={1}>
            <Typography variant="body2" color="text.secondary">
              X Axis
            </Typography>
            <Typography variant="h6">{currentSpeed.x}</Typography>
          </Box>
          <Box flex={1}>
            <Typography variant="body2" color="text.secondary">
              Y Axis
            </Typography>
            <Typography variant="h6">{currentSpeed.y}</Typography>
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

      {/* Emergency Stop */}
      <Button
        variant="contained"
        color="error"
        fullWidth
        onClick={handleEmergencyStop}
        disabled={!isMoving}
      >
        EMERGENCY STOP
      </Button>

      {/* Instructions */}
      <Paper sx={{ p: 2, mt: 2, bgcolor: "action.hover" }}>
        <Typography variant="subtitle2" mb={1}>
          Instructions
        </Typography>
        <Typography variant="body2" color="text.secondary">
          • Use the <strong>left analog stick</strong> to control X/Y movement
          <br />
          • Adjust the speed multiplier for finer or coarser control
          <br />
          • Movement speed updates every 200ms while the stick is active
          <br />
          • Release the stick to center position to stop movement
          <br />• Use the emergency stop button to halt all movement immediately
        </Typography>
      </Paper>
    </Box>
  );
}
