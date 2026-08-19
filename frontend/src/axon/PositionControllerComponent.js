import React, { useEffect, useRef, useState } from "react";

import {
  Box,
  Button,
  ButtonGroup,
  Typography,
  useTheme,
  useMediaQuery,
} from "@mui/material";

import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";
import apiPositionerControllerMovePositionerForever from "../backendapi/apiPositionerControllerMovePositionerForever.js";

const validXYStepSizes = [10, 100, 1000];
const validZStepSizes = [50, 100, 500];
const STORAGE_KEY = "imswitch-stage-control-step-sizes";

//##################################################################################
const PositionControllerComponent = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));

  const [xyStepSize, setXYStepSize] = useState(() => {
    try {
      const saved = JSON.parse(
        window.localStorage.getItem(STORAGE_KEY) || "{}",
      );
      return validXYStepSizes.includes(saved.xy) ? saved.xy : 100;
    } catch {
      return 100;
    }
  });

  const [zStepSize, setZStepSize] = useState(() => {
    try {
      const saved = JSON.parse(
        window.localStorage.getItem(STORAGE_KEY) || "{}",
      );
      return validZStepSizes.includes(saved.z) ? saved.z : 100;
    } catch {
      return 100;
    }
  });

  const keyMoveDistance = 100; // Distance for keyboard single press
  const zCoarseDistance = 500; // Coarse step for Z axis (PageUp/Down)
  const continuousMoveSpeed = 5000; // Speed for continuous movement

  useEffect(() => {
    if (!validXYStepSizes.includes(xyStepSize)) {
      setXYStepSize(100);
    }
    if (!validZStepSizes.includes(zStepSize)) {
      setZStepSize(100);
    }
  }, [xyStepSize, zStepSize]);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ xy: xyStepSize, z: zStepSize }),
      );
    } catch {
      // Ignore storage write issues in private browsing or restricted envs.
    }
  }, [xyStepSize, zStepSize]);

  // Track pressed keys, their timers, and whether continuous mode was triggered
  const keyTimersRef = useRef({});
  const keyPressedRef = useRef({});
  const continuousModeTriggeredRef = useRef({}); // Track if continuous mode was activated

  // Button long-press state: { timer, continuousMode, active, axis, speed, singleDist }
  const buttonPressRef = useRef({
    timer: null,
    continuousMode: false,
    active: false,
    axis: null,
    speed: 0,
    singleDist: 0,
  });

  //##################################################################################
  const movePositioner = (axis, dist) => {
    apiPositionerControllerMovePositioner({
      axis,
      dist,
      isAbsolute: false,
    })
      .then((positionerResponse) => {
        console.log(`Move ${axis} by ${dist} successful:`, positionerResponse);
      })
      .catch((error) => {
        console.log(`Move ${axis} by ${dist} error:`, error);
      });
  };

  //##################################################################################
  // Move positioner continuously (forever mode)
  const movePositionerForever = (axis, speed, is_stop) => {
    apiPositionerControllerMovePositionerForever({
      axis,
      speed,
      is_stop,
    })
      .then((positionerResponse) => {
        console.log(
          `Move forever ${axis} speed ${speed} stop=${is_stop}:`,
          positionerResponse,
        );
      })
      .catch((error) => {
        console.log(`Move forever ${axis} error:`, error);
      });
  };

  //##################################################################################
  // Generic button long-press handlers (short press = single step, long press = move forever)
  const handleButtonDown = (axis, speed, singleDist) => {
    const bp = buttonPressRef.current;
    bp.active = true;
    bp.continuousMode = false;
    bp.axis = axis;
    bp.speed = speed;
    bp.singleDist = singleDist;
    if (bp.timer) clearTimeout(bp.timer);

    bp.timer = setTimeout(() => {
      bp.continuousMode = true;
      bp.timer = null;
      movePositionerForever(axis, speed, false);
    }, 1000);
  };

  const handleButtonUp = () => {
    const bp = buttonPressRef.current;
    if (!bp.active) {
      return;
    }

    if (bp.timer) {
      clearTimeout(bp.timer);
      bp.timer = null;
    }

    if (bp.continuousMode) {
      movePositionerForever(bp.axis, bp.speed, true); // stop
      bp.continuousMode = false;
    } else {
      movePositioner(bp.axis, bp.singleDist);
    }

    bp.active = false;
    bp.axis = null;
    bp.speed = 0;
    bp.singleDist = 0;
  };

  //##################################################################################
  // Keyboard event handlers
  const handleKeyDown = (event) => {
    if (
      [
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
        "PageUp",
        "PageDown",
      ].includes(event.key)
    ) {
      event.preventDefault();
    }

    if (event.repeat) {
      return;
    }

    if (keyPressedRef.current[event.key]) {
      return;
    }

    keyPressedRef.current[event.key] = true;
    continuousModeTriggeredRef.current[event.key] = false;

    if (keyTimersRef.current[event.key]) {
      clearTimeout(keyTimersRef.current[event.key]);
    }

    keyTimersRef.current[event.key] = setTimeout(() => {
      continuousModeTriggeredRef.current[event.key] = true;
      delete keyTimersRef.current[event.key];

      let axis = null;
      let speed = continuousMoveSpeed;

      switch (event.key) {
        case "ArrowLeft":
          axis = "X";
          speed = -continuousMoveSpeed;
          break;
        case "ArrowRight":
          axis = "X";
          speed = continuousMoveSpeed;
          break;
        case "ArrowUp":
          axis = "Y";
          speed = -continuousMoveSpeed;
          break;
        case "ArrowDown":
          axis = "Y";
          speed = continuousMoveSpeed;
          break;
        case "PageUp":
          axis = "Z";
          speed = continuousMoveSpeed;
          break;
        case "PageDown":
          axis = "Z";
          speed = -continuousMoveSpeed;
          break;
        default:
          return;
      }

      if (axis) {
        movePositionerForever(axis, speed, false);
      }
    }, 1000);
  };

  //##################################################################################
  const handleKeyUp = (event) => {
    if (!keyPressedRef.current[event.key]) {
      return;
    }

    let axis = null;
    let dist = keyMoveDistance;

    switch (event.key) {
      case "ArrowLeft":
        axis = "X";
        dist = -keyMoveDistance;
        break;
      case "ArrowRight":
        axis = "X";
        dist = keyMoveDistance;
        break;
      case "ArrowUp":
        axis = "Y";
        dist = keyMoveDistance;
        break;
      case "ArrowDown":
        axis = "Y";
        dist = -keyMoveDistance;
        break;
      case "PageUp":
        axis = "Z";
        dist = zCoarseDistance;
        break;
      case "PageDown":
        axis = "Z";
        dist = -zCoarseDistance;
        break;
      default:
        keyPressedRef.current[event.key] = false;
        delete continuousModeTriggeredRef.current[event.key];
        if (keyTimersRef.current[event.key]) {
          clearTimeout(keyTimersRef.current[event.key]);
          delete keyTimersRef.current[event.key];
        }
        return;
    }

    const wasContinuousMode = continuousModeTriggeredRef.current[event.key];

    if (keyTimersRef.current[event.key]) {
      clearTimeout(keyTimersRef.current[event.key]);
      delete keyTimersRef.current[event.key];
    }

    if (wasContinuousMode) {
      if (axis) {
        movePositionerForever(axis, continuousMoveSpeed, true);
      }
    } else if (axis) {
      movePositioner(axis, dist);
    }

    keyPressedRef.current[event.key] = false;
    delete continuousModeTriggeredRef.current[event.key];
  };

  //##################################################################################
  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);

      Object.values(keyTimersRef.current).forEach((timer) =>
        clearTimeout(timer),
      );
      keyTimersRef.current = {};
      keyPressedRef.current = {};
      continuousModeTriggeredRef.current = {};
    };
  }, []);

  //##################################################################################
  const buttonSize = isMobile ? 60 : 48;
  const buttonStyle = {
    minHeight: buttonSize,
    minWidth: buttonSize,
    maxHeight: buttonSize,
    maxWidth: buttonSize,
    fontSize: isMobile ? "1.2rem" : "0.9rem",
    touchAction: "manipulation",
    userSelect: "none",
    padding: 0,
  };

  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 1,
          mb: 1,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
          <Typography variant="caption" sx={{ fontWeight: 700, opacity: 0.8 }}>
            XY
          </Typography>
          <ButtonGroup
            size="small"
            sx={{ height: 24 }}
            aria-label="XY step size"
          >
            {validXYStepSizes.map((value) => (
              <Button
                key={value}
                variant={xyStepSize === value ? "contained" : "outlined"}
                onClick={() => setXYStepSize(value)}
                sx={{
                  minWidth: 0,
                  px: 0.75,
                  py: 0,
                  fontSize: "0.65rem",
                  lineHeight: 1,
                }}
              >
                {value}
              </Button>
            ))}
          </ButtonGroup>
        </Box>

        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
          <Typography variant="caption" sx={{ fontWeight: 700, opacity: 0.8 }}>
            Z
          </Typography>
          <ButtonGroup
            size="small"
            sx={{ height: 24 }}
            aria-label="Z step size"
          >
            {validZStepSizes.map((value) => (
              <Button
                key={value}
                variant={zStepSize === value ? "contained" : "outlined"}
                onClick={() => setZStepSize(value)}
                sx={{
                  minWidth: 0,
                  px: 0.75,
                  py: 0,
                  fontSize: "0.65rem",
                  lineHeight: 1,
                }}
              >
                {value}
              </Button>
            ))}
          </ButtonGroup>
        </Box>
      </Box>

      <div
        className="arrow-container"
        style={{
          padding: isMobile ? "16px" : "10px",
          display: "grid",
          gridTemplateColumns: `repeat(3, ${buttonSize}px)`,
          gridTemplateRows: `repeat(2, ${buttonSize}px)`,
          gap: isMobile ? "8px" : "4px",
          width: "fit-content",
        }}
      >
        <Button
          variant="contained"
          onMouseDown={() =>
            handleButtonDown("Z", -continuousMoveSpeed, -zStepSize)
          }
          onMouseUp={handleButtonUp}
          onMouseLeave={handleButtonUp}
          onTouchStart={() =>
            handleButtonDown("Z", -continuousMoveSpeed, -zStepSize)
          }
          onTouchEnd={handleButtonUp}
          onTouchCancel={handleButtonUp}
          sx={buttonStyle}
        >
          Z-
        </Button>
        <Button
          variant="contained"
          onMouseDown={() =>
            handleButtonDown("Y", -continuousMoveSpeed, -xyStepSize)
          }
          onMouseUp={handleButtonUp}
          onMouseLeave={handleButtonUp}
          onTouchStart={() =>
            handleButtonDown("Y", -continuousMoveSpeed, -xyStepSize)
          }
          onTouchEnd={handleButtonUp}
          onTouchCancel={handleButtonUp}
          sx={buttonStyle}
        >
          Y↑
        </Button>
        <Button
          variant="contained"
          onMouseDown={() =>
            handleButtonDown("Z", continuousMoveSpeed, zStepSize)
          }
          onMouseUp={handleButtonUp}
          onMouseLeave={handleButtonUp}
          onTouchStart={() =>
            handleButtonDown("Z", continuousMoveSpeed, zStepSize)
          }
          onTouchEnd={handleButtonUp}
          onTouchCancel={handleButtonUp}
          sx={buttonStyle}
        >
          Z+
        </Button>
        <Button
          variant="contained"
          onMouseDown={() =>
            handleButtonDown("X", -continuousMoveSpeed, -xyStepSize)
          }
          onMouseUp={handleButtonUp}
          onMouseLeave={handleButtonUp}
          onTouchStart={() =>
            handleButtonDown("X", -continuousMoveSpeed, -xyStepSize)
          }
          onTouchEnd={handleButtonUp}
          onTouchCancel={handleButtonUp}
          sx={buttonStyle}
        >
          X←
        </Button>
        <Button
          variant="contained"
          onMouseDown={() =>
            handleButtonDown("Y", continuousMoveSpeed, xyStepSize)
          }
          onMouseUp={handleButtonUp}
          onMouseLeave={handleButtonUp}
          onTouchStart={() =>
            handleButtonDown("Y", continuousMoveSpeed, xyStepSize)
          }
          onTouchEnd={handleButtonUp}
          onTouchCancel={handleButtonUp}
          sx={buttonStyle}
        >
          Y↓
        </Button>
        <Button
          variant="contained"
          onMouseDown={() =>
            handleButtonDown("X", continuousMoveSpeed, xyStepSize)
          }
          onMouseUp={handleButtonUp}
          onMouseLeave={handleButtonUp}
          onTouchStart={() =>
            handleButtonDown("X", continuousMoveSpeed, xyStepSize)
          }
          onTouchEnd={handleButtonUp}
          onTouchCancel={handleButtonUp}
          sx={buttonStyle}
        >
          X→
        </Button>
      </div>
    </Box>
  );
};
//##################################################################################
export default PositionControllerComponent;
