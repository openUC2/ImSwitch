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

  // Long-press state per axis. One shared record meant that pressing a second
  // button while the first was held overwrote the axis, so the stop command
  // targeted the wrong one and the original move never stopped.
  const buttonPressRef = useRef({});

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
  // The step goes out on press, not on release: waiting for the 1 s
  // long-press timer to rule out a hold meant nothing happened while the
  // button was down, which is the "clicks register with a delay" complaint.
  // Holding still starts continuous travel from the same handler.
  const handleButtonDown = (axis, speed, singleDist, event) => {
    // Pointer capture delivers the release even if the finger slides off the
    // button, so a held move can never be left running.
    event?.currentTarget?.setPointerCapture?.(event.pointerId);

    const presses = buttonPressRef.current;
    if (presses[axis]?.active) return; // already held; ignore a second press
    movePositioner(axis, singleDist);

    presses[axis] = {
      active: true,
      continuousMode: false,
      speed,
      timer: setTimeout(() => {
        presses[axis].continuousMode = true;
        presses[axis].timer = null;
        movePositionerForever(axis, speed, false);
      }, 1000),
    };
  };

  const handleButtonUp = (axis) => {
    const press = buttonPressRef.current[axis];
    if (!press?.active) return;

    if (press.timer) clearTimeout(press.timer);
    if (press.continuousMode) {
      movePositionerForever(axis, press.speed, true); // stop
    }
    delete buttonPressRef.current[axis];
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

    // Keyboard steps use the same selectors as the on-screen buttons.
    let axis = null;
    let dist = xyStepSize;

    switch (event.key) {
      case "ArrowLeft":
        axis = "X";
        dist = -xyStepSize;
        break;
      case "ArrowRight":
        axis = "X";
        dist = xyStepSize;
        break;
      case "ArrowUp":
        axis = "Y";
        dist = xyStepSize;
        break;
      case "ArrowDown":
        axis = "Y";
        dist = -xyStepSize;
        break;
      case "PageUp":
        axis = "Z";
        dist = zStepSize;
        break;
      case "PageDown":
        axis = "Z";
        dist = -zStepSize;
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
  // Keys are handled on this pad's own element (see onKeyDown/onKeyUp below),
  // not on window: the wrapper mounts a pad in both the PiP window and the
  // camera viewport, and two window listeners moved the stage twice per press.
  useEffect(() => {
    const keyTimers = keyTimersRef.current;
    return () => {
      Object.values(keyTimers).forEach((timer) => clearTimeout(timer));
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
    <Box
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onKeyUp={handleKeyUp}
      sx={{ outline: "none" }}
    >
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
          onPointerDown={(event) =>
            handleButtonDown("Z", -continuousMoveSpeed, -zStepSize, event)
          }
          onPointerUp={() => handleButtonUp("Z")}
          onPointerCancel={() => handleButtonUp("Z")}
          sx={buttonStyle}
        >
          Z-
        </Button>
        <Button
          variant="contained"
          onPointerDown={(event) =>
            handleButtonDown("Y", -continuousMoveSpeed, -xyStepSize, event)
          }
          onPointerUp={() => handleButtonUp("Y")}
          onPointerCancel={() => handleButtonUp("Y")}
          sx={buttonStyle}
        >
          Y↑
        </Button>
        <Button
          variant="contained"
          onPointerDown={(event) =>
            handleButtonDown("Z", continuousMoveSpeed, zStepSize, event)
          }
          onPointerUp={() => handleButtonUp("Z")}
          onPointerCancel={() => handleButtonUp("Z")}
          sx={buttonStyle}
        >
          Z+
        </Button>
        <Button
          variant="contained"
          onPointerDown={(event) =>
            handleButtonDown("X", -continuousMoveSpeed, -xyStepSize, event)
          }
          onPointerUp={() => handleButtonUp("X")}
          onPointerCancel={() => handleButtonUp("X")}
          sx={buttonStyle}
        >
          X←
        </Button>
        <Button
          variant="contained"
          onPointerDown={(event) =>
            handleButtonDown("Y", continuousMoveSpeed, xyStepSize, event)
          }
          onPointerUp={() => handleButtonUp("Y")}
          onPointerCancel={() => handleButtonUp("Y")}
          sx={buttonStyle}
        >
          Y↓
        </Button>
        <Button
          variant="contained"
          onPointerDown={(event) =>
            handleButtonDown("X", continuousMoveSpeed, xyStepSize, event)
          }
          onPointerUp={() => handleButtonUp("X")}
          onPointerCancel={() => handleButtonUp("X")}
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
