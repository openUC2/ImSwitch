import React, { useEffect, useRef } from "react";

import { Button, useTheme, useMediaQuery } from "@mui/material";

import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";
import apiPositionerControllerMovePositionerForever from "../backendapi/apiPositionerControllerMovePositionerForever.js";

//##################################################################################
const PositionControllerComponent = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));

  const moveDistance = 500; //TODO adjust
  const zoomDistance = 100; //TODO adjust
  const keyMoveDistance = 100; // Distance for keyboard single press
  const zCoarseDistance = 500; // Coarse step for Z axis (PageUp/Down)
  const continuousMoveSpeed = 5000; // Speed for continuous movement

  // Track pressed keys, their timers, and whether continuous mode was triggered
  const keyTimersRef = useRef({});
  const keyPressedRef = useRef({});
  const continuousModeTriggeredRef = useRef({}); // Track if continuous mode was activated

  // Button long-press state: { timer, continuousMode, axis, speed, singleDist }
  const buttonPressRef = useRef({ timer: null, continuousMode: false });

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
          positionerResponse
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
  };

  //##################################################################################
  // Keyboard event handlers
  const handleKeyDown = (event) => {
    // Prevent default browser behavior for arrow keys and page keys
    if (
      ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "PageUp", "PageDown"].includes(event.key)
    ) {
      event.preventDefault();
    }

    // Ignore browser-generated key-repeat events (fires periodically while key is held)
    if (event.repeat) {
      return;
    }

    // Ignore if key is already pressed (belt-and-suspenders guard)
    if (keyPressedRef.current[event.key]) {
      return;
    }

    console.log(`Key down: ${event.key}`);

    // Initialize all state for this key press
    keyPressedRef.current[event.key] = true;
    continuousModeTriggeredRef.current[event.key] = false; // Reset continuous mode flag

    // Clear any existing timer for this key (just in case)
    if (keyTimersRef.current[event.key]) {
      clearTimeout(keyTimersRef.current[event.key]);
    }

    // Set a timer for 1 second - if still pressed, switch to continuous mode
    keyTimersRef.current[event.key] = setTimeout(() => {
      console.log(`Key ${event.key} held for 1s - starting continuous mode`);

      // Mark that continuous mode was triggered
      continuousModeTriggeredRef.current[event.key] = true;

      // Remove the timer reference since it has fired
      delete keyTimersRef.current[event.key];

      // Key held for more than 1 second, start continuous movement
      let axis = null;
      let speed = continuousMoveSpeed;

      switch (event.key) {
        case "ArrowLeft":
          axis = "X";
          speed = -continuousMoveSpeed; // Negative for left
          break;
        case "ArrowRight":
          axis = "X";
          speed = continuousMoveSpeed; // Positive for right
          break;
        case "ArrowUp":
          axis = "Y";
          speed = -continuousMoveSpeed; // Positive for up
          break;
        case "ArrowDown":
          axis = "Y";
          speed = continuousMoveSpeed; // Negative for down
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
        movePositionerForever(axis, speed, false); // Start continuous movement
      }
    }, 1000); // 1 second delay
  };

  //##################################################################################
  const handleKeyUp = (event) => {
    if (!keyPressedRef.current[event.key]) {
      return;
    }

    console.log(`Key up: ${event.key}`);

    // Determine axis first
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
        // Clean up state even for unhandled keys
        keyPressedRef.current[event.key] = false;
        delete continuousModeTriggeredRef.current[event.key];
        if (keyTimersRef.current[event.key]) {
          clearTimeout(keyTimersRef.current[event.key]);
          delete keyTimersRef.current[event.key];
        }
        return;
    }

    // Check if continuous mode was triggered
    const wasContinuousMode = continuousModeTriggeredRef.current[event.key];
    console.log(`Key ${event.key} - continuous mode was: ${wasContinuousMode}`);

    // Clean up timer if it still exists
    if (keyTimersRef.current[event.key]) {
      clearTimeout(keyTimersRef.current[event.key]);
      delete keyTimersRef.current[event.key];
    }

    if (wasContinuousMode) {
      // Continuous mode was active, just stop it
      console.log(`Stopping continuous mode for ${event.key}`);
      if (axis) {
        movePositionerForever(axis, continuousMoveSpeed, true);
      }
    } else {
      // Do a single move (only if continuous mode was NOT triggered)
      console.log(`Single move for ${event.key}`);
      if (axis) {
        movePositioner(axis, dist);
      }
    }

    // Clean up ALL tracking state for this key
    keyPressedRef.current[event.key] = false;
    delete continuousModeTriggeredRef.current[event.key];

    console.log(`Key ${event.key} state cleaned up`);
  };

  //##################################################################################
  // Add and remove keyboard event listeners
  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    // Cleanup function to remove listeners and clear timers
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);

      // Clear all timers on unmount
      Object.values(keyTimersRef.current).forEach((timer) =>
        clearTimeout(timer)
      );
      keyTimersRef.current = {};
      keyPressedRef.current = {};
      continuousModeTriggeredRef.current = {};
    };
  }, []); // Empty dependency array means this effect runs once on mount

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
        onMouseDown={() => handleButtonDown("Z", -continuousMoveSpeed, -zoomDistance)}
        onMouseUp={handleButtonUp}
        onMouseLeave={handleButtonUp}
        onTouchStart={() => handleButtonDown("Z", -continuousMoveSpeed, -zoomDistance)}
        onTouchEnd={handleButtonUp}
        onTouchCancel={handleButtonUp}
        sx={buttonStyle}
      >
        Z-
      </Button>
      <Button
        variant="contained"
        onMouseDown={() => handleButtonDown("Y", -continuousMoveSpeed, -moveDistance)}
        onMouseUp={handleButtonUp}
        onMouseLeave={handleButtonUp}
        onTouchStart={() => handleButtonDown("Y", -continuousMoveSpeed, -moveDistance)}
        onTouchEnd={handleButtonUp}
        onTouchCancel={handleButtonUp}
        sx={buttonStyle}
      >
        Y↑
      </Button>
      <Button
        variant="contained"
        onMouseDown={() => handleButtonDown("Z", continuousMoveSpeed, zoomDistance)}
        onMouseUp={handleButtonUp}
        onMouseLeave={handleButtonUp}
        onTouchStart={() => handleButtonDown("Z", continuousMoveSpeed, zoomDistance)}
        onTouchEnd={handleButtonUp}
        onTouchCancel={handleButtonUp}
        sx={buttonStyle}
      >
        Z+
      </Button>
      <Button
        variant="contained"
        onMouseDown={() => handleButtonDown("X", -continuousMoveSpeed, -moveDistance)}
        onMouseUp={handleButtonUp}
        onMouseLeave={handleButtonUp}
        onTouchStart={() => handleButtonDown("X", -continuousMoveSpeed, -moveDistance)}
        onTouchEnd={handleButtonUp}
        onTouchCancel={handleButtonUp}
        sx={buttonStyle}
      >
        X←
      </Button>
      <Button
        variant="contained"
        onMouseDown={() => handleButtonDown("Y", continuousMoveSpeed, moveDistance)}
        onMouseUp={handleButtonUp}
        onMouseLeave={handleButtonUp}
        onTouchStart={() => handleButtonDown("Y", continuousMoveSpeed, moveDistance)}
        onTouchEnd={handleButtonUp}
        onTouchCancel={handleButtonUp}
        sx={buttonStyle}
      >
        Y↓
      </Button>
      <Button
        variant="contained"
        onMouseDown={() => handleButtonDown("X", continuousMoveSpeed, moveDistance)}
        onMouseUp={handleButtonUp}
        onMouseLeave={handleButtonUp}
        onTouchStart={() => handleButtonDown("X", continuousMoveSpeed, moveDistance)}
        onTouchEnd={handleButtonUp}
        onTouchCancel={handleButtonUp}
        sx={buttonStyle}
      >
        X→
      </Button>
    </div>
  );
};
//##################################################################################
export default PositionControllerComponent;
