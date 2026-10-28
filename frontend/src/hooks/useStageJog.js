// src/hooks/useStageJog.js
// Press-and-hold stage jogging, shared by the on-stream pad
// (axon/PositionControllerComponent) and the touch stage panel.
//
// A press sends one relative step immediately (no waiting to rule out a hold,
// which made taps feel laggy). Holding then either
//   - "continuous": starts a constant-velocity move after `holdDelayMs` and
//     stops it on release, or
//   - "repeat": keeps sending single steps. Each step is bounded, so nothing
//     keeps moving once the finger is lifted — used for Z, where an unstopped
//     move drives the objective into the sample.
// Pointer capture delivers the release even if the finger slides off the
// button, and any move still held when the component unmounts is stopped.
import { useCallback, useEffect, useRef } from "react";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";
import apiPositionerControllerMovePositionerForever from "../backendapi/apiPositionerControllerMovePositionerForever.js";

const REPEAT_INTERVAL_MS = 300;

export default function useStageJog({ holdDelayMs = 1000 } = {}) {
  const pressesRef = useRef({});

  const step = useCallback((axis, dist) => {
    apiPositionerControllerMovePositioner({ axis, dist, isAbsolute: false })
      .catch((error) => console.error(`Move ${axis} by ${dist} failed:`, error));
  }, []);

  const moveForever = useCallback((axis, speed, isStop) => {
    apiPositionerControllerMovePositionerForever({ axis, speed, is_stop: isStop })
      .catch((error) => console.error(`Move forever ${axis} failed:`, error));
  }, []);

  const release = useCallback(
    (axis) => {
      const press = pressesRef.current[axis];
      if (!press) return;
      clearTimeout(press.timer);
      clearInterval(press.interval);
      if (press.continuous) moveForever(axis, press.speed, true);
      delete pressesRef.current[axis];
    },
    [moveForever],
  );

  /**
   * @param axis   "X" | "Y" | "Z" | "A"
   * @param dist   signed step in µm, sent on press
   * @param speed  signed velocity for the continuous hold (mode "continuous")
   * @param mode   "continuous" | "repeat" | "single"
   */
  const press = useCallback(
    (axis, { dist, speed = 0, mode = "continuous" }, event) => {
      if (event?.button > 0) return; // right/middle mouse button
      event?.currentTarget?.setPointerCapture?.(event.pointerId);
      if (pressesRef.current[axis]) return; // already held

      step(axis, dist);
      const entry = { continuous: false, speed };
      if (mode === "continuous") {
        entry.timer = setTimeout(() => {
          entry.continuous = true;
          moveForever(axis, speed, false);
        }, holdDelayMs);
      } else if (mode === "repeat") {
        entry.timer = setTimeout(() => {
          entry.interval = setInterval(() => step(axis, dist), REPEAT_INTERVAL_MS);
        }, holdDelayMs);
      }
      pressesRef.current[axis] = entry;
    },
    [holdDelayMs, moveForever, step],
  );

  // Never leave a held move running behind an unmounted pad.
  useEffect(
    () => () => Object.keys(pressesRef.current).forEach((axis) => release(axis)),
    [release],
  );

  /** Spread onto a button: `<Button {...jogHandlers("X", {...})}>`. */
  const jogHandlers = useCallback(
    (axis, options) => ({
      onPointerDown: (event) => press(axis, options, event),
      onPointerUp: () => release(axis),
      onPointerCancel: () => release(axis),
      onLostPointerCapture: () => release(axis),
      // A long press must not open the browser's context menu on touch.
      onContextMenu: (event) => event.preventDefault(),
    }),
    [press, release],
  );

  return { step, press, release, jogHandlers };
}
