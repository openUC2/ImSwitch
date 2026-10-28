// src/components/objective/useObjectiveActions.js
// Everything the objective UIs do to the backend, in one place: switching
// slots (with the per-objective illumination presets), jogging the turret and
// focus, storing positions / optics, homing and the switch speed.
//
// Used by the full ObjectiveController page and the compact ObjectiveSwitcher
// in the Live View, which previously each carried their own copy of the
// switching logic (and disagreed on when a switch was "done").
import { useCallback, useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice.js";
import * as laserSlice from "../../state/slices/LaserSlice.js";
import * as stormSlice from "../../state/slices/STORMSlice.js";
import * as detectorParametersSlice from "../../state/slices/DetectorParametersSlice.js";
import { setNotification } from "../../state/slices/NotificationSlice.js";
import { getConnectionSettingsState } from "../../state/slices/ConnectionSettingsSlice";
import apiObjectiveControllerMoveToObjective from "../../backendapi/apiObjectiveControllerMoveToObjective.js";
import apiObjectiveControllerSetPositions from "../../backendapi/apiObjectiveControllerSetPositions.js";
import apiObjectiveControllerCalibrateObjective from "../../backendapi/apiObjectiveControllerCalibrateObjective.js";
import apiObjectiveControllerSetObjectiveParameters from "../../backendapi/apiObjectiveControllerSetObjectiveParameters.js";
import apiObjectiveControllerSetMoveSpeed from "../../backendapi/apiObjectiveControllerSetMoveSpeed.js";
import apiPositionerControllerMovePositioner from "../../backendapi/apiPositionerControllerMovePositioner.js";
import apiPositionerControllerGetPositions from "../../backendapi/apiPositionerControllerGetPositions.js";
import fetchObjectiveControllerGetStatus from "../../middleware/fetchObjectiveControllerGetStatus.js";
import fetchObjectiveControllerGetCurrentObjective from "../../middleware/fetchObjectiveControllerGetCurrentObjective.js";
import {
  rememberObjectiveIllumination,
  restoreObjectiveIllumination,
} from "../../middleware/objectiveIlluminationPresets.js";

// moveToObjective only acknowledges the request; the turret moves on a backend
// thread and sigObjectiveChanged reports the new slot when it arrives.
const SWITCH_TIMEOUT_MS = 15000;

const errorText = (e) => e?.response?.data?.detail || e?.message || String(e);

export default function useObjectiveActions() {
  const dispatch = useDispatch();
  const { ip: hostIP, apiPort: hostPort } = useSelector(getConnectionSettingsState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const laserState = useSelector(laserSlice.getLaserState);

  const [pendingSlot, setPendingSlot] = useState(null);
  const timeoutRef = useRef(null);
  const pendingRef = useRef(null);
  pendingRef.current = pendingSlot;

  const notify = useCallback(
    (message, type = "info") => dispatch(setNotification({ message, type })),
    [dispatch],
  );

  const refresh = useCallback(() => {
    fetchObjectiveControllerGetStatus(dispatch);
    fetchObjectiveControllerGetCurrentObjective(dispatch);
  }, [dispatch]);

  const clearSwitchTimeout = () => {
    clearTimeout(timeoutRef.current);
    timeoutRef.current = null;
  };

  // The backend's sigObjectiveChanged updates currentObjective: switch done.
  useEffect(() => {
    if (objectiveState.currentObjective != null) {
      clearSwitchTimeout();
      setPendingSlot(null);
    }
  }, [objectiveState.currentObjective]);

  useEffect(() => clearSwitchTimeout, []);

  const switchTo = useCallback(
    async (slot, { withZ = true } = {}) => {
      if (pendingRef.current !== null) return;
      if (objectiveState.slotConfigured?.[slot] === false) {
        notify(
          "This slot is not configured yet — set its name and magnification first.",
          "warning",
        );
        return;
      }
      if (objectiveState.currentObjective === slot) {
        notify("This objective is already in the beam.", "info");
        return;
      }
      try {
        await rememberObjectiveIllumination({
          objectiveSlot: objectiveState.currentObjective,
          laserState,
          hostIP,
          hostPort,
        });
        setPendingSlot(slot);
        await apiObjectiveControllerMoveToObjective(slot, !withZ);
        const restored = await restoreObjectiveIllumination({
          objectiveSlot: slot,
          hostIP,
          hostPort,
          dispatch,
          laserSlice,
          stormSlice,
          detectorParametersSlice,
        });
        if (restored?.errors?.length > 0) {
          notify(
            `Objective switched, but some illumination values could not be restored: ${restored.errors.join("; ")}`,
            "warning",
          );
        }
        clearSwitchTimeout();
        timeoutRef.current = setTimeout(() => {
          timeoutRef.current = null;
          if (pendingRef.current === null) return;
          setPendingSlot(null);
          refresh();
          notify(
            "Objective switch timed out. Check the hardware connection and try again.",
            "error",
          );
        }, SWITCH_TIMEOUT_MS);
      } catch (e) {
        clearSwitchTimeout();
        setPendingSlot(null);
        refresh();
        notify(`Objective switch failed: ${errorText(e)}`, "error");
      }
    },
    [objectiveState.slotConfigured, objectiveState.currentObjective, laserState, hostIP, hostPort, dispatch, notify, refresh],
  );

  /** Relative move of one axis in µm (non-blocking). */
  const jog = useCallback(
    (axis, dist) =>
      apiPositionerControllerMovePositioner({
        axis,
        dist,
        isAbsolute: false,
        isBlocking: false,
      }).catch((e) => notify(`Move ${axis} failed: ${errorText(e)}`, "error")),
    [notify],
  );

  /** Fresh A/Z from the backend (the socket only pushes while moving). */
  const readCurrentAZ = useCallback(async () => {
    const data = await apiPositionerControllerGetPositions();
    const stage =
      data?.ESP32Stage || data?.VirtualStage || Object.values(data || {})[0];
    if (!stage) throw new Error("No positioner reported a position");
    dispatch(objectiveSlice.setCurrentA(stage.A));
    dispatch(objectiveSlice.setCurrentZ(stage.Z));
    return { a: Number(stage.A), z: Number(stage.Z) };
  }, [dispatch]);

  /** key: "x0" | "x1" (turret A position) or "z0" | "z1" (par-focal Z). */
  const storePosition = useCallback(
    async (key, value) => {
      try {
        await apiObjectiveControllerSetPositions({ [key]: value, isBlocking: false });
        refresh();
        notify("Position stored", "success");
      } catch (e) {
        notify(`Could not store the position: ${errorText(e)}`, "error");
      }
    },
    [notify, refresh],
  );

  /** optics: { name, magnification, NA, pixelsize } — empty fields are skipped. */
  const saveOptics = useCallback(
    async (slot, optics) => {
      const params = { objectiveSlot: slot };
      if (optics.name) params.objectiveName = optics.name;
      if (optics.magnification !== "" && optics.magnification != null)
        params.magnification = parseInt(optics.magnification, 10);
      if (optics.NA !== "" && optics.NA != null) params.NA = parseFloat(optics.NA);
      if (optics.pixelsize !== "" && optics.pixelsize != null)
        params.pixelsize = parseFloat(optics.pixelsize);
      try {
        await apiObjectiveControllerSetObjectiveParameters(params);
        refresh();
        notify(`Slot ${slot + 1} optics saved`, "success");
        return true;
      } catch (e) {
        notify(`Could not save the optics: ${errorText(e)}`, "error");
        return false;
      }
    },
    [notify, refresh],
  );

  /** Home the turret (A) against its endstop. */
  const home = useCallback(async () => {
    try {
      await apiObjectiveControllerCalibrateObjective();
      refresh();
      notify("Turret homing started", "info");
    } catch (e) {
      notify(`Homing failed: ${errorText(e)}`, "error");
    }
  }, [notify, refresh]);

  const saveMoveSpeed = useCallback(
    async (speed) => {
      try {
        await apiObjectiveControllerSetMoveSpeed(speed);
        dispatch(objectiveSlice.setMoveSpeed(speed));
        notify(`Switch speed set to ${speed}`, "success");
      } catch (e) {
        notify(`Could not set the switch speed: ${errorText(e)}`, "error");
      }
    },
    [dispatch, notify],
  );

  return {
    objectiveState,
    pendingSlot,
    isSwitching: pendingSlot !== null,
    switchTo,
    jog,
    readCurrentAZ,
    storePosition,
    saveOptics,
    home,
    saveMoveSpeed,
    refresh,
    notify,
  };
}
