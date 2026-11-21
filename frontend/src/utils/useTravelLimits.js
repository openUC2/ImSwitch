// Travel limits of the first positioner (user frame, µm) and the last move the
// backend refused (sigStageLimit). WebSocketHandler fires STAGE_LIMIT_EVENT.
import { useEffect, useState } from "react";
import apiPositionerControllerGetStageTravelLimits from "../backendapi/apiPositionerControllerGetStageTravelLimits";

export const STAGE_LIMIT_EVENT = "uc2:stage-limit";

export default function useTravelLimits() {
  const [limits, setLimits] = useState({}); // { X: { min, max, enabled }, ... }
  const [refusal, setRefusal] = useState(null); // { message, at }

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      apiPositionerControllerGetStageTravelLimits()
        .then((all) => !cancelled && setLimits(Object.values(all || {})[0] || {}))
        .catch(() => {});
    load();
    const timer = setInterval(load, 30000); // limits change on calibration / motor settings
    const onLimit = (e) => {
      setRefusal({ message: e.detail?.message || "Move refused", at: Date.now() });
      load();
    };
    window.addEventListener(STAGE_LIMIT_EVENT, onLimit);
    return () => {
      cancelled = true;
      clearInterval(timer);
      window.removeEventListener(STAGE_LIMIT_EVENT, onLimit);
    };
  }, []);

  return { limits, refusal };
}
