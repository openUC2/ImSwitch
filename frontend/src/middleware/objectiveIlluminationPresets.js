const STORAGE_KEY = "imswitch.objectiveIlluminationPresets.v1";

const readStoredPresets = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (_e) {
    return {};
  }
};

const writeStoredPresets = (presets) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets || {}));
  } catch (_e) {
    /* quota or disabled storage — ignore */
  }
};

const snapshotLaserState = (laserState) => {
  const lasers = laserState?.lasers || {};
  return Object.entries(lasers).reduce(
    (accumulator, [laserName, laserData]) => {
      if (!laserName || !laserData || typeof laserData !== "object") {
        return accumulator;
      }

      accumulator[laserName] = {
        power: laserData.power ?? null,
        enabled: laserData.enabled ?? null,
      };
      return accumulator;
    },
    {},
  );
};

const fetchDetectorState = async ({ hostIP, hostPort }) => {
  try {
    const response = await fetch(
      `${hostIP}:${hostPort}/imswitch/api/SettingsController/getDetectorParameters`,
    );
    if (!response.ok) {
      return { exposure: null, gain: null };
    }

    const data = await response.json();
    return {
      exposure: data?.exposure ?? null,
      gain: data?.gain ?? null,
    };
  } catch (_e) {
    return { exposure: null, gain: null };
  }
};

export const rememberObjectiveIllumination = async ({
  objectiveSlot,
  laserState,
  hostIP,
  hostPort,
}) => {
  if (objectiveSlot === null || objectiveSlot === undefined) return;

  const lasers = snapshotLaserState(laserState);
  const detector = await fetchDetectorState({ hostIP, hostPort });
  if (
    Object.keys(lasers).length === 0 &&
    detector.exposure == null &&
    detector.gain == null
  ) {
    return;
  }

  const presets = readStoredPresets();
  presets[String(objectiveSlot)] = {
    objectiveSlot,
    savedAt: new Date().toISOString(),
    lasers,
    detector,
  };
  writeStoredPresets(presets);
};

export const restoreObjectiveIllumination = async ({
  objectiveSlot,
  hostIP,
  hostPort,
  dispatch,
  laserSlice,
  stormSlice,
  detectorParametersSlice,
}) => {
  if (objectiveSlot === null || objectiveSlot === undefined) {
    return { restored: false, errors: [] };
  }

  const presets = readStoredPresets();
  const preset = presets[String(objectiveSlot)];
  if (!preset?.lasers || typeof preset.lasers !== "object") {
    return { restored: false, errors: [] };
  }

  const errors = [];

  const detector = preset.detector || {};
  if (detector.exposure !== null && detector.exposure !== undefined) {
    try {
      const response = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorExposureTime?exposureTime=${detector.exposure}`,
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      if (stormSlice?.setExposureTime) {
        dispatch(stormSlice.setExposureTime(Number(detector.exposure)));
      }
      if (detectorParametersSlice?.updateParameter) {
        dispatch(
          detectorParametersSlice.updateParameter({
            key: "exposure",
            value: Number(detector.exposure),
          }),
        );
      }
    } catch (error) {
      errors.push(`detector exposure: ${error.message || error}`);
    }
  }

  if (detector.gain !== null && detector.gain !== undefined) {
    try {
      const response = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/SettingsController/setDetectorGain?gain=${detector.gain}`,
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      if (detectorParametersSlice?.updateParameter) {
        dispatch(
          detectorParametersSlice.updateParameter({
            key: "gain",
            value: Number(detector.gain),
          }),
        );
      }
    } catch (error) {
      errors.push(`detector gain: ${error.message || error}`);
    }
  }

  for (const [laserName, laserData] of Object.entries(preset.lasers)) {
    if (!laserName || !laserData) continue;

    const encodedLaserName = encodeURIComponent(laserName);
    const { power, enabled } = laserData;

    if (power !== null && power !== undefined) {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/LaserController/setLaserValue?laserName=${encodedLaserName}&value=${power}`,
        );
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        dispatch(laserSlice.setLaserPower({ laserName, power: Number(power) }));
      } catch (error) {
        errors.push(`laser ${laserName} power: ${error.message || error}`);
      }
    }

    if (enabled !== null && enabled !== undefined) {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/LaserController/setLaserActive?laserName=${encodedLaserName}&active=${enabled}`,
        );
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        dispatch(laserSlice.setLaserEnabled({ laserName, enabled: !!enabled }));
      } catch (error) {
        errors.push(`laser ${laserName} active: ${error.message || error}`);
      }
    }
  }

  return { restored: true, errors };
};
