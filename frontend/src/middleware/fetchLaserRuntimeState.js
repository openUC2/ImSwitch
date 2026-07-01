const normalizeLaserEnabled = (enabledRaw) =>
  enabledRaw === true ||
  enabledRaw === 1 ||
  enabledRaw === "1" ||
  enabledRaw === "true";

const normalizeLaserPower = (powerRaw) => {
  const parsedPower = Number(powerRaw);
  return Number.isFinite(parsedPower) ? parsedPower : 0;
};

/**
 * Fetch per-laser runtime state (power + enabled) from LaserController.
 * Non-default kinds are returned as neutral values and are not queried.
 */
const fetchLaserRuntimeState = async ({
  hostIP,
  hostPort,
  sources = [],
  kinds = [],
}) => {
  if (!hostIP || !hostPort || !Array.isArray(sources) || sources.length === 0) {
    return [];
  }

  return Promise.all(
    sources.map(async (laserName, index) => {
      const kind = kinds[index] || "default";

      if (kind !== "default") {
        return {
          laserName,
          kind,
          power: 0,
          enabled: false,
          ok: true,
        };
      }

      const encodedLaserName = encodeURIComponent(laserName);

      try {
        const [valueResponse, activeResponse] = await Promise.all([
          fetch(
            `${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserValue?laserName=${encodedLaserName}`,
          ),
          fetch(
            `${hostIP}:${hostPort}/imswitch/api/LaserController/getLaserActive?laserName=${encodedLaserName}`,
          ),
        ]);

        if (!valueResponse.ok || !activeResponse.ok) {
          throw new Error(
            `Laser sync failed for ${laserName}: value=${valueResponse.status}, active=${activeResponse.status}`,
          );
        }

        const [powerRaw, enabledRaw] = await Promise.all([
          valueResponse.json(),
          activeResponse.json(),
        ]);

        return {
          laserName,
          kind,
          power: normalizeLaserPower(powerRaw),
          enabled: normalizeLaserEnabled(enabledRaw),
          ok: true,
        };
      } catch (error) {
        return {
          laserName,
          kind,
          power: 0,
          enabled: false,
          ok: false,
          error,
        };
      }
    }),
  );
};

export default fetchLaserRuntimeState;