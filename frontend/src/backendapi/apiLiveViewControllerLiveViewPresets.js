import createAxiosInstance from "./createAxiosInstance";

/** Saved live-view presets of this instrument: an array, or null if none were ever saved. */
export const apiLiveViewControllerGetLiveViewPresets = async () => {
  const response = await createAxiosInstance().get(
    "/LiveViewController/getLiveViewPresets",
  );
  return response.data?.presets ?? null;
};

/** Replace the instrument's live-view presets with this array. */
export const apiLiveViewControllerSetLiveViewPresets = async (presets) => {
  const response = await createAxiosInstance().post(
    "/LiveViewController/setLiveViewPresets",
    { presets },
  );
  return response.data;
};
