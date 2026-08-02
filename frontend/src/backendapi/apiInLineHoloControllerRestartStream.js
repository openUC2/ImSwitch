// src/backendapi/apiInLineHoloControllerRestartStream.js
// Reset the inline-holo MJPEG stream (drop queued frames, reset counters)

import createAxiosInstance from "./createAxiosInstance";

/**
 * Tell the backend to drop any queued processed frames and reset stream-
 * health counters. The caller should re-mount the MJPEG <img> with a fresh
 * query string afterwards so the browser opens a new connection.
 * @returns {Promise<Object>}
 */
const apiInLineHoloControllerRestartStream = async () => {
  const instance = createAxiosInstance();
  try {
    const response = await instance.get(
      "/InLineHoloController/restart_stream_inlineholo"
    );
    return response.data;
  } catch (error) {
    console.error("Error restarting inline holo stream:", error);
    throw error;
  }
};

export default apiInLineHoloControllerRestartStream;
