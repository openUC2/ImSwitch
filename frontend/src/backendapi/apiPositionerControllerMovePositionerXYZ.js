// src/backendapi/apiPositionerControllerMovePositionerXYZ.js
//
// Move several stage axes in ONE request. Prefer this over firing
// apiPositionerControllerMovePositioner once per axis: two back-to-back
// single-axis requests race each other on the wire and make the stage stutter
// (and pick up backlash) instead of travelling a straight line. The backend
// issues a single coordinated command on stages that support it and falls back
// to sequential axes on stages that don't — same semantics either way.

import createAxiosInstance from "./createAxiosInstance";

/**
 * @param {object}  params
 * @param {string} [params.positionerName] Defaults to the first positioner.
 * @param {number} [params.x] Target for X in µm. Omit to leave the axis alone.
 * @param {number} [params.y] Target for Y in µm.
 * @param {number} [params.z] Target for Z in µm.
 * @param {number} [params.a] Target for A in µm.
 * @param {boolean} [params.isAbsolute=true] false => values are relative offsets.
 * @param {boolean} [params.isBlocking=false] Wait for the motion to finish.
 * @param {number} [params.speed] µm/s applied to every commanded axis.
 * @returns {Promise<object>} { positionerName, movedAxes, target, position, isAbsolute, isBlocking }
 */
const apiPositionerControllerMovePositionerXYZ = async ({
  positionerName,
  x,
  y,
  z,
  a,
  isAbsolute = true,
  isBlocking = false,
  speed,
} = {}) => {
  try {
    const axiosInstance = createAxiosInstance();

    // Only send the axes the caller actually specified — an omitted axis must
    // stay put, and 0 is a valid target (so check for null/undefined, not
    // falsiness).
    const params = { isAbsolute, isBlocking };
    if (positionerName) params.positionerName = positionerName;
    if (x !== undefined && x !== null) params.x = x;
    if (y !== undefined && y !== null) params.y = y;
    if (z !== undefined && z !== null) params.z = z;
    if (a !== undefined && a !== null) params.a = a;
    if (speed !== undefined && speed !== null) params.speed = speed;

    const response = await axiosInstance.get(
      "/PositionerController/movePositionerXYZ",
      { params },
    );
    return response.data;
  } catch (error) {
    console.error("Error moving positioner (XYZ):", error);
    throw error;
  }
};

export default apiPositionerControllerMovePositionerXYZ;
