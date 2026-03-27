// src/backendapi/apiGoniometerControllerMeasureManual.js
import createAxiosInstance from "./createAxiosInstance";

const apiGoniometerControllerMeasureManual = async (points) => {
  const instance = createAxiosInstance();
  const response = await instance.post(
    "/GoniometerController/measure_manual_goniometer",
    null,
    {
      params: {
        baseline_x1: points.baseline_x1,
        baseline_y1: points.baseline_y1,
        baseline_x2: points.baseline_x2,
        baseline_y2: points.baseline_y2,
        tangent_x: points.tangent_x,
        tangent_y: points.tangent_y,
      },
    }
  );
  return response.data;
};

export default apiGoniometerControllerMeasureManual;
