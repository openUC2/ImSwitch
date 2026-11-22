import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import CameraHealthBanner from "./CameraHealthBanner";
import apiGetCameraHealth from "../backendapi/apiLiveViewControllerGetCameraHealth";

jest.mock("../backendapi/apiLiveViewControllerGetCameraHealth", () => jest.fn());
jest.mock("../backendapi/apiUC2ConfigControllerRestartImSwitch", () => jest.fn(() => Promise.resolve()));

test("shows a stand-in camera with a restart button, nothing when healthy", async () => {
  apiGetCameraHealth.mockResolvedValueOnce({
    Main: { placeholder: true, stalled: false, message: "Camera Main did not open." },
    Side: { placeholder: false, stalled: false, message: "" },
  });
  render(<CameraHealthBanner />);
  await waitFor(() => expect(screen.getByText(/Camera Main did not open/)).toBeTruthy());
  expect(screen.getByText("Restart ImSwitch")).toBeTruthy();
  expect(screen.queryByText(/Side/)).toBeNull();
});
