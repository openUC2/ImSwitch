import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import CameraSettingsDialog from "./CameraSettingsDialog";
import apiGetTree from "../backendapi/apiSettingsControllerGetDetectorParameterTree";
import apiSetParam from "../backendapi/apiSettingsControllerSetDetectorParameterValue";
import apiSetBinning from "../backendapi/apiSettingsControllerSetDetectorBinning";
import apiGetDetectorNames from "../backendapi/apiSettingsControllerGetDetectorNames";

// Explicit factories (hoisted above the imports by babel-jest): the real
// modules pull in axios (ESM), which CRA's jest transform leaves untouched
// inside node_modules.
jest.mock("../backendapi/apiSettingsControllerGetDetectorParameterTree", () => ({
  __esModule: true,
  default: jest.fn(),
}));
jest.mock("../backendapi/apiSettingsControllerSetDetectorParameterValue", () => ({
  __esModule: true,
  default: jest.fn(),
}));
jest.mock("../backendapi/apiSettingsControllerSetDetectorBinning", () => ({
  __esModule: true,
  default: jest.fn(),
}));
jest.mock("../backendapi/apiSettingsControllerGetDetectorNames", () => ({
  __esModule: true,
  default: jest.fn(),
}));

const TREE = {
  detectorName: "WidefieldCamera",
  model: "ToupTek ITR3CMOS",
  isRGB: false,
  isConnected: true,
  isMock: false,
  cameraType: "Toupcam",
  sensorWidth: 4096,
  sensorHeight: 3000,
  currentWidth: 4096,
  currentHeight: 3000,
  pixelSizeUm: [1, 2.4, 2.4],
  binning: 1,
  supportedBinnings: [1, 2, 3, 4],
  groups: [
    {
      name: "Misc",
      parameters: [
        {
          name: "exposure",
          value: 100,
          type: "number",
          units: "ms",
          editable: true,
          min: 0.1,
          max: 5000,
        },
        {
          name: "blacklevel",
          value: 0,
          type: "number",
          units: "arb.u.",
          editable: true,
          min: null,
          max: null,
        },
        {
          name: "exposure_mode",
          value: "manual",
          type: "list",
          options: ["manual", "auto"],
          editable: true,
        },
      ],
    },
    {
      name: "Cooling",
      parameters: [
        {
          name: "temperature",
          value: -9.5,
          type: "number",
          units: "°C",
          editable: false,
          min: null,
          max: null,
        },
        {
          name: "target_temperature",
          value: -10,
          type: "number",
          units: "°C",
          editable: true,
          min: null,
          max: null,
        },
      ],
    },
  ],
};

beforeEach(() => {
  jest.clearAllMocks();
  apiGetTree.mockResolvedValue(TREE);
  apiSetParam.mockResolvedValue(TREE);
  apiSetBinning.mockResolvedValue({ status: "ok", binning: 2 });
  apiGetDetectorNames.mockResolvedValue(["WidefieldCamera"]);
});

describe("CameraSettingsDialog", () => {
  it("shows the camera summary and every parameter group", async () => {
    render(<CameraSettingsDialog open onClose={() => {}} />);

    await screen.findByText(/ToupTek ITR3CMOS/);
    expect(screen.getByText("Sensor 4096 × 3000")).toBeInTheDocument();
    expect(screen.getByText("Toupcam")).toBeInTheDocument();
    expect(screen.getByText("Misc")).toBeInTheDocument();
    expect(screen.getByText("Cooling")).toBeInTheDocument();
    expect(screen.getByLabelText("exposure (ms)")).toBeInTheDocument();
  });

  it("renders read-only values (sensor temperature) as disabled fields", async () => {
    render(<CameraSettingsDialog open onClose={() => {}} />);

    const temperature = await screen.findByLabelText("temperature (°C)");
    expect(temperature).toBeDisabled();
    expect(temperature).toHaveValue("-9.500");
  });

  it("sends a parameter update to the backend when a field is committed", async () => {
    render(<CameraSettingsDialog open onClose={() => {}} />);

    const exposure = await screen.findByLabelText("exposure (ms)");
    fireEvent.change(exposure, { target: { value: "250" } });
    fireEvent.blur(exposure);

    await waitFor(() =>
      expect(apiSetParam).toHaveBeenCalledWith({
        detectorName: "WidefieldCamera",
        name: "exposure",
        value: 250,
      }),
    );
  });

  it("applies binning through the dedicated endpoint", async () => {
    render(<CameraSettingsDialog open onClose={() => {}} />);

    await screen.findByText(/ToupTek ITR3CMOS/);
    fireEvent.mouseDown(screen.getByLabelText("Binning"));
    fireEvent.click(await screen.findByText("2 × 2"));

    await waitFor(() =>
      expect(apiSetBinning).toHaveBeenCalledWith({
        binning: 2,
        detectorName: "WidefieldCamera",
      }),
    );
  });

  it("surfaces backend errors instead of failing silently", async () => {
    apiGetTree.mockRejectedValueOnce(new Error("camera offline"));
    render(<CameraSettingsDialog open onClose={() => {}} />);

    expect(
      await screen.findByText(/Could not load camera parameters: camera offline/),
    ).toBeInTheDocument();
  });
});
