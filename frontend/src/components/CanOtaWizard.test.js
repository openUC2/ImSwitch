import React from "react";
import { Provider } from "react-redux";
import { act, render, screen } from "@testing-library/react";
import store from "../state/store";
import * as canOtaSlice from "../state/slices/canOtaSlice";
import CanOtaWizard from "./CanOtaWizard";

jest.mock("../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer", () => () =>
  Promise.resolve({ firmware_server_url: "http://fw/firmware" }));
jest.mock("../backendapi/apiUC2ConfigControllerGetOTADeviceMapping", () => () =>
  Promise.resolve({ status: "success", mapping: { master: 1, motors: { Y: 12 }, laser: { laser_0: 20 } } }));

// The wizard is CAN streaming only: five steps, no method or WiFi step.
test("walks the five CAN OTA steps", async () => {
  render(
    <Provider store={store}>
      <CanOtaWizard open onClose={() => {}} />
    </Provider>,
  );
  expect(await screen.findByText("Firmware Server Configuration")).toBeInTheDocument();
  for (const label of ["Firmware Server", "Scan Devices", "Select Devices", "Update Progress", "Complete"]) {
    expect(screen.getAllByText(label).length).toBeGreaterThan(0);
  }
  expect(screen.queryByText(/WiFi/)).not.toBeInTheDocument();

  act(() => {
    store.dispatch(canOtaSlice.setScannedDevices([
      { canId: 12, deviceTypeStr: "motor", statusStr: "idle", fwVersion: "v1" },
    ]));
    store.dispatch(canOtaSlice.setCurrentStep(1));
  });
  expect(screen.getByText("Scan for CAN Devices")).toBeInTheDocument();
  expect(screen.getByText("MOTOR Y (CAN ID: 12)")).toBeInTheDocument(); // label from the mapping

  act(() => store.dispatch(canOtaSlice.setCurrentStep(2)));
  expect(screen.getByText("Select Devices to Update")).toBeInTheDocument();

  act(() => store.dispatch(canOtaSlice.setCurrentStep(3)));
  expect(screen.getByRole("button", { name: /start/i })).toBeInTheDocument();

  act(() => store.dispatch(canOtaSlice.nextStep())); // progress -> completion is reachable
  expect(store.getState().canOtaState.currentStep).toBe(4);
});
