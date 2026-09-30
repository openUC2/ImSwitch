import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import FirmwareUpdateDialog from "./FirmwareUpdateDialog";
import apiStart from "../backendapi/apiUC2ConfigControllerStartFirmwareUpdate";
import apiStatus from "../backendapi/apiUC2ConfigControllerGetFirmwareUpdateStatus";

jest.mock("../backendapi/apiUC2ConfigControllerCheckFirmwareUpdates");
jest.mock("../backendapi/apiUC2ConfigControllerStartFirmwareUpdate");
jest.mock("../backendapi/apiUC2ConfigControllerGetFirmwareUpdateStatus");
jest.mock("../backendapi/apiUC2ConfigControllerCancelFirmwareUpdate");
jest.mock("react-redux", () => ({
  useSelector: (select) =>
    select({
      canOtaState: { updateProgress: { 12: { progress: 40, message: "Page 86/215 - 17.6 KB/s" } } },
      usbFlash: {},
    }),
}));

const NEW = "v2026.0.0-beta.4-7-gbd08544-t20260930125929";
const CHECK = {
  server_version: NEW,
  updates_available: 2,
  devices: [
    { canId: 1, connection: "usb", deviceTypeStr: "UC2_canopen_master", installed_version: "UC2-ESP v2.0",
      filename: "esp32_UC2_canopen_master_release.bin", available_version: NEW, update_status: "update_available" },
    { canId: 12, connection: "can", deviceTypeStr: "motor", installed_version: "UC2-ESP v2.0",
      filename: "esp32_UC2_canopen_slave_motor_release_motY.bin", available_version: NEW, update_status: "update_available" },
    { canId: 30, connection: "can", deviceTypeStr: "led", installed_version: NEW,
      filename: "esp32_UC2_canopen_slave_led_release.bin", available_version: NEW, update_status: "up_to_date" },
    { canId: 20, connection: "can", deviceTypeStr: "laser", installed_version: null,
      filename: "esp32_UC2_canopen_slave_laser_release.bin", available_version: NEW, update_status: "unreachable" },
  ],
};

beforeEach(() => {
  apiStatus.mockResolvedValue({ state: "idle", steps: [] });
});

const openWithCheck = async () => {
  render(<FirmwareUpdateDialog open onClose={() => {}} initialCheck={CHECK} />);
  return screen.findByRole("button", { name: "Update 2 boards" });
};

test("preselects the outdated boards and sends them, master flagged separately", async () => {
  const startButton = await openWithCheck();
  expect(screen.getByLabelText("update motor 12")).toBeChecked();
  expect(screen.getByLabelText("update UC2_canopen_master (USB master)")).toBeChecked();
  expect(screen.getByLabelText("update led 30")).not.toBeChecked();
  expect(screen.getByLabelText("update laser 20")).toBeDisabled();

  apiStart.mockResolvedValue({
    status: "started",
    steps: [{ canId: 12, connection: "can", deviceTypeStr: "motor", from_version: "UC2-ESP v2.0",
              to_version: NEW, status: "running", message: "" }],
  });
  fireEvent.click(startButton);

  await waitFor(() => expect(apiStart).toHaveBeenCalledWith([12], true));
  expect(await screen.findByText("Page 86/215 - 17.6 KB/s")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Stop after current board" })).toBeInTheDocument();
});

test("shows why the backend refused to start", async () => {
  const startButton = await openWithCheck();
  apiStart.mockResolvedValue({ status: "refused", reasons: ["An experiment is running."] });
  fireEvent.click(startButton);
  expect(await screen.findByText("An experiment is running.")).toBeInTheDocument();
});

test("reopening during an update shows the running update", async () => {
  apiStatus.mockResolvedValue({
    state: "running",
    message: "Updating motor 12…",
    steps: [{ canId: 12, connection: "can", deviceTypeStr: "motor", from_version: "UC2-ESP v2.0",
              to_version: NEW, status: "running", message: "" }],
  });
  render(<FirmwareUpdateDialog open onClose={() => {}} initialCheck={CHECK} />);
  expect(await screen.findByText("Updating motor 12…")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Hide (keeps running)" })).toBeInTheDocument();
});
