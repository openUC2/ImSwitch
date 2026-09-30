import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import FirmwareVersionsPanel from "./FirmwareVersionsPanel";
import { firmwareUpdateStatus } from "./firmwareStatus";
import apiUC2ConfigControllerCheckFirmwareUpdates from "../backendapi/apiUC2ConfigControllerCheckFirmwareUpdates";

jest.mock("../backendapi/apiUC2ConfigControllerCheckFirmwareUpdates");
jest.mock("../backendapi/apiUC2ConfigControllerGetFirmwareCheckOnConnect", () => () =>
  Promise.resolve({ enabled: false }));
jest.mock("../backendapi/apiUC2ConfigControllerSetFirmwareCheckOnConnect");
jest.mock("./FirmwareUpdateDialog", () => () => null);

const SERVER = "v2026.0.0-beta.4-6-g2108590-t20260930092622";

// Same cases as test_uc2config_firmware_versions.py::test_update_status, so the
// wizard's client-side rule cannot drift from the backend's.
test.each([
  [SERVER, SERVER, "up_to_date"],
  ["UC2-ESP v2.0", SERVER, "update_available"],
  [undefined, SERVER, "update_available"],
  ["v2026.0.0-beta.4-5-gaaaaaaa-t20260901000000", SERVER, "update_available"],
  ["v2026.0.0-beta.4-7-gbbbbbbb-t20261001000000-dirty", SERVER, "device_newer"],
  ["v2026.1.0", SERVER, "update_available"],
  [SERVER, null, "unknown"],
])("firmwareUpdateStatus(%s, %s) = %s", (installed, available, expected) => {
  expect(firmwareUpdateStatus(installed, available)).toBe(expected);
});

test("lists installed vs available per board after a check", async () => {
  apiUC2ConfigControllerCheckFirmwareUpdates.mockResolvedValue({
    status: "success",
    server_version: SERVER,
    updates_available: 1,
    devices: [
      { canId: 1, deviceTypeStr: "UC2_canopen_master", connection: "usb",
        installed_version: SERVER, available_version: SERVER, update_status: "up_to_date" },
      { canId: 11, deviceTypeStr: "motor", connection: "can",
        installed_version: "UC2-ESP v2.0", available_version: SERVER, update_status: "update_available" },
    ],
  });
  render(<FirmwareVersionsPanel />);
  fireEvent.click(screen.getByRole("button", { name: /check firmware versions/i }));

  expect(await screen.findByText("Update available")).toBeInTheDocument();
  expect(screen.getByText("Up to date")).toBeInTheDocument();
  expect(screen.getByText("UC2-ESP v2.0")).toBeInTheDocument();
  expect(screen.getByText(/1 board\(s\) run a different firmware/)).toBeInTheDocument();
});
