import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import FirmwareUpdateDialog from "./FirmwareUpdateDialog";
import apiStart from "../backendapi/apiUC2ConfigControllerStartFirmwareUpdate";
import apiStatus from "../backendapi/apiUC2ConfigControllerGetFirmwareUpdateStatus";
import apiCheck from "../backendapi/apiUC2ConfigControllerCheckFirmwareUpdates";
import apiGetSource from "../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer";
import apiReleases from "../backendapi/apiUC2ConfigControllerListFirmwareReleases";
import apiSetRelease from "../backendapi/apiUC2ConfigControllerSetOTAFirmwareRelease";

jest.mock("../backendapi/apiUC2ConfigControllerCheckFirmwareUpdates");
jest.mock("../backendapi/apiUC2ConfigControllerStartFirmwareUpdate");
jest.mock("../backendapi/apiUC2ConfigControllerGetFirmwareUpdateStatus");
jest.mock("../backendapi/apiUC2ConfigControllerCancelFirmwareUpdate");
jest.mock("../backendapi/apiUC2ConfigControllerSetOTAFirmwareServer");
jest.mock("../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer");
jest.mock("../backendapi/apiUC2ConfigControllerListFirmwareReleases");
jest.mock("../backendapi/apiUC2ConfigControllerSetOTAFirmwareRelease");
jest.mock("../backendapi/apiUC2ConfigControllerReassignCANId");
jest.mock("./UsbFlashWizard", () => () => <div>usb flash panel</div>);
jest.mock("react-redux", () => ({
  useSelector: (select) =>
    select({
      canOtaState: { updateProgress: { 12: { progress: 40, message: "Page 86/215 - 17.6 KB/s" } } },
      usbFlash: { isFlashing: false },
    }),
}));

const NEW = "v2026.0.0-beta.4-7-gbd08544-t20260930125929";
const device = (canId, connection, deviceTypeStr, installed, status, available = NEW) => ({
  canId, connection, deviceTypeStr, installed_version: installed, update_status: status,
  filename: `esp32_${deviceTypeStr}.bin`, available_version: available,
});
const CHECK = {
  firmware_server: "http://fw/firmware",
  server_reachable: true,
  server_version: NEW,
  updates_available: 2,
  devices: [
    device(1, "usb", "UC2_canopen_master", "UC2-ESP v2.0", "update_available"),
    device(12, "can", "motor", "UC2-ESP v2.0", "update_available"),
    device(30, "can", "led", NEW, "up_to_date"),
    device(20, "can", "laser", null, "unreachable"),
  ],
};
const RUNNING = {
  state: "running",
  message: "Updating motor 12…",
  steps: [{ canId: 12, connection: "can", deviceTypeStr: "motor", from_version: "UC2-ESP v2.0",
            to_version: NEW, status: "running", message: "" }],
};

beforeEach(() => {
  apiStatus.mockResolvedValue({ state: "idle", steps: [] });
});

const openCan = async (check = CHECK) => {
  render(<FirmwareUpdateDialog open onClose={() => {}} initialCheck={check} initialMethod="can" />);
  await screen.findByLabelText("update motor 12"); // the check result is shown
  return screen.getByRole("button", { name: /^Update \d+ boards?$/ });
};

test("asks for the method first; USB opens the flash panel", async () => {
  render(<FirmwareUpdateDialog open onClose={() => {}} />);
  expect(screen.getByText("Over the CAN bus")).toBeInTheDocument();
  fireEvent.click(screen.getByText("Over a USB cable"));
  expect(await screen.findByText("usb flash panel")).toBeInTheDocument();
  expect(screen.getByText("Update firmware over a USB cable")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "← Other method" }));
  expect(screen.getByText("Over the CAN bus")).toBeInTheDocument();
});

test("the CAN bus needs a connected master", () => {
  render(<FirmwareUpdateDialog open onClose={() => {}} canAvailable={false} />);
  expect(screen.getByText("Needs a connected master board.")).toBeInTheDocument();
  expect(screen.getByText("Over the CAN bus").closest("button")).toBeDisabled();
});

test("preselects the outdated boards and sends them, master flagged separately", async () => {
  const startButton = await openCan();
  expect(startButton).toHaveTextContent("Update 2 boards");
  expect(screen.getByLabelText("update motor 12")).toBeChecked();
  expect(screen.getByLabelText("update UC2_canopen_master (USB master)")).toBeChecked();
  expect(screen.getByLabelText("update led 30")).not.toBeChecked();
  expect(screen.getByLabelText("update laser 20")).toBeDisabled();

  apiStart.mockResolvedValue({ status: "started", steps: RUNNING.steps });
  fireEvent.click(startButton);

  await waitFor(() => expect(apiStart).toHaveBeenCalledWith([12], true));
  expect(await screen.findByText("Page 86/215 - 17.6 KB/s")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Stop after current board" })).toBeInTheDocument();
});

test("shows why the backend refused to start", async () => {
  const startButton = await openCan();
  apiStart.mockResolvedValue({ status: "refused", reasons: ["An experiment is running."] });
  fireEvent.click(startButton);
  expect(await screen.findByText("An experiment is running.")).toBeInTheDocument();
});

test("without version.json boards are chosen by hand", async () => {
  const unversioned = {
    ...CHECK, server_version: null, updates_available: 0,
    devices: [device(12, "can", "motor", "UC2-ESP v2.0", "unknown", null),
              device(13, "can", "motor", "UC2-ESP v2.0", "no_firmware", null)],
  };
  const startButton = await openCan(unversioned);
  expect(screen.getByText(/has no version.json/)).toBeInTheDocument();
  expect(startButton).toBeDisabled(); // nothing preselected
  fireEvent.click(screen.getByLabelText("update motor 12"));
  expect(screen.getByLabelText("update motor 13")).toBeDisabled();
  expect(screen.getByRole("button", { name: "Update 1 board" })).toBeEnabled();
});

test("reopening during an update shows the running update", async () => {
  apiStatus.mockResolvedValue(RUNNING);
  render(<FirmwareUpdateDialog open onClose={() => {}} />);
  expect(await screen.findByText("Updating motor 12…")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Hide (keeps running)" })).toBeInTheDocument();
});

test("switches the firmware source to a GitHub release and checks again", async () => {
  apiGetSource.mockResolvedValue({ source: "local", firmware_server_url: "http://fw/firmware",
                                   github_repo: "youseetoo/uc2-esp32", github_release: null });
  apiReleases.mockResolvedValue({ status: "success", releases: [
    { tag: "v2026.01.19-3", prerelease: false, published_at: "2026-01-19T16:16:09Z",
      image_count: 0 },
    { tag: "v2026.0.0-beta.4", prerelease: true, published_at: "2026-09-18T06:56:12Z",
      image_count: 36 },
  ] });
  apiSetRelease.mockResolvedValue({ status: "success", release_tag: "v2026.0.0-beta.4" });
  apiCheck.mockResolvedValue(CHECK);
  await openCan();
  fireEvent.click(screen.getByRole("button", { name: "Change" }));
  fireEvent.click(await screen.findByRole("button", { name: "GitHub release" }));
  fireEvent.mouseDown(screen.getByLabelText("Release"));
  expect(await screen.findByRole("option", { name: "Newest release (v2026.0.0-beta.4)" }))
    .toBeInTheDocument();
  // a release with only web-flasher images cannot be chosen
  expect(screen.getByRole("option", { name: /v2026.01.19-3 · .* · no ImSwitch images/ }))
    .toHaveAttribute("aria-disabled", "true");
  fireEvent.click(screen.getByRole("option", { name: /v2026.0.0-beta.4 · pre-release/ }));
  fireEvent.click(screen.getByRole("button", { name: "Use" }));
  await waitFor(() => expect(apiSetRelease).toHaveBeenCalledWith("v2026.0.0-beta.4"));
  await waitFor(() => expect(apiCheck).toHaveBeenCalled());
  expect(screen.queryByRole("button", { name: "Use" })).not.toBeInTheDocument();
});
