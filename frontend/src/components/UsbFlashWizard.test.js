import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";

import UsbFlashWizard from "./UsbFlashWizard";
import usbFlashReducer, * as usbFlashSlice from "../state/slices/usbFlashSlice";
import apiRecommend from "../backendapi/apiUC2ConfigControllerGetRecommendedFirmware";
import apiServer from "../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer";
import apiPorts from "../backendapi/apiUC2ConfigControllerListSerialPorts";
import apiFiles from "../backendapi/apiUC2ConfigControllerListAllFirmwareFiles";
import apiReleases from "../backendapi/apiUC2ConfigControllerListFirmwareReleases";
import apiSetRelease from "../backendapi/apiUC2ConfigControllerSetOTAFirmwareRelease";

jest.mock("../backendapi/apiUC2ConfigControllerGetRecommendedFirmware");
jest.mock("../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer");
jest.mock("../backendapi/apiUC2ConfigControllerListSerialPorts");
jest.mock("../backendapi/apiUC2ConfigControllerListAllFirmwareFiles");
jest.mock("../backendapi/apiUC2ConfigControllerListFirmwareReleases");
jest.mock("../backendapi/apiUC2ConfigControllerSetOTAFirmwareRelease");

const MASTER = "esp32_UC2_canopen_master_release.bin";
const file = (filename) => ({ filename, size: 1024 * 900, mod_time: "t", url: `http://fw/${filename}` });
const FILES = [file("esp32_UC2_3.bin"), file("esp32_seeed_xiao_esp32s3.bin"), file(MASTER)];

const mountAtFirmwareStep = () => {
  const store = configureStore({ reducer: { usbFlash: usbFlashReducer } });
  store.dispatch(usbFlashSlice.setFirmwareFiles(FILES));
  store.dispatch(usbFlashSlice.setCurrentStep(1));
  render(<Provider store={store}><UsbFlashWizard open onClose={() => {}} /></Provider>);
  return store;
};

beforeEach(() => {
  apiServer.mockResolvedValue({ firmware_server_url: "http://fw" });
  apiPorts.mockResolvedValue([{ device: "/dev/ttyACM0", description: "XIAO" }]);
});

test("detecting the connected board recommends and selects its image", async () => {
  apiRecommend.mockResolvedValue({
    status: "success", source: "imswitch", port: "/dev/ttyUSB0", chip: "esp32",
    identity: { pindef: "UC2_canopen_master", fwImage: MASTER, fwVersion: "v1", canId: 1 },
    recommended: { filename: MASTER, merged: null, source: "reported",
                   reason: "the board reports it was built as this image",
                   file: { ...file(MASTER), version: "v2" } },
  });
  const store = mountAtFirmwareStep();
  fireEvent.click(screen.getByRole("button", { name: "Detect firmware" }));
  await waitFor(() => expect(apiRecommend).toHaveBeenCalledWith(""));
  expect(await screen.findByText("Recommended for the detected board")).toBeInTheDocument();
  expect(screen.getByText(/the board reports it was built as this image/)).toBeInTheDocument();
  expect(screen.getByText(/Update available/)).toBeInTheDocument(); // v1 on the board, v2 on the server
  expect(store.getState().usbFlash.selectedFirmware.filename).toBe(MASTER);
  // the recommended file is listed first
  expect(screen.getAllByRole("radio")[0]).toHaveAttribute("value", MASTER);
});

test("no matching image: the reason is shown and nothing is selected", async () => {
  apiRecommend.mockResolvedValue({
    status: "success", source: "port", port: "/dev/ttyACM0",
    identity: { pindef: "UC2_4", fwVersion: "" },
    recommended: { filename: null, reason: "No image on the server matches this board (expected esp32_UC2_4_release.bin)." },
  });
  const store = mountAtFirmwareStep();
  fireEvent.click(screen.getByRole("button", { name: "Detect firmware" }));
  expect(await screen.findByText(/expected esp32_UC2_4_release.bin/)).toBeInTheDocument();
  expect(store.getState().usbFlash.selectedFirmware).toBeNull();
});

test("the first step applies a GitHub release and lists its images", async () => {
  apiServer.mockResolvedValue({ source: "github", firmware_server_url: "http://fw",
                                github_repo: "youseetoo/uc2-esp32", github_release: "latest" });
  apiReleases.mockResolvedValue({ status: "success", releases: [] });
  apiRecommend.mockResolvedValue({ status: "error", message: "not connected" });
  apiSetRelease.mockResolvedValueOnce({ status: "error", message: "rate limit reached" });
  apiFiles.mockResolvedValue({ status: "success", files: FILES });
  const store = configureStore({ reducer: { usbFlash: usbFlashReducer } });
  render(<Provider store={store}><UsbFlashWizard open onClose={() => {}} /></Provider>);
  const next = await screen.findByRole("button", { name: "Next" });
  await waitFor(() => expect(next).toBeEnabled());
  fireEvent.click(next);
  expect(await screen.findByText("rate limit reached")).toBeInTheDocument();
  expect(store.getState().usbFlash.currentStep).toBe(0); // a refused source does not advance

  apiSetRelease.mockResolvedValueOnce({ status: "success", release_tag: "v2026.0.0-beta.4" });
  fireEvent.click(next);
  await waitFor(() => expect(store.getState().usbFlash.currentStep).toBe(1));
  expect(apiSetRelease).toHaveBeenLastCalledWith("latest");
  expect(store.getState().usbFlash.firmwareFiles).toHaveLength(3);
});
