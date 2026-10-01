import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";

import UsbFlashWizard from "./UsbFlashWizard";
import usbFlashReducer, * as usbFlashSlice from "../state/slices/usbFlashSlice";
import apiRecommend from "../backendapi/apiUC2ConfigControllerGetRecommendedFirmware";
import apiServer from "../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer";
import apiPorts from "../backendapi/apiUC2ConfigControllerListSerialPorts";

jest.mock("../backendapi/apiUC2ConfigControllerGetRecommendedFirmware");
jest.mock("../backendapi/apiUC2ConfigControllerGetOTAFirmwareServer");
jest.mock("../backendapi/apiUC2ConfigControllerListSerialPorts");

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
