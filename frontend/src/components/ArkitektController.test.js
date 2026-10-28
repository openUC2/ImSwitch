import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";

import ArkitektController from "./ArkitektController";
import arkitektReducer, * as arkitektSlice from "../state/slices/ArkitektSlice";
import languageReducer from "../state/slices/LanguageSlice";
import * as api from "../backendapi/apiArkitektController";

jest.mock("../backendapi/apiArkitektController");

const ACTIONS = [
  { name: "move_stage", title: "Move Stage", description: "Moves one axis.", moves: true, images: false },
  { name: "acquire_frame", title: "Acquire Frame", description: "", moves: false, images: true },
];
const status = (overrides) => ({
  state: "unbound", available: true, enabled: true, url: "http://nas.local", appName: "imswitch",
  appVersion: "2.0.0", hasStoredLogin: false, insecureUrl: true, allowInsecureTransport: false,
  autoConnect: true, useMikro: true, services: ["mikro"], actions: ACTIONS, activity: [],
  uploadCount: 0, ...overrides,
});

const mountPanel = (initial) => {
  const store = configureStore({
    reducer: { arkitektState: arkitektReducer, languageState: languageReducer },
  });
  api.apiArkitektGetStatus.mockResolvedValue(initial);
  api.apiArkitektGetUploads.mockResolvedValue([]);
  render(<Provider store={store}><ArkitektController /></Provider>);
  return store;
};

afterEach(() => jest.clearAllMocks());

test("binds to the entered server and warns about plain http", async () => {
  mountPanel(status());
  const bind = await screen.findByRole("button", { name: "Bind microscope" });
  expect(screen.getByText(/allow the insecure login under Settings/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Arkitekt server"), { target: { value: "https://lab.example" } });
  api.apiArkitektBind.mockResolvedValue(status({ state: "connecting", status: "started" }));
  fireEvent.click(bind);
  await waitFor(() => expect(api.apiArkitektBind).toHaveBeenCalledWith("https://lab.example"));
  expect(screen.getByText("Move Stage")).toBeInTheDocument();
  expect(screen.getByText("moves hardware")).toBeInTheDocument();
});

test("shows the device code and the approval link while waiting", async () => {
  mountPanel(status({
    state: "awaiting_login", userCode: "ABCD-1234", serverName: "Lab NAS",
    approveUrl: "http://nas.local/f/configure/ABCD-1234", loginStartedAt: new Date().toISOString(),
  }));
  expect(await screen.findByText("ABCD-1234")).toBeInTheDocument();
  const link = screen.getByRole("link", { name: /Open approval page/ });
  expect(link).toHaveAttribute("href", "http://nas.local/f/configure/ABCD-1234");
  expect(link).toHaveAttribute("target", "_blank");
  api.apiArkitektCancel.mockResolvedValue(status());
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(api.apiArkitektCancel).toHaveBeenCalled());
});

test("unbind asks first, then forgets the stored login", async () => {
  mountPanel(status({ hasStoredLogin: true }));
  expect(await screen.findByRole("button", { name: "Connect" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Unbind" }));
  expect(screen.getByText("Unbind this microscope?")).toBeInTheDocument();
  api.apiArkitektUnbind.mockResolvedValue(status({ hasStoredLogin: false, message: "Unbound." }));
  fireEvent.click(screen.getAllByRole("button", { name: "Unbind" }).pop());
  await waitFor(() => expect(api.apiArkitektUnbind).toHaveBeenCalled());
});

test("connected: live state, remote calls and the images sent", async () => {
  const store = mountPanel(status({
    state: "connected", serverName: "Lab NAS", boundSince: "2026-10-01T12:00:00",
    deviceId: "pi-frame-3", hasStoredLogin: true,
    publishedState: { x_um: 100, y_um: 200, z_um: 50, illumination_on: "LED", exposure_ms: 10,
                      running_action: "run_tile_scan" },
    activity: [{ id: 2, action: "run_tile_scan", arguments: { range_x_um: 1000 }, status: "running",
                 startedAt: "2026-10-01T12:01:00", results: 3 },
               { id: 1, action: "move_stage", arguments: { axis: "X" }, status: "failed",
                 startedAt: "2026-10-01T12:00:30", error: "RuntimeError: Refused: an experiment is running in ImSwitch." }],
  }));
  expect(await screen.findByText("X 100.0 · Y 200.0 · Z 50.0 µm")).toBeInTheDocument();
  expect(screen.getByText("pi-frame-3")).toBeInTheDocument();
  expect(screen.getByText("running · 3 images")).toBeInTheDocument();
  expect(screen.getByText(/Refused: an experiment is running/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Disconnect" })).toBeInTheDocument();

  store.dispatch(arkitektSlice.addUpload({ id: 9, name: "Tile 000_000", positionUm: { x: 70, y: 185 },
                                           shape: [60, 80], thumbnail: "data:image/jpeg;base64,AA==" }));
  expect(await screen.findByAltText("Tile 000_000")).toBeInTheDocument();
  expect(screen.getByText("70, 185 µm · 60×80")).toBeInTheDocument();
});

test("without the package it says what to install", async () => {
  mountPanel(status({ state: "unavailable", available: false, message: "not installed" }));
  expect(await screen.findByText('pip install "arkitekt[rekuest,mikro]"')).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Bind microscope" })).toBeDisabled();
});

test("socket status updates keep the actions and the call log", () => {
  let state = arkitektReducer(undefined, arkitektSlice.setStatus(status({ activity: [{ id: 1, status: "done" }] })));
  state = arkitektReducer(state, arkitektSlice.setStatus({ state: "connected" }));
  expect(state.status.actions).toHaveLength(2);
  expect(state.activity).toHaveLength(1);
  state = arkitektReducer(state, arkitektSlice.upsertActivity({ id: 1, status: "failed" }));
  state = arkitektReducer(state, arkitektSlice.upsertActivity({ id: 2, status: "running" }));
  expect(state.activity.map((e) => [e.id, e.status])).toEqual([[2, "running"], [1, "failed"]]);
});
