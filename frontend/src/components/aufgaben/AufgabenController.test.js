import React from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";

import AufgabenController from "./AufgabenController";
import aufgabenReducer, * as A from "../../state/slices/AufgabenSlice";
import languageReducer from "../../state/slices/LanguageSlice";
import * as api from "../../backendapi/apiAufgabenController";
import catalog from "../../__tests__/fixtures/aufgabenCatalog.json";

jest.mock("../../backendapi/apiAufgabenController");
jest.mock("../../backendapi/apiPositionerControllerGetPositions");

const mount = () => {
  const store = configureStore({
    reducer: { aufgabenState: aufgabenReducer, languageState: languageReducer },
  });
  render(
    <Provider store={store}>
      <AufgabenController />
    </Provider>,
  );
  return store;
};

beforeEach(() => {
  api.apiAufgabenGetCatalog.mockResolvedValue(catalog);
  api.apiAufgabenGetStatus.mockResolvedValue({ state: "idle" });
  api.apiAufgabenListJobs.mockResolvedValue([]);
  api.apiAufgabenListRuns.mockResolvedValue([]);
});
afterEach(() => jest.clearAllMocks());

test("palette clicks build the outline and the step form follows the schema", async () => {
  const store = mount();
  fireEvent.click(await screen.findByText("Loop over positions"));
  fireEvent.click(screen.getByText("Capture"));
  expect(store.getState().aufgabenState.job.tasks[0].body[0].task).toBe("capture");

  fireEvent.click(screen.getByTestId("aufgaben-node-loop_positions"));
  // the required positions parameter is a resource picker with a "New" button
  fireEvent.click(screen.getByRole("button", { name: "New positions resource" }));
  const job = store.getState().aufgabenState.job;
  expect(job.resources.positions).toEqual({ kind: "positions", positions: [] });
  expect(job.tasks[0].params.source).toBe("@positions");
});

test("validation marks the outline; Start needs the ask-first steps confirmed", async () => {
  const store = mount();
  fireEvent.click(await screen.findByText("Home axis"));
  api.apiAufgabenValidateJob.mockResolvedValue({
    ok: true, issues: [{ level: "warning", node: "home_axis", message: "homes X" }],
    ask_first: ["home_axis"], moves_xy: true, moves_z: false, emits_light: false,
  });
  fireEvent.click(screen.getByRole("button", { name: "Start" }));
  const dialog = await screen.findByRole("dialog");
  expect(within(dialog).getByText(/moves the stage in X and Y/)).toBeInTheDocument();
  const start = within(dialog).getByRole("button", { name: "Start" });
  expect(start).toBeDisabled();
  fireEvent.click(within(dialog).getByRole("checkbox"));
  api.apiAufgabenStartJob.mockResolvedValue({ run_id: "r1", folder: "/data/Aufgaben/x" });
  fireEvent.click(start);
  await waitFor(() => expect(api.apiAufgabenStartJob).toHaveBeenCalled());
  const [job, parameters, confirmed] = api.apiAufgabenStartJob.mock.calls[0];
  expect(job.tasks[0].task).toBe("home_axis");
  expect(parameters).toEqual({});
  expect(confirmed).toBe(true);
  expect(store.getState().aufgabenState.validation.ask_first).toEqual(["home_axis"]);
});

test("a job with errors is not started", async () => {
  mount();
  fireEvent.click(await screen.findByText("Wait"));
  api.apiAufgabenValidateJob.mockResolvedValue({
    ok: false, issues: [{ level: "error", node: "wait", message: "seconds is required" }],
    ask_first: [], moves_xy: false, moves_z: false, emits_light: false,
  });
  fireEvent.click(screen.getByRole("button", { name: "Start" }));
  expect(await screen.findByText(/1 errors/)).toBeInTheDocument();
  expect(screen.getByLabelText("Error")).toBeInTheDocument();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

test("the operator answers the job's question", async () => {
  const store = mount();
  await screen.findByText("Loop over positions");
  store.dispatch(A.aufgabenEvent({ type: "question", node: "q", text: "Slide clamped?", buttons: ["Yes", "No"] }));
  api.apiAufgabenAnswerQuestion.mockResolvedValue({ answered: "Yes" });
  fireEvent.click(await screen.findByRole("button", { name: "Yes" }));
  await waitFor(() => expect(api.apiAufgabenAnswerQuestion).toHaveBeenCalledWith("Yes"));
  await waitFor(() => expect(store.getState().aufgabenState.question).toBeNull());
});

test("explains a microscope without the controller", async () => {
  api.apiAufgabenGetCatalog.mockRejectedValue({ response: { status: 404 } });
  mount();
  expect(await screen.findByText("This microscope does not run the AufgabenController.")).toBeInTheDocument();
});

test("steps this microscope cannot run stay out of the palette but are explained", async () => {
  const store = mount();
  await screen.findByText("Loop over positions");
  expect(screen.queryByText("Pump volume")).not.toBeInTheDocument();
  store.dispatch(A.loadJob({ name: "flow", tasks: [
    { id: "pump", task: "pump_volume", params: { pump: "p", volume_ul: 5 } }] }));
  expect(await screen.findByText("not on this microscope")).toBeInTheDocument();
  fireEvent.click(screen.getByTestId("aufgaben-node-pump"));
  expect(await screen.findByText(/Needs a pump, which this microscope does not have/)).toBeInTheDocument();
});
