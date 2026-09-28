import React from "react";
import { render, screen } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import ObjectiveTurretView from "./ObjectiveTurretView";
import objectiveReducer from "../state/slices/ObjectiveSlice";
import positionReducer from "../state/slices/PositionSlice";

// Two parfocal-ish objectives on a turret whose slots sit at A = 1000 / 21000.
const STATUS = {
  currentObjective: 1,
  posX0: 1000,
  posX1: 21000,
  posZ0: 500,
  posZ1: 520,
  availableObjectivesNames: ["4x", "20x"],
  availableObjectiveMagnifications: [4, 20],
  availableObjectiveNAs: [0.1, 0.4],
  availableObjectivePixelSizes: [1.6, 0.325],
  slotConfigured: [true, true],
  hasMotor: true,
};

const renderTurret = (objective, position) => {
  const store = configureStore({
    reducer: { objectiveState: objectiveReducer, position: positionReducer },
    preloadedState: {
      objectiveState: { ...objectiveReducer(undefined, { type: "init" }), ...STATUS, ...objective },
      position: { ...positionReducer(undefined, { type: "init" }), ...position },
    },
  });
  return render(
    <Provider store={store}>
      <ObjectiveTurretView />
    </Provider>,
  );
};

test("the slot whose position matches A is in the beam", () => {
  renderTurret({}, { a: 21000, z: 520 });
  expect(screen.getByText("In the beam: 20x")).toBeInTheDocument();
  expect(screen.getByText("Z 520 µm · at stored focus")).toBeInTheDocument();
  expect(screen.getByText("20× · NA 0.4 · 0.325 µm/px")).toBeInTheDocument();
});

test("A away from both slots means the turret is between slots", () => {
  renderTurret({}, { a: 11000, z: 520 });
  expect(screen.getByText("Turret between slots (A 11,000 µm)")).toBeInTheDocument();
});

test("a slot in the beam that the backend does not report is flagged", () => {
  renderTurret({ currentObjective: 0 }, { a: 21000, z: 500 });
  expect(screen.getByText("In the beam: 20x (backend: 4x)")).toBeInTheDocument();
});

test("Z is shown relative to the stored focus of the objective in the beam", () => {
  renderTurret({}, { a: 21000, z: 532.3 });
  expect(screen.getByText("Z 532.3 µm · +12.3 µm from stored focus")).toBeInTheDocument();
});

test("without an A motor the reported objective is the one in the beam", () => {
  renderTurret({ hasMotor: false, currentObjective: 0 }, { a: 0, z: 0 });
  expect(screen.getByText("In the beam: 4x")).toBeInTheDocument();
});

test("unset focus positions and unconfigured slots are said so", () => {
  renderTurret(
    { currentObjective: 0, posZ0: 0, posZ1: 0, slotConfigured: [true, false] },
    { a: 1000, z: 7 },
  );
  expect(screen.getByText("Z 7 µm · no stored focus")).toBeInTheDocument();
  expect(screen.getByText("not configured")).toBeInTheDocument();
  expect(screen.getByText("empty")).toBeInTheDocument();
});
