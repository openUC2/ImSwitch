import React from "react";
import { render, screen, fireEvent, within } from "@testing-library/react";
import PositionControllerComponent from "./PositionControllerComponent";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";
import apiPositionerControllerMovePositionerForever from "../backendapi/apiPositionerControllerMovePositionerForever.js";

jest.mock("../backendapi/apiPositionerControllerMovePositioner.js", () =>
  jest.fn(() => Promise.resolve({})),
);
jest.mock("../backendapi/apiPositionerControllerMovePositionerForever.js", () =>
  jest.fn(() => Promise.resolve({})),
);

describe("PositionControllerComponent", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    window.localStorage.clear();
  });

  test("keeps one XY step value selected by default", () => {
    render(<PositionControllerComponent />);

    const xyGroup = screen.getByText("XY").closest("div");
    const xyButtons = within(xyGroup).getAllByRole("button");
    const selectedXY = xyButtons.find(
      (button) =>
        button.textContent === "100" &&
        button.className.includes("MuiButton-contained"),
    );

    expect(selectedXY).toBeTruthy();
  });

  test("restores the last selected step values after remount", () => {
    window.localStorage.setItem(
      "imswitch-stage-control-step-sizes",
      JSON.stringify({ xy: 10, z: 500 }),
    );

    const { unmount } = render(<PositionControllerComponent />);
    const xyGroup = screen.getByText("XY").closest("div");
    const zGroup = screen.getByText("Z").closest("div");

    expect(
      within(xyGroup).getByRole("button", { name: "10" }).className,
    ).toContain("MuiButton-contained");
    expect(
      within(zGroup).getByRole("button", { name: "500" }).className,
    ).toContain("MuiButton-contained");

    unmount();
    render(<PositionControllerComponent />);

    expect(
      within(screen.getByText("XY").closest("div")).getByRole("button", {
        name: "10",
      }).className,
    ).toContain("MuiButton-contained");
    expect(
      within(screen.getByText("Z").closest("div")).getByRole("button", {
        name: "500",
      }).className,
    ).toContain("MuiButton-contained");
  });

  test("does not move the stage on mouse leave without an active press", () => {
    render(<PositionControllerComponent />);

    const button = screen.getByRole("button", { name: "X→" });

    fireEvent.mouseEnter(button);
    fireEvent.mouseLeave(button);

    expect(apiPositionerControllerMovePositioner).not.toHaveBeenCalled();
    expect(apiPositionerControllerMovePositionerForever).not.toHaveBeenCalled();
  });
});
