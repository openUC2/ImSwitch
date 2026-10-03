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
    // CRA's jest config sets resetMocks, which strips the factory
    // implementations above — restore them or the component awaits undefined.
    apiPositionerControllerMovePositioner.mockResolvedValue({});
    apiPositionerControllerMovePositionerForever.mockResolvedValue({});
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

  test("moves on press, not on release", () => {
    render(<PositionControllerComponent />);
    const button = screen.getByRole("button", { name: "X→" });

    fireEvent.pointerDown(button);
    expect(apiPositionerControllerMovePositioner).toHaveBeenCalledTimes(1);
    expect(apiPositionerControllerMovePositioner).toHaveBeenCalledWith(
      expect.objectContaining({ axis: "X", dist: 100 }),
    );

    fireEvent.pointerUp(button);
    expect(apiPositionerControllerMovePositioner).toHaveBeenCalledTimes(1);
  });

  test("one press is one step even when several axes are held", () => {
    render(<PositionControllerComponent />);
    const x = screen.getByRole("button", { name: "X→" });
    const y = screen.getByRole("button", { name: "Y↑" });

    fireEvent.pointerDown(x);
    fireEvent.pointerDown(y); // second axis while the first is still down
    fireEvent.pointerUp(x);
    fireEvent.pointerUp(y);

    const axes = apiPositionerControllerMovePositioner.mock.calls.map(
      ([args]) => args.axis,
    );
    expect(axes).toEqual(["X", "Y"]);
    expect(apiPositionerControllerMovePositionerForever).not.toHaveBeenCalled();
  });

  test("a repeated press on a held button does not queue a second step", () => {
    render(<PositionControllerComponent />);
    const button = screen.getByRole("button", { name: "X→" });

    fireEvent.pointerDown(button);
    fireEvent.pointerDown(button);
    expect(apiPositionerControllerMovePositioner).toHaveBeenCalledTimes(1);
  });

  test("keyboard steps use the selected step size, not a hardcoded one", () => {
    window.localStorage.setItem(
      "imswitch-stage-control-step-sizes",
      JSON.stringify({ xy: 10, z: 500 }),
    );
    const { container } = render(<PositionControllerComponent />);

    fireEvent.keyDown(container.firstChild, { key: "ArrowRight" });
    fireEvent.keyUp(container.firstChild, { key: "ArrowRight" });

    expect(apiPositionerControllerMovePositioner).toHaveBeenCalledWith(
      expect.objectContaining({ axis: "X", dist: 10 }),
    );
  });

  test("arrow keys on the document do not reach a pad that is not focused", () => {
    // The wrapper mounts a pad in both the PiP window and the camera viewport;
    // window-level listeners moved the stage twice per press.
    render(<PositionControllerComponent />);

    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    fireEvent.keyUp(document.body, { key: "ArrowRight" });

    expect(apiPositionerControllerMovePositioner).not.toHaveBeenCalled();
  });
});
