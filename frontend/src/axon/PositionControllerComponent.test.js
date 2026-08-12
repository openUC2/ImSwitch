import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
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
