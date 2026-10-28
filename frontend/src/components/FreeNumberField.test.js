import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import FreeNumberField from "./FreeNumberField";

const setup = (props = {}) => {
  const onCommit = jest.fn();
  render(<FreeNumberField label="Exposure (ms)" value={10} onCommit={onCommit} {...props} />);
  const input = screen.getByLabelText("Exposure (ms)");
  input.focus();
  fireEvent.change(input, { target: { value: "25" } });
  return { onCommit, input };
};

test("Enter commits the typed value", () => {
  // Regression: onKeyDown lands on TextField's wrapper div, so blurring
  // e.currentTarget was a no-op and nothing was committed until the user
  // clicked somewhere else.
  const { onCommit, input } = setup();
  fireEvent.keyDown(input, { key: "Enter" });
  expect(onCommit).toHaveBeenCalledWith(25);
});

test("leaving the field commits the typed value", () => {
  const { onCommit, input } = setup();
  fireEvent.blur(input);
  expect(onCommit).toHaveBeenCalledWith(25);
});

test("values above max are clamped on commit", () => {
  const { onCommit, input } = setup({ max: 20 });
  fireEvent.keyDown(input, { key: "Enter" });
  expect(onCommit).toHaveBeenCalledWith(20);
  expect(input.value).toBe("20");
});
