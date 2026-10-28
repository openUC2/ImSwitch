import React, { useState } from "react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import uiPreferencesReducer from "../../state/slices/UIPreferencesSlice";
import TouchKeypadHost from "./TouchKeypadHost";

const renderWithPad = (ui, { onScreenKeypad = "on" } = {}) => {
  const store = configureStore({
    reducer: { uiPreferencesState: uiPreferencesReducer },
    preloadedState: {
      uiPreferencesState: { touchMode: "auto", onScreenKeypad },
    },
  });
  return render(
    <Provider store={store}>
      <TouchKeypadHost />
      {ui}
    </Provider>,
  );
};

// Commits on blur from its own draft state — the pattern of FreeNumberField,
// which only works if the pad lets React re-render between value and blur.
function BlurCommitField({ onCommit, initial = "5", ...inputProps }) {
  const [draft, setDraft] = useState(initial);
  return (
    <input
      aria-label="Speed"
      inputMode="decimal"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => onCommit(draft)}
      {...inputProps}
    />
  );
}

const press = (label) => fireEvent.click(screen.getByRole("button", { name: label }));
const focus = (el) => act(() => el.focus());

describe("TouchKeypadHost", () => {
  test("opens for a numeric field and commits the typed value on OK", async () => {
    const onCommit = jest.fn();
    renderWithPad(<BlurCommitField onCommit={onCommit} />);
    const input = screen.getByLabelText("Speed");

    focus(input);
    expect(screen.getByRole("dialog", { name: "Number pad: Speed" })).toBeInTheDocument();

    // The shown value is selected: the first digit replaces it.
    press("1");
    press("2");
    press("OK");

    await waitFor(() => expect(onCommit).toHaveBeenCalledWith("12"));
    expect(input.value).toBe("12");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    // The soft-keyboard suppression is undone.
    expect(input.getAttribute("inputmode")).toBe("decimal");
  });

  test("cancel leaves the field unchanged", async () => {
    const onCommit = jest.fn();
    renderWithPad(<BlurCommitField onCommit={onCommit} />);
    const input = screen.getByLabelText("Speed");

    focus(input);
    press("9");
    press("Cancel");

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(input.value).toBe("5");
    expect(onCommit).not.toHaveBeenCalledWith("9");
  });

  test("respects min/max and offers presets", async () => {
    const onCommit = jest.fn();
    renderWithPad(
      <BlurCommitField
        onCommit={onCommit}
        min="1"
        max="100"
        data-keypad-presets="10,50"
      />,
    );
    focus(screen.getByLabelText("Speed"));

    press("2");
    press("0");
    press("0");
    expect(screen.getByText("Maximum is 100")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "OK" })).toBeDisabled();

    fireEvent.click(screen.getByText("50"));
    press("OK");
    await waitFor(() => expect(onCommit).toHaveBeenCalledWith("50"));
  });

  test("ignores text fields and stays off when disabled", () => {
    renderWithPad(<input aria-label="Name" type="text" />);
    focus(screen.getByLabelText("Name"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  test("does nothing when the pad is switched off", () => {
    renderWithPad(<BlurCommitField onCommit={() => {}} />, { onScreenKeypad: "off" });
    focus(screen.getByLabelText("Speed"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
