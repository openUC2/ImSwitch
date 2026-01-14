import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";
import WiFiController from "./WiFiController";
import connectionSettingsReducer from "../state/slices/ConnectionSettingsSlice";

const createMockStore = (hostIP = "http://192.168.1.1") => {
  return configureStore({
    reducer: {
      connectionSettingsState: connectionSettingsReducer,
    },
    preloadedState: {
      connectionSettingsState: {
        ip: hostIP,
        apiPort: "80",
        websocketPort: "80",
      },
    },
  });
};

const renderWithStore = (store) => {
  return render(
    <Provider store={store}>
      <WiFiController />
    </Provider>
  );
};

describe("WiFiController", () => {
  it("renders without crashing", () => {
    const store = createMockStore();
    renderWithStore(store);

    expect(
      screen.getByText("Internet Access Configuration")
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Configure internet access and network settings/i)
    ).toBeInTheDocument();
  });

  it("displays iframe with correct URL from connection settings", () => {
    const testIP = "http://test-device.local";
    const store = createMockStore(testIP);
    renderWithStore(store);

    const iframe = screen.getByTitle("Internet Access Configuration");
    expect(iframe).toBeInTheDocument();
    expect(iframe).toHaveAttribute("src", `${testIP}/admin/panel/internet`);
  });

  it("displays info alert with button to open in new tab", () => {
    const store = createMockStore();
    renderWithStore(store);

    expect(
      screen.getByText(/If the page doesn't load below/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/open it in a new tab/i)).toBeInTheDocument();
  });

  it("opens admin panel in new tab when button is clicked", () => {
    const testIP = "http://192.168.1.100";
    const store = createMockStore(testIP);
    renderWithStore(store);

    // Mock window.open
    const mockOpen = jest.fn();
    window.open = mockOpen;

    const openButton = screen.getByText(/open it in a new tab/i);
    fireEvent.click(openButton);

    expect(mockOpen).toHaveBeenCalledWith(
      `${testIP}/admin/panel/internet`,
      "_blank"
    );
  });

  it("constructs correct admin panel URL based on connection settings", () => {
    const testCases = [
      "http://localhost",
      "http://192.168.1.1",
      "https://my-device.local",
    ];

    testCases.forEach((testIP) => {
      const store = createMockStore(testIP);
      const { rerender } = render(
        <Provider store={store}>
          <WiFiController />
        </Provider>
      );

      const iframe = screen.getByTitle("Internet Access Configuration");
      expect(iframe).toHaveAttribute("src", `${testIP}/admin/panel/internet`);

      rerender(<div />); // Clean up for next iteration
    });
  });
});
