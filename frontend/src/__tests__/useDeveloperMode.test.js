import { renderHook, act } from "@testing-library/react";
import { useDeveloperMode } from "../utils/useDeveloperMode";

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// Mock Redux hooks for testing
const mockDispatch = jest.fn();
jest.mock("react-redux", () => ({
  useDispatch: () => mockDispatch,
}));

describe("useDeveloperMode", () => {
  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();
    mockDispatch.mockClear();

    // Reset localStorage mock to return null by default
    localStorageMock.getItem.mockReset();
    localStorageMock.setItem.mockReset();
    localStorageMock.removeItem.mockReset();
    localStorageMock.getItem.mockReturnValue(null);

    // Clear console.log spy if it exists
    if (console.log.mockRestore) {
      console.log.mockRestore();
    }
  });

  afterEach(() => {
    // Clean up event listeners
    document.removeEventListener("keydown", () => {});
    document.removeEventListener("keyup", () => {});
  });

  describe("Initial State", () => {
    test("should initialize with developer mode disabled", () => {
      const { result } = renderHook(() => useDeveloperMode());

      expect(result.current.isDeveloperMode).toBe(false);
    });

    test("should initialize from localStorage if previously enabled", () => {
      localStorageMock.getItem.mockReturnValue("true");

      const { result } = renderHook(() => useDeveloperMode());

      expect(result.current.isDeveloperMode).toBe(true);
      expect(localStorageMock.getItem).toHaveBeenCalledWith(
        "imswitch_developer_mode"
      );
    });
  });

  describe("Manual Activation", () => {
    test("should activate developer mode manually", () => {
      const consoleSpy = jest
        .spyOn(console, "log")
        .mockImplementation(() => {});
      const { result } = renderHook(() => useDeveloperMode());

      act(() => {
        result.current.activateDeveloperMode();
      });

      expect(result.current.isDeveloperMode).toBe(true);
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "imswitch_developer_mode",
        "true"
      );
      expect(consoleSpy).toHaveBeenCalledWith(
        "🔧 ImSwitch Developer Mode Activated"
      );
      expect(mockDispatch).toHaveBeenCalledWith({
        type: "notification/setNotification",
        payload: {
          message: "🔧 Developer Mode Activated",
          type: "success",
        },
      });

      consoleSpy.mockRestore();
    });

    test("should deactivate developer mode manually", () => {
      localStorageMock.getItem.mockReturnValue("true");
      const consoleSpy = jest
        .spyOn(console, "log")
        .mockImplementation(() => {});
      const { result } = renderHook(() => useDeveloperMode());

      act(() => {
        result.current.deactivateDeveloperMode();
      });

      expect(result.current.isDeveloperMode).toBe(false);
      expect(localStorageMock.removeItem).toHaveBeenCalledWith(
        "imswitch_developer_mode"
      );
      expect(consoleSpy).toHaveBeenCalledWith("Developer Mode Deactivated");
      expect(mockDispatch).toHaveBeenCalledWith({
        type: "notification/setNotification",
        payload: {
          message: "Developer Mode Deactivated",
          type: "info",
        },
      });

      consoleSpy.mockRestore();
    });
  });

  describe("Keyboard Shortcuts", () => {
    test("should activate on Konami code sequence", () => {
      const consoleSpy = jest
        .spyOn(console, "log")
        .mockImplementation(() => {});
      const { result } = renderHook(() => useDeveloperMode());

      // Simulate Konami code: ↑↑↓↓←→←→BA
      const konamiKeys = [
        "ArrowUp",
        "ArrowUp",
        "ArrowDown",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "ArrowLeft",
        "ArrowRight",
        "KeyB",
        "KeyA",
      ];

      act(() => {
        konamiKeys.forEach((key) => {
          const event = new KeyboardEvent("keydown", {
            code: key,
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            metaKey: false,
          });
          window.dispatchEvent(event);
        });
      });

      expect(result.current.isDeveloperMode).toBe(true);
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "imswitch_developer_mode",
        "true"
      );

      consoleSpy.mockRestore();
    });

    test("should activate on Ctrl+Shift+DEV sequence", () => {
      const consoleSpy = jest
        .spyOn(console, "log")
        .mockImplementation(() => {});
      const { result } = renderHook(() => useDeveloperMode());

      // Simulate Ctrl+Shift+D+E+V
      act(() => {
        ["KeyD", "KeyE", "KeyV"].forEach((key) => {
          const event = new KeyboardEvent("keydown", {
            code: key,
            ctrlKey: true,
            shiftKey: true,
            altKey: false,
            metaKey: false,
          });
          window.dispatchEvent(event);
        });
      });

      expect(result.current.isDeveloperMode).toBe(true);
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "imswitch_developer_mode",
        "true"
      );

      consoleSpy.mockRestore();
    });

    test("should not activate on partial sequences", () => {
      const { result } = renderHook(() => useDeveloperMode());

      // Simulate partial Konami code
      act(() => {
        ["ArrowUp", "ArrowDown", "KeyB"].forEach((key) => {
          const event = new KeyboardEvent("keydown", {
            code: key,
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            metaKey: false,
          });
          window.dispatchEvent(event);
        });
      });

      expect(result.current.isDeveloperMode).toBe(false);
      expect(localStorageMock.setItem).not.toHaveBeenCalled();
    });

    test("should reset dev sequence when Ctrl+Shift is released", () => {
      const { result } = renderHook(() => useDeveloperMode());

      act(() => {
        // Press Ctrl+Shift+D
        const event1 = new KeyboardEvent("keydown", {
          code: "KeyD",
          ctrlKey: true,
          shiftKey: true,
        });
        window.dispatchEvent(event1);

        // Release Ctrl+Shift
        const event2 = new KeyboardEvent("keyup", {
          code: "KeyD",
          ctrlKey: false,
          shiftKey: false,
        });
        window.dispatchEvent(event2);

        // Press E (should not trigger as sequence was reset)
        const event3 = new KeyboardEvent("keydown", {
          code: "KeyE",
          ctrlKey: false,
          shiftKey: false,
        });
        window.dispatchEvent(event3);
      });

      expect(result.current.isDeveloperMode).toBe(false);
    });

    test("should not attach listeners when developer mode is active", () => {
      localStorageMock.getItem.mockReturnValue("true");
      const addEventListenerSpy = jest.spyOn(window, "addEventListener");

      renderHook(() => useDeveloperMode());

      expect(addEventListenerSpy).not.toHaveBeenCalled();

      addEventListenerSpy.mockRestore();
    });
  });

  describe("Development Environment", () => {
    test("should log instructions in development mode when not active", () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = "development";

      const consoleSpy = jest
        .spyOn(console, "log")
        .mockImplementation(() => {});

      renderHook(() => useDeveloperMode());

      expect(consoleSpy).toHaveBeenCalledWith(
        "🔧 ImSwitch Developer Backdoor Available:"
      );
      expect(consoleSpy).toHaveBeenCalledWith("  • Konami Code: ↑↑↓↓←→←→BA");
      expect(consoleSpy).toHaveBeenCalledWith(
        "  • Quick Access: Ctrl+Shift+D+E+V"
      );
      expect(consoleSpy).toHaveBeenCalledWith(
        "  • Enables settings access when backend offline"
      );

      process.env.NODE_ENV = originalEnv;
      consoleSpy.mockRestore();
    });

    test("should provide debug info in development mode", () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = "development";

      const { result } = renderHook(() => useDeveloperMode());

      expect(result.current.debugInfo).toEqual({
        konamiProgress: 0,
        devProgress: 0,
      });

      process.env.NODE_ENV = originalEnv;
    });

    test("should not provide debug info in production mode", () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = "production";

      const { result } = renderHook(() => useDeveloperMode());

      expect(result.current.debugInfo).toBeNull();

      process.env.NODE_ENV = originalEnv;
    });
  });

  describe("Memory Management", () => {
    test("should limit sequence memory to prevent memory leaks", () => {
      const { result } = renderHook(() => useDeveloperMode());

      act(() => {
        // Send 20 random keys (more than the 15-key limit for Konami)
        for (let i = 0; i < 20; i++) {
          const event = new KeyboardEvent("keydown", {
            code: `Key${String.fromCharCode(65 + (i % 26))}`, // KeyA, KeyB, etc.
            ctrlKey: false,
            shiftKey: false,
            altKey: false,
            metaKey: false,
          });
          window.dispatchEvent(event);
        }
      });

      // Should not crash or use excessive memory
      expect(result.current.isDeveloperMode).toBe(false);
    });
  });
});
