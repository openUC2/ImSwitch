import { useState, useEffect, useCallback } from "react";
import { useDispatch } from "react-redux";
import { setNotification } from "../state/slices/NotificationSlice.js";

/**
 * Developer Mode Hook
 *
 * Enables hidden developer access via keyboard shortcuts:
 * - Konami Code: ↑↑↓↓←→←→BA (ArrowUp ArrowUp ArrowDown ArrowDown ArrowLeft ArrowRight ArrowLeft ArrowRight KeyB KeyA)
 * - Quick Access: Ctrl+Shift+D+E+V
 *
 * When activated, enables access to backend-dependent features even when offline
 * Persists in localStorage until page reload
 */

// Konami Code sequence
const KONAMI_CODE = [
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

// Quick developer access (Ctrl+Shift+D+E+V)
const DEV_SEQUENCE = ["KeyD", "KeyE", "KeyV"];

export const useDeveloperMode = () => {
  const dispatch = useDispatch();
  const [isDeveloperMode, setIsDeveloperMode] = useState(false);
  const [konamiSequence, setKonamiSequence] = useState([]);
  const [devSequence, setDevSequence] = useState([]);
  const [ctrlShiftPressed, setCtrlShiftPressed] = useState(false);

  // Initialize from localStorage
  useEffect(() => {
    const savedMode = localStorage.getItem("imswitch_developer_mode");
    if (savedMode === "true") {
      setIsDeveloperMode(true);
    }
  }, []);

  // Activate developer mode
  const activateDeveloperMode = useCallback(() => {
    setIsDeveloperMode(true);
    localStorage.setItem("imswitch_developer_mode", "true");

    // Visual feedback
    console.log("🔧 ImSwitch Developer Mode Activated");

    // Show notification via Redux
    dispatch(
      setNotification({
        message: "🔧 Developer Mode Activated",
        type: "success",
      })
    );
  }, [dispatch]);

  // Deactivate developer mode
  const deactivateDeveloperMode = useCallback(() => {
    setIsDeveloperMode(false);
    localStorage.removeItem("imswitch_developer_mode");

    console.log("Developer Mode Deactivated");

    // Show notification via Redux
    dispatch(
      setNotification({
        message: "Developer Mode Deactivated",
        type: "info",
      })
    );
  }, [dispatch]);

  // Handle keyboard events
  useEffect(() => {
    const handleKeyDown = (event) => {
      const key = event.code;

      // Track Ctrl+Shift state for dev sequence
      if (event.ctrlKey && event.shiftKey) {
        setCtrlShiftPressed(true);

        // Update dev sequence
        setDevSequence((prev) => {
          const newSequence = [...prev, key];

          // Check if we match the dev sequence
          if (newSequence.length >= DEV_SEQUENCE.length) {
            const lastThree = newSequence.slice(-DEV_SEQUENCE.length);
            if (JSON.stringify(lastThree) === JSON.stringify(DEV_SEQUENCE)) {
              activateDeveloperMode();
              return [];
            }
          }

          // Keep only last 5 keys to avoid memory issues
          return newSequence.slice(-5);
        });
      } else {
        setCtrlShiftPressed(false);
        setDevSequence([]);
      }

      // Update Konami sequence (only if no modifiers pressed)
      if (
        !event.ctrlKey &&
        !event.shiftKey &&
        !event.altKey &&
        !event.metaKey
      ) {
        setKonamiSequence((prev) => {
          const newSequence = [...prev, key];

          // Check if we match the Konami code
          if (newSequence.length >= KONAMI_CODE.length) {
            const lastTen = newSequence.slice(-KONAMI_CODE.length);
            if (JSON.stringify(lastTen) === JSON.stringify(KONAMI_CODE)) {
              activateDeveloperMode();
              return [];
            }
          }

          // Keep only last 15 keys to avoid memory issues
          return newSequence.slice(-15);
        });
      }
    };

    const handleKeyUp = (event) => {
      // Reset dev sequence when Ctrl+Shift is released
      if (!event.ctrlKey || !event.shiftKey) {
        setCtrlShiftPressed(false);
        setDevSequence([]);
      }
    };

    // Only attach listeners if not in developer mode
    if (!isDeveloperMode) {
      window.addEventListener("keydown", handleKeyDown);
      window.addEventListener("keyup", handleKeyUp);
    }

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [isDeveloperMode, activateDeveloperMode]);

  // Clear sequences when developer mode is activated
  useEffect(() => {
    if (isDeveloperMode) {
      setKonamiSequence([]);
      setDevSequence([]);
    }
  }, [isDeveloperMode]);

  return {
    isDeveloperMode,
    activateDeveloperMode,
    deactivateDeveloperMode,
    // For debugging - remove in production
    debugInfo:
      process.env.NODE_ENV === "development"
        ? {
            konamiProgress: konamiSequence.length,
            devProgress: ctrlShiftPressed ? devSequence.length : 0,
          }
        : null,
  };
};

export default useDeveloperMode;
