// src/hooks/useDeviceProfile.js
// One place that decides "is this a touchscreen / a phone / a small screen".
//
// The environment part (pointer type, screen size class, OS) is a module-level
// store shared by every caller, so a resize re-renders consumers only when one
// of the derived flags flips — not on every pixel. The user's preference
// (UIPreferencesSlice: auto / on / off) is layered on top.
//
//   touchUI        larger touch targets (theme) + touch-specific behaviour
//   compactLayout  stream-first layouts: controls docked next to / under the
//                  live image instead of in a side panel that hides it
//   phoneLayout    narrow portrait screen (stack everything vertically)
//   keypad         show the on-screen number pad for numeric fields
import { useSyncExternalStore } from "react";
import { useSelector } from "react-redux";
import { getUIPreferences } from "../state/slices/UIPreferencesSlice";

const hasWindow = typeof window !== "undefined";

const matches = (query) =>
  hasWindow &&
  typeof window.matchMedia === "function" &&
  window.matchMedia(query).matches;

// Phones and tablets bring their own soft keyboard; a Raspberry Pi kiosk
// (X11/Wayland Linux with a touchscreen) does not.
const detectMobileOS = () => {
  if (!hasWindow) return false;
  const ua = navigator.userAgent || "";
  const iPadOS =
    navigator.platform === "MacIntel" && (navigator.maxTouchPoints || 0) > 1;
  return /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || iPadOS;
};

const widthClassOf = (w) => {
  if (w < 600) return "xs";
  if (w < 900) return "sm";
  if (w < 1100) return "md";
  if (w < 1400) return "lg";
  return "xl";
};

const computeEnvironment = () => {
  if (!hasWindow) {
    return {
      isTouch: false,
      hasTouch: false,
      isMobileOS: false,
      widthClass: "lg",
      isShort: false,
      isPortrait: false,
    };
  }
  const w = window.innerWidth;
  const h = window.innerHeight;
  const hasTouch =
    matches("(any-pointer: coarse)") || (navigator.maxTouchPoints || 0) > 0;
  // Primary input is a finger: either the primary pointer is coarse, or the
  // device can touch and nothing can hover (no mouse attached).
  const isTouch =
    matches("(pointer: coarse)") || (hasTouch && matches("(hover: none)"));
  return {
    isTouch,
    hasTouch,
    isMobileOS: detectMobileOS(),
    widthClass: widthClassOf(w),
    // The 7" Pi display is 800x480; below ~560 px of height the desktop
    // layout (64 px top bar + stacked panels) no longer fits.
    isShort: h < 560,
    isPortrait: h > w,
  };
};

let snapshot = null;
const listeners = new Set();
const mediaQueries = hasWindow
  ? [
      "(pointer: coarse)",
      "(any-pointer: coarse)",
      "(hover: none)",
    ].map((q) =>
      typeof window.matchMedia === "function" ? window.matchMedia(q) : null,
    )
  : [];

const sameEnvironment = (a, b) =>
  !!a &&
  !!b &&
  Object.keys(a).every((key) => a[key] === b[key]);

const refresh = () => {
  const next = computeEnvironment();
  if (!sameEnvironment(next, snapshot)) {
    snapshot = next;
    listeners.forEach((listener) => listener());
  }
};

const subscribe = (listener) => {
  if (listeners.size === 0 && hasWindow) {
    window.addEventListener("resize", refresh);
    window.addEventListener("orientationchange", refresh);
    mediaQueries.forEach((mq) => mq?.addEventListener?.("change", refresh));
  }
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && hasWindow) {
      window.removeEventListener("resize", refresh);
      window.removeEventListener("orientationchange", refresh);
      mediaQueries.forEach((mq) =>
        mq?.removeEventListener?.("change", refresh),
      );
    }
  };
};

const getSnapshot = () => {
  if (!snapshot) snapshot = computeEnvironment();
  return snapshot;
};

/** Raw device facts, without the user's preference applied. */
export const useDeviceEnvironment = () =>
  useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

/** Pure combination of environment + preferences (exported for tests). */
export const resolveDeviceProfile = (env, prefs) => {
  const phone = env.widthClass === "xs";
  const touchUI =
    prefs.touchMode === "on" ||
    (prefs.touchMode === "auto" && (env.isTouch || phone));
  const compactLayout =
    prefs.touchMode !== "off" &&
    (phone ||
      env.widthClass === "sm" ||
      env.isShort ||
      (touchUI && env.widthClass === "md"));
  const keypad =
    prefs.onScreenKeypad === "on" ||
    (prefs.onScreenKeypad === "auto" && touchUI && !env.isMobileOS);
  return {
    ...env,
    touchUI,
    compactLayout,
    phoneLayout: phone,
    keypad,
  };
};

export default function useDeviceProfile() {
  const env = useDeviceEnvironment();
  const prefs = useSelector(getUIPreferences);
  return resolveDeviceProfile(env, prefs);
}
