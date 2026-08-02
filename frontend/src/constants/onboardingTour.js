/**
 * Onboarding / intro-tour configuration.
 *
 * This is the single place to edit the first-run guided tour. It is consumed by
 * components/OnboardingTour.jsx which drives intro.js (https://introjs.com).
 *
 * HOW TO EDIT
 * -----------
 * - Each entry below is one step in the tour. Just change `title` / `intro`
 *   (HTML is allowed in `intro`) to reword a step.
 * - `group` groups consecutive steps. The tour shows the group name in the
 *   tooltip header and offers a "Skip section" button so users can jump over a
 *   whole group of button-level steps.
 * - `target` is a CSS selector for the element to highlight. We use
 *   `[data-tour="..."]` attributes that are placed on the relevant components.
 *   Set `target: null` for an intro/outro step that floats in the center.
 * - `optional: true` means: if the target element is not currently in the DOM
 *   (e.g. the Objective panel is hidden when no objective controller exists, or
 *   the right panel is collapsed on mobile), the step is silently skipped
 *   instead of breaking the tour.
 * - `position` is the tooltip placement relative to the target
 *   ("top" | "bottom" | "left" | "right" | "auto"). Omit for floating steps.
 *
 * Bump TOUR_VERSION when you make changes worth re-showing to users who have
 * already completed an older tour.
 */

export const TOUR_VERSION = 1;

export const ONBOARDING_TOUR_STEPS = [
  // ---------------------------------------------------------------- Welcome --
  {
    id: "welcome",
    group: "Welcome",
    target: null,
    title: "Welcome to your openUC2 Microscope",
    intro:
      "This quick tour points out the main areas of the live view. " +
      "Use <b>Next</b> / <b>Back</b> to move through it, <b>Skip section</b> to " +
      "jump over a group, or the <b>✕</b> to close the tour at any time.<br/><br/>" +
      "You can restart it later from the <b>⚙ Settings</b> menu (top-right).",
  },

  // -------------------------------------------------------------- Live View --
  {
    id: "live-view",
    group: "Live View",
    target: '[data-tour="live-view"]',
    title: "Live View",
    intro:
      "The live camera stream is shown here. Use the on-image arrow buttons to " +
      "nudge the stage, scroll to pan and pinch/scroll to zoom into the image.",
    position: "right",
  },
  {
    id: "detector-tabs",
    group: "Live View",
    target: '[data-tour="detector-tabs"]',
    title: "Detector selection",
    intro:
      "If your microscope has several cameras/detectors, switch between their " +
      "live streams here.",
    position: "bottom",
    optional: true,
  },

  // ------------------------------------------------------ Live View settings --
  {
    id: "stream-controls",
    group: "Live View settings",
    target: '[data-tour="stream-controls"]',
    title: "Stream controls",
    intro:
      "<b>Start</b> / <b>Stop</b> the live stream and toggle the <b>Histogram</b> " +
      "overlay to inspect pixel intensities.",
    position: "top",
  },
  {
    id: "stream-settings-button",
    group: "Live View settings",
    target: '[data-tour="stream-settings-button"]',
    title: "Stream settings",
    intro:
      "Open advanced stream settings — image format (binary / JPEG / WebRTC), " +
      "compression and performance options for the live view.",
    position: "left",
    optional: true,
  },
  {
    id: "capture-controls",
    group: "Live View settings",
    target: '[data-tour="capture-controls"]',
    title: "Snap & record",
    intro:
      "Save a single snapshot or record a video/stack. Pick the file format, " +
      "add an optional description and jump to the saved file with " +
      "<b>Go to Folder</b>.",
    position: "top",
    optional: true,
  },

  // --------------------------------------------------- Camera control (cam) --
  {
    id: "camera-controls",
    group: "Camera control",
    target: '[data-tour="camera-controls"]',
    title: "Detector parameters",
    intro:
      "Adjust camera exposure, gain and black level here. Values update live and " +
      "can also be driven automatically by the backend.",
    position: "top",
    optional: true,
  },

  // ----------------------------------------------------------------- Stage --
  {
    id: "stage-control",
    group: "Stage",
    target: '[data-tour="stage-control"]',
    title: "Stage control",
    intro:
      "Move the XYZ(A) stage. Switch between a precise <b>Axis View</b>, a " +
      "<b>Joystick</b> and a <b>Virtual Joystick</b>, set step sizes and home " +
      "the axes.",
    position: "left",
    optional: true,
  },

  // ------------------------------------------------------------- Autofocus --
  {
    id: "autofocus",
    group: "Autofocus",
    target: '[data-tour="autofocus"]',
    title: "Autofocus",
    intro:
      "Run an automatic focus sweep. Choose the focus mode, set the Z range and " +
      "resolution, then start the routine to bring the sample into focus.",
    position: "left",
    optional: true,
  },

  // ---------------------------------------------------------- Illumination --
  {
    id: "illumination",
    group: "Illumination",
    target: '[data-tour="illumination"]',
    title: "Illumination",
    intro:
      "Control lasers and LEDs: enable a source and set its intensity. Changes " +
      "are applied to the sample in real time.",
    position: "left",
    optional: true,
  },

  // ------------------------------------------------------------- Objective --
  {
    id: "objective",
    group: "Objective",
    target: '[data-tour="objective"]',
    title: "Objective",
    intro:
      "Switch between configured objective lenses. The pixel calibration and " +
      "field of view update to match the selected objective.",
    position: "left",
    optional: true,
  },

  // ------------------------------------------------------------ Navigation --
  {
    id: "sidebar",
    group: "Navigation",
    target: '[data-tour="sidebar"]',
    title: "Navigation sidebar",
    intro:
      "All tools live here, grouped into <b>Essentials</b>, <b>Apps</b>, " +
      "<b>Calibration</b> and more. Click a group to expand it, then pick a tool.",
    position: "right",
    optional: true,
  },
  {
    id: "app-manager",
    group: "Navigation",
    target: '[data-tour="app-manager"]',
    title: "App Manager (app store)",
    intro:
      "Customize your workspace: enable or disable the apps that appear in the " +
      "sidebar so you only see the tools you actually use.",
    position: "right",
    optional: true,
  },

  // -------------------------------------------------------------- Settings --
  {
    id: "settings-menu",
    group: "Settings",
    target: '[data-tour="settings-menu"]',
    title: "Settings & connection",
    intro:
      "Click here to unfold the settings menu: backend connection, system & " +
      "motor settings, WiFi, firmware updates, dark mode — and to restart this " +
      "tour from <b>Start Intro Tour</b>.",
    position: "bottom",
  },

  // ----------------------------------------------------------------- Outro --
  {
    id: "done",
    group: "Done",
    target: null,
    title: "You're all set",
    intro:
      "That's the quick overview. Explore at your own pace — you can reopen this " +
      "tour any time from the <b>⚙ Settings</b> menu via <b>Start Intro Tour</b>." +
      "In case you have questions, reach out to us via mail to support@openuc2.com",
  },
];
