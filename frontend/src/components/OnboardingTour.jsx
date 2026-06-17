import { useCallback, useEffect, useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import Button from "@mui/material/Button";
import introJs from "intro.js";
import "intro.js/introjs.css";
import "./OnboardingTour.css";

import {
  ONBOARDING_TOUR_STEPS,
  TOUR_VERSION,
} from "../constants/onboardingTour.js";
import {
  getOnboardingState,
  completeTour,
} from "../state/slices/OnboardingSlice.js";
import { getThemeState } from "../state/slices/ThemeSlice.js";

// Plugin (view) on which the tour is anchored. The first-run tour only makes
// sense on the Live View, where all highlighted elements live.
const TOUR_PLUGIN = "LiveView";

// Selector for the element that must exist before we launch (proves the Live
// View is mounted). Used to retry briefly while the view is still rendering.
const ANCHOR_SELECTOR = '[data-tour="live-view"]';

// Static intro.js options shared by every launch.
const INTRO_OPTIONS = {
  showProgress: true,
  showBullets: false,
  showStepNumbers: false,
  exitOnOverlayClick: false,
  exitOnEsc: true,
  keyboardNavigation: true,
  disableInteraction: true,
  scrollToElement: true,
  nextLabel: "Next",
  prevLabel: "Back",
  doneLabel: "Done",
  skipLabel: "✕",
  overlayOpacity: 0.6,
  tooltipClass: "imswitch-introjs-tooltip",
  highlightClass: "imswitch-introjs-highlight",
  // Allow simple HTML (bold, line breaks, the group chip) inside step text.
  tooltipRenderAsHtml: true,
};

const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[char],
  );

/**
 * Resolve the declarative tour config into intro.js steps.
 *
 * - Targeted steps whose element is not in the DOM are dropped (so optional
 *   panels like Objective, or a collapsed mobile right panel, don't break the
 *   tour).
 * - Floating steps (target === null) are always kept.
 * Returns the intro.js `steps` plus a parallel `groups` array used to power the
 * "Skip section" button.
 */
const buildSteps = () => {
  const steps = [];
  const groups = [];
  const targets = [];

  for (const step of ONBOARDING_TOUR_STEPS) {
    let element;
    if (step.target) {
      const found = document.querySelector(step.target);
      if (!found) {
        if (!step.optional) {
          // eslint-disable-next-line no-console
          console.warn(
            `[OnboardingTour] required step "${step.id}" target not found: ${step.target}`,
          );
        }
        continue;
      }
      element = found;
    }

    steps.push({
      element, // undefined => intro.js renders a centered floating tooltip
      title: step.title,
      intro: `<div class="imswitch-tour-group">${escapeHtml(step.group)}</div>${step.intro}`,
      ...(element ? { position: step.position || "auto" } : {}),
    });
    groups.push(step.group);
    targets.push(step.target || null);
  }

  return { steps, groups, targets };
};

/**
 * For each step index, the index of the first step belonging to the *next*
 * group (or null if it is the last group). Drives the "Skip section" button.
 */
const computeNextGroupStart = (groups) => {
  const result = new Array(groups.length).fill(null);
  for (let i = 0; i < groups.length; i += 1) {
    for (let j = i + 1; j < groups.length; j += 1) {
      if (groups[j] !== groups[i]) {
        result[i] = j;
        break;
      }
    }
  }
  return result;
};

/**
 * Resolve the current 0-based step index.
 *
 * intro.js' getCurrentStep() is unreliable inside the v8 legacy wrapper, so we
 * primarily match the currently highlighted element back to our steps array and
 * only fall back to getCurrentStep() for floating steps (no element).
 */
const resolveCurrentStep = (tour, steps, targets) => {
  const highlighted = document.querySelector(".introjs-showElement");
  if (highlighted) {
    // Prefer matching by the data-tour selector: robust against React
    // re-renders that may replace the originally captured DOM node.
    let index = targets.findIndex((t) => t && highlighted.matches(t));
    if (index < 0) {
      index = steps.findIndex((step) => step.element === highlighted);
    }
    if (index >= 0) return index;
  }
  const fallback =
    typeof tour.getCurrentStep === "function" ? tour.getCurrentStep() : null;
  return typeof fallback === "number" ? fallback : null;
};

/**
 * Given the current step, decide whether a "Skip section" affordance should be
 * shown and where it should jump to. Returns { label, toStep } (toStep is the
 * 1-based step number intro.js' goToStep expects) or null.
 *
 * We render this as a separate React overlay rather than injecting into the
 * intro.js tooltip, because intro.js v8 renders its tooltip reactively and would
 * discard any node we append to it.
 */
const computeSkipSection = (current, nextGroupStart, groups) => {
  if (current == null) return null;
  const start = nextGroupStart[current];
  // Only offer "skip section" when there is more than one step left in the
  // current group (otherwise it would just duplicate the Next button).
  if (start == null || start - current - 1 < 1) return null;
  return { label: `Skip to ${groups[start]}`, toStep: start + 1 };
};

/**
 * Headless component (renders nothing) that owns the intro.js first-run tour.
 *
 * Mount it once near the top of the app. It:
 *  - auto-starts the tour on the very first visit to the Live View,
 *  - re-starts on demand when `startTour()` is dispatched (settings menu),
 *  - persists a "was done" flag via the onboarding slice so it does not nag.
 */
export default function OnboardingTour({ selectedPlugin }) {
  const dispatch = useDispatch();
  const { completed, version, runToken } = useSelector(getOnboardingState);
  const { isDarkMode } = useSelector(getThemeState);

  const activeTourRef = useRef(null); // the running intro.js instance, if any
  const autoStartedRef = useRef(false); // guard: auto-start only once per mount
  // Capture the runToken present at mount so a persisted value does not trigger
  // a spurious manual start on reload — only later increments should launch.
  const lastRunTokenRef = useRef(runToken);

  // "Skip section" overlay button state: { label, toStep } while a group with
  // further steps is active, otherwise null.
  const [skipSection, setSkipSection] = useState(null);

  const launch = useCallback(
    function launch(attempt = 0) {
      // Wait until the Live View is actually rendered (handles slow mounts and
      // the navigate-then-start race on manual restart).
      if (!document.querySelector(ANCHOR_SELECTOR)) {
        if (attempt < 10) {
          window.setTimeout(() => launch(attempt + 1), 400);
        }
        return;
      }

      // Never run two tours at once.
      if (activeTourRef.current) return;

      const { steps, groups, targets } = buildSteps();
      if (steps.length === 0) return;

      const nextGroupStart = computeNextGroupStart(groups);
      const tour = introJs();
      activeTourRef.current = tour;

      if (isDarkMode) {
        document.body.classList.add("imswitch-tour-dark");
      }

      tour.setOptions({ ...INTRO_OPTIONS, steps });

      tour.onAfterChange(function onAfterChange() {
        const current = resolveCurrentStep(tour, steps, targets);
        setSkipSection(computeSkipSection(current, nextGroupStart, groups));
      });

      let finished = false;
      const finish = () => {
        if (finished) return;
        finished = true;
        activeTourRef.current = null;
        setSkipSection(null);
        document.body.classList.remove("imswitch-tour-dark");
        dispatch(completeTour({ version: TOUR_VERSION }));
      };

      tour.onComplete(finish);
      tour.onExit(finish);

      tour.start();
    },
    [dispatch, isDarkMode],
  );

  // First-run auto-start: only on the Live View, only while the tour has not
  // been completed for the current TOUR_VERSION.
  useEffect(() => {
    if (selectedPlugin !== TOUR_PLUGIN) return undefined;
    if (autoStartedRef.current) return undefined;
    const needsTour = !completed || version < TOUR_VERSION;
    if (!needsTour) return undefined;

    autoStartedRef.current = true;
    const timer = window.setTimeout(() => launch(), 700);
    return () => window.clearTimeout(timer);
  }, [selectedPlugin, completed, version, launch]);

  // Manual restart: fired by startTour() (runToken increments).
  useEffect(() => {
    if (runToken === lastRunTokenRef.current) return undefined;
    lastRunTokenRef.current = runToken;
    const timer = window.setTimeout(() => launch(), 300);
    return () => window.clearTimeout(timer);
  }, [runToken, launch]);

  // Cleanup on unmount: tear down a running tour and remove the theme class.
  useEffect(() => {
    return () => {
      if (activeTourRef.current) {
        try {
          activeTourRef.current.exit(true);
        } catch (error) {
          // ignore – component is going away anyway
        }
        activeTourRef.current = null;
      }
      document.body.classList.remove("imswitch-tour-dark");
    };
  }, []);

  // "Skip section" affordance — rendered as a fixed overlay above the intro.js
  // layer so it is not discarded by intro.js' reactive tooltip rendering.
  if (!skipSection) return null;

  return (
    <Button
      variant="contained"
      size="small"
      disableElevation
      onClick={() => {
        const tour = activeTourRef.current;
        if (tour && typeof tour.goToStep === "function") {
          tour.goToStep(skipSection.toStep);
        }
      }}
      sx={{
        position: "fixed",
        bottom: 20,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 10000002, // above the intro.js overlay/tooltip layers
        textTransform: "none",
        boxShadow: 3,
      }}
    >
      {skipSection.label} ⏭
    </Button>
  );
}
