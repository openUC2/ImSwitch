import { createSlice } from "@reduxjs/toolkit";

/**
 * Onboarding / intro-tour state.
 *
 * This slice tracks whether the first-run guided tour (intro.js) has already
 * been shown to the user ("was done" flag). It is persisted (see store.js
 * whitelist) so the tour only auto-starts on the very first launch.
 *
 * - completed : the user has finished or dismissed the tour at least once.
 * - version   : the TOUR_VERSION that was completed. Bumping TOUR_VERSION in
 *               constants/onboardingTour.js re-shows the tour after a major UI
 *               change, even for users who already completed an older version.
 * - runToken  : incremented to *manually* (re)start the tour (e.g. from the
 *               settings menu "Start Intro Tour" entry). The OnboardingTour
 *               component watches this value and launches when it changes.
 *               Not meaningful across reloads – only its change matters.
 */
const initialOnboardingState = {
  completed: false,
  version: 0,
  runToken: 0,
};

const onboardingSlice = createSlice({
  name: "onboardingState",
  initialState: initialOnboardingState,
  reducers: {
    // Request a (re)start of the tour. Increments runToken so the
    // OnboardingTour component picks up the change and launches the tour.
    startTour: (state) => {
      state.runToken += 1;
    },
    // Mark the tour as seen. Pass { version } to record which tour version was
    // completed. Called when the tour is finished, skipped or otherwise exited.
    completeTour: (state, action) => {
      state.completed = true;
      const version = action.payload?.version;
      if (typeof version === "number") {
        state.version = version;
      }
    },
    // Forget that the tour was shown so it auto-starts again on next visit.
    resetTour: (state) => {
      state.completed = false;
      state.version = 0;
    },
  },
});

export const { startTour, completeTour, resetTour } = onboardingSlice.actions;

// Selector helper
export const getOnboardingState = (state) => state.onboardingState;

export default onboardingSlice.reducer;
