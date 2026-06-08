import onboardingReducer, {
  startTour,
  completeTour,
  resetTour,
  getOnboardingState,
} from "../state/slices/OnboardingSlice";

describe("OnboardingSlice", () => {
  const initialState = { completed: false, version: 0, runToken: 0 };

  test("returns the initial state", () => {
    expect(onboardingReducer(undefined, { type: "@@INIT" })).toEqual(
      initialState,
    );
  });

  test("startTour increments runToken without touching completed", () => {
    const afterFirst = onboardingReducer(initialState, startTour());
    expect(afterFirst.runToken).toBe(1);
    expect(afterFirst.completed).toBe(false);

    const afterSecond = onboardingReducer(afterFirst, startTour());
    expect(afterSecond.runToken).toBe(2);
  });

  test("completeTour sets completed and records the version", () => {
    const next = onboardingReducer(initialState, completeTour({ version: 3 }));
    expect(next.completed).toBe(true);
    expect(next.version).toBe(3);
  });

  test("completeTour without a version keeps the previous version", () => {
    const seeded = { completed: false, version: 2, runToken: 5 };
    const next = onboardingReducer(seeded, completeTour());
    expect(next.completed).toBe(true);
    expect(next.version).toBe(2);
    expect(next.runToken).toBe(5);
  });

  test("resetTour clears the completed flag and version", () => {
    const completedState = { completed: true, version: 4, runToken: 7 };
    const next = onboardingReducer(completedState, resetTour());
    expect(next.completed).toBe(false);
    expect(next.version).toBe(0);
  });

  test("getOnboardingState selects the onboarding slice", () => {
    const rootState = { onboardingState: initialState };
    expect(getOnboardingState(rootState)).toBe(initialState);
  });
});
