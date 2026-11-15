import reducer, { disableApp, selectEnabledApps } from "../state/slices/appManagerSlice";

const stateWith = (appManager) => ({ appManager });

test("essential apps show up also in a list saved before they became essential", () => {
  const saved = { ...reducer(undefined, { type: "@@init" }), enabledApps: ["liveView", "stageMap"] };
  const enabled = selectEnabledApps(stateWith(saved));
  expect(enabled).toEqual(expect.arrayContaining(["aufgaben", "fileManager", "liveView", "stageMap"]));
  expect(selectEnabledApps(stateWith(saved))).toBe(enabled);          // memoised
});

test("Aufgaben is on by default and cannot be switched off", () => {
  let state = reducer(undefined, { type: "@@init" });
  expect(state.enabledApps).toContain("aufgaben");
  state = reducer(state, disableApp("aufgaben"));
  expect(state.enabledApps).toContain("aufgaben");
});
