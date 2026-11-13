import catalog from "./fixtures/aufgabenCatalog.json";
import reducer, * as A from "../state/slices/AufgabenSlice";

const run = (...actions) => actions.reduce(reducer, reducer(undefined, { type: "@@init" }));

test("palette clicks build a nested job and select the new step", () => {
  const state = run(
    A.setCatalog(catalog),
    A.addNode("loop_positions"),
    A.addNode("capture"),                 // into the selected loop
    A.selectNode(null),
    A.addNode("wait"),                    // top level
  );
  expect(state.job.tasks.map((n) => n.id)).toEqual(["loop_positions", "wait"]);
  expect(state.job.tasks[0].body[0]).toMatchObject({ id: "capture", task: "capture" });
  expect(state.selectedId).toBe("wait");
  expect(state.dirty).toBe(true);
});

test("removing a step selects its neighbour; edits drop a stale validation", () => {
  let state = run(A.setCatalog(catalog), A.addNode("log"), A.addNode("wait"));
  state = reducer(state, A.setValidation({ ok: true, issues: [] }));
  state = reducer(state, A.removeNode("wait"));
  expect(state.selectedId).toBe("log");
  expect(state.validation).toBeNull();
});

test("engine events update the status, the question and a bounded log", () => {
  let state = run(A.setStatus({ state: "running", node: "a", frames: 3 }));
  state = reducer(state, A.aufgabenEvent({ type: "node", node: "img", task: "capture" }));
  expect(state.status).toMatchObject({ state: "running", node: "img", frames: 3 });
  expect(state.events).toHaveLength(0);
  state = reducer(state, A.aufgabenEvent({ type: "question", node: "q", text: "Ready?", buttons: ["Yes"] }));
  expect(state.question).toEqual({ node: "q", text: "Ready?", buttons: ["Yes"] });
  for (let i = 0; i < 350; i += 1) {
    state = reducer(state, A.aufgabenEvent({ type: "log", level: "info", message: `m${i}` }));
  }
  expect(state.events).toHaveLength(300);
  expect(state.events[299].message).toBe("m349");
  state = reducer(state, A.aufgabenEvent({ type: "state", state: "finished", message: "" }));
  expect(state.question).toBeNull();
  expect(A.isActive(state.status)).toBe(false);
});
