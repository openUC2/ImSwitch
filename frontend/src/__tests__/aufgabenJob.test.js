// Pure job-document helpers of the Aufgaben app, against a catalogue the
// backend produced (fixtures/aufgabenCatalog.json, pruned $defs).
import catalog from "./fixtures/aufgabenCatalog.json";
import {
  emptyJob,
  findNode,
  indentNode,
  insertNode,
  moveNode,
  newNode,
  newNodeId,
  outdentNode,
  paramFields,
  parseNumberInput,
  placeNode,
  removeNode,
  renameNode,
  renameResource,
  catalogById,
  duplicateNode,
  paramSummary,
} from "../utils/aufgabenJob";

const byId = catalogById(catalog);
const ids = (nodes) => nodes.map((n) => n.id);

const sampleJob = () => {
  const job = emptyJob();
  job.tasks = [
    { id: "loop", task: "loop_positions", params: { source: "@pts" }, body: [
      { id: "img", task: "capture", params: {} },
    ] },
    { id: "wait", task: "wait", params: { seconds: 1 } },
    { id: "note", task: "log", params: { message: "x" } },
  ];
  return job;
};

test("new nodes get unique ids, required defaults and their branches", () => {
  const job = sampleJob();
  expect(newNodeId(job, "capture")).toBe("capture");
  expect(newNodeId(job, "wait")).toBe("wait_2");
  const lapse = newNode(job, byId.time_lapse);
  expect(lapse).toMatchObject({ id: "time_lapse", task: "time_lapse", body: [], idle: [] });
  expect(newNode(job, byId.home_axis).params).toEqual({ axis: "X" });
  expect(newNode(job, byId.condition).params).toEqual({ expr: "" });
});

test("a palette click lands in the selected container, after a step, or at the end", () => {
  const job = sampleJob();
  insertNode(job, { id: "a", task: "log" }, "loop", catalog);
  expect(ids(job.tasks[0].body)).toEqual(["img", "a"]);
  insertNode(job, { id: "b", task: "log" }, "wait", catalog);
  expect(ids(job.tasks)).toEqual(["loop", "wait", "b", "note"]);
  insertNode(job, { id: "c", task: "log" }, null, catalog);
  expect(ids(job.tasks)).toEqual(["loop", "wait", "b", "note", "c"]);
});

test("move, indent, outdent and place keep every node exactly once", () => {
  const job = sampleJob();
  moveNode(job, "note", -1);
  expect(ids(job.tasks)).toEqual(["loop", "note", "wait"]);
  expect(indentNode(job, "note", catalog)).toBe(true);           // into the loop
  expect(ids(job.tasks[0].body)).toEqual(["img", "note"]);
  expect(indentNode(job, "loop", catalog)).toBe(false);          // first node
  expect(indentNode(job, "note", catalog)).toBe(false);          // capture is no container
  expect(outdentNode(job, "note")).toBe(true);
  expect(ids(job.tasks)).toEqual(["loop", "note", "wait"]);
  expect(placeNode(job, "loop", "loop", "body", 0)).toBe(false); // not into itself
  expect(placeNode(job, "wait", "loop", "body", 0)).toBe(true);
  expect(ids(job.tasks[0].body)).toEqual(["wait", "img"]);
  expect(placeNode(job, "wait", "loop", "body", 2)).toBe(true);   // below itself
  expect(ids(job.tasks[0].body)).toEqual(["img", "wait"]);
  expect(removeNode(job, "img").node.id).toBe("img");
  expect(findNode(job, "img")).toBeNull();
});

test("renaming updates loop references and refuses duplicates", () => {
  const job = sampleJob();
  job.tasks[2] = { id: "out", task: "break", params: { loop: "loop" } };
  job.tasks[1].every = { nth: 2, of: "loop" };
  expect(renameNode(job, "loop", "wait")).toBe(false);
  expect(renameNode(job, "loop", "2bad")).toBe(false);
  expect(renameNode(job, "loop", "positions")).toBe(true);
  expect(job.tasks[2].params.loop).toBe("positions");
  expect(job.tasks[1].every.of).toBe("positions");
});

test("parameter fields follow the catalogue schema", () => {
  const kinds = (id) => Object.fromEntries(paramFields(byId[id]).map((f) => [f.name, f.kind]));
  expect(kinds("move_relative")).toEqual({
    dx_um: "number", dy_um: "number", dz_um: "number", speed_um_s: "number",
  });
  const relative = paramFields(byId.move_relative);
  expect(relative[0].unit).toBe("µm");
  expect(relative[3].nullable).toBe(true);
  expect(kinds("set_illumination")).toEqual({ source: "choice", intensity: "number", on: "boolean" });
  expect(paramFields(byId.set_illumination)[0].choices).toEqual(["LED", "Laser488"]);
  expect(kinds("condition")).toEqual({ expr: "expression" });
  expect(kinds("question")).toMatchObject({ text: "string", buttons: "strings" });
  const lapse = paramFields(byId.time_lapse);
  expect(lapse.find((f) => f.name === "plan")).toMatchObject({
    kind: "resource", resourceKinds: ["time_plan"], required: true,
  });
  expect(lapse.find((f) => f.name === "until").kind).toBe("expression");
  expect(lapse.find((f) => f.name === "on_overrun").choices).toEqual(["run_now", "skip"]);
});

test("number inputs accept numbers, expressions and references", () => {
  expect(parseNumberInput("1,5")).toBe(1.5);
  expect(parseNumberInput("7", true)).toBe(7);
  expect(parseNumberInput("")).toBeUndefined();
  expect(parseNumberInput("${n * 2}")).toBe("${n * 2}");
  expect(parseNumberInput("@z")).toBe("@z");
});

test("renaming a resource rewrites its references", () => {
  const job = sampleJob();
  job.resources = { pts: { kind: "positions", positions: [] }, z: { kind: "z_plan", plan: {} } };
  expect(renameResource(job, "pts", "z")).toBe(false);
  expect(renameResource(job, "pts", "sites")).toBe(true);
  expect(Object.keys(job.resources)).toEqual(["sites", "z"]);
  expect(job.tasks[0].params.source).toBe("@sites");
});

test("duplicating copies children with fresh ids", () => {
  const job = sampleJob();
  const copy = duplicateNode(job, "loop");
  expect(ids(job.tasks)).toEqual(["loop", "loop_2", "wait", "note"]);
  expect(copy.body[0].id).toBe("img_2");
  expect(paramSummary({ source: "@pts", n: 3, plan: { a: 1 } })).toBe("source=@pts  n=3  plan");
});

test("a code parameter gets the code editor", () => {
  const entry = {
    id: "python_script",
    schema: { properties: { code: { type: "string", "x-widget": "code" } }, required: ["code"] },
  };
  expect(paramFields(entry)[0]).toMatchObject({ name: "code", kind: "code", required: true });
});
