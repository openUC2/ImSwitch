// src/utils/aufgabenJob.js
// Pure helpers for Aufgaben job documents (schema "aufgaben.job/1"): the node
// tree, node ids, and the parameter fields of a catalogue entry.
//
// The tree helpers mutate the job they are given, so they work on an immer
// draft inside a reducer and on a plain copy in tests. A node is
// {id, task, params, body, orelse, idle, enabled, every, on_error, label};
// containers keep their children in the branches their catalogue entry lists.

export const BRANCHES = ["body", "orelse", "idle"];
export const BRANCH_TITLES = { body: "Do", orelse: "Else", idle: "While waiting" };

export const emptyJob = () => ({
  schema: "aufgaben.job/1",
  name: "New job",
  description: "",
  parameters: {},
  variables: {},
  resources: {},
  storage: { formats: ["ome-zarr"], layout: "auto" },
  tasks: [],
  metadata: {},
});

export const catalogById = (catalog) =>
  Object.fromEntries((catalog || []).map((entry) => [entry.id, entry]));

export const branchesOf = (entry) => (entry && entry.branches) || [];

// Every node with its parent list and index, depth first.
export function* walk(nodes, parent = null, branch = null) {
  for (let index = 0; index < (nodes || []).length; index += 1) {
    const node = nodes[index];
    yield { node, list: nodes, index, parent, branch };
    for (const name of BRANCHES) {
      if (node[name] && node[name].length) yield* walk(node[name], node, name);
    }
  }
}

export const findNode = (job, id) => {
  for (const entry of walk(job.tasks)) if (entry.node.id === id) return entry;
  return null;
};

const ID_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
export const isValidId = (id) => ID_RE.test(id || "");

// A fresh id from the task name: "capture", "capture_2", ...
export const newNodeId = (job, task) => {
  const used = new Set([...walk(job.tasks)].map((e) => e.node.id));
  const base = (task || "step").replace(/[^A-Za-z0-9_]/g, "_").replace(/^([0-9])/, "_$1");
  if (!used.has(base)) return base;
  for (let n = 2; ; n += 1) if (!used.has(`${base}_${n}`)) return `${base}_${n}`;
};

// Parameter defaults from the schema, so a new node validates where it can.
export const defaultParams = (entry) => {
  const schema = (entry && entry.schema) || {};
  const params = {};
  for (const name of schema.required || []) {
    const prop = (schema.properties || {})[name] || {};
    if (prop.enum && prop.enum.length) params[name] = prop.enum[0];
    else if (prop.type === "boolean") params[name] = false;
    else if (prop.type === "number" || prop.type === "integer") params[name] = 0;
    else if (prop.type === "array") params[name] = [];
    else if (prop.type === "string") params[name] = "";
  }
  return params;
};

export const newNode = (job, entry) => {
  const node = { id: newNodeId(job, entry.id), task: entry.id, params: defaultParams(entry) };
  for (const branch of branchesOf(entry)) node[branch] = [];
  return node;
};

// Where a palette click lands: inside the selected container (first branch),
// after the selected step, or at the end of the job.
export const insertNode = (job, node, selectedId, catalog) => {
  const target = selectedId ? findNode(job, selectedId) : null;
  if (!target) {
    job.tasks.push(node);
    return;
  }
  const branches = branchesOf(catalogById(catalog)[target.node.task]);
  if (branches.length) {
    const branch = branches[0];
    target.node[branch] = target.node[branch] || [];
    target.node[branch].push(node);
  } else {
    target.list.splice(target.index + 1, 0, node);
  }
};

export const removeNode = (job, id) => {
  const target = findNode(job, id);
  if (target) target.list.splice(target.index, 1);
  return target;
};

export const updateNode = (job, id, patch) => {
  const target = findNode(job, id);
  if (target) Object.assign(target.node, patch);
};

export const setParam = (job, id, name, value) => {
  const target = findNode(job, id);
  if (!target) return;
  const params = { ...(target.node.params || {}) };
  if (value === undefined) delete params[name];
  else params[name] = value;
  target.node.params = params;
};

// Rename a node and the references to it (`every.of`, `break`/`move_to_item`
// loop targets); expressions that name it are left alone and the validator
// reports them.
export const renameNode = (job, id, newId) => {
  if (!isValidId(newId) || findNode(job, newId)) return false;
  for (const { node } of walk(job.tasks)) {
    if (node.id === id) node.id = newId;
    if (node.every && node.every.of === id) node.every = { ...node.every, of: newId };
    if (node.params && node.params.loop === id) node.params = { ...node.params, loop: newId };
  }
  return true;
};

export const moveNode = (job, id, delta) => {
  const target = findNode(job, id);
  if (!target) return;
  const to = target.index + delta;
  if (to < 0 || to >= target.list.length) return;
  const [node] = target.list.splice(target.index, 1);
  target.list.splice(to, 0, node);
};

// Into the previous sibling's first branch, at its end.
export const indentNode = (job, id, catalog) => {
  const target = findNode(job, id);
  if (!target || target.index === 0) return false;
  const previous = target.list[target.index - 1];
  const branches = branchesOf(catalogById(catalog)[previous.task]);
  if (!branches.length) return false;
  const [node] = target.list.splice(target.index, 1);
  previous[branches[0]] = previous[branches[0]] || [];
  previous[branches[0]].push(node);
  return true;
};

// Out of the parent's branch, right after the parent.
export const outdentNode = (job, id) => {
  const target = findNode(job, id);
  if (!target || !target.parent) return false;
  const parent = findNode(job, target.parent.id);
  const [node] = target.list.splice(target.index, 1);
  parent.list.splice(parent.index + 1, 0, node);
  return true;
};

// Move a node into another container's branch (or the job's top level when
// parentId is null) at an index; refuses to move a node into itself.
export const placeNode = (job, id, parentId, branch, index) => {
  const target = findNode(job, id);
  if (!target) return false;
  if (parentId && (parentId === id || findNode({ tasks: [target.node] }, parentId))) return false;
  const [node] = target.list.splice(target.index, 1);
  const list = parentId ? (findNode(job, parentId).node[branch] ||= []) : job.tasks;
  list.splice(Math.max(0, Math.min(index, list.length)), 0, node);
  return true;
};

// -- parameter fields ---------------------------------------------------------

const resolve = (schema, prop) => {
  if (prop && prop.$ref) {
    const name = prop.$ref.split("/").pop();
    return { ...((schema.$defs || {})[name] || {}), ...prop, $ref: undefined };
  }
  return prop || {};
};

// One form field per parameter of a catalogue entry:
// {name, title, kind, required, unit, choices, default, description, resourceKinds}
// kind: "number" | "integer" | "boolean" | "choice" | "string" | "expression"
//       | "strings" | "resource" | "json"
export const paramFields = (entry) => {
  const schema = (entry && entry.schema) || {};
  const required = new Set(schema.required || []);
  const refKinds = (entry && entry.ref_kinds) || {};
  const exprFields = new Set((entry && entry.expr_fields) || []);
  return Object.entries(schema.properties || {}).map(([name, raw]) => {
    let prop = resolve(schema, raw);
    const options = (prop.anyOf || []).filter((o) => o.type !== "null");
    const nullable = (prop.anyOf || []).some((o) => o.type === "null");
    if (options.length === 1) prop = { ...resolve(schema, options[0]), ...prop, anyOf: undefined };
    const field = {
      name,
      title: prop.title || name,
      required: required.has(name),
      nullable,
      unit: prop["x-unit"] || raw["x-unit"] || null,
      default: prop.default,
      description: prop.description || raw.description || "",
      resourceKinds: refKinds[name] || null,
      minimum: prop.minimum ?? prop.exclusiveMinimum,
      maximum: prop.maximum ?? prop.exclusiveMaximum,
    };
    const choices = prop.enum || raw.enum;
    if (exprFields.has(name)) field.kind = "expression";
    else if (choices) Object.assign(field, { kind: "choice", choices });
    else if (prop.type === "boolean") field.kind = "boolean";
    else if (prop.type === "integer") field.kind = "integer";
    else if (prop.type === "number") field.kind = "number";
    else if (prop.type === "string") field.kind = "string";
    else if (prop.type === "array" && prop.items && prop.items.type === "string")
      field.kind = "strings";
    else field.kind = field.resourceKinds ? "resource" : "json";
    return field;
  });
};

// Values may also be "${expr}" expressions or "@resource" references.
export const isExpression = (value) =>
  typeof value === "string" && value.startsWith("${") && value.endsWith("}");
export const isReference = (value) =>
  typeof value === "string" && value.startsWith("@") && !value.startsWith("@@");

// Text typed into a number field: a number, an expression, or nothing.
export const parseNumberInput = (text, integer = false) => {
  const trimmed = (text || "").trim();
  if (trimmed === "") return undefined;
  if (trimmed.startsWith("${") || trimmed.startsWith("@")) return trimmed;
  const value = integer ? parseInt(trimmed, 10) : parseFloat(trimmed.replace(",", "."));
  return Number.isNaN(value) ? trimmed : value;
};

// -- resources ------------------------------------------------------------------

// Starting values for a new resource of each kind (valid against the job schema).
export const RESOURCE_TEMPLATES = {
  positions: { kind: "positions", positions: [] },
  capture: { kind: "capture", channels: [{ config: "", exposure: 10 }] },
  z_plan: { kind: "z_plan", plan: { range: 10, step: 1 } },
  time_plan: { kind: "time_plan", plan: { interval: 60, loops: 10 } },
  grid: { kind: "grid", plan: { rows: 2, columns: 2, fov_width: 500, fov_height: 500 } },
  autofocus: { kind: "autofocus", mode: "software", range_um: 50, step_um: 5 },
  regions: { kind: "regions", regions: [] },
  focus_map: { kind: "focus_map", method: "spline" },
};

// Rename a resource and every "@name" reference to it in node parameters.
export const renameResource = (job, name, newName) => {
  const resources = job.resources || {};
  if (!isValidId(newName) || resources[newName] || !resources[name]) return false;
  job.resources = Object.fromEntries(
    Object.entries(resources).map(([key, value]) => [key === name ? newName : key, value]),
  );
  const replace = (value) => {
    if (value === `@${name}`) return `@${newName}`;
    return Array.isArray(value) ? value.map(replace) : value;
  };
  for (const { node } of walk(job.tasks)) {
    const params = node.params || {};
    node.params = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, replace(v)]));
  }
  return true;
};

export const resourcesOfKind = (job, kinds) =>
  Object.entries(job.resources || {})
    .filter(([, resource]) => !kinds || kinds.includes(resource.kind))
    .map(([name]) => name);

// Issues of one node from a validation report.
export const issuesFor = (validation, id) =>
  ((validation && validation.issues) || []).filter((issue) => issue.node === id);
