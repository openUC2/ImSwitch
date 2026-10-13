// src/state/slices/AufgabenSlice.js
// The Aufgaben app: the catalogue, the job being edited (persisted, so a
// reload keeps the draft), its validation, and the running job's status,
// events and open question (fed by sigAufgabenEvent and getAufgabenStatus).

import { createSlice } from "@reduxjs/toolkit";

import * as tree from "../../utils/aufgabenJob";

const MAX_EVENTS = 300;
const ACTIVE = ["running", "paused"];

const initialState = {
  catalog: [],
  catalogError: "",
  job: tree.emptyJob(),
  selectedId: null,
  validation: null,
  dirty: false,
  status: { state: "idle" },
  events: [],
  progress: null,
  question: null,
  jobs: [],
  runs: [],
};

const edited = (state) => {
  state.dirty = true;
  state.validation = null;
};

const aufgabenSlice = createSlice({
  name: "aufgaben",
  initialState,
  reducers: {
    setCatalog: (state, action) => {
      state.catalog = action.payload || [];
      state.catalogError = "";
    },
    setCatalogError: (state, action) => {
      state.catalogError = action.payload;
    },

    // -- the job document --------------------------------------------------
    loadJob: (state, action) => {
      state.job = { ...tree.emptyJob(), ...action.payload };
      state.selectedId = null;
      state.validation = null;
      state.dirty = false;
    },
    newJob: (state) => {
      state.job = tree.emptyJob();
      state.selectedId = null;
      state.validation = null;
      state.dirty = false;
    },
    markSaved: (state) => {
      state.dirty = false;
    },
    setJobField: (state, action) => {
      const { field, value } = action.payload;
      state.job[field] = value;
      edited(state);
    },
    setStorage: (state, action) => {
      state.job.storage = { ...state.job.storage, ...action.payload };
      edited(state);
    },
    addNode: (state, action) => {
      const entry = state.catalog.find((e) => e.id === action.payload);
      if (!entry) return;
      const node = tree.newNode(state.job, entry);
      tree.insertNode(state.job, node, state.selectedId, state.catalog);
      state.selectedId = node.id;
      edited(state);
    },
    removeNode: (state, action) => {
      const removed = tree.removeNode(state.job, action.payload);
      if (removed) {
        const { list, index, parent } = removed;
        state.selectedId = (list[index] || list[index - 1] || parent || {}).id || null;
        edited(state);
      }
    },
    updateNode: (state, action) => {
      tree.updateNode(state.job, action.payload.id, action.payload.patch);
      edited(state);
    },
    setNodeParam: (state, action) => {
      const { id, name, value } = action.payload;
      tree.setParam(state.job, id, name, value);
      edited(state);
    },
    renameNode: (state, action) => {
      const { id, newId } = action.payload;
      if (tree.renameNode(state.job, id, newId)) {
        if (state.selectedId === id) state.selectedId = newId;
        edited(state);
      }
    },
    moveNode: (state, action) => {
      tree.moveNode(state.job, action.payload.id, action.payload.delta);
      edited(state);
    },
    indentNode: (state, action) => {
      if (tree.indentNode(state.job, action.payload, state.catalog)) edited(state);
    },
    outdentNode: (state, action) => {
      if (tree.outdentNode(state.job, action.payload)) edited(state);
    },
    placeNode: (state, action) => {
      const { id, parentId, branch, index } = action.payload;
      if (tree.placeNode(state.job, id, parentId, branch, index)) edited(state);
    },
    selectNode: (state, action) => {
      state.selectedId = action.payload;
    },
    setResource: (state, action) => {
      const { name, resource } = action.payload;
      state.job.resources = { ...state.job.resources, [name]: resource };
      edited(state);
    },
    removeResource: (state, action) => {
      const resources = { ...state.job.resources };
      delete resources[action.payload];
      state.job.resources = resources;
      edited(state);
    },
    renameResource: (state, action) => {
      const { name, newName } = action.payload;
      if (tree.renameResource(state.job, name, newName)) edited(state);
    },
    setValidation: (state, action) => {
      state.validation = action.payload;
    },

    // -- runs ----------------------------------------------------------------
    setStatus: (state, action) => {
      const status = action.payload || { state: "idle" };
      state.status = status;
      state.question = status.question || null;
      if (!ACTIVE.includes(status.state)) state.progress = null;
    },
    aufgabenEvent: (state, action) => {
      const event = action.payload || {};
      switch (event.type) {
        case "state":
          state.status = { ...state.status, state: event.state, message: event.message };
          if (!ACTIVE.includes(event.state)) {
            state.question = null;
            state.progress = null;
          }
          break;
        case "node":
          state.status = { ...state.status, node: event.node };
          return;   // too frequent for the log
        case "question":
          state.question = { node: event.node, text: event.text, buttons: event.buttons };
          break;
        case "task_progress":
          state.progress = { percentage: event.percentage, message: event.message };
          return;
        default:
          break;
      }
      state.events.push({ ...event, time: Date.now() });
      if (state.events.length > MAX_EVENTS) state.events.splice(0, state.events.length - MAX_EVENTS);
    },
    clearEvents: (state) => {
      state.events = [];
    },
    answered: (state) => {
      state.question = null;
    },
    setJobs: (state, action) => {
      state.jobs = action.payload || [];
    },
    setRuns: (state, action) => {
      state.runs = action.payload || [];
    },
  },
});

export const {
  setCatalog,
  setCatalogError,
  loadJob,
  newJob,
  markSaved,
  setJobField,
  setStorage,
  addNode,
  removeNode,
  updateNode,
  setNodeParam,
  renameNode,
  moveNode,
  indentNode,
  outdentNode,
  placeNode,
  selectNode,
  setResource,
  removeResource,
  renameResource,
  setValidation,
  setStatus,
  aufgabenEvent,
  clearEvents,
  answered,
  setJobs,
  setRuns,
} = aufgabenSlice.actions;

export const getAufgabenState = (state) => state.aufgabenState;
export const isActive = (status) => ACTIVE.includes((status || {}).state);

export default aufgabenSlice.reducer;
