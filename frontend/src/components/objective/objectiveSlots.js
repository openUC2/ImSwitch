// src/components/objective/objectiveSlots.js
// The two turret slots as plain objects, derived from ObjectiveSlice. Slots
// are 0-based (ObjectiveController.moveToObjective); the UI shows them 1-based.

const num = (v) => (v === null || v === undefined || v === "" ? NaN : Number(v));

export const getSlots = (obj) =>
  [0, 1].map((i) => ({
    index: i,
    label: `Slot ${i + 1}`,
    name: obj.availableObjectivesNames?.[i] || `Obj ${i + 1}`,
    magnification: num(obj.availableObjectiveMagnifications?.[i]),
    na: num(obj.availableObjectiveNAs?.[i]),
    pixelSize: num(obj.availableObjectivePixelSizes?.[i]),
    configured: obj.slotConfigured?.[i] !== false,
    turretPosition: num(i === 0 ? obj.posX0 : obj.posX1), // A axis, µm
    focusPosition: num(i === 0 ? obj.posZ0 : obj.posZ1), // Z axis, µm
  }));

export const fmt = (v, digits = 1) =>
  Number.isFinite(Number(v)) && v !== "" && v !== null
    ? Number(v).toLocaleString(undefined, { maximumFractionDigits: digits })
    : "—";

/** "4× / 0.10" style short optics label, or the slot name when unknown. */
export const opticsLabel = (slot) =>
  slot.magnification > 0
    ? `${fmt(slot.magnification, 0)}×${slot.na > 0 ? ` / ${fmt(slot.na, 2)}` : ""}`
    : slot.name;
