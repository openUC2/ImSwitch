export const SUPPORTED_GAIN_VALUES = [
  0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 23,
];

// The Andor (and other MMCore cameras) accept a wide, continuous gain range
// (up to ~300), so gain is now a free numeric field. Parse and clamp to the
// valid range rather than snapping to a small discrete list. SUPPORTED_GAIN_VALUES
// is kept for reference/back-compat but no longer constrains input.
export const MIN_GAIN_VALUE = 0;
export const MAX_GAIN_VALUE = 300;

export const normalizeGainValue = (rawValue) => {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) return null;

  return Math.min(MAX_GAIN_VALUE, Math.max(MIN_GAIN_VALUE, parsed));
};
