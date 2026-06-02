export const SUPPORTED_GAIN_VALUES = [
  0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 23,
];

export const normalizeGainValue = (rawValue) => {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) return null;

  return SUPPORTED_GAIN_VALUES.reduce((closest, current) => {
    return Math.abs(current - parsed) < Math.abs(closest - parsed)
      ? current
      : closest;
  }, SUPPORTED_GAIN_VALUES[0]);
};
