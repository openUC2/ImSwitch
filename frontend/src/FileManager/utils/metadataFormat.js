import { getDataSize } from "./getDataSize";

export const LABEL_MAP = {
  format: "Format",
  shape: "Shape",
  axes: "Axes",
  dtype: "Data type",
  pixelType: "Pixel type",
  pixelSizeUm: "Pixel size",
  dimensions: "Dimensions",
  channels: "Channels",
  acquisitionDate: "Acquired",
  exposure: "Exposure",
  stagePosition: "Stage position",
  description: "Description",
  sizeBytes: "Size on disk",
  metadataError: "Warning",
};

export const formatPixelSize = (ps) => {
  if (!ps) return null;
  const parts = [];
  for (const axis of ["x", "y", "z"]) {
    if (ps[axis] != null) parts.push(`${axis.toUpperCase()}: ${ps[axis]} µm`);
  }
  return parts.join(", ");
};

export const formatDimensions = (dims) => {
  if (!dims) return null;
  return Object.entries(dims)
    .map(([k, v]) => `${k}: ${v}`)
    .join(", ");
};

export const formatExposure = (exp) => {
  if (!exp) return null;
  return `${exp.value} ${exp.unit}`;
};

export const formatStagePosition = (pos) => {
  if (!pos) return null;
  return Object.entries(pos)
    .map(([k, v]) => `${k.toUpperCase()}: ${Number(v).toFixed(1)}`)
    .join(", ");
};

export const formatSimpleValue = (v) => {
  if (v === true) return "Yes";
  if (v === false) return "No";
  if (typeof v === "number")
    return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (Array.isArray(v)) return v.join(", ");
  return String(v);
};

export const formatValue = (key, value) => {
  if (value == null) return null;
  switch (key) {
    case "pixelSizeUm":
      return formatPixelSize(value);
    case "dimensions":
      return formatDimensions(value);
    case "exposure":
      return formatExposure(value);
    case "stagePosition":
      return formatStagePosition(value);
    case "sizeBytes":
      return value > 0 ? getDataSize(value) : null;
    case "shape":
      return Array.isArray(value) ? value.join(" × ") : String(value);
    case "channels":
      return Array.isArray(value) ? value.join(", ") : String(value);
    case "description":
      return "___DESCRIPTION___";
    default:
      return String(value);
  }
};

export const groupAcquisitionAttributes = (attrs) => {
  const groups = {};
  for (const [key, value] of Object.entries(attrs)) {
    const parts = key.split(":");
    const groupKey =
      parts.length >= 2 ? `${parts[0]}:${parts[1]}` : parts[0];
    const label = parts.length >= 3 ? parts.slice(2).join(":") : key;
    if (!groups[groupKey]) groups[groupKey] = [];
    groups[groupKey].push({ label, value });
  }
  return groups;
};
