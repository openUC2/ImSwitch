/**
 * Experiment Designer - Dimension-based experiment configuration UI
 * 
 * This module provides a modern, progressive disclosure UI for configuring
 * multidimensional microscopy experiments. Parameters are organized around
 * explicit acquisition dimensions:
 * 
 * - Positions: Well plate / XY canvas selection
 * - Channels: Illumination sources, exposure, gain
 * - Z / Focus: Single Z, Autofocus, Z-Stack options
 * - Time: Time-lapse configuration
 * - Tiling: Overlap and scan pattern settings
 * - Output: File format selection
 * 
 * Each dimension can be enabled/disabled independently, and only shows
 * relevant parameters when active. This reduces cognitive load and makes
 * the experiment configuration workflow more intuitive.
 */

// Main container component
export { default as ExperimentDesigner } from "./ExperimentDesigner";

// Dimension bar navigation
export { default as DimensionBar } from "./DimensionBar";

// Summary panel
export { default as ExperimentSummary } from "./ExperimentSummary";

// Individual dimension components
export { default as PositionsDimension } from "./PositionsDimension";
export { default as ChannelsDimension } from "./ChannelsDimension";
export { default as ZFocusDimension } from "./ZFocusDimension";
export { default as TimeDimension } from "./TimeDimension";
export { default as TilingDimension } from "./TilingDimension";
export { default as OutputDimension } from "./OutputDimension";

// Re-export dimension constants
export { DIMENSIONS, Z_FOCUS_MODES } from "../../state/slices/ExperimentUISlice";
