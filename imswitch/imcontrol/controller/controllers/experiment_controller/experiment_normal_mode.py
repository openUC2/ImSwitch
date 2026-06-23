"""
Normal mode implementation for ExperimentController.

This module handles experiment execution where Python controls each step
of the scanning process using workflow steps for precise control.
"""

import os
import re
from typing import List, Dict, Any, Optional
import numpy as np

from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowStep
from imswitch.imcontrol.model.io import OMEWriter, OMEROConnectionParams
from imswitch.imcontrol.model.io.ome_writers import write_plate_metadata_sidecar
from .experiment_mode_base import ExperimentModeBase


# Module-level constant so every method in this file can reference the same
# tuple.  Used by:
#   - execute_experiment (size-of-channel calc)
#   - _setup_ome_writers (channel-name expansion)
#   - _create_tile_workflow_steps (per-position DPC step expansion)
# Order matters: it's the order frames are acquired and saved per XY
# position, and the OME channel-name ordering ("DPC_top", "DPC_bottom", ...).
DPC_SUB_DIRS = ("top", "bottom", "left", "right")


def _sanitize_name(name: str, max_len: int = 40) -> str:
    """Make a position/area name safe to use as a file/folder path component.

    Keeps alphanumerics, dot, dash and underscore; collapses every other run of
    characters to a single dash; trims separators and length. Returns "" for an
    empty/None input so callers can fall back to the legacy index-only naming
    (keeping the output byte-identical when no name is set).
    """
    s = re.sub(r"[^0-9A-Za-z._-]+", "-", str(name or "").strip())
    s = s.strip("-_.")
    return s[:max_len]


class ExperimentNormalMode(ExperimentModeBase):
    """
    Normal mode experiment execution.

    In normal mode, Python controls each step: moving stage, setting illumination,
    acquiring frames, and saving data. This provides maximum flexibility and
    precise control over the experiment sequence.
    """

    def execute_experiment(self,
                         snake_tiles: List[List[Dict]] = None,
                         illumination_intensities: List[float] = None,
                         illumination_sources: List[str] = None,
                         isRGB: bool = False,
                         shared_file_writers: Optional[List[OMEWriter]] = None,
                         omero_connection_params: Optional[OMEROConnectionParams] = None,
                         shared_omero_key: Optional[str] = None,
                         ctx=None,
                         **kwargs) -> Dict[str, Any]:
        """
        Execute experiment in normal mode.

        Preferred call form (used by ``ExperimentController.startWellplateExperiment``)::

            self.execute_experiment(ctx=execution_context,
                                    shared_file_writers=...,
                                    omero_connection_params=...,
                                    shared_omero_key=...)

        The historical kwargs-based form is still supported for any
        external callers; when ``ctx`` is provided it is flattened to
        kwargs via ``ExecutionContext.to_kwargs()``.

        Args:
            ctx: Optional :class:`ExecutionContext` carrying snake tiles,
                Z offsets, resolved illumination/exposure/gain lists,
                output paths, the validated ``Experiment`` model, and the
                current timepoint index.
            snake_tiles, illumination_intensities, illumination_sources,
            isRGB: legacy positional/kwarg fallback when ``ctx`` is None.
            shared_file_writers: Optional shared writers from previous timepoints (for timelapse).
            omero_connection_params: Optional OMERO connection parameters for streaming upload.
            shared_omero_key: Optional key for shared OMERO uploader.
            **kwargs: Additional parameters (z_positions, exposures, gains, etc.).

        Returns:
            Dictionary with execution results including workflow steps and file writers
        """
        # If a typed ExecutionContext was provided, flatten it onto kwargs
        # so the existing body below stays unchanged.  Caller-supplied
        # kwargs win over ctx values to keep the override path open.
        if ctx is not None:
            ctx_kwargs = ctx.to_kwargs()
            snake_tiles = ctx_kwargs.pop("snake_tiles")
            illumination_intensities = ctx_kwargs.pop("illumination_intensities")
            illumination_sources = ctx_kwargs.pop("illumination_sources")
            isRGB = ctx_kwargs.pop("isRGB", isRGB)
            for _key, _val in ctx_kwargs.items():
                kwargs.setdefault(_key, _val)

        self._logger.debug("Normal mode is enabled. Creating workflow steps for precise control.")

        # Extract parameters
        z_positions = kwargs.get('z_positions', [0])
        exposures = kwargs.get('exposures', [100])
        gains = kwargs.get('gains', [1])
        # Per-source kind tags + kind-specific params (radius/RGB for the
        # LED-matrix synthetic channels).  Default to "default"/empty so
        # callers that don't yet thread these through keep working.
        illumination_kinds = kwargs.get('illumination_kinds') or []
        illumination_params = kwargs.get('illumination_params') or {}
        if not illumination_kinds:
            illumination_kinds = ["default"] * len(illumination_sources or [])
        # Defence-in-depth against array misalignment: build a name→kind dict
        # so every downstream lookup goes by name, never by index.  Earlier
        # bugs collapsed illumination_sources without touching the kinds
        # array, which then mapped a synthetic "LED Matrix Ring" source to
        # the kind "default" at index 0 and tried to drive it through
        # set_laser_power (which has no laser by that name).  With this dict
        # any future asymmetric filtering is harmless.
        def _kind_for(name: str) -> str:
            # Prefer explicit mapping from the parallel arrays as supplied;
            # only fall back to "default" if the source is genuinely unknown.
            for _n, _k in zip(illumination_sources or [], illumination_kinds or []):
                if _n == name:
                    return _k
            return "default"
        # Effective channel-frame count: ring contributes 1, dpc contributes 4,
        # default contributes 1.  Used to size the OME writer's channel axis
        # so DPC's four sub-frames all fit.  (DPC_SUB_DIRS is the module-level
        # constant defined at the top of this file.)
        def _effective_frames_for_kind(kind: str) -> int:
            return 4 if kind == "dpc" else 1
        exp_name = kwargs.get('exp_name', 'experiment')
        dir_path = kwargs.get('dir_path', '')
        m_file_name = kwargs.get('m_file_name', 'experiment')
        t = kwargs.get('t', 0)  # time index
        is_auto_focus = kwargs.get('is_auto_focus', False)
        autofocus_min = kwargs.get('autofocus_min', 0)
        autofocus_max = kwargs.get('autofocus_max', 0)
        autofocus_step_size = kwargs.get('autofocus_step_size', 1)
        autofocus_illumination_channel = kwargs.get('autofocus_illumination_channel', '')
        autofocus_mode = kwargs.get('autofocus_mode', 'software')  # 'hardware' or 'software'
        autofocus_software_method = kwargs.get('autofocus_software_method', 'scan')  # 'scan' or 'hillClimbing'
        autofocus_hc_initial_step = kwargs.get('autofocus_hc_initial_step', 20.0)
        autofocus_hc_min_step = kwargs.get('autofocus_hc_min_step', 1.0)
        autofocus_hc_step_reduction = kwargs.get('autofocus_hc_step_reduction', 0.5)
        autofocus_hc_max_iterations = kwargs.get('autofocus_hc_max_iterations', 50)
        autofocus_max_attempts = kwargs.get('autofocus_max_attempts', 2)
        autofocus_target_focus_setpoint = kwargs.get('autofocus_target_focus_setpoint', None)
        initial_z_position = kwargs.get('initial_z_position', None)
        t_period = kwargs.get('t_period', 1)
        # Timing parameters from GUI (used for settle/exposure waits)
        t_pre_s = kwargs.get('t_pre_s', 0.09)  # Pre-exposure settle time in seconds
        t_post_s = kwargs.get('t_post_s', 0.05)  # Post-exposure time in seconds
        # New parameters for multi-timepoint support
        n_times = kwargs.get('n_times', 1)  # Total number of time points
        # Illumination mode: keep illumination on for entire acquisition?
        keep_illumination_on = kwargs.get('keep_illumination_on', False)

        # Initialize workflow components
        workflow_steps = []
        file_writers = []
        step_id = 0

        # Set up OME writers for each tile
        # If shared_file_writers provided (timelapse), reuse them; otherwise create new
        if shared_file_writers is not None:
            file_writers = shared_file_writers
            self._logger.debug(f"Reusing {len(file_writers)} shared file writers from previous timepoint")
        else:
            file_writers = self._setup_ome_writers(
                snake_tiles, t, exp_name, dir_path, m_file_name,
                z_positions, illumination_intensities, isRGB=isRGB,
                omero_connection_params=omero_connection_params,
                shared_omero_key=shared_omero_key,
                n_times=n_times,
                illumination_kinds=illumination_kinds,
            )
        # Compute writer offset for multi-timepoint experiments.
        # Each timepoint creates its own set of writers, so the flat
        # file_writers list is indexed as: t * num_tiles + tile_index.
        is_single_tiff_mode = getattr(self.controller, '_ome_write_single_tiff', False)
        writer_offset = t * len(snake_tiles) if not is_single_tiff_mode else 0

        # If keep_illumination_on, turn on all active illumination sources once
        # at the beginning instead of toggling per frame.
        # LED-matrix synthetic channels ("ring"/"dpc") are excluded — they must
        # be toggled per-frame because each frame uses a different pattern
        # (and DPC explicitly cycles 4 patterns per position).
        if keep_illumination_on:
            for illu_index, illu_source in enumerate(illumination_sources):
                illu_intensity = illumination_intensities[illu_index] if illu_index < len(illumination_intensities) else 0
                illu_kind = _kind_for(illu_source)  # name-based lookup; immune to array misalignment
                if illu_intensity > 0 and illu_kind == "default":
                    workflow_steps.append(WorkflowStep(
                        name=f"Turn on illumination (continuous): {illu_source}",
                        step_id=step_id,
                        main_func=self.controller.set_laser_power,
                        main_params={"power": illu_intensity, "channel": illu_source},
                        post_funcs=[self.controller.wait_time],
                        post_params={"seconds": t_pre_s},
                    ))
                    step_id += 1

        # Create workflow steps for each tile
        for position_center_index, tiles in enumerate(snake_tiles):
            step_id = self._create_tile_workflow_steps(
                tiles, position_center_index, step_id, workflow_steps,
                z_positions, initial_z_position, illumination_sources, illumination_intensities,
                exposures, gains, t, is_auto_focus, autofocus_min,
                autofocus_max, autofocus_step_size, autofocus_illumination_channel,
                autofocus_mode, autofocus_max_attempts, autofocus_target_focus_setpoint, n_times,
                autofocus_software_method=autofocus_software_method,
                autofocus_hc_initial_step=autofocus_hc_initial_step,
                autofocus_hc_min_step=autofocus_hc_min_step,
                autofocus_hc_step_reduction=autofocus_hc_step_reduction,
                autofocus_hc_max_iterations=autofocus_hc_max_iterations,
                t_pre_s=t_pre_s, t_post_s=t_post_s,
                writer_offset=writer_offset,
                keep_illumination_on=keep_illumination_on,
                illumination_kinds=illumination_kinds,
                illumination_params=illumination_params,
            )

        # Add finalization steps
        step_id = self._add_finalization_steps(
            workflow_steps, step_id, snake_tiles, illumination_sources,
            illumination_intensities, t_period, t, n_times,
            writer_offset=writer_offset,
            keep_illumination_on=keep_illumination_on,
            illumination_kinds=illumination_kinds,
        )

        # Add step to set LED status to idle when done
        workflow_steps.append(WorkflowStep(
            step_id=step_id,
            name="Set LED status to idle",
            main_func=self.controller.set_led_status,
            main_params={"status": "idle"})
        )
        step_id += 1

        # Save experiment protocol to JSON exactly once, on the first timepoint.
        # The file name has no _t{NNNN} suffix so it is stable for the whole
        # experiment and is written only once regardless of n_times.
        if t == 0:
            protocol_data = {
                "experiment_name": exp_name,
                "experiment_mode": "normal",
                "directory": dir_path,
                "filename": m_file_name,
                "total_timepoints": n_times,
                "tile_count": len(snake_tiles),
                "step_count": step_id,
                "snake_tiles": snake_tiles,
                "z_positions": z_positions,
                "illumination_sources": illumination_sources,
                "illumination_intensities": illumination_intensities,
                "illumination_kinds": illumination_kinds,
                "illumination_params": illumination_params,
                "exposures": exposures,
                "gains": gains,
                "autofocus": {
                    "enabled": is_auto_focus,
                    "min": autofocus_min,
                    "max": autofocus_max,
                    "step_size": autofocus_step_size,
                    "channel": autofocus_illumination_channel,
                    "mode": autofocus_mode,
                    "software_method": autofocus_software_method,
                    "max_attempts": autofocus_max_attempts,
                    "target_focus_setpoint": autofocus_target_focus_setpoint,
                    "hc_initial_step": autofocus_hc_initial_step,
                    "hc_min_step": autofocus_hc_min_step,
                    "hc_step_reduction": autofocus_hc_step_reduction,
                    "hc_max_iterations": autofocus_hc_max_iterations,
                },
                "workflow_steps": [self._serialize_workflow_step(step) for step in workflow_steps]
            }
            # Protocol file path — no _t{NNNN} suffix
            protocol_file_path = os.path.join(dir_path, m_file_name)
            self.save_experiment_protocol(protocol_data, protocol_file_path, mode="normal")

        return {
            "status": "workflow_created",
            "mode": "normal",
            "workflow_steps": workflow_steps,
            "file_writers": file_writers,
            "step_count": step_id
        }

    def _serialize_workflow_step(self, step) -> Dict[str, Any]:
        """
        Serialize a WorkflowStep object to a dictionary for JSON export.

        Args:
            step: WorkflowStep object

        Returns:
            Dictionary representation of the workflow step
        """
        return {
            "step_id": step.step_id,
            "name": step.name,
            "main_func": step.main_func.__name__ if hasattr(step.main_func, '__name__') else str(step.main_func),
            "main_params": step.main_params,
            "pre_funcs": [f.__name__ if hasattr(f, '__name__') else str(f) for f in step.pre_funcs],
            "pre_params": step.pre_params,
            "post_funcs": [f.__name__ if hasattr(f, '__name__') else str(f) for f in step.post_funcs],
            "post_params": step.post_params,
            "max_retries": step.max_retries
        }

    def setup_shared_ome_writers(self,
                                snake_tiles: List[List[Dict]],
                                exp_name: str,
                                dir_path: str,
                                m_file_name: str,
                                z_positions: List[float],
                                illumination_intensities: List[float],
                                isRGB: bool = False,
                                omero_connection_params: Optional[OMEROConnectionParams] = None,
                                shared_omero_key: Optional[str] = None,
                                n_times: int = 1) -> List[OMEWriter]:
        """
        Set up shared OME writers for multi-timepoint experiments.

        This method creates writers that can be reused across multiple timepoints
        in a timelapse experiment. The writers are configured to handle all
        timepoints without creating new files for each timepoint.

        Args:
            snake_tiles: List of tiles containing scan points
            exp_name: Experiment name
            dir_path: Directory path for saving
            m_file_name: Base filename
            z_positions: List of Z positions
            illumination_intensities: List of illumination values
            isRGB: Whether images are RGB
            omero_connection_params: Optional OMERO connection parameters
            shared_omero_key: Optional key for shared OMERO uploader
            n_times: Total number of time points

        Returns:
            List of OMEWriter instances to be reused across timepoints
        """
        return self._setup_ome_writers(
            snake_tiles=snake_tiles,
            t=0,  # Initial timepoint
            exp_name=exp_name,
            dir_path=dir_path,
            m_file_name=m_file_name,
            z_positions=z_positions,
            illumination_intensities=illumination_intensities,
            isRGB=isRGB,
            omero_connection_params=omero_connection_params,
            shared_omero_key=shared_omero_key,
            n_times=n_times,
        )
    def _setup_ome_writers(self,
                          snake_tiles: List[List[Dict]],
                          t: int,
                          exp_name: str,
                          dir_path: str,
                          m_file_name: str,
                          z_positions: List[float],
                          illumination_intensities: List[float],
                          isRGB: bool,
                          omero_connection_params: Optional[OMEROConnectionParams] = None,
                          shared_omero_key: Optional[str] = None,
                          n_times: int = 1,
                          illumination_kinds: Optional[List[str]] = None) -> List[OMEWriter]:

        """
        Set up OME writers for each tile.

        Args:
            snake_tiles: List of tiles containing scan points
            t: Time index
            exp_name: Experiment name
            dir_path: Directory path for saving
            m_file_name: Base filename
            z_positions: List of Z positions
            illumination_intensities: List of illumination values
            isRGB: Whether images are RGB
            omero_connection_params: Optional OMERO connection parameters
            shared_omero_key: Optional key for shared OMERO uploader
            n_times: Total number of time points

        Returns:
            List of OMEWriter instances
        """
        file_writers = []

        # Create shared directory for individual TIFFs - all tiles go under experiment folder
        # Structure: dir_path/m_file_name/tiles/timepoint_XXXX/
        shared_individual_tiffs_dir = None  # No longer needed - OMEFileStorePaths handles it internally

        # Original behavior: create separate writers for each tile position
        # but use shared individual_tiffs directory
        wells_used: List[tuple] = []
        labware_load_name: Optional[str] = None
        condition_labels: Dict[str, str] = {}
        for position_center_index, tiles in enumerate(snake_tiles):
            experiment_name = f"{t}_{exp_name}_{position_center_index}"
            # Per-area position name (frontend pointList → ScanArea.areaName,
            # carried onto every tile by generate_snake_tiles). Appended to the
            # base name, additively: the timestamp/experiment/index prefix is
            # preserved so existing index-based tooling still resolves the files,
            # while the file, the per-area folder (both derived from this base
            # path) and the OME/OMERO image name now carry the position name.
            # Empty name => byte-identical to the legacy index-only naming.
            area_name = _sanitize_name(tiles[0].get("areaName") or tiles[0].get("name")) if tiles else ""
            m_file_path = os.path.join(
                dir_path,
                m_file_name + str(position_center_index) + "_" + experiment_name + "_" + area_name + ".ome.tif"
            )
            self._logger.debug(f"OME-TIFF path: {m_file_path}")

            # Create file paths with shared individual_tiffs directory
            file_paths = self.create_ome_file_paths(m_file_path.replace(".ome.tif", ""), shared_individual_tiffs_dir)

            # Calculate tile and grid parameters
            tile_shape = (self.controller.mDetector._shape[-1], self.controller.mDetector._shape[-2])
            grid_shape, grid_geometry = self.calculate_grid_parameters(tiles)

            # Create writer configuration.
            # Effective channel count: active normal/ring channels contribute
            # one frame each; an active DPC channel contributes four (one per
            # half-illumination quadrant).  We have to size the OME writer's
            # channel axis to fit the expanded total so DPC sub-frames land
            # in distinct channel slots rather than overwriting each other.
            # We also build the parallel channel_names list so OME-Zarr root
            # metadata (omero.channels[].label) shows "DPC_top" etc. instead
            # of generic "Channel_0", which makes the resulting stores
            # immediately readable in napari/Fiji without manual renaming.
            _ints = list(illumination_intensities) if illumination_intensities is not None else []
            _sources_for_naming: List[str] = []
            try:
                _sources_for_naming = list(getattr(self.controller, '_illuminationSources', []) or [])
            except Exception:
                _sources_for_naming = []
            # Look kinds up by source name (defence-in-depth against array
            # misalignment — the parallel arrays SHOULD match here, but
            # name-based dispatch keeps us correct even when they don't).
            _kind_by_name_local = {
                n: k for n, k in zip(_sources_for_naming or [], list(illumination_kinds or []))
            }
            n_channels = 0
            channel_names_expanded: List[str] = []
            for src_idx, _intensity in enumerate(_ints):
                if _intensity is None or _intensity <= 0:
                    continue
                src_name = (
                    _sources_for_naming[src_idx]
                    if src_idx < len(_sources_for_naming)
                    else f"Channel_{src_idx}"
                )
                _kind = _kind_by_name_local.get(src_name, "default")
                if _kind == "dpc":
                    for _d in DPC_SUB_DIRS:
                        channel_names_expanded.append(f"DPC_{_d}")
                    n_channels += 4
                elif _kind == "ring":
                    channel_names_expanded.append("Ring")
                    n_channels += 1
                else:
                    channel_names_expanded.append(src_name)
                    n_channels += 1
            if n_channels == 0:
                # Fallback to legacy behaviour when nothing is active.
                n_channels = max(1, int(sum(np.array(_ints) > 0)))
                channel_names_expanded = None  # let OMEWriter default to "Channel_N"
            write_omero = omero_connection_params is not None and getattr(self.controller, '_ome_write_omero', False)
            writer_config = self.create_writer_config(
                write_tiff=self.controller._ome_write_tiff,
                write_zarr=self.controller._ome_write_zarr,
                write_stitched_tiff=self.controller._ome_write_stitched_tiff,
                write_tiff_single=False,  # Disable single TIFF for multi-tile mode
                write_individual_tiffs=self.controller._ome_write_individual_tiffs,
                write_omero=write_omero,
                min_period=0.1,  # Faster for normal mode
                n_time_points=n_times,
                n_z_planes=len(z_positions),
                n_channels=n_channels,
                channel_names=channel_names_expanded,
            )

            # Extract per-well labware metadata from the first tile in this group.
            # Tiles within a single position_center_index share a well, so the
            # first one is representative.
            well_metadata: Optional[Dict[str, Any]] = None
            if tiles:
                first = tiles[0]
                w_row = first.get("wellRow")
                w_col = first.get("wellColumn")
                w_load = first.get("labwareLoadName")
                w_cond = first.get("conditionLabel")
                if w_row or w_col or w_load:
                    well_metadata = {
                        "wellRow": w_row,
                        "wellColumn": w_col,
                        "labwareLoadName": w_load,
                        "conditionLabel": w_cond,
                    }
                    if w_row and w_col is not None:
                        wells_used.append((str(w_row), str(int(w_col))))
                    if w_load and labware_load_name is None:
                        labware_load_name = w_load
                    if w_cond and w_row and w_col is not None:
                        condition_labels[f"{w_row}{int(w_col)}"] = w_cond

            # Create OME writer
            ome_writer = OMEWriter(
                file_paths=file_paths,
                tile_shape=tile_shape,
                grid_shape=grid_shape,
                grid_geometry=grid_geometry,
                config=writer_config,
                logger=self._logger,
                isRGB=isRGB,
                omero_connection_params=omero_connection_params,
                shared_omero_key=shared_omero_key,
                well_metadata=well_metadata,
                # Clean position name for the OME/OMERO image-name metadata
                # (falls back to the file basename when no name is set).
                image_name=area_name or None,
            )
            file_writers.append(ome_writer)

        # Emit OME-NGFF plate sidecar once per acquisition (only on the first
        # timepoint to avoid clobbering on every loop iteration).
        if labware_load_name and t == 0 and self.controller.labware_manager is not None:
            try:
                lab = self.controller.labware_manager.get(labware_load_name)
                if lab is not None:
                    rows = list(lab.rows)
                    columns = [str(c) for c in lab.columns]
                    write_plate_metadata_sidecar(
                        output_dir=dir_path,
                        plate_name=labware_load_name,
                        rows=rows,
                        columns=columns,
                        wells_used=wells_used,
                        extra={
                            "imswitch_labware": {
                                "loadName": labware_load_name,
                                "conditionLabels": condition_labels or None,
                            }
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - sidecar is best-effort
                self._logger.warning(f"Failed to write plate metadata sidecar: {exc}")

        return file_writers

    def _create_tile_workflow_steps(self,
                                  tiles: List[Dict],
                                  position_center_index: int,
                                  step_id: int,
                                  workflow_steps: List[WorkflowStep],
                                  z_positions: List[float],
                                  initial_z_position: float,
                                  illumination_sources: List[str],
                                  illumination_intensities: List[float],
                                  exposures: List[float],
                                  gains: List[float],
                                  t: int,
                                  is_auto_focus: bool,
                                  autofocus_min: float,
                                  autofocus_max: float,
                                  autofocus_step_size: float,
                                  autofocus_illumination_channel: str,
                                  autofocus_mode: str,
                                    autofocus_max_attempts: int,
                                    autofocus_target_focus_setpoint: float,
                                  n_times: int,
                                  t_pre_s: float = 0.09,
                                  t_post_s: float = 0.05,
                                  writer_offset: int = 0,
                                  keep_illumination_on: bool = False,
                                  autofocus_software_method: str = "scan",
                                  autofocus_hc_initial_step: float = 20.0,
                                  autofocus_hc_min_step: float = 1.0,
                                  autofocus_hc_step_reduction: float = 0.5,
                                  autofocus_hc_max_iterations: int = 50,
                                  illumination_kinds: Optional[List[str]] = None,
                                  illumination_params: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
        """
        Create workflow steps for a single tile.

        Args:
            tiles: List of points in the tile
            position_center_index: Index of the tile
            step_id: Current step ID
            workflow_steps: List to append workflow steps to
            z_positions: List of Z positions
            illumination_sources: List of illumination source names
            illumination_intensities: List of illumination values
            exposures: List of exposure times
            gains: List of gain values
            t: Time index
            is_auto_focus: Whether autofocus is enabled
            autofocus_min: Minimum autofocus position
            autofocus_max: Maximum autofocus position
            autofocus_step_size: Autofocus step size
            autofocus_illumination_channel: Selected illumination channel for autofocus
            autofocus_mode: Autofocus mode ('hardware' or 'software')
            autofocus_max_attempts,
            autofocus_target_focus_setpoint,


        Returns:
            Updated step ID
        """
        # Get scan range information
        min_x, max_x, min_y, max_y, _, _ = self.compute_scan_ranges([tiles])
        m_pixel_size = self.controller.detectorPixelSize[-1] if hasattr(self.controller, 'detectorPixelSize') else 1.0

        # Iterate over positions in the tile
        for m_index, m_point in enumerate(tiles):
            try:
                name = f"Move to point {m_point['iterator']}"
            except Exception:
                name = f"Move to point {m_point['x']}, {m_point['y']}"

            # Determine per-point Z base: prefer the point's own z, else fall
            # back to the global initial Z captured at experiment start.
            # z=0.0 is the frontend default for sub-tiles that have no explicit
            # per-point Z – treat it the same as None so we always use the real
            # stage Z (initial_z_position) in that case.
            # (The "override per-group Z with current Z" Tiling toggle is applied
            # entirely on the frontend, which rewrites each position's Z before
            # sending; here we just consume the coordinates as-is.)
            point_z_origin = m_point.get("z")
            if point_z_origin is None or point_z_origin == 0.0:
                point_z_origin = initial_z_position

            # Move to XY position
            workflow_steps.append(WorkflowStep(
                name=name,
                step_id=step_id,
                main_func=self.controller.move_stage_xy,
                main_params={"posX": m_point["x"], "posY": m_point["y"], "relative": False},
            ))
            step_id += 1

            # Focus map Z lookup (happens after XY move to know the coordinate).
            # Gate on `_focus_map_active` (set per-experiment from
            # focusMap.enabled + apply_during_scan) so a map left over in the
            # manager from a PREVIOUS run is never silently re-applied to an
            # experiment that did not request focus mapping – e.g. after the
            # user cleared the focus map.  Also require fitted maps to exist.
            focus_map_z = None
            if getattr(self.controller, "_focus_map_active", False) and self.controller.focus_map_manager.get_all():
                # Resolve the group_id used when storing the focus map.
                # _run_focus_map_phase stores under sa.areaId (e.g. "area_0").
                focus_map_group_id = (
                    m_point.get("centerIndex")
                    or m_point.get("areaName")
                    or m_point.get("wellId")
                    or "default"
                )
                focus_map_z = self.controller.apply_focus_map_z(
                    x=m_point["x"], y=m_point["y"], group_id=focus_map_group_id
                )
                if focus_map_z is None and not getattr(self.controller, '_focus_map_fit_by_region', True):
                    focus_map_z = self.controller.apply_focus_map_z(
                        x=m_point["x"], y=m_point["y"], group_id="global"
                    )
                if focus_map_z is None:
                    focus_map_z = self.controller.apply_focus_map_z(
                        x=m_point["x"], y=m_point["y"], group_id="manual"
                    )

            # -------------------------------------------------------------------
            # SINGLE SOURCE OF TRUTH for Z positions.
            # z_positions are PURE RELATIVE OFFSETS sent by the frontend (e.g.
            # [-10, -8, ..., 10] for a Z-stack, or [0.0] for single Z).
            # base_z is the absolute reference for this tile:
            #   1. focus_map_z  – interpolated surface (highest priority)
            #   2. point_z_origin – per-scan-area Z from the frontend
            #   3. initial_z_position – global Z measured at experiment start
            # effective_z_positions = [base_z + offset for offset in z_positions]
            # -------------------------------------------------------------------
            base_z = focus_map_z if focus_map_z is not None else point_z_origin
            effective_z_positions = [base_z + offset for offset in z_positions]

            # If focus map provides a Z, emit a dedicated Z-move so the stage
            # settles at the predicted position before any Z-stack or capture.
            if focus_map_z is not None:
                settle_ms = getattr(self.controller, '_focus_map_settle_ms', 0)
                settle_s = settle_ms / 1000.0 if settle_ms > 0 else 0
                workflow_steps.append(WorkflowStep(
                    name=f"Focus map Z → {focus_map_z:.2f}",
                    step_id=step_id,
                    main_func=self.controller.move_stage_z,
                    main_params={"posZ": focus_map_z, "relative": False},
                    post_funcs=[self.controller.wait_time] if settle_s > 0 else None,
                    post_params={"seconds": settle_s} if settle_s > 0 else None,
                ))
                step_id += 1

            # Perform autofocus if enabled (runs after focus map Z move if both active)
            if is_auto_focus:
                workflow_steps.append(WorkflowStep(
                    name="Autofocus",
                    step_id=step_id,
                    main_func=self.controller.autofocus,
                    main_params={
                        "minZ": autofocus_min,
                        "maxZ": autofocus_max,
                        "stepSize": autofocus_step_size,
                        "illuminationChannel": autofocus_illumination_channel,
                        "max_attempts": autofocus_max_attempts,
                        "target_focus_setpoint": autofocus_target_focus_setpoint,
                        "mode": autofocus_mode,
                        "af_software_method": autofocus_software_method,
                        "af_hc_initial_step": autofocus_hc_initial_step,
                        "af_hc_min_step": autofocus_hc_min_step,
                        "af_hc_step_reduction": autofocus_hc_step_reduction,
                        "af_hc_max_iterations": autofocus_hc_max_iterations,
                    },
                ))
                step_id += 1

            # Iterate over Z planes.
            # z_positions are pure relative offsets; effective_z_positions = base_z + offset.
            # For multi-plane Z-stacks, add a Z-move step for every plane.
            # For single Z (z_positions = [0.0]), add one Z-move to base_z so the
            # stage is always at the correct absolute Z before capture, even after
            # a prior tile's focus-map or autofocus moved it elsewhere.
            is_z_stack = len(effective_z_positions) > 1
            for index_z, i_z in enumerate(effective_z_positions):
                # Move Z for every plane in a Z-stack, or once for single-Z
                if is_z_stack or index_z == 0:
                    workflow_steps.append(WorkflowStep(
                        name=f"Move to Z {'plane ' + str(index_z) if is_z_stack else 'base position'} ({i_z:.1f} µm)",
                        step_id=step_id,
                        main_func=self.controller.move_stage_z,
                        main_params={"posZ": i_z, "relative": False},
                        pre_funcs=[self.controller.wait_time],
                        pre_params={"seconds": t_pre_s},
                    ))
                    step_id += 1

                # Iterate over illumination sources.
                # effective_channel_index walks the OME channel axis: each
                # active normal/ring source consumes one slot, an active DPC
                # source consumes four (one per direction). This must match
                # the n_channels computed in _setup_ome_writers, otherwise
                # DPC sub-frames overwrite each other in the Zarr store.
                effective_channel_index = 0
                _params = illumination_params or {}
                # Build local name→kind map for this loop.  Looking up by name
                # rather than parallel-index protects against any upstream
                # filtering that drops one array but not the other.
                _local_kind_by_name = {
                    n: k for n, k in zip(illumination_sources or [], illumination_kinds or [])
                }
                for illu_index, illu_source in enumerate(illumination_sources):
                    illu_intensity = illumination_intensities[illu_index] if illu_index < len(illumination_intensities) else 0
                    illu_kind = _local_kind_by_name.get(illu_source, "default")
                    if illu_intensity <= 0:
                        continue

                    # Apply per-channel Z offset for chromatic shift compensation.
                    # i_z is already the absolute Z position; add the channel offset on top.
                    if focus_map_z is not None:
                        channel_offset_z = 0.0
                        _fm_cfg = getattr(self.controller, '_focus_map_config', None)
                        if _fm_cfg and _fm_cfg.channel_offsets:
                            channel_offset_z = _fm_cfg.channel_offsets.get(illu_source, 0.0)
                        if channel_offset_z != 0.0:
                            adjusted_z = i_z + channel_offset_z
                            workflow_steps.append(WorkflowStep(
                                name=f"Channel Z offset ({illu_source}: {channel_offset_z:+.1f} µm)",
                                step_id=step_id,
                                main_func=self.controller.move_stage_z,
                                main_params={"posZ": adjusted_z, "relative": False},
                            ))
                            step_id += 1

                    # Per-channel exposure/gain (shared across DPC sub-frames).
                    exposure_time = exposures[illu_index] if illu_index < len(exposures) else exposures[0]
                    gain = gains[illu_index] if illu_index < len(gains) else gains[0]

                    is_single_tiff_mode = getattr(self.controller, '_ome_write_single_tiff', False)
                    writer_index = t if is_single_tiff_mode else (writer_offset + position_center_index)

                    if illu_kind == "default":
                        # ---- Conventional laser/LED channel: legacy behaviour ----
                        # Turn on illumination - use tPre as settle time after activation.
                        # Skip per-frame toggle when keep_illumination_on; light is already on.
                        if not keep_illumination_on:
                            workflow_steps.append(WorkflowStep(
                                name="Turn on illumination",
                                step_id=step_id,
                                main_func=self.controller.set_laser_power,
                                main_params={"power": illu_intensity, "channel": illu_source},
                                post_funcs=[self.controller.wait_time],
                                post_params={"seconds": t_pre_s},
                            ))
                            step_id += 1

                        workflow_steps.append(WorkflowStep(
                            name="Acquire frame",
                            step_id=step_id,
                            main_func=self.controller.acquire_frame,
                            main_params={"channel": "Mono"},
                            post_funcs=[self.controller.save_frame_ome],
                            pre_funcs=[self.controller.set_exposure_time_gain],
                            pre_params={"exposure_time": exposure_time, "gain": gain},
                            post_params={
                                "posX": m_point["x"],
                                "posY": m_point["y"],
                                "posZ": i_z,
                                "iX": m_point["iX"],
                                "iY": m_point["iY"],
                                "pixel_size": m_pixel_size,
                                "minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y,
                                "channel": illu_source,
                                "time_index": t,
                                "tile_index": m_index,
                                "position_center_index": writer_index,
                                "runningNumber": step_id,
                                "illuminationChannel": illu_source,
                                "illuminationValue": illu_intensity,
                                "z_index": index_z,
                                "channel_index": effective_channel_index,
                            },
                        ))
                        step_id += 1
                        effective_channel_index += 1

                        # Turn off illumination only if multiple sources (for switching between them).
                        if not keep_illumination_on:
                            workflow_steps.append(WorkflowStep(
                                name="Turn off illumination",
                                step_id=step_id,
                                main_func=self.controller.set_laser_power,
                                main_params={"power": 0, "channel": illu_source},
                            ))
                            step_id += 1

                    elif illu_kind == "ring":
                        # ---- LED-matrix ring channel: one frame at given radius ----
                        ring_params = _params.get(illu_source, {}) or {}
                        radius = int(ring_params.get("radius", 8))
                        r_int = int(ring_params.get("intensityR", illu_intensity))
                        g_int = int(ring_params.get("intensityG", illu_intensity))
                        b_int = int(ring_params.get("intensityB", illu_intensity))
                        ring_channel_name = ring_params.get("channelName") or "Ring"

                        workflow_steps.append(WorkflowStep(
                            name=f"LED matrix ring r={radius}",
                            step_id=step_id,
                            main_func=self.controller.set_led_matrix_pattern,
                            main_params={
                                "kind": "ring",
                                "radius": radius,
                                "intensity_r": r_int,
                                "intensity_g": g_int,
                                "intensity_b": b_int,
                                "settle_s": t_pre_s,
                            },
                        ))
                        step_id += 1

                        workflow_steps.append(WorkflowStep(
                            name=f"Acquire frame ({ring_channel_name})",
                            step_id=step_id,
                            main_func=self.controller.acquire_frame,
                            main_params={"channel": "Mono"},
                            post_funcs=[self.controller.save_frame_ome],
                            pre_funcs=[self.controller.set_exposure_time_gain],
                            pre_params={"exposure_time": exposure_time, "gain": gain},
                            post_params={
                                "posX": m_point["x"],
                                "posY": m_point["y"],
                                "posZ": i_z,
                                "iX": m_point["iX"],
                                "iY": m_point["iY"],
                                "pixel_size": m_pixel_size,
                                "minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y,
                                "channel": ring_channel_name,
                                "time_index": t,
                                "tile_index": m_index,
                                "position_center_index": writer_index,
                                "runningNumber": step_id,
                                "illuminationChannel": ring_channel_name,
                                "illuminationValue": max(r_int, g_int, b_int),
                                "illuminationPattern": "ring",
                                "illuminationRadius": radius,
                                "illuminationRGB": (r_int, g_int, b_int),
                                "z_index": index_z,
                                "channel_index": effective_channel_index,
                            },
                        ))
                        step_id += 1
                        effective_channel_index += 1

                        # Always turn the LED matrix off after the snap so the
                        # next step starts dark.  No "keep_illumination_on" for
                        # synthetic channels — patterns must switch per frame.
                        workflow_steps.append(WorkflowStep(
                            name="LED matrix off",
                            step_id=step_id,
                            main_func=self.controller.set_led_matrix_pattern,
                            main_params={"kind": "off"},
                        ))
                        step_id += 1

                    elif illu_kind == "dpc":
                        # ---- LED-matrix DPC channel: 4 frames, one per direction ----
                        dpc_params = _params.get(illu_source, {}) or {}
                        # DPC's documented gotcha: lower RGB values beat with
                        # the rolling shutter, so default to full 255 across
                        # whichever colour the user picked.
                        r_int = int(dpc_params.get("intensityR", 0))
                        g_int = int(dpc_params.get("intensityG", illu_intensity))
                        b_int = int(dpc_params.get("intensityB", 0))

                        for direction in DPC_SUB_DIRS:
                            sub_channel_name = f"DPC_{direction}"

                            workflow_steps.append(WorkflowStep(
                                name=f"LED matrix halves: {direction}",
                                step_id=step_id,
                                main_func=self.controller.set_led_matrix_pattern,
                                main_params={
                                    "kind": "halves",
                                    "direction": direction,
                                    "intensity_r": r_int,
                                    "intensity_g": g_int,
                                    "intensity_b": b_int,
                                    "settle_s": t_pre_s,
                                },
                            ))
                            step_id += 1

                            workflow_steps.append(WorkflowStep(
                                name=f"Acquire frame ({sub_channel_name})",
                                step_id=step_id,
                                main_func=self.controller.acquire_frame,
                                main_params={"channel": "Mono"},
                                post_funcs=[self.controller.save_frame_ome],
                                pre_funcs=[self.controller.set_exposure_time_gain],
                                pre_params={"exposure_time": exposure_time, "gain": gain},
                                post_params={
                                    "posX": m_point["x"],
                                    "posY": m_point["y"],
                                    "posZ": i_z,
                                    "iX": m_point["iX"],
                                    "iY": m_point["iY"],
                                    "pixel_size": m_pixel_size,
                                    "minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y,
                                    "channel": sub_channel_name,
                                    "time_index": t,
                                    "tile_index": m_index,
                                    "position_center_index": writer_index,
                                    "runningNumber": step_id,
                                    "illuminationChannel": sub_channel_name,
                                    "illuminationValue": max(r_int, g_int, b_int),
                                    "illuminationPattern": "dpc",
                                    "illuminationDirection": direction,
                                    "illuminationRGB": (r_int, g_int, b_int),
                                    "z_index": index_z,
                                    "channel_index": effective_channel_index,
                                },
                            ))
                            step_id += 1
                            effective_channel_index += 1

                        workflow_steps.append(WorkflowStep(
                            name="LED matrix off",
                            step_id=step_id,
                            main_func=self.controller.set_led_matrix_pattern,
                            main_params={"kind": "off"},
                        ))
                        step_id += 1

                    else:
                        # Unknown kind — log + skip rather than fail the whole run.
                        self._logger.warning(
                            f"Unknown illumination kind '{illu_kind}' for source "
                            f"'{illu_source}'; skipping this channel."
                        )

            # After a Z-stack, return to base_z so the next tile starts from a
            # known Z reference (important for timelapse repeatability).
            if is_z_stack:
                workflow_steps.append(WorkflowStep(
                    name=f"Return to base Z after Z-stack ({base_z:.1f} µm)",
                    step_id=step_id,
                    main_func=self.controller.move_stage_z,
                    main_params={"posZ": base_z, "relative": False},
                    pre_funcs=[self.controller.wait_time],
                    pre_params={"seconds": t_pre_s},
                ))
                step_id += 1



        # Finalize OME writer for this tile (skip in single TIFF mode since we only have one writer)
        # Always finalize tile writers since each timepoint creates its own writers.
        # Use writer_offset to address the correct writer in the flat list.
        is_single_tiff_mode = getattr(self.controller, '_ome_write_single_tiff', False)
        should_finalize_tile = not is_single_tiff_mode

        if should_finalize_tile:
            actual_writer_index = writer_offset + position_center_index
            workflow_steps.append(WorkflowStep(
                name=f"Finalize OME writer for tile {position_center_index} (timepoint {t})",
                step_id=step_id,
                main_func=self.controller.dummy_main_func,
                main_params={},
                post_funcs=[self.controller.finalize_tile_ome_writer],
                post_params={"tile_index": actual_writer_index},
            ))
            step_id += 1

        return step_id

    def _add_finalization_steps(self,
                              workflow_steps: List[WorkflowStep],
                              step_id: int,
                              snake_tiles: List[List[Dict]], # TODO: not needed
                              illumination_sources: List[str],
                              illumination_intensities: List[float],
                              t_period: float,
                              t: int, n_times: int,
                              writer_offset: int = 0,
                              keep_illumination_on: bool = False,
                              illumination_kinds: Optional[List[str]] = None) -> int:
        """
        Add finalization workflow steps.

        Args:
            workflow_steps: List to append workflow steps to
            step_id: Current step ID
            snake_tiles: List of tiles (for context)
            illumination_sources: List of illumination source names
            illumination_intensities: List of illumination values
            t_period: Time period to wait
            t: Current time point index

        Returns:
            Updated step ID
        """
        # Finalize OME writers for this timepoint.
        # Per-tile finalization already handled individual tile writers above.
        # On the last timepoint, do a full cleanup pass (no time_index → finalizes all).
        # On intermediate timepoints, skip this step since tile writers are already finalized.
        is_last_timepoint = (t == n_times - 1)
        if is_last_timepoint:
            workflow_steps.append(WorkflowStep(
                name=f"Finalize all OME writers (final cleanup)",
                step_id=step_id,
                main_func=self.controller.dummy_main_func,
                main_params={},
                post_funcs=[self.controller.finalize_current_ome_writer],
                post_params={}  # No time_index → finalizes all remaining writers
            ))
            step_id += 1

        # Turn off all illuminations.
        # When keep_illumination_on is active, only turn off on the very last
        # timepoint so the light stays on between timepoints for speed.
        should_turn_off = (not keep_illumination_on) or is_last_timepoint
        if should_turn_off:
            # Name-based kind lookup (parallel-array indexing was historically
            # brittle here when passthrough mode collapsed illumination_sources).
            _local_kind_by_name = {
                n: k for n, k in zip(illumination_sources or [], illumination_kinds or [])
            }
            _has_synthetic = False
            for illu_index, illu_source in enumerate(illumination_sources):
                illu_intensity = illumination_intensities[illu_index] if illu_index < len(illumination_intensities) else 0
                illu_kind = _local_kind_by_name.get(illu_source, "default")
                if illu_intensity <= 0:
                    continue

                if illu_kind == "default":
                    workflow_steps.append(WorkflowStep(
                        name="Turn off illumination",
                        step_id=step_id,
                        main_func=self.controller.set_laser_power,
                        main_params={"power": 0, "channel": illu_source},
                    ))
                    step_id += 1
                else:
                    # ring / dpc share the LED matrix; one off-call covers all.
                    _has_synthetic = True

            if _has_synthetic:
                workflow_steps.append(WorkflowStep(
                    name="LED matrix off (finalization)",
                    step_id=step_id,
                    main_func=self.controller.set_led_matrix_pattern,
                    main_params={"kind": "off"},
                ))
                step_id += 1

        # Add timing calculation for proper period control (for all timepoints except implicit last)
        if n_times > 1:
            workflow_steps.append(WorkflowStep(
                name=f"Calculate and wait for proper time period (timepoint {t})",
                step_id=step_id,
                main_func=self.controller.dummy_main_func,
                main_params={},
                pre_funcs=[self.controller.wait_for_next_timepoint],
                pre_params={"timepoint": t, "t_period": t_period}
            ))
            step_id += 1

        # On the very last timepoint, optionally return the stage to the
        # pre-scan XYZ position.  Gated on the "Return to origin" Tiling toggle
        # (default off → the stage stays at the last acquired tile).  When on,
        # restore the full XYZ (including Z) the microscope was at before the
        # scan started.
        if is_last_timepoint and getattr(self.controller, "_return_to_origin", False):
            workflow_steps.append(WorkflowStep(
                name="Return to initial XYZ position",
                step_id=step_id,
                main_func=self.controller.return_to_initial_position,
                main_params={"include_z_position": True},
            ))
            step_id += 1

        return step_id
