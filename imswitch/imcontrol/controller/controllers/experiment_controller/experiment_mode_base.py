"""
Base class for experiment execution modes.

This module provides common functionality shared between performance mode
and normal mode experiment execution.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod

# Import OME writers from the new io location
from imswitch.imcontrol.model.io import (
    OMEWriterConfig,
    OMEFileStorePaths,
    OMEROConnectionParams,
    is_omero_available,
)
from imswitch.imcontrol.model.io.ome_writers import write_plate_metadata_sidecar

# DPC half-illumination sub-frames, in acquisition (= OME channel) order.
DPC_SUB_DIRS = ("top", "bottom", "left", "right")


class ExperimentModeBase(ABC):
    """
    Base class for experiment execution modes.
    
    Provides common functionality for both performance mode and normal mode
    experiment execution, including parameter processing, scan range computation,
    and OME writer configuration.
    """

    def __init__(self, experiment_controller):
        """Initialize the base mode with reference to the main controller."""
        self.controller = experiment_controller
        self._logger = experiment_controller._logger

    def compute_scan_ranges(self, snake_tiles: List[List[Dict]]) -> Tuple[float, float, float, float, float, float]:
        """
        Compute scan ranges from snake tiles.
        
        Args:
            snake_tiles: List of tiles containing point dictionaries
            
        Returns:
            Tuple of (minX, maxX, minY, maxY, diffX, diffY)
        """
        # Flatten all point dictionaries from all tiles to compute scan range
        all_points = [pt for tile in snake_tiles for pt in tile]
        minX = min(pt["x"] for pt in all_points)
        maxX = max(pt["x"] for pt in all_points)
        minY = min(pt["y"] for pt in all_points)
        maxY = max(pt["y"] for pt in all_points)

        # compute step between two adjacent points in X/Y
        uniqueX = np.unique([pt["x"] for pt in all_points])
        uniqueY = np.unique([pt["y"] for pt in all_points])

        if len(uniqueX) == 1:
            diffX = 0
        else:
            diffX = np.diff(uniqueX).min()

        if len(uniqueY) == 1:
            diffY = 0
        else:
            diffY = np.diff(uniqueY).min()

        return minX, maxX, minY, maxY, diffX, diffY

    def create_ome_file_paths(self, base_path: str, shared_individual_tiffs_dir: str = None) -> 'OMEFileStorePaths':
        """
        Create OME file storage paths.
        
        Args:
            base_path: Base path for the writer's files
            shared_individual_tiffs_dir: Optional shared directory for individual TIFFs across all timepoints
            
        Returns:
            OMEFileStorePaths instance
        """
        return OMEFileStorePaths(base_path, shared_individual_tiffs_dir)

    def create_writer_config(self,
                           write_tiff: bool = False,
                           write_zarr: bool = True,
                           write_stitched_tiff: bool = True,
                           write_tiff_single: bool = False,
                           write_individual_tiffs: bool = False,
                           write_omero: bool = False,
                           omero_queue_size: int = 100,
                           min_period: float = 0.2,
                           n_time_points: int = 1,
                           n_z_planes: int = 1,
                           n_channels: int = 1,
                           channel_names: Optional[List[str]] = None) -> OMEWriterConfig:
        """
        Create OME writer configuration.
        
        Args:
            write_tiff: Whether to write individual TIFF files
            write_zarr: Whether to write OME-Zarr format
            write_stitched_tiff: Whether to write stitched TIFF
            write_tiff_single: Whether to append tiles to a single TIFF file
            write_individual_tiffs: Whether to write individual TIFF files with position-based naming
            write_omero: Whether to stream tiles to OMERO server
            omero_queue_size: Max tiles to queue for OMERO upload
            min_period: Minimum period between writes
            n_time_points: Number of time points
            n_z_planes: Number of Z planes
            n_channels: Number of channels
            
        Returns:
            OMEWriterConfig instance
        """
        pixel_size = self.controller.detectorPixelSize[-1] if hasattr(self.controller, 'detectorPixelSize') else 1.0

        return OMEWriterConfig(
            write_tiff=write_tiff,
            write_zarr=write_zarr,
            write_stitched_tiff=write_stitched_tiff,
            write_tiff_single=write_tiff_single,
            write_individual_tiffs=write_individual_tiffs,
            write_omero=write_omero,
            omero_queue_size=omero_queue_size,
            min_period=min_period,
            pixel_size=pixel_size,
            n_time_points=n_time_points,
            n_z_planes=n_z_planes,
            n_channels=n_channels,
            channel_names=channel_names,
        )

    def tile_shape(self) -> Tuple[int, int]:
        """(width, height) of one camera frame."""
        shape = getattr(self.controller.mDetector, "_shape", (512, 512))
        return (shape[-1], shape[-2])

    @staticmethod
    def channel_layout(intensities, sources, kinds) -> Tuple[int, Optional[List[str]]]:
        """Writer channel axis: one slot per active channel, four for a DPC channel.

        Kinds are looked up by source name so a misaligned kinds list cannot
        put a DPC sub-frame into the wrong slot. (1, None) when nothing is
        active, which lets OMEWriter default to "Channel_N".
        """
        sources = list(sources or [])
        kind_by_name = dict(zip(sources, list(kinds or [])))
        names: List[str] = []
        for i, intensity in enumerate(intensities or []):
            if intensity is None or intensity <= 0:
                continue
            name = sources[i] if i < len(sources) else f"Channel_{i}"
            kind = kind_by_name.get(name, "default")
            if kind == "dpc":
                names += [f"DPC_{d}" for d in DPC_SUB_DIRS]
            elif kind == "ring":
                names.append("Ring")
            else:
                names.append(name)
        return (len(names), names) if names else (1, None)

    @staticmethod
    def well_metadata(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Per-well labware metadata of one region, None when the region is not a well."""
        if not (meta.get("wellRow") or meta.get("wellColumn") or meta.get("labwareLoadName")):
            return None
        return {k: meta.get(k) for k in ("wellRow", "wellColumn", "labwareLoadName", "conditionLabel")}

    def write_plate_sidecar(self, snake_tiles, region_meta: Dict[str, Dict[str, Any]], dir_path: str) -> None:
        """Best-effort OME-NGFF plate sidecar listing the wells this run touches."""
        wells_used, condition_labels, labware_load_name = [], {}, None
        for tiles in snake_tiles:
            meta = region_meta.get(tiles[0]["region_id"], {}) if tiles else {}
            row, col, load, cond = (meta.get(k) for k in ("wellRow", "wellColumn", "labwareLoadName", "conditionLabel"))
            if load and labware_load_name is None:
                labware_load_name = load
            if row and col is not None:
                wells_used.append((str(row), str(int(col))))
                if cond:
                    condition_labels[f"{row}{int(col)}"] = cond
        if not labware_load_name or getattr(self.controller, "labware_manager", None) is None:
            return
        try:
            lab = self.controller.labware_manager.get(labware_load_name)
            if lab is None:
                return
            write_plate_metadata_sidecar(
                output_dir=dir_path, plate_name=labware_load_name,
                rows=list(lab.rows), columns=[str(c) for c in lab.columns], wells_used=wells_used,
                extra={"imswitch_labware": {"loadName": labware_load_name,
                                            "conditionLabels": condition_labels or None}},
            )
        except Exception as exc:  # the sidecar is best-effort
            self._logger.warning(f"Failed to write plate metadata sidecar: {exc}")

    def prepare_omero_connection_params(self) -> Optional[OMEROConnectionParams]:
        """
        Prepare OMERO connection parameters from ExperimentManager config.
        
        Returns:
            OMEROConnectionParams if OMERO is enabled and available, None otherwise.
        """
        if not is_omero_available():
            self._logger.debug("OMERO not available (omero-py not installed)")
            return None

        exp_manager = self.controller.experimentManager
        if exp_manager is None:
            self._logger.debug("ExperimentManager not available")
            return None

        if not getattr(exp_manager, 'omeroEnabled', False):
            self._logger.debug("OMERO not enabled in ExperimentManager config")
            return None

        # Get OMERO connection parameters from ExperimentManager
        host = getattr(exp_manager, 'omeroServerUrl', None)
        if not host:
            self._logger.warning("OMERO enabled but no server URL configured")
            return None

        return OMEROConnectionParams(
            host=host,
            port=getattr(exp_manager, 'omeroPort', 4064),
            username=getattr(exp_manager, 'omeroUsername', ''),
            password=getattr(exp_manager, 'omeroPassword', ''),
            group_id=getattr(exp_manager, 'omeroGroupId', None),
            project_id=getattr(exp_manager, 'omeroProjectId', None),
            dataset_id=getattr(exp_manager, 'omeroDatasetId', None),
            connection_timeout=getattr(exp_manager, 'omeroConnectionTimeout', 30),
            upload_timeout=getattr(exp_manager, 'omeroUploadTimeout', 300),
        )

    def prepare_illumination_parameters(self, illumination_intensities: List[float],
                                        illumination_sources: Optional[List[str]] = None) -> List[float]:
        """Intensities per ESP32 laser channel (``illumination[0..4]`` of the stagescan).

        ``illumination_intensities[i]`` belongs to ``illumination_sources[i]``
        (the channel plan the frontend selected). Each source is looked up by
        name among the controller's laser managers and lands on that manager's
        ``channel_index``; without a source list the i-th laser manager is
        assumed, which is only right when the frontend sends every laser in
        config order. Sources that are not a laser (LED-matrix synthetic
        channels) or channels outside 0..4 are skipped with a warning instead
        of shifting the others.
        """
        n_channels = 5  # the ESP32 stagescan exposes illumination[0..4]
        intensity_list = [0] * n_channels
        managers = list(getattr(self.controller, "availableIlluminations", []) or [])
        by_name = {getattr(m, "name", None): m for m in managers}

        for i, intensity in enumerate(illumination_intensities or []):
            if intensity is None or intensity <= 0:
                continue
            source = illumination_sources[i] if illumination_sources and i < len(illumination_sources) else None
            manager = by_name.get(source) if source is not None else (managers[i] if i < len(managers) else None)
            if manager is None:
                self._logger.warning(f"Performance mode: illumination '{source or i}' is not a laser channel; skipped")
                continue
            channel = getattr(manager, "channel_index", None)
            if not isinstance(channel, int):
                channel = managers.index(manager)
            if not 0 <= channel < n_channels:
                self._logger.warning(f"Performance mode: '{source or i}' has channel_index {channel}, "
                                     f"outside the firmware's 0..{n_channels - 1}; skipped")
                continue
            intensity_list[channel] = intensity
        return intensity_list

    def calculate_grid_parameters(self, tiles: List[Dict]) -> Tuple[Tuple[int, int], Tuple[float, float, float, float]]:
        """
        Calculate grid parameters from tile list.
        
        Args:
            tiles: List of point dictionaries
            
        Returns:
            Tuple of (grid_shape, grid_geometry)
        """
        all_points = []
        for point in tiles:
            if point is not None:
                try:all_points.append([point["x"], point["y"]])
                except Exception as e: self._logger.error(f"Error processing point {point}: {e}")

        if all_points:
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            x_start, x_end = min(x_coords), max(x_coords)
            y_start, y_end = min(y_coords), max(y_coords)
            unique_x = sorted(set(x_coords))
            unique_y = sorted(set(y_coords))
            x_step = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 100.0
            y_step = unique_y[1] - unique_y[0] if len(unique_y) > 1 else 100.0
            nx, ny = len(unique_x), len(unique_y)
            grid_shape = (nx, ny)
            grid_geometry = (x_start, y_start, x_step, y_step)
        else:
            grid_shape = (1, 1)
            grid_geometry = (0, 0, 100, 100)

        return grid_shape, grid_geometry

    @abstractmethod
    def execute_experiment(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the experiment. Must be implemented by subclasses.
        
        Args:
            **kwargs: Experiment parameters
            
        Returns:
            Dictionary with execution results
        """
        pass

    def save_experiment_protocol(self, 
                                protocol_data: Dict[str, Any],
                                file_path: str,
                                mode: str = "unknown") -> str:
        """
        Save experiment protocol and parameters to JSON file.
        
        Args:
            protocol_data: Dictionary containing experiment parameters and steps
            file_path: Base path for the experiment data
            mode: Experiment mode ('normal' or 'performance')
            
        Returns:
            Path to the saved protocol JSON file

        Raises:
            Whatever json.dumps raises. Serializing before opening matters:
            open("w") truncates, so serializing inside the with-block left a
            0-byte protocol behind on any failure. The error is not swallowed —
            a protocol we cannot write means the run is not reproducible.
        """
        protocol_file = file_path + "_protocol.json"

        protocol_data["timestamp"] = datetime.now().isoformat()
        protocol_data["mode"] = mode
        protocol_data["imswitch_version"] = getattr(self.controller, 'version', 'unknown')

        payload = json.dumps(protocol_data, indent=2, default=self._json_serializer)

        os.makedirs(os.path.dirname(protocol_file), exist_ok=True)
        with open(protocol_file, 'w') as f:
            f.write(payload)

        self._logger.info(f"Experiment protocol saved (one-time): {protocol_file}")
        return protocol_file
    
    def _json_serializer(self, obj):
        """
        Custom JSON serializer for objects not serializable by default.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        # Handle numpy types
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle callable functions (store name only)
        elif callable(obj):
            return f"<function: {obj.__name__}>"
        
        # Handle objects with __dict__
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        
        # Fallback to string representation
        else:
            return str(obj)
