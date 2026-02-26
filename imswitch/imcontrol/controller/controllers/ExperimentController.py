from datetime import datetime
import time
from fastapi import HTTPException
import numpy as np
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import os
import threading

from imswitch.imcommon.framework import Signal
from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowContext, WorkflowsManager
from imswitch.imcontrol.model.managers.MDASequenceManager import MDASequenceManager
from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from .wellplate_layouts import get_predefined_layouts, get_layout_by_name

try:
    IS_ASHLAR_AVAILABLE = True
except Exception:
    IS_ASHLAR_AVAILABLE = False

# Attempt to use OME-Zarr
try:
    IS_OMEZARR_AVAILABLE = True # TODO: True
except Exception:
    IS_OMEZARR_AVAILABLE = False

# Import OME writers from new io location
from imswitch.imcontrol.model.io import OMEWriter, OMEWriterConfig, OMEFileStorePaths
from imswitch.imcontrol.controller.controllers.experiment_controller import (
    ExperimentPerformanceMode,
    ExperimentNormalMode,
)
from imswitch.imcontrol.model.focus_map import FocusMap, FocusMapManager, FocusMapResult
from imswitch.imcontrol.model.overview_registration import OverviewRegistrationService, PixelPoint, StagePoint, SlotDefinition


from pydantic import Field

# -----------------------------------------------------------
# Reuse the existing sub-models:
# -----------------------------------------------------------


class FocusMapFromPointsRequest(BaseModel):
    """Request body for computeFocusMapFromPoints endpoint."""
    points: List[Dict[str, float]]
    group_id: str = "manual"
    group_name: str = "Manual Points"
    method: str = "rbf"
    smoothing_factor: float = 0.1
    z_offset: float = 0.0
    clamp_enabled: bool = False
    z_min: float = 0.0
    z_max: float = 0.0


class NeighborPoint(BaseModel):
    x: float
    y: float
    iX: int
    iY: int

class Point(BaseModel):
    id: Optional[str] = None  # Allow string IDs from frontend
    name: str
    x: float
    y: float
    iX: int = 0
    iY: int = 0
    neighborPointList: List[NeighborPoint] = Field(default_factory=list)
    wellId: Optional[str] = None  # NEW: Well association
    areaType: Optional[str] = None  # NEW: Area type (well, free_scan, etc.)

# NEW: Models for pre-calculated scan coordinates
class ScanPosition(BaseModel):
    """Single position in a scan area"""
    index: int
    x: float
    y: float
    iX: int
    iY: int

class ScanBounds(BaseModel):
    """Bounding box for a scan area"""
    minX: float
    maxX: float
    minY: float
    maxY: float
    width: float
    height: float

class CenterPosition(BaseModel):
    """Center position of a scan area"""
    x: float
    y: float

class ScanArea(BaseModel):
    """Pre-calculated scan area with ordered positions"""
    areaId: str
    areaName: str
    areaType: str = "free_scan"  # well, free_scan, etc.
    wellId: Optional[str] = None
    centerPosition: CenterPosition
    bounds: ScanBounds
    scanPattern: str = "raster"  # snake or raster (calculated in frontend, positions already ordered)
    positions: List[ScanPosition]

class ScanMetadata(BaseModel):
    """Metadata for the entire scan"""
    totalPositions: int
    fovX: float
    fovY: float
    overlapWidth: float = 0.0  # Used by frontend for coordinate calculation
    overlapHeight: float = 0.0  # Used by frontend for coordinate calculation
    scanPattern: str = "raster"  # Positions are already ordered per this pattern from frontend

class ParameterValue(BaseModel):
    illumination: Union[List[str], str] = None # X, Y, nX, nY
    illuIntensities: Union[List[Optional[int]], Optional[int]] = None
    brightfield: bool = 0,
    darkfield: bool = 0,
    differentialPhaseContrast: bool = 0,
    timeLapsePeriod: float
    numberOfImages: int
    autoFocus: bool
    autoFocusMin: float
    autoFocusMax: float
    autoFocusStepSize: float
    autoFocusIlluminationChannel: str = "" # Selected illumination channel for autofocus
    autoFocusMode: str = "software" # "software" (Z-sweep) or "hardware" (one-shot using FocusLock)
    autofocus_target_focus_setpoint: float = None
    autofocus_max_attempts: int = 2
    zStack: bool
    zStackMin: float
    zStackMax: float
    zStackStepSize: Union[List[float], float] = 1.
    exposureTimes: Union[List[float], float] = None
    gains: Union[List[float], float] = None
    speed: float = 20000.0
    performanceMode: bool = False
    # Performance mode advanced settings
    performanceTriggerMode: str = Field("hardware", description="Trigger mode: 'hardware' (external TTL) or 'software' (callback-based)")
    performanceTPreMs: float = Field(90.0, description="Pre-exposure settle time in milliseconds")
    performanceTPostMs: float = Field(50.0, description="Post-exposure/acquisition time in milliseconds")
    ome_write_tiff: bool = Field(False, description="Whether to write OME-TIFF files")
    ome_write_zarr: bool = Field(True, description="Whether to write OME-Zarr files")
    ome_write_stitched_tiff: bool = Field(False, description="Whether to write stitched OME-TIFF files")
    ome_write_individual_tiffs: bool = Field(False, description="Whether to write individual TIFF files per frame")

class FocusMapConfig(BaseModel):
    """Configuration for optional focus mapping (Z surface estimation over XY)."""
    enabled: bool = Field(False, description="Enable focus mapping before acquisition")

    # Grid generation
    rows: int = Field(3, description="Number of grid rows for focus measurement")
    cols: int = Field(3, description="Number of grid columns for focus measurement")
    add_margin: bool = Field(False, description="Shrink grid inward to avoid edge effects")

    # Fit strategy
    fit_by_region: bool = Field(True, description="Fit per well / scan region (True) or global (False)")
    use_manual_map: bool = Field(False, description="Reuse a pre-existing manual/global map for all groups via interpolation instead of measuring per group")
    method: str = Field("spline", description="Fit method: spline, rbf, or constant")
    smoothing_factor: float = Field(0.1, description="Smoothing factor for surface fit")

    # Runtime behavior
    apply_during_scan: bool = Field(True, description="Move Z per XY using focus map during acquisition")
    z_offset: float = Field(0.0, description="Global Z offset applied to interpolated values")
    clamp_enabled: bool = Field(False, description="Clamp interpolated Z to min/max range")
    z_min: float = Field(0.0, description="Minimum allowed Z value when clamping")
    z_max: float = Field(0.0, description="Maximum allowed Z value when clamping")

    # Autofocus integration
    autofocus_profile: Optional[str] = Field(None, description="Reference to AF controller preset")
    settle_ms: int = Field(0, description="Extra settle time in ms after Z move")
    store_debug_artifacts: bool = Field(True, description="Store focus points + fit stats as JSON")
    channel_offsets: Optional[Dict[str, float]] = Field(default=None, description="Per-illumination-channel Z offset (µm)")

    # Autofocus parameters – passed through to doAutofocusBackground
    af_range: float = Field(100.0, description="Autofocus Z range (±µm from current Z)")
    af_resolution: float = Field(10.0, description="Autofocus step size (µm)")
    af_cropsize: int = Field(2048, description="Crop size for focus quality algorithm")
    af_algorithm: str = Field("LAPE", description="Focus quality algorithm: LAPE, GLVA, JPEG")
    af_settle_time: float = Field(0.1, description="Settle time (s) after each Z step")
    af_static_offset: float = Field(0.0, description="Static Z offset applied after autofocus (µm)")
    af_two_stage: bool = Field(False, description="Use two-stage autofocus (coarse + fine)")
    af_n_gauss: int = Field(0, description="Gaussian kernel size for focus algorithm")
    af_illumination_channel: str = Field("", description="Illumination channel for autofocus")
    af_mode: str = Field("software", description="Autofocus mode: software (Z-sweep) or hardware (FocusLock)")
    af_max_attempts: int = Field(2, description="Max retry attempts for hardware autofocus")
    af_target_setpoint: Optional[float] = Field(None, description="Target focus setpoint for hardware AF")

    # Scan areas – passed from the frontend so that computeFocusMap knows the
    # correct XY bounds even when no experiment has been started yet.
    scan_areas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of scan area dicts with areaId, areaName, bounds (minX/maxX/minY/maxY)"
    )


class Experiment(BaseModel):
    # From your old "Experiment" BaseModel:
    name: str
    parameterValue: ParameterValue
    pointList: List[Point] = Field(default_factory=list)

    # NEW: Pre-calculated scan data from frontend
    scanAreas: Optional[List[ScanArea]] = None
    scanMetadata: Optional[ScanMetadata] = None

    # Focus mapping configuration (disabled by default)
    focusMap: Optional[FocusMapConfig] = Field(default=None, description="Optional focus mapping configuration")

    # From your old "ExperimentModel":
    timepoints: int = Field(1, description="Number of timepoints for time-lapse")

    # -----------------------------------------------------------
    # A helper to produce the "configuration" dict
    # -----------------------------------------------------------
    def to_configuration(self) -> dict:
        """
        Convert this Experiment into a dict structure that your Zarr writer or
        scanning logic can easily consume.
        """
        config = {
            "experiment": {
                "MicroscopeState": {
                    "timepoints": self.timepoints,
                },
                # TODO: Complete it again
            },
        }
        return config


# MDA-related models for useq-schema integration
class MDAChannelConfig(BaseModel):
    """Configuration for an MDA channel."""
    name: str = Field(..., description="Channel name/identifier")
    exposure: Optional[float] = Field(100.0, description="Exposure time in milliseconds")
    power: Optional[float] = Field(100.0, description="Laser/illumination power")

class MDASequenceRequest(BaseModel):
    """Request to start an MDA experiment using useq-schema."""
    channels: List[MDAChannelConfig] = Field(..., description="List of channel configurations")
    z_range: Optional[float] = Field(None, description="Total Z range to scan (µm)")
    z_step: Optional[float] = Field(None, description="Z step size (µm)")
    time_points: int = Field(1, description="Number of time points")
    time_interval: float = Field(1.0, description="Interval between time points (seconds)")
    save_directory: Optional[str] = Field(None, description="Directory to save data")
    experiment_name: str = Field("MDA_Experiment", description="Name of the experiment")

class MDASequenceInfo(BaseModel):
    """Information about an MDA sequence."""
    total_events: int
    channels: List[str]
    z_positions: List[float]
    time_points: List[int]
    axis_order: tuple
    estimated_duration_minutes: float

class ExperimentWorkflowParams(BaseModel):
    """Parameters for the experiment workflow."""

    # Illumination parameters
    illuSources: List[str] = Field(default_factory=list, description="List of illumination sources")
    illuSourceMinIntensities: List[float] = Field(default_factory=list, description="Minimum intensities for each source")
    illuSourceMaxIntensities: List[float] = Field(default_factory=list, description="Maximum intensities for each source")
    illuIntensities: List[float] = Field(default_factory=list, description="Intensities for each source")

    # Camera parameters
    exposureTimes: List[float] = Field(default_factory=list, description="Exposure times for each source")
    gains: List[float] = Field(default_factory=list, description="gains settings for each source")

    # Feature toggles
    isDPCpossible: bool = Field(False, description="Whether DPC is possible")
    isDarkfieldpossible: bool = Field(False, description="Whether darkfield is possible")

    # timelapse parameters
    timeLapsePeriodMin: float = Field(0, description="Minimum time for a timelapse series")
    timeLapsePeriodMax: float = Field(100000000, description="Maximum time for a timelapse series in seconds")
    numberOfImagesMin: int = Field(0, description="Minimum time for a timelapse series")
    numberOfImagesMax: int = Field(0, description="Minimum time for a timelapse series")
    autofocusMinFocusPosition: float = Field(-10000, description="Minimum autofocus position")
    autofocusMaxFocusPosition: float = Field(10000, description="Maximum autofocus position")
    autofocusStepSizeMin: float = Field(1, description="Minimum autofocus position")
    autofocusStepSizeMax: float = Field(1000, description="Maximum autofocus position")
    zStackMinFocusPosition: float = Field(0, description="Minimum Z-stack position")
    zStackMaxFocusPosition: float = Field(10000, description="Maximum Z-stack position")
    zStackStepSizeMin: float = Field(1, description="Minimum Z-stack position")
    zStackStepSizeMax: float = Field(1000, description="Maximum Z-stack position")
    performanceMode: bool = Field(False, description="Whether to use performance mode for the experiment - this would be executing the scan on the Cpp hardware directly, not on the Python side.")



class ExperimentController(ImConWidgetController):
    """Linked to ExperimentWidget."""

    sigExperimentWorkflowUpdate = Signal()
    sigExperimentImageUpdate = Signal(str, np.ndarray, bool, list, bool)  # (detectorName, image, init, scale, isCurrentDetector)
    sigUpdateOMEZarrStore = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # initialize variables
        self.tWait = 0.1
        self.workflow_manager = WorkflowsManager()
        self.mda_manager = MDASequenceManager()

        # set default values
        self.SPEED_Y = self.SPEED_Y_default = 20000
        self.SPEED_X = self.SPEED_X_default = 20000
        self.SPEED_Z = self.SPEED_Z_default = 10000
        self.ACCELERATION = 1000000

        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.mDetector = self._master.detectorsManager[allDetectorNames[0]]
        self.isRGB = self.mDetector._camera.isRGB
        self.detectorPixelSize = self.mDetector.pixelSizeUm

        # select lasers
        self.allIlluNames = self._master.lasersManager.getAllDeviceNames()
        self.availableIlluminations = []
        for iDevice in self.allIlluNames:
            # laser maanger
            self.availableIlluminations.append(self._master.lasersManager[iDevice])

        # select stage
        self.allPositionerNames = self._master.positionersManager.getAllDeviceNames()[0]
        try:
            self.mStage = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]
        except:
            self.mStage = None

        # stop if some external signal (e.g. memory full is triggered)
        self._commChannel.sigExperimentStop.connect(self.stopExperiment)

        # TODO: Adjust parameters
        # define changeable Experiment parameters as ExperimentWorkflowParams
        self.ExperimentParams = ExperimentWorkflowParams()
        self.ExperimentParams.illuSources = self.allIlluNames
        self.ExperimentParams.illuSourceMinIntensities = []
        self.ExperimentParams.illuSourceMaxIntensities = []
        self.ExperimentParams.illuIntensities = [0]*len(self.allIlluNames)
        self.ExperimentParams.exposureTimes = [0]*len(self.allIlluNames)
        self.ExperimentParams.gains = [0]*len(self.allIlluNames)
        self.ExperimentParams.isDPCpossible = False
        self.ExperimentParams.isDarkfieldpossible = False
        self.ExperimentParams.performanceMode = False
        for laserN in self.availableIlluminations:
            self.ExperimentParams.illuSourceMinIntensities.append(laserN.valueRangeMin)
            self.ExperimentParams.illuSourceMaxIntensities.append(laserN.valueRangeMax)
        '''
        For Fast Scanning - Performance Mode -> Parameters will be sent to the hardware directly
        requires hardware triggering
        '''
        # where to dump the TIFFs ----------------------------------------------
        save_dir = dirtools.UserFileDirs.getValidatedDataPath()
        self.save_dir  = os.path.join(save_dir, "ExperimentController")
        # ensure all subfolders are generated:
        os.makedirs(self.save_dir) if not os.path.exists(self.save_dir) else None

        # writer thread control -------------------------------------------------
        self._writer_thread   = None
        self._writer_thread_ome = None
        self._current_ome_writer = None  # For normal mode OME writing
        self._stop_writer_evt = threading.Event()

        # fast stage scanning parameters ----------------------------------------
        self.fastStageScanIsRunning = False

        # OME writer configuration -----------------------------------------------
        self._ome_write_tiff = False
        self._ome_write_zarr = True
        self._ome_write_stitched_tiff = False
        self._ome_write_individual_tiffs = False
        self._ome_write_single_tiff = False

        # Initialize experiment execution modes
        self.performance_mode = ExperimentPerformanceMode(self)
        self.normal_mode = ExperimentNormalMode(self)

        # Initialize focus map manager
        self.focus_map_manager = FocusMapManager(logger=self._logger)

        # Initialize overview camera registration service
        self._overview_registration = OverviewRegistrationService()
        self._overview_camera = None
        self._overview_camera_name = None
        try:
            if hasattr(self._setupInfo, 'PixelCalibration') and hasattr(self._setupInfo.PixelCalibration, 'ObservationCamera'):
                obs_cam_name = self._setupInfo.PixelCalibration.ObservationCamera
                if obs_cam_name and obs_cam_name in allDetectorNames:
                    self._overview_camera = self._master.detectorsManager[obs_cam_name]
                    self._overview_camera_name = obs_cam_name
                    self._logger.info(f"Overview camera initialized: {obs_cam_name}")
        except Exception as e:
            self._logger.warning(f"Could not initialize overview camera: {e}")

        # Initialize omero  parameters  # TODO: Maybe not needed!
        self.omero_url = self._master.experimentManager.omeroServerUrl
        self.omero_username = self._master.experimentManager.omeroUsername
        self.omero_password = self._master.experimentManager.omeroPassword
        self.omero_port = self._master.experimentManager.omeroPort

    @APIExport(requestType="GET")
    def getHardwareParameters(self):
        return self.ExperimentParams

    @APIExport(requestType="GET")
    def getAvailableWellplateLayouts(self):
        """
        Get list of available pre-defined wellplate layouts.

        Returns:
            Dict with layout names as keys and layout metadata as values
        """
        try:
            layouts = get_predefined_layouts()
            return {
                name: {
                    "name": layout.name,
                    "description": layout.description,
                    "rows": layout.rows,
                    "cols": layout.cols,
                    "well_count": len(layout.wells),
                    "well_spacing_x": layout.well_spacing_x,
                    "well_spacing_y": layout.well_spacing_y
                }
                for name, layout in layouts.items()
            }
        except Exception as e:
            self._logger.error(f"Failed to get wellplate layouts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @APIExport(requestType="GET")
    def getWellplateLayout(self, layout_name: str, offset_x: float = 0, offset_y: float = 0):
        """
        Get a specific wellplate layout with optional offset parameters.

        Args:
            layout_name: Name of the layout (e.g., '96-well-standard', '384-well-standard')
            offset_x: X offset in micrometers (default: 0)
            offset_y: Y offset in micrometers (default: 0)

        Returns:
            Complete wellplate layout definition including all wells
        """
        try:
            layout = get_layout_by_name(layout_name, offset_x=offset_x, offset_y=offset_y)
            if not layout:
                raise HTTPException(status_code=404, detail=f"Layout '{layout_name}' not found")
            return layout.dict()
        except HTTPException:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get wellplate layout '{layout_name}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @APIExport(requestType="POST")
    def generateCustomWellplateLayout(self, layout_params: dict):
        """
        Generate a custom wellplate layout with specified parameters.

        Args:
            layout_params: Dictionary with layout parameters:
                - name: str (required)
                - rows: int (required)
                - cols: int (required)
                - well_spacing_x: float (required, micrometers)
                - well_spacing_y: float (required, micrometers)
                - well_shape: str ('circle' or 'rectangle', default: 'circle')
                - well_radius: float (micrometers, for circular wells)
                - well_width: float (micrometers, for rectangular wells)
                - well_height: float (micrometers, for rectangular wells)
                - offset_x: float (default: 0)
                - offset_y: float (default: 0)
                - description: str (default: '')

        Returns:
            Complete wellplate layout definition
        """
        try:
            layout = get_layout_by_name("custom", **layout_params)
            if not layout:
                raise HTTPException(status_code=400, detail="Invalid layout parameters")
            return layout.dict()
        except HTTPException:
            raise
        except Exception as e:
            self._logger.error(f"Failed to generate custom wellplate layout: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @APIExport(requestType="GET")
    def getOMEROConfig(self):
        """Get current OMERO configuration from the experiment manager."""
        try:
            if hasattr(self._master, 'experimentManager'):
                return self._master.experimentManager.getOmeroConfig()
            else:
                return {"error": "ExperimentManager not available"}
        except Exception as e:
            self._logger.error(f"Failed to get OMERO config: {e}")
            return {"error": str(e)}

    @APIExport(requestType="POST")
    def setOMEROConfig(self, config: dict):
        """Set OMERO configuration via the experiment manager."""
        try:
            if hasattr(self._master, 'experimentManager'):
                self._master.experimentManager.setOmeroConfig(config)
                return {"success": True, "message": "OMERO configuration updated"}
            else:
                return {"error": "ExperimentManager not available"}
        except Exception as e:
            self._logger.error(f"Failed to set OMERO config: {e}")
            return {"error": str(e)}

    @APIExport(requestType="GET")
    def isOMEROEnabled(self):
        """Check if OMERO integration is enabled."""
        try:
            if hasattr(self._master, 'experimentManager'):
                return {"enabled": self._master.experimentManager.isOmeroEnabled()}
            else:
                return {"enabled": False, "error": "ExperimentManager not available"}
        except Exception as e:
            self._logger.error(f"Failed to check OMERO status: {e}")
            return {"enabled": False, "error": str(e)}

    @APIExport(requestType="GET")
    def getOMEROConnectionParams(self):
        """Get OMERO connection parameters (excluding password for security)."""
        try:
            if hasattr(self._master, 'experimentManager'):
                params = self._master.experimentManager.getOmeroConnectionParams()
                if params:
                    # Remove password for security when returning via API
                    safe_params = params.copy()
                    safe_params["password"] = "***"
                    return safe_params
                else:
                    return {"error": "OMERO not enabled"}
            else:
                return {"error": "ExperimentManager not available"}
        except Exception as e:
            self._logger.error(f"Failed to get OMERO connection params: {e}")
            return {"error": str(e)}

    @APIExport(requestType="GET")
    def getOMEWriterConfig(self):
        """Get current OME writer configuration."""
        return {
            "write_tiff": getattr(self, '_ome_write_tiff', False),
            "write_zarr": getattr(self, '_ome_write_zarr', True),
            "write_stitched_tiff": getattr(self, '_ome_write_stitched_tiff', False),
            "write_single_tiff": getattr(self, '_ome_write_single_tiff', False),
            "write_individual_tiffs": getattr(self, '_ome_write_individual_tiffs', False)
        }


    def set_led_status(self, status: str = "idle"):
        """
        Set LED matrix status if available.

        Args:
            status: Status string - "idle", "rainbow" (busy), "error", etc.
        """
        try:
            # Check if LED matrix manager is available
            if hasattr(self._master, 'LEDMatrixsManager'):
                led_names = self._master.LEDMatrixsManager.getAllDeviceNames()
                if led_names and len(led_names) > 0:
                    # Set status on first LED matrix
                    led_matrix = self._master.LEDMatrixsManager[led_names[0]]
                    led_matrix.setStatus(status=status)
                    self._logger.debug(f"LED status set to: {status}")
        except Exception as e:
            self._logger.debug(f"Could not set LED status: {e}")

    def get_num_xy_steps(self, pointList):
        # we don't consider the center point as this .. well in the center
        if len(pointList) == 0:
            return 1,1
        all_iX = []
        all_iY = []
        for point in pointList:
            all_iX.append(point.iX)
            all_iY.append(point.iY)
        min_iX, max_iX = min(all_iX), max(all_iX)
        min_iY, max_iY = min(all_iY), max(all_iY)

        num_x_steps = (max_iX - min_iX) + 1
        num_y_steps = (max_iY - min_iY) + 1

        return num_x_steps, num_y_steps

    def generate_snake_tiles(self, mExperiment):
        """
        Generate tiles from experiment with pre-calculated coordinates.

        The frontend now calculates ALL coordinates including scan order.
        This method simply converts the scanAreas format to the internal tiles format.

        Args:
            mExperiment: Experiment object containing scanAreas with pre-calculated positions

        Returns:
            List of tiles, where each tile is a list of coordinate dictionaries
        """
        tiles = []

        # New workflow: Use pre-calculated coordinates from scanAreas
        if mExperiment.scanAreas:
            self._logger.info("Using pre-calculated coordinates from frontend scanAreas")

            # Convert scanAreas to tiles format
            for area in mExperiment.scanAreas:
                # Extract positions from scan area - already ordered by frontend
                tile_positions = []
                for pos in area.positions:
                    tile_positions.append({
                        "iterator": pos.index,
                        "centerIndex": area.areaId,
                        "iX": pos.iX,
                        "iY": pos.iY,
                        "x": pos.x,
                        "y": pos.y,
                        "wellId": area.wellId,
                        "areaName": area.areaName,
                        "areaType": area.areaType
                    })

                if tile_positions:
                    tiles.append(tile_positions)

            self._logger.info(f"Loaded {len(tiles)} scan areas with {sum(len(t) for t in tiles)} total positions")
            return tiles

        # Fallback: Use pointList with pre-ordered neighborPointList
        elif mExperiment.pointList:
            self._logger.info("Using coordinates from pointList")

            for iCenter, centerPoint in enumerate(mExperiment.pointList):
                if not centerPoint.neighborPointList:
                    # Single point - no neighbors
                    tile_positions = [{
                        "iterator": 0,
                        "centerIndex": iCenter,
                        "iX": 0,
                        "iY": 0,
                        "x": centerPoint.x,
                        "y": centerPoint.y,
                        "wellId": centerPoint.wellId,
                        "areaName": centerPoint.name,
                        "areaType": centerPoint.areaType or 'free_scan'
                    }]
                else:
                    # Use pre-ordered neighbor list (no sorting!)
                    tile_positions = []
                    for idx, neighbor in enumerate(centerPoint.neighborPointList):
                        tile_positions.append({
                            "iterator": idx,
                            "centerIndex": iCenter,
                            "iX": neighbor.iX,
                            "iY": neighbor.iY,
                            "x": neighbor.x,
                            "y": neighbor.y,
                            "wellId": centerPoint.wellId,
                            "areaName": centerPoint.name,
                            "areaType": centerPoint.areaType or 'free_scan'
                        })

                tiles.append(tile_positions)

            self._logger.info(f"Loaded {len(tiles)} tiles from pointList")
            return tiles

        # No coordinates provided - create single point at current position
        else:
            self._logger.warning("No scan coordinates provided. Using current stage position.")

            # Get current stage position
            current_position = self.mStage.getPosition()
            current_x = current_position.get("X", 0)
            current_y = current_position.get("Y", 0)

            fallback_tile = [{
                "iterator": 0,
                "centerIndex": 0,
                "iX": 0,
                "iY": 0,
                "x": current_x,
                "y": current_y,
                "wellId": None,
                "areaName": "Current Position",
                "areaType": "free_scan"
            }]
            tiles.append(fallback_tile)
            return tiles

    @APIExport()
    def getLastScanAsOMEZARR(self):
        """ Returns the last OME-Zarr folder as a zipped file for download. """
        try:
            return self.getOmeZarrUrl()
        except Exception as e:
            self._logger.error(f"Error while getting last scan as OME-Zarr: {e}")
            raise HTTPException(status_code=500, detail="Error while getting last scan as OME-Zarr.")

    @APIExport(requestType="GET")
    def getExperimentStatus(self):
        """Get the current status of running experiments."""
        # Check workflow manager status (normal mode)
        if self.ExperimentParams.performanceMode:
            # Check performance mode status
            workflow_status = self.performance_mode.get_scan_status()
        else:
            # Check normal mode status
            workflow_status = self.workflow_manager.get_status()
        #
        return workflow_status

    @APIExport(requestType="POST")
    def startWellplateExperiment(self, mExperiment: Experiment):
        # Extract key parameters
        exp_name = mExperiment.name
        p = mExperiment.parameterValue

        # Timelapse-related
        nTimes = p.numberOfImages
        tPeriod = p.timeLapsePeriod

        # Z-steps -related
        isZStack = p.zStack
        zStackMin = p.zStackMin
        zStackMax = p.zStackMax
        zStackStepSize = p.zStackStepSize

        # Illumination-related
        illuSources = p.illumination
        illuminationIntensities = p.illuIntensities
        if type(illuminationIntensities) is not List  and type(illuminationIntensities) is not list: illuminationIntensities = [p.illuIntensities]
        if type(illuSources) is not List  and type(illuSources) is not list: illuSources = [p.illumination]
        isDarkfield = p.darkfield # TODO: Needs to be implemented
        isBrightfield = p.brightfield
        isDPC = p.differentialPhaseContrast

        # check if any of the illumination sources is turned on, if not, return error
        if not any(illuminationIntensities):
            return HTTPException(status_code=400, detail="No illumination sources are turned on. Please set at least one illumination source intensity.")

        # check if we want to use performance mode
        self.ExperimentParams.performanceMode = p.performanceMode
        performanceMode = p.performanceMode

        # camera-related
        gains = p.gains
        exposures = p.exposureTimes
        if p.speed <= 0:
            self.SPEED_X = self.SPEED_X_default
            self.SPEED_Y = self.SPEED_Y_default
            self.SPEED_Z = self.SPEED_Z_default
        else:
            self.SPEED_X = p.speed
            self.SPEED_Y = p.speed
            self.SPEED_Z = p.speed

        # Autofocus Related
        isAutoFocus = p.autoFocus
        autofocusMax = p.autoFocusMax
        autofocusMin = p.autoFocusMin
        autofocusStepSize = p.autoFocusStepSize
        autofocusIlluminationChannel = getattr(p, 'autoFocusIlluminationChannel', "") or ""
        autofocusMode = getattr(p, 'autoFocusMode', 'software')  # Default to software if not specified
        autofocus_target_focus_setpoint = getattr(p, 'autofocus_target_focus_setpoint', None)
        autofocus_max_attempts = getattr(p, 'autofocus_max_attempts', 2)

        # pre-check gains/exposures  if they are lists and have same lengths as illuminationsources
        if type(gains) is not List and type(gains) is not list: gains = [gains]
        if type(exposures) is not List and type(exposures) is not list: exposures = [exposures]
        if len(gains) != len(illuSources): gains = [-1]*len(illuSources)
        if len(exposures) != len(illuSources): exposures = [exposures[0]]*len(illuSources)


        # Check if another workflow is running
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        # Set LED status to rainbow (busy)
        self.set_led_status("rainbow")

        # Start the detector if not already running
        if not self.mDetector._running:
            self.mDetector.startAcquisition()

        # Store scan areas for focus map API access
        self._last_scan_areas = None
        self._focus_map_fit_by_region = True
        self._focus_map_settle_ms = 0
        if mExperiment.scanAreas:
            self._last_scan_areas = [
                {
                    "areaId": sa.areaId,
                    "areaName": sa.areaName,
                    "bounds": sa.bounds.dict() if hasattr(sa.bounds, 'dict') else {
                        "minX": sa.bounds.minX, "maxX": sa.bounds.maxX,
                        "minY": sa.bounds.minY, "maxY": sa.bounds.maxY,
                    },
                }
                for sa in mExperiment.scanAreas
            ]

        # ── Focus Mapping (optional) ──────────────────────────────────────
        focusMapConfig = mExperiment.focusMap
        if focusMapConfig is not None and focusMapConfig.enabled:
            self._logger.info("Focus mapping enabled – computing Z surface per scan group")
            self._focus_map_fit_by_region = focusMapConfig.fit_by_region
            self._focus_map_settle_ms = focusMapConfig.settle_ms
            self._run_focus_map_phase(mExperiment, focusMapConfig)

        # Generate the list of points to scan from pre-calculated coordinates
        snake_tiles = self.generate_snake_tiles(mExperiment) # TODO: Is this still needed?
        # remove none values from all_points list
        snake_tiles = [[pt for pt in tile if pt is not None] for tile in snake_tiles]

        # Generate Z-positions
        currentZ = self.mStage.getPosition()["Z"]
        if isZStack:
            z_positions = np.arange(zStackMin, zStackMax + zStackStepSize, zStackStepSize) + currentZ
        else:
            z_positions = [currentZ]  # Get current Z position

        # Prepare directory and filename for saving
        timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        drivePath = dirtools.UserFileDirs.getValidatedDataPath()
        dirPath = os.path.join(drivePath, 'ExperimentController', timeStamp)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        mFileName = f"{timeStamp}_{exp_name}"

        workflowSteps = []
        file_writers = []  # Initialize outside the loop for context storage

        # OME writer-related
        self._ome_write_tiff = p.ome_write_tiff
        self._ome_write_zarr = p.ome_write_zarr
        self._ome_write_stitched_tiff = p.ome_write_stitched_tiff
        self._ome_write_single_tiff = getattr(p, 'ome_write_single_tiff', False)  # Default to False if not specified
        self._ome_write_individual_tiffs = getattr(p, 'ome_write_individual_tiffs', False)  # Default to False if not specified

        # determine if each sub scan in snake_tiles is a single tile or a multi-tile scan - if single image we should squah them in a single TIF (e.g. by appending )
        is_single_tile_scan = all(len(tile) == 1 for tile in snake_tiles)
        if is_single_tile_scan:
            self._ome_write_stitched_tiff = False  # Disable stitched TIFF for single tile scans
            self._ome_write_single_tiff = True   # Enable single TIFF writing
        else:
            self._ome_write_single_tiff = False


        # Decide which execution mode to use
        if performanceMode and self.performance_mode.is_hardware_capable():
            # Execute in performance mode
            experiment_params = {
                'mExperiment': mExperiment,
                'tPeriod': tPeriod,
                'nTimes': nTimes
            }
            result = self.performance_mode.execute_experiment(
                snake_tiles=snake_tiles,
                illumination_intensities=illuminationIntensities,
                experiment_params=experiment_params
            )
            return {"status": "running", "mode": "performance"}
        else:
            # Execute in normal mode using workflow
            all_workflow_steps = []
            all_file_writers = []

            # Get the initial Z position at the start of each timepoint
            initial_z_position = self.mStage.getPosition()["Z"]

            for t in range(nTimes):
                experiment_params = {
                    'mExperiment': mExperiment,
                    'tPeriod': tPeriod,
                    'nTimes': nTimes
                }

                result = self.normal_mode.execute_experiment(
                    snake_tiles=snake_tiles,
                    illumination_intensities=illuminationIntensities,
                    illumination_sources=illuSources,
                    z_positions=z_positions,
                    initial_z_position=initial_z_position,
                    exposures=exposures,
                    gains=gains,
                    exp_name=exp_name,
                    dir_path=dirPath,
                    m_file_name=mFileName,
                    t=t,
                    n_times=nTimes,  # Pass total number of time points
                    is_auto_focus=isAutoFocus,
                    autofocus_min=autofocusMin,
                    autofocus_max=autofocusMax,
                    autofocus_step_size=autofocusStepSize,
                    autofocus_illumination_channel=autofocusIlluminationChannel,
                    autofocus_mode=autofocusMode,  # Pass autofocus mode
                    autofocus_target_focus_setpoint=autofocus_target_focus_setpoint,
                    autofocus_max_attempts=autofocus_max_attempts,
                    t_period=tPeriod,
                    isRGB=self.mDetector._isRGB,
                    t_pre_s=p.performanceTPreMs / 1000.0,  # Convert ms to seconds
                    t_post_s=p.performanceTPostMs / 1000.0,  # Convert ms to seconds
                )

                # Append workflow steps and file writers to the accumulated lists
                all_workflow_steps.extend(result["workflow_steps"])
                all_file_writers.extend(result["file_writers"])

            # Use the accumulated workflow steps and file writers
            workflowSteps = all_workflow_steps
            file_writers = all_file_writers
            # Create workflow progress handler
            def sendProgress(payload):
                self.sigExperimentWorkflowUpdate.emit(payload)

            # Create workflow and context
            from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext
            wf = Workflow(workflowSteps, self.workflow_manager)
            context = WorkflowContext()

            # Set metadata
            context.set_metadata("experimentName", exp_name)
            context.set_metadata("nTimes", nTimes)
            context.set_metadata("tPeriod", tPeriod)
            # Add timing information for proper period calculation
            import time
            context.set_metadata("experiment_start_time", time.time())
            context.set_metadata("timepoint_times", {})  # Track timing for each timepoint

            # Store file_writers in context
            if len(file_writers) > 0:
                context.set_object("file_writers", file_writers)
            context.on("progress", sendProgress)
            context.on("rgb_stack", sendProgress)

            # Start the workflow
            self.workflow_manager.start_workflow(wf, context)

        return {"status": "running"}

    def computeScanRanges(self, snake_tiles):
        """Compute scan ranges - delegated to base class method."""
        return self.performance_mode.compute_scan_ranges(snake_tiles)




    ########################################
    # Hardware-related functions
    ########################################
    def acquire_frame(self, channel: str, frameSync: int = 3):
        self._logger.debug(f"Acquiring frame on channel {channel}")

        # ensure we get a fresh frame (frameSync=3 to account for exposure/gain register latency)
        timeoutFrameRequest = 1 # seconds # TODO: Make dependent on exposure time
        cTime = time.time()

        lastFrameNumber=-1
        while(1):
            # get frame and frame number to get one that is newer than the one with illumination off eventually
            mFrame, currentFrameNumber = self.mDetector.getLatestFrame(returnFrameNumber=True)
            if lastFrameNumber==-1:
                # first round
                lastFrameNumber = currentFrameNumber
            if time.time()-cTime> timeoutFrameRequest:
                # in case exposure time is too long we need break at one point
                if mFrame is None:
                    mFrame = self.mDetector.getLatestFrame(returnFrameNumber=False)
                break
            if currentFrameNumber <= lastFrameNumber+frameSync:
                time.sleep(0.01) # off-load CPU
            else:
                break
        return mFrame

    def set_exposure_time_gain(self, exposure_time: float, gain: float, context: WorkflowContext, metadata: Dict[str, Any]):
        if gain and gain >=0:
            self._commChannel.sharedAttrs.sigAttributeSet(['Detector', None, None, "gain"], gain)  # [category, detectorname, ROI1, ROI2] attribute, value
            self._logger.debug(f"Setting gain to {gain}")
        if exposure_time and exposure_time >0:
            self._commChannel.sharedAttrs.sigAttributeSet(['Detector', None, None, "exposureTime"],exposure_time) # category, detectorname, attribute, value
            self._logger.debug(f"Setting exposure time to {exposure_time}")

    def dummy_main_func(self):
        self._logger.debug("Dummy main function called")
        return True

    def autofocus_hardware(self, target_focus_setpoint: Optional[float] = None, max_attempts=2, illuminationChannel: str = "") -> Optional[float]:
        """Perform hardware-based one-shot autofocus using FocusLockController.

        This is significantly faster than software autofocus because it:
        - Captures only ONE frame from dedicated autofocus camera
        - Uses pre-calibrated linear relationship (focus metric → Z position)
        - No Z-sweep required

        Similar to Seafront laser autofocus approach.

        Args:
            illuminationChannel: Selected illumination channel for autofocus (currently unused)

        Returns:
            float: Best focus Z position in µm, or None if autofocus failed
        """
        self._logger.debug("Performing hardware-based one-shot autofocus...")

        # Get the focus lock controller
        try:
            focusLockController = self._master.getController('FocusLock')
        except Exception as e:
            self._logger.warning(f"FocusLockController not available: {e}")
            return None

        if focusLockController is None:
            self._logger.warning("FocusLockController not available - skipping hardware autofocus")
            return None

        # Check if calibration exists
        try:
            calib_status = focusLockController.getCalibrationStatus()
            if not calib_status.get('calibrated', False):
                self._logger.error("Hardware autofocus requires calibration. Please run focus calibration first.")
                return None
        except Exception as e:
            self._logger.error(f"Failed to check calibration status: {e}")
            return None

        # Perform one-shot autofocus
        try:
            result = focusLockController.performOneStepAutofocus(
                target_focus_setpoint=target_focus_setpoint,
                move_to_focus=True,
                max_attempts=max_attempts,
                threshold_um=0.5,
                in_background=False
            )
            if result.get('success', False):
                target_z = result.get('z_offset')
                self._logger.info(
                    f"Hardware autofocus successful: "
                    f"Z={target_z:.2f}µm, error={result.get('final_error_um', 0):.3f}µm, "
                    f"attempts={result.get('num_attempts', 0)}"
                )
                return target_z
            else:
                error_msg = result.get('error', 'Unknown error')
                self._logger.error(f"Hardware autofocus failed: {error_msg}")
                return None

        except Exception as e:
            self._logger.error(f"Hardware autofocus exception: {e}")
            return None

    def autofocus(self, minZ: float=0, maxZ: float=0, stepSize: float=0,
                  illuminationChannel: str="", mode: str="software",
                  max_attempts: int=2,
                  target_focus_setpoint: Optional[float] = None,
                  af_range: float = 100.0,
                  af_resolution: float = 10.0,
                  af_cropsize: int = 2048,
                  af_algorithm: str = "LAPE",
                  af_settle_time: float = 0.1,
                  af_static_offset: float = 0.0,
                  af_two_stage: bool = False,
                  af_n_gauss: int = 0) -> Optional[float]:
        """Perform autofocus using either hardware or software method.

        Args:
            minZ: Legacy minimum Z position (overridden by af_range)
            maxZ: Legacy maximum Z position (overridden by af_range)
            stepSize: Legacy step size (overridden by af_resolution)
            illuminationChannel: Selected illumination channel for autofocus
            mode: "hardware" (fast, one-shot) or "software" (slow, Z-sweep)
            max_attempts: Max retry attempts for hardware AF
            target_focus_setpoint: Target setpoint for hardware AF
            af_range: Autofocus Z range ±µm from current Z
            af_resolution: Z step size for autofocus
            af_cropsize: Crop size for focus algorithm
            af_algorithm: Focus quality algorithm (LAPE, GLVA, JPEG)
            af_settle_time: Settle time in seconds
            af_static_offset: Static Z offset after autofocus
            af_two_stage: Use two-stage (coarse+fine) autofocus
            af_n_gauss: Gaussian kernel size

        Returns:
            float: Best focus Z position, or None if autofocus failed
        """
        self._logger.debug(
            f"Performing autofocus (mode={mode}) with parameters "
            f"af_range={af_range}, af_resolution={af_resolution}, "
            f"af_algorithm={af_algorithm}, channel={illuminationChannel}"
        )

        # Route to appropriate autofocus method
        if mode == "hardware":
            return self.autofocus_hardware(target_focus_setpoint=target_focus_setpoint,
                                           max_attempts=max_attempts,
                                           illuminationChannel=illuminationChannel)
        else:
            return self.autofocus_software(
                af_range=af_range,
                af_resolution=af_resolution,
                af_cropsize=af_cropsize,
                af_algorithm=af_algorithm,
                af_settle_time=af_settle_time,
                af_static_offset=af_static_offset,
                af_two_stage=af_two_stage,
                af_n_gauss=af_n_gauss,
                illuminationChannel=illuminationChannel,
                minZ=minZ,
                maxZ=maxZ,
                stepSize=stepSize,
            )

    def autofocus_software(self, af_range: float = 100.0, af_resolution: float = 10.0,
                           af_cropsize: int = 2048, af_algorithm: str = "LAPE",
                           af_settle_time: float = 0.1, af_static_offset: float = 0.0,
                           af_two_stage: bool = False, af_n_gauss: int = 0,
                           illuminationChannel: str = "",
                           minZ: float = 0, maxZ: float = 0, stepSize: float = 0):
        """Perform software-based autofocus using AutofocusController (Z-sweep).

        All parameters are passed through from FocusMapConfig or experiment settings.
        Legacy minZ/maxZ/stepSize params are kept for backward compatibility but
        are overridden by af_range/af_resolution when those are non-default.

        Returns:
            float: Best focus Z position, or None if autofocus failed
        """
        self._logger.debug(
            "Performing software autofocus (Z-sweep)... "
            "af_range=%s, af_resolution=%s, af_algorithm=%s, af_cropsize=%s, "
            "af_settle_time=%s, af_n_gauss=%s, af_two_stage=%s, illumination=%s",
            af_range, af_resolution, af_algorithm, af_cropsize,
            af_settle_time, af_n_gauss, af_two_stage, illuminationChannel
        )

        # Get the autofocus controller
        autofocusController = self._master.getController('Autofocus')

        if autofocusController is None:
            self._logger.warning("AutofocusController not available - skipping autofocus")
            return None

        # Set illumination if specified
        if illuminationChannel and hasattr(self, '_master') and hasattr(self._master, 'lasersManager'):
            try:
                self._logger.debug(f"Setting illumination channel {illuminationChannel} for autofocus")
            except Exception as e:
                self._logger.warning(f"Failed to set illumination channel {illuminationChannel}: {e}")

        try:
            # Determine rangez: prefer af_range, fall back to legacy minZ/maxZ
            if af_range > 0:
                rangez = af_range
            elif maxZ > minZ:
                rangez = abs(maxZ - minZ) / 2.0
            else:
                rangez = 50.0

            # Determine resolution: prefer af_resolution, fall back to legacy stepSize
            resolutionz = af_resolution if af_resolution > 0 else (stepSize if stepSize > 0 else 10.0)

            # Call autofocus – use autoFocus which starts doAutofocusBackground
            # in a thread with proper state management and validation.
            # Then wait for the AF thread to finish before reading the result.
            autofocusController.autoFocus(
                rangez=rangez,
                resolutionz=resolutionz,
                defocusz=0,
                tSettle=af_settle_time,
                isDebug=False,
                nGauss=af_n_gauss,
                nCropsize=af_cropsize,
                focusAlgorithm=af_algorithm,
                static_offset=af_static_offset,
                twoStage=af_two_stage
            )

            # Wait for autofocus thread to finish (it runs in _AutofocusThead)
            af_thread = getattr(autofocusController, '_AutofocusThead', None)
            if af_thread is not None and af_thread.is_alive():
                af_thread.join(timeout=120)  # 2 min timeout

            # Read the resulting Z position
            result = self.mStage.getPosition().get("Z", None)

            self._logger.debug("Autofocus completed successfully")
            return result

        except Exception as e:
            self._logger.error(f"Autofocus failed: {e}")
            return None

    def wait_time(self, seconds: int, context: WorkflowContext, metadata: Dict[str, Any]):
        import time
        time.sleep(seconds)

    def wait_for_next_timepoint(self, timepoint: int, t_period: float, context: WorkflowContext, metadata: Dict[str, Any]):
        """
        Wait for the proper time interval between timepoints, accounting for measurement time.

        Args:
            timepoint: Current timepoint index
            t_period: Target period between timepoints in seconds
            context: WorkflowContext containing timing information
            metadata: Metadata dictionary
        """
        import time

        current_time = time.time()
        experiment_start_time = context.get_metadata("experiment_start_time", current_time)
        timepoint_times = context.get_metadata("timepoint_times", {})

        # Calculate expected time for this timepoint
        expected_time = experiment_start_time + (timepoint + 1) * t_period

        # Store timing information for this timepoint
        timepoint_times[str(timepoint)] = current_time
        context.set_metadata("timepoint_times", timepoint_times)

        # Calculate how long to wait
        wait_time = max(0, expected_time - current_time)

        if wait_time > 0:
            self._logger.info(f"Waiting {wait_time:.2f}s for next timepoint (timepoint {timepoint})")
            time.sleep(wait_time)
        else:
            self._logger.warning(f"Timepoint {timepoint} is running {abs(wait_time):.2f}s behind schedule")
            # Small delay to prevent issues
            time.sleep(0.01)

    def save_frame_ome(self, context: WorkflowContext, metadata: Dict[str, Any], **kwargs):
        """
        Saves a single frame using the unified OME writer (both stitched TIFF and OME-Zarr).

        Args:
            context: WorkflowContext containing relevant data.
            metadata: A dictionary containing the image data and other metadata.
            **kwargs: Additional keyword arguments, including tile position, channel, etc.
        """
        # Get the latest image from the camera
        img = metadata.get("result")
        if img is None:
            self._logger.debug("No image found in metadata!")
            return

        # Get tile index to identify the correct OME writer
        position_center_index = kwargs.get("position_center_index")
        if position_center_index is None:
            self._logger.error("No position_center_index provided for OME writer lookup")
            metadata["frame_saved"] = False
            return

        # Prepare metadata for OME writer
        ome_metadata = {
            "x": kwargs.get("posX", 0),
            "y": kwargs.get("posY", 0),
            "z": kwargs.get("posZ", 0),
            "runningNumber": kwargs.get("runningNumber", 0),
            "illuminationChannel": kwargs.get("illuminationChannel", "unknown"),
            "illuminationValue": kwargs.get("illuminationValue", 0),
            "tile_index": kwargs.get("tile_index", 0),
            "time_index": kwargs.get("time_index", 0),
            "z_index": kwargs.get("z_index", 0),
            "channel_index": kwargs.get("channel_index", 0),
        }
        
        # Enrich metadata with MetadataHub data if available
        try:
            if hasattr(self._master, 'metadataHub') and self._master.metadataHub is not None:
                detector_name = self._master.detectorsManager.getAllDeviceNames()[0]
                
                # Get objective info from hub
                hub_global = self._master.metadataHub.get_latest(flat=True, filter_category='Objective')
                for key, value_dict in hub_global.items():
                    if 'PixelSizeUm' in key:
                        ome_metadata['objective_pixel_size_um'] = value_dict.get('value')
                    elif 'Name' in key:
                        ome_metadata['objective_name'] = value_dict.get('value')
                    elif 'Magnification' in key:
                        ome_metadata['objective_magnification'] = value_dict.get('value')
                    elif 'NA' in key:
                        ome_metadata['objective_na'] = value_dict.get('value')
                
                # Get detector context (includes isRGB, exposure, etc.)
                detector_ctx = self._master.metadataHub.get_detector(detector_name)
                if detector_ctx:
                    ome_metadata['detector_is_rgb'] = detector_ctx.is_rgb
                    if detector_ctx.exposure_ms:
                        ome_metadata['exposure_ms'] = detector_ctx.exposure_ms
                    if detector_ctx.pixel_size_um:
                        ome_metadata['pixel_size_um'] = detector_ctx.pixel_size_um
        except Exception as e:
            self._logger.debug(f"Could not enrich OME metadata from MetadataHub: {e}")

        try:
            # Get file_writers list from context
            file_writers = context.get_object("file_writers")
            if file_writers is None or position_center_index >= len(file_writers):
                self._logger.error(f"No OME writer found for tile index {position_center_index}")
                metadata["frame_saved"] = False
                return

            # Write frame using the specific OME writer from the list
            ome_writer = file_writers[position_center_index]
            chunk_info = ome_writer.write_frame(img, ome_metadata)
            if ome_writer.store:
                data_path = dirtools.UserFileDirs.getValidatedDataPath()
                self.setOmeZarrUrl(ome_writer.store.split(data_path)[-1])  # Update OME-Zarr URL in context
            # Emit signal for frontend updates if Zarr chunk was written
            if chunk_info and "rel_chunk" in chunk_info:
                sigZarrDict = {
                    "event": "zarr_chunk",
                    "path": chunk_info["rel_chunk"],
                    "zarr": str(self.getOmeZarrUrl())
                }
                self.sigUpdateOMEZarrStore.emit(sigZarrDict)

            metadata["frame_saved"] = True
        except Exception as e:
            self._logger.error(f"Error saving OME frame: {e}")
            metadata["frame_saved"] = False

        '''
        if tiff_writer is None:
            self._logger.debug("No TIFF writer found in context!")
            return
        img = metadata["result"]
        # append the image to the tiff file
        try:
            tiff_writer.write(img)
            metadata["frame_saved"] = True
        except Exception as e:
            self._logger.error(f"Error saving TIFF: {e}")
            metadata["frame_saved"] = False
        '''

    def close_ome_zarr_store(self, omezarr_store):
        # If you need to do anything special (like flush) for the store, do it here.
        # Otherwise, Zarr’s FS-store or disk-store typically closes on its own.
        # This function can be effectively a no-op if you do not require extra steps.
        try:
            if omezarr_store:
                omezarr_store.close()
            else:
                self._logger.debug("OME-Zarr store not found in context.")
            return
        except Exception as e:
            self._logger.error(f"Error closing OME-Zarr store: {e}")
            raise e

    def save_frame_ome_zarr(self, context: Dict[str, Any], metadata: Dict[str, Any], **kwargs):
        """
        Saves a single frame (tile) to an OME-Zarr store, handling coordinate transformation mismatch.

        Args:
            context: A dictionary containing the OME-Zarr store and other relevant data.
            metadata: A dictionary containing the image data and other metadata.
            **kwargs: Additional keyword arguments, including tile position, channel, etc.
        """
        if not IS_OMEZARR_AVAILABLE:
            # self._logger.error("OME-Zarr is not available.")
            return
        omeZarrStore = context.get_object("omezarr_store")
        if omeZarrStore is None:
            raise ValueError("OME-Zarr store not found in context.")

        img = metadata.get("result")
        if img is None:
            return

        posX = kwargs.get("posX", 0)
        posY = kwargs.get("posY", 0)
        posZ = kwargs.get("posZ", 0)
        channel_str = kwargs.get("channel", "Mono")

        # 3) Write the frame with stage coords:
        if 0:
            omeZarrStore.write(img, x=posX, y=posY, z=posZ)
        else:
            # TODO: This is not working as the posY and posX are in microns, but the OME-Zarr store expects pixel coordinates.
            # Convert to pixel coordinates
            omeZarrStore.write_tile(img, t=0, c=0, z=0, y_start=posY, x_start=posX)

        time.sleep(0.01)


    def set_laser_power(self, power: float, channel: str):
        if channel not in self.allIlluNames:
            self._logger.error(f"Channel {channel} not found in available lasers: {self.allIlluNames}")
            return None
        self._master.lasersManager[channel].setValue(power, getReturn=True)
        if self._master.lasersManager[channel].enabled == 0:
            self._master.lasersManager[channel].setEnabled(1, getReturn=True)
        self._logger.debug(f"Setting laser power to {power} for channel {channel}")
        return power



    def move_stage_xy(self, posX: float = None, posY: float = None, relative: bool = False):
        # {"task":"/motor_act",     "motor":     {         "steppers": [             { "stepperid": 1, "position": -1000, "speed": 30000, "isabs": 0, "isaccel":1, "isen":0, "accel":500000}     ]}}
        self._logger.info(f"Moving stage to X={posX}, Y={posY}")
        #if posY and posX is None:
        self.mStage.move(value=(posX, posY), speed=(self.SPEED_X_default, self.SPEED_Y_default), axis="XY", is_absolute=not relative, is_blocking=True, acceleration=self.ACCELERATION)
        #newPosition = self.mStage.getPosition()
        #self._commChannel.sigUpdateMotorPosition.emit([posX, posY])
        return (posX, posY) # TODO: Need to adjust in case of relative move

    def move_stage_z(self, posZ: float, relative: bool = False, maxSpeedZ=5000):
        self._logger.info(f"Moving stage to Z={posZ}")
        self.mStage.move(value=posZ, speed=np.min((self.SPEED_Z, maxSpeedZ)), axis="Z", is_absolute=not relative, is_blocking=True)
        #newPosition = self.mStage.getPosition()
        #self._commChannel.sigUpdateMotorPosition.emit([newPosition["Z"]])
        return posZ # TODO: Need to adjust in case of relative move

    def set_detector_parameter(self, parameter: str, value: Any):
        """Set a detector parameter."""
        try:
            if hasattr(self.mDetector, 'setParameter'):
                self.mDetector.setParameter(parameter, value)
            elif hasattr(self.mDetector, '_camera') and hasattr(self.mDetector._camera, 'setParameter'):
                self.mDetector._camera.setParameter(parameter, value)
            else:
                self._logger.warning(f"Cannot set detector parameter {parameter} - method not available")
        except Exception as e:
            self._logger.error(f"Error setting detector parameter {parameter} to {value}: {e}")


    @APIExport()
    def pauseWorkflow(self):
        """Pause the workflow. Only works in normal mode."""
        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check if performance mode is running
        performance_status = self.performance_mode.get_scan_status()
        if performance_status["running"]:
            return {"status": "error", "message": "Cannot pause experiment in performance mode"}

        if workflow_status == "running":
            return self.workflow_manager.pause_workflow()
        else:
            return {"status": "error", "message": f"Cannot pause in current state: {workflow_status}"}

    @APIExport()
    def resumeExperiment(self):
        """Resume the experiment. Only works in normal mode."""
        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check if performance mode is running
        performance_status = self.performance_mode.get_scan_status()
        if performance_status["running"]:
            return {"status": "error", "message": "Cannot resume experiment in performance mode"}

        if workflow_status == "paused":
            return self.workflow_manager.resume_workflow()
        else:
            return {"status": "error", "message": f"Cannot resume in current state: {workflow_status}"}

    @APIExport()
    def stopExperiment(self):
        """Stop the experiment. Works for both normal and performance modes."""
        # Abort any in-progress focus map computation first so
        # the synchronous _run_focus_map_phase loop exits early.
        try:
            self.focus_map_manager.request_abort()
        except Exception:
            pass

        # Check workflow manager status (normal mode)
        workflow_status = self.workflow_manager.get_status()["status"]

        # Check performance mode status
        performance_status = self.performance_mode.get_scan_status()

        results = {}

        # Stop workflow if running
        if workflow_status in ["running", "paused", "stopping"]:
            results["workflow"] = self.workflow_manager.stop_workflow()

        # Stop performance mode if running
        if performance_status["running"]:
            results["performance"] = self.performance_mode.stop_scan()

        # Set LED status to idle
        self.set_led_status("idle")

        # If nothing was running, return appropriate message
        if not results:
            return "No experiments are currently running"

        return results


    @APIExport()
    def forceStopExperiment(self):
        """Force stop the experiment. Works for both normal and performance modes."""
        results = {}

        # Force stop workflow
        try:
            self.workflow_manager.stop_workflow()
            del self.workflow_manager
            self.workflow_manager = WorkflowsManager()
            results["workflow"] = {"status": "force_stopped", "message": "Workflow force stopped"}
        except Exception as e:
            results["workflow"] = {"status": "error", "message": f"Error force stopping workflow: {e}"}
            self.set_led_status("error")

        # Force stop performance mode
        try:
            results["performance"] = self.performance_mode.force_stop_scan()
        except Exception as e:
            results["performance"] = {"status": "error", "message": f"Error force stopping performance mode: {e}"}
            self.set_led_status("error")

        # Set LED status to idle if no errors
        if all(r.get("status") != "error" for r in results.values()):
            self.set_led_status("idle")

        return results


    """Couples a 2‑D stage scan with external‑trigger camera acquisition.

    • Puts the connected ``CameraHIK`` into *external* trigger mode
      (one exposure per TTL rising edge on LINE0).
    • Runs ``positioner.start_stage_scanning``.
    • Pops every frame straight from the camera ring‑buffer and writes it to
      disk as ``000123.tif`` (frame‑id used as filename).

    Assumes the micro‑controller (or the positioner itself) raises a TTL pulse
    **after** arriving at each grid co‑ordinate.
    """

    def setOmeZarrUrl(self, url):
        """Set the OME-Zarr URL for the experiment."""
        self._omeZarrUrl = url
        self._logger.info(f"OME-Zarr URL set to: {self._omeZarrUrl}")

    def getOmeZarrUrl(self):
        """Get the OME-Zarr URL for the experiment."""
        if self._omeZarrUrl is None:
            return -1
        return self._omeZarrUrl

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------
    @APIExport(runOnUIThread=False)
    def startFastStageScanAcquisition(self,
                      xstart:float=0, xstep:float=500, nx:int=10,
                      ystart:float=0, ystep:float=500, ny:int=10,
                      zstart:float=0, zstep:float=0, nz:int=1,
                      tsettle:float=90, tExposure:float=50,
                      illumination:List[int]=None, led:float=None,
                      tPeriod:int=1, nTimes:int=1,
                      isSnakeScan:bool=True):
        """Full workflow: arm camera ➔ launch writer ➔ execute scan.
        
        Args:
            xstart: Starting X position
            xstep: Step size in X direction
            nx: Number of steps in X
            ystart: Starting Y position  
            ystep: Step size in Y direction
            ny: Number of steps in Y
            zstart: Starting Z position (for Z-stacking)
            zstep: Step size in Z direction (0 = no Z-stacking)
            nz: Number of Z planes (1 = single plane)
            tsettle: Settle time after movement (ms)
            tExposure: Exposure time (ms)
            illumination: List of illumination channel intensities (e.g., [100, 50, 0, 75, 0])
            led: LED intensity (0-255)
            tPeriod: Period between time points (s)
            nTimes: Number of time points
            isSnakeScan: If True, apply snake (serpentine) scan pattern; if False, use raster
        """
        self.fastStageScanIsRunning = True
        self._stop() # ensure all prior runs are stopped
        self.move_stage_xy(posX=xstart, posY=ystart, relative=False)

        # Turn off all illumination channels before starting scan
        self._switch_off_all_illumination()

        # compute the metadata for the stage scan (e.g. x/y coordinates and illumination channels)
        # stage will start at xstart, ystart and move in steps of xstep, ystep in snake scan logic

        # Ensure illumination is a list
        if illumination is None:
            illumination = []
        
        # Build illumination dict for metadata (maintain backward compatibility)
        illum_dict = {}
        for i, val in enumerate(illumination[:5]):  # Take up to 5 channels
            illum_dict[f"illumination{i}"] = val
        illum_dict["led"] = led

        # Count how many illumination entries are valid (not None and > 0)
        nIlluminations = sum(val is not None and val > 0 for val in illumination) + (1 if led and led > 0 else 0)
        nScan = max(nIlluminations, 1)
        total_frames = nx * ny * nz * nScan
        self._logger.info(f"Stage-scan: {nx}×{ny}×{nz} ({total_frames} frames)")
        
        def addDataPoint(metadataList, x, y, z, illuminationChannel, illuminationValue, runningNumber):
            """Helper function to add metadata for each position."""
            metadataList.append({
                "x": x,
                "y": y,
                "z": z,
                "illuminationChannel": illuminationChannel,
                "illuminationValue": illuminationValue,
                "runningNumber": runningNumber
            })
            return metadataList
        # This corresponds to the metadataList in the UC2-ESP firmware e.g. here: https://github.com/youseetoo/uc2-esp32/blob/9addafaca538186e642e97f50c248df661a74637/main/src/motor/StageScan.cpp#L602
        metadataList = []
        runningNumber = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    z = zstart + iz * zstep
                    x = xstart + ix * xstep
                    y = ystart + iy * ystep
                    # Snake pattern: reverse X direction on odd rows
                    if isSnakeScan and iy % 2 == 1:
                        x = xstart + (nx - 1 - ix) * xstep
                
                    # If there's at least one valid illumination or LED set, take only one image as "default"
                    if nIlluminations == 0:
                        runningNumber += 1
                        addDataPoint(metadataList, x, y, z, "default", -1, runningNumber)
                    else:
                        # Otherwise take an image for each illumination channel > 0
                        for channel, value in illum_dict.items():
                            if value is not None and value > 0:
                                runningNumber += 1
                                addDataPoint(metadataList, x, y, z, channel, value, runningNumber)
        # 2. start writer thread ----------------------------------------------
        nLastTime = time.time()
        for iTime in range(nTimes):
            saveOMEZarr = True
            nTimePoints = nTimes
            nZPlanes = nz

            if saveOMEZarr:
                # ------------------------------------------------------------------+
                # 2. open OME-Zarr canvas                                           |
                # ──────────────────────────────────────────────────────────────────+
                timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.mFilePath = os.path.join(self.save_dir,  f"{timeStamp}_FastStageScan")
                # create directory if it does not exist and file paths
                omezarr_store = OMEFileStorePaths(self.mFilePath)
                data_path = dirtools.UserFileDirs.getValidatedDataPath()
                self.setOmeZarrUrl(self.mFilePath.split(data_path)[-1]+".ome.zarr")
                self._writer_thread_ome = threading.Thread(
                    target=self._writer_loop_ome, args=(omezarr_store, total_frames, metadataList, xstart, ystart, xstep, ystep, nx, ny, 0, nTimePoints, nZPlanes, nIlluminations),
                    daemon=True)
                self._stop_writer_evt.clear()
                self._writer_thread_ome.start()
            else:
                # Non-performance mode: also use _writer_loop_ome but configure for TIFF stitching
                self._stop_writer_evt.clear()
                self._writer_thread_ome = threading.Thread(
                    target=self._writer_loop_ome,
                    args=(omezarr_store, total_frames, metadataList, xstart, ystart, xstep, ystep, nx, ny, 0.2, True, True, False, nTimePoints, nZPlanes, nIlluminations),  # is_tiff=True, write_stitched_tiff=True, is_performance_mode=False
                    daemon=True
                )
                self._writer_thread_ome.start()
            
            # Pad illumination list to 5 channels if needed
            illumination_padded = (illumination + [0] * 5)[:5] if illumination else [0, 0, 0, 0, 0]
            
            # 3. execute stage scan (blocks until finished) ------------------------
            self.fastStageScanIsRunning = True  # Set flag to indicate scan is running
            self.mStage.start_stage_scanning(
                xstart=0, xstep=xstep, nx=nx, # we choose xstart/ystart = 0 since this means we start from here in the positive direction with nsteps
                ystart=0, ystep=ystep, ny=ny,
                zstart=0, zstep=zstep, nz=nz,  # Z-stacking parameters
                tsettle=tsettle, tExposure=tExposure,
                illumination=tuple(illumination_padded), led=led,
            )
            # Wait for time period or until scan is stopped
            while nLastTime + tPeriod < time.time() and self.fastStageScanIsRunning:
                time.sleep(0.1) # TODO: This fails / breaks immediately if there is no timealpse as the start_stage_scanning runs in the background and does not give a signal if it'S done
        return self.getOmeZarrUrl()  # return relative path to the data directory

    def _switch_off_all_illumination(self) -> None:
        """
        Turn off all illumination sources before starting scan.
        This ensures clean state for hardware-controlled illumination.
        """
        try:
            # Try to access laser manager
            if hasattr(self, '_master') and hasattr(self._master, 'lasersManager'):
                for laser_name in self._master.lasersManager.getAllDeviceNames():
                    try:
                        self._master.lasersManager[laser_name].setEnabled(False)
                        self._master.lasersManager[laser_name].setValue(0)
                    except Exception as e:
                        self._logger.debug(f"Could not turn off laser {laser_name}: {e}")
            
            self._logger.debug("All illumination sources switched off before scan")
        except Exception as e:
            self._logger.warning(f"Error switching off illumination: {e}")

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------
    def _writer_loop_ome(
        self,
        mFilePath: OMEFileStorePaths,
        n_expected: int,
        metadata_list: list[dict],
        x_start: float,
        y_start: float,
        x_step: float,
        y_step: float,
        nx: int,
        ny: int,
        min_period: float = 0.2,
        is_tiff: bool = False,
        write_stitched_tiff: bool = True,  # Enable stitched TIFF by default
        is_performance_mode: bool = True,  # New parameter to distinguish modes
        nTimePoints: int = 1,
        nZ_planes: int = 1,
        nIlluminations: int = 1
    ):
        """
        Bulk-writer for both fast stage scan (performance mode) and normal stage scan.

        Stores every frame in multiple formats:
        • individual OME-TIFF tile (debug/backup) - optional
        • single chunked OME-Zarr mosaic (browser streaming)
        • stitched OME-TIFF file (Fiji compatible) - optional

        Parameters
        ----------
        n_expected      total number of frames that will arrive
        metadata_list   list with x, y, illuminationChannel, … for each frame-id
        x_start … ny    grid geometry (needed to locate each tile in the canvas)
        is_performance_mode  whether using hardware triggering or workflow-based acquisition
        """
        # Set up unified OME writer
        tile_shape = (self.mDetector._shape[-1], self.mDetector._shape[-2])  # (height, width)
        grid_shape = (nx, ny)
        grid_geometry = (x_start, y_start, x_step, y_step)
        writer_config = OMEWriterConfig(
            write_tiff=is_tiff,
            write_zarr=self._ome_write_zarr,
            write_stitched_tiff=write_stitched_tiff,
            write_tiff_single=self._ome_write_single_tiff,
            write_individual_tiffs=self._ome_write_individual_tiffs,
            min_period=min_period,
            pixel_size=self.detectorPixelSize[-1] if hasattr(self, 'detectorPixelSize') else 1.0,
            n_time_points=nTimePoints,
            n_z_planes= nZ_planes,
            n_channels = nIlluminations
        )

        ome_writer = OMEWriter(
            file_paths=mFilePath,
            tile_shape=tile_shape,
            grid_shape=grid_shape,
            grid_geometry=grid_geometry,
            config=writer_config,
            logger=self._logger
        )

        # ------------------------------------------------------------- main loop
        saved = 0
        self._logger.info(f"Writer thread started → {mFilePath.base_dir}")

        if is_performance_mode:
            # Performance mode: get frames from camera buffer
            while saved < n_expected and not self._stop_writer_evt.is_set(): # TODO: we have that already checked in the exerpiment_performance_mode
                frames, ids = self.mDetector.getChunk()  # empties camera buffer

                if frames.size == 0:
                    time.sleep(0.005)
                    continue

                for frame, fid in zip(frames, ids):
                    meta = metadata_list[fid] if fid < len(metadata_list) else None
                    if not meta:
                        self._logger.warning(f"missing metadata for frame-id {fid}")
                        self._stop_writer_evt.set()
                        break # end experiment - TODO: should send a signal?

                    # Write frame using unified writer
                    chunk_info = ome_writer.write_frame(frame, meta)
                    saved += 1

                    # emit signal to tell frontend about the new chunk
                    if chunk_info and "rel_chunk" in chunk_info:
                        sigZarrDict = {
                            "event": "zarr_chunk",
                            "path": chunk_info["rel_chunk"],
                            "zarr": str(self.getOmeZarrUrl())  # e.g. /files/…/FastStageScan.ome.zarr
                        }
                        self.sigUpdateOMEZarrStore.emit(sigZarrDict)
        else:
            # Normal mode: frames are provided via external queue or workflow
            # This is a placeholder - actual implementation depends on how frames are provided
            self._logger.info("Normal mode writer started - waiting for frames via workflow")
            # In normal mode, frames will be written via separate calls to write_frame

        self._logger.info(f"Writer thread finished ({saved}/{n_expected}) tiles under : {mFilePath.base_dir}")

        # Finalize writing (build pyramids, etc.)
        ome_writer.finalize()

        # Store writer reference for normal mode
        if not is_performance_mode:
            self._current_ome_writer = ome_writer
            return  # Don't reset camera in normal mode


    def write_frame_to_ome_writer(self, frame, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Write a frame to the current OME writer (for normal mode scanning).

        Args:
            frame: Image data as numpy array
            metadata: Dictionary containing position and other metadata

        Returns:
            Dictionary with information about the written chunk (for Zarr)
        """
        if self._current_ome_writer is not None:
            return self._current_ome_writer.write_frame(frame, metadata)
        else:
            self._logger.warning("No OME writer available for frame writing")
            return None

    def finalize_tile_ome_writer(self, context: WorkflowContext, metadata: Dict[str, Any], tile_index: int):
        """Finalize the OME writer for a specific tile."""
        file_writers = context.get_object("file_writers")

        if file_writers is not None and tile_index < len(file_writers):
            ome_writer = file_writers[tile_index]
            try:
                self._logger.info(f"Finalizing OME writer for tile: {tile_index}")
                ome_writer.finalize()
                self._logger.info(f"OME writer finalized for tile {tile_index}")
            except Exception as e:
                self._logger.error(f"Error finalizing OME writer for tile {tile_index}: {e}")
        else:
            self._logger.warning(f"No OME writer found for tile index when finalizing writer: {tile_index}")

    def finalize_current_ome_writer(self, context: WorkflowContext = None, metadata: Dict[str, Any] = None, *args, **kwargs):
        """Finalize all OME writers and clean up."""
        # TODO: This is misleading as the method closes all filewriters, not just the current one.
        # Finalize OME writers from context (normal mode)
        # optional: extract time_point from args/kwargs if needed
        time_index = kwargs.get("time_index", None)

        if context is not None:
            # Get file_writers list and finalize all
            file_writers = context.get_object("file_writers")
            if file_writers is not None:
                if time_index is not None:
                    self._logger.info(f"Finalizing OME writers for time point {time_index}")
                    try:
                        ome_writer = file_writers[time_index]
                        ome_writer.finalize()
                        self._logger.info(f"OME writer finalized for time point {time_index}")
                    except Exception as e:
                        self._logger.error(f"Error finalizing OME writer for time point {time_index}: {e}")
                else:
                    for i, ome_writer in enumerate(file_writers):
                        try:
                            self._logger.info(f"Finalizing OME writer for tile {i}")
                            ome_writer.finalize()
                        except Exception as e:
                            self._logger.error(f"Error finalizing OME writer for tile {i}: {e}")
                    # Clear the list from context
                    context.remove_object("file_writers")

        # Also finalize the instance variable if it exists (performance mode)
        if self._current_ome_writer is not None:
            try:
                self._current_ome_writer.finalize()
                self._current_ome_writer = None
            except Exception as e:
                self._logger.error(f"Error finalizing current OME writer: {e}")

    def _stop(self):
        """Abort the acquisition gracefully."""
        self._stop_writer_evt.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2)
        if self._writer_thread_ome is not None:
            self._writer_thread_ome.join(timeout=2)
        self.mDetector.stopAcquisition()

    @APIExport(runOnUIThread=False)
    def stopFastStageScanAcquisition(self):
        """Stop the stage scan acquisition and writer thread."""
        self.mStage.stop_stage_scanning()
        self.fastStageScanIsRunning = False
        self._logger.info("Stopping stage scan acquisition...")
        self._stop()
        self._logger.info("Stage scan acquisition stopped.")

    @APIExport(runOnUIThread=False)
    def startFastStageScanAcquisitionFilePath(self) -> str:
        """Returns the file path of the last saved fast stage scan."""
        if hasattr(self, 'fastStageScanFilePath') and self.fastStageScanFilePath is not None:
            return self.fastStageScanFilePath
        else:
            return "No fast stage scan available yet"

    # MDA (Multi-Dimensional Acquisition) Methods using useq-schema

    @APIExport()
    def get_mda_capabilities(self) -> Dict[str, Any]:
        """Get information about MDA capabilities and available channels."""
        return {
            "mda_available": self.mda_manager.is_available(),
            "available_channels": self.allIlluNames,
            "available_detectors": self._master.detectorsManager.getAllDeviceNames(),
            "stage_available": self.mStage is not None
        }

    @APIExport(requestType="POST")
    def start_mda_experiment(self, request: MDASequenceRequest) -> Dict[str, Any]:
        """
        Start an MDA experiment using useq-schema.
        
        This provides a modern, standardized interface for multi-dimensional 
        acquisition experiments.
        """
        if not self.mda_manager.is_available():
            raise HTTPException(status_code=400, detail="useq-schema not available")

        # Check if another workflow is running
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        try:
            # Convert channel configurations to useq format
            channel_names = [ch.name for ch in request.channels]
            exposure_times = {ch.name: ch.exposure for ch in request.channels}

            # Create MDA sequence
            sequence = self.mda_manager.create_simple_sequence(
                channels=channel_names,
                z_range=request.z_range,
                z_step=request.z_step,
                time_points=request.time_points,
                time_interval=request.time_interval,
                exposure_times=exposure_times
            )

            # Get sequence info for logging/validation
            seq_info = self.mda_manager.get_sequence_info(sequence)
            self._logger.info(f"Starting MDA experiment: {seq_info}")

            # Start the detector if not already running
            if not self.mDetector._running:
                self.mDetector.startAcquisition()

            # Create controller function mapping for MDA conversion
            controller_functions = {
                'move_stage_xy': self.move_stage_xy,
                'move_stage_z': self.move_stage_z,
                'set_laser_power': self.set_laser_power,
                'set_detector_parameter': self.set_detector_parameter,
                'snap_image': self.snap_image_with_metadata
            }

            # Convert MDA sequence to workflow steps
            workflow_steps = self.mda_manager.convert_sequence_to_workflow_steps(
                sequence, controller_functions
            )

            # Setup data directory
            if request.save_directory:
                dirPath = request.save_directory
            else:
                timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                drivePath = dirtools.UserFileDirs.Data
                dirPath = os.path.join(drivePath, 'MDAExperiments', timeStamp)

            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # Create workflow progress handler
            def sendProgress(payload):
                self.sigExperimentWorkflowUpdate.emit()

            # Create workflow and context
            from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext
            wf = Workflow(workflow_steps, self.workflow_manager)
            context = WorkflowContext()

            # Set metadata
            context.set_metadata("experiment_name", request.experiment_name)
            context.set_metadata("sequence_info", seq_info)
            context.set_metadata("save_directory", dirPath)
            context.set_metadata("channels", channel_names)
            context.set_metadata("experiment_start_time", time.time())

            context.on("progress", sendProgress)

            # Start the workflow
            result = self.workflow_manager.start_workflow(wf, context)

            return {
                "status": result["status"],
                "sequence_info": seq_info,
                "save_directory": dirPath,
                "estimated_duration_minutes": seq_info["estimated_duration_minutes"]
            }

        except Exception as e:
            self._logger.error(f"Error starting MDA experiment: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error starting MDA experiment: {str(e)}")

    @APIExport(requestType="POST")
    def get_mda_sequence_info(self, request: MDASequenceRequest) -> MDASequenceInfo:
        """Get information about an MDA sequence without starting it."""
        if not self.mda_manager.is_available():
            raise HTTPException(status_code=400, detail="useq-schema not available")

        try:
            # Convert channel configurations to useq format
            channel_names = [ch.name for ch in request.channels]
            exposure_times = {ch.name: ch.exposure for ch in request.channels}

            # Create MDA sequence
            sequence = self.mda_manager.create_simple_sequence(
                channels=channel_names,
                z_range=request.z_range,
                z_step=request.z_step,
                time_points=request.time_points,
                time_interval=request.time_interval,
                exposure_times=exposure_times
            )

            # Get sequence info
            info = self.mda_manager.get_sequence_info(sequence)
            return MDASequenceInfo(**info)

        except Exception as e:
            self._logger.error(f"Error getting MDA sequence info: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting MDA sequence info: {str(e)}")

    @APIExport(requestType="POST")
    def run_native_mda_sequence(self, sequence_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a native useq-schema MDASequence from JSON.
        
        This endpoint accepts a native useq.MDASequence serialized as JSON (dict),
        following the EXACT pattern from pymmcore-plus and raman-mda-engine.
        
        Args:
            sequence_dict: Native useq.MDASequence serialized to dict/JSON with fields:
                - metadata: Dict with arbitrary experiment metadata
                - stage_positions: List of (x, y, z) tuples or AbsolutePosition dicts
                - grid_plan: Dict with GridRowsColumns configuration
                - channels: List of Channel dicts with 'config' and optional 'exposure'
                - time_plan: Dict with 'interval' and 'loops' for TIntervalLoops
                - z_plan: Dict with 'range' and 'step' for ZRangeAround
                - autofocus_plan: Dict with autofocus configuration
                - axis_order: String like "tpcz" defining acquisition order
                - keep_shutter_open_across: Tuple of axes to keep shutter open
        
        Returns:
            Dict with execution status and sequence information
            
        Example request body:
            {
                "metadata": {"experiment": "test", "user": "researcher"},
                "stage_positions": [[100.0, 100.0, 30.0], [200.0, 150.0, 35.0]],
                "channels": [
                    {"config": "BF", "exposure": 50.0},
                    {"config": "DAPI", "exposure": 100.0}
                ],
                "time_plan": {"interval": 1, "loops": 20},
                "z_plan": {"range": 4.0, "step": 0.5},
                "axis_order": "tpcz"
            }
        """
        if not self.mda_manager.is_available():
            raise HTTPException(status_code=400, detail="useq-schema not available. Install with: pip install useq-schema")

        try:
            from useq import MDASequence

            # Parse the native useq-schema JSON into an MDASequence object
            # useq-schema's pydantic models can parse from dict
            self._logger.info(f"Received native MDA sequence: {sequence_dict.keys()}")
            sequence = MDASequence(**sequence_dict)

            self._logger.info(f"Parsed MDASequence: {len(list(sequence))} events, axis_order={sequence.axis_order}")

            # Register the MDA manager with ImSwitch hardware
            if not self.mda_manager._detector_manager:
                self.mda_manager.register(
                    detector_manager=self._master.detectorsManager,
                    positioners_manager=self._master.positionersManager,
                    lasers_manager=self._master.lasersManager,
                    autofocus_manager=getattr(self._master, 'autofocusManager', None)
                )
                self._logger.info("Registered MDA engine with ImSwitch hardware managers")

            # Get sequence info
            seq_info = self.mda_manager.get_sequence_info(sequence)
            self._logger.info(f"Sequence info: {seq_info}")

            # Setup data directory
            metadata = sequence.metadata if hasattr(sequence, 'metadata') and sequence.metadata else {}
            experiment_name = metadata.get('experiment_name', 'MDA_Experiment')

            timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drivePath = dirtools.UserFileDirs.Data
            dirPath = os.path.join(drivePath, 'NativeMDA', experiment_name, timeStamp)

            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
                self._logger.info(f"Created output directory: {dirPath}")

            # Run the MDA sequence using the native engine
            # This runs in a background thread to not block the API
            import threading

            def run_sequence():
                try:
                    self._logger.info("Starting native MDA sequence execution")
                    self.mda_manager.run_mda(sequence, output_path=dirPath)
                    self._logger.info("Native MDA sequence completed successfully")
                except Exception as e:
                    self._logger.error(f"Error during MDA execution: {str(e)}", exc_info=True)

            # Start execution thread
            thread = threading.Thread(target=run_sequence, daemon=False)
            thread.start()

            return {
                "status": "started",
                "sequence_info": seq_info,
                "save_directory": dirPath,
                "estimated_duration_minutes": seq_info["estimated_duration_minutes"],
                "message": "Native MDA sequence started in background thread"
            }

        except Exception as e:
            self._logger.error(f"Error starting native MDA sequence: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error starting native MDA sequence: {str(e)}")

    def snap_image_with_metadata(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Snap an image with MDA metadata."""
        # This is a wrapper around the existing snap functionality
        # that includes MDA-specific metadata handling
        image = self.mDetector.getLatestFrame()

        # Store metadata with the image (implementation depends on storage backend)
        # For now, just log it
        self._logger.debug(f"Captured MDA image with metadata: {metadata}")

        # Emit signal for live view update
        self.sigExperimentImageUpdate.emit(
            self.mDetector.name,
            image,
            False,  # not init
            [1, 1, 1, 1],  # scale
            True  # is current detector
        )

        return image

    # ================================================================
    # Focus Map – internal helpers
    # ================================================================

    def _run_focus_map_phase(self, mExperiment, config: FocusMapConfig):
        """
        Run focus mapping before the main acquisition loop.

        Iterates over all scan areas (groups), measures focus on a grid,
        and fits a Z surface for each group.  Called from startWellplateExperiment
        when focusMap.enabled == True.
        """
        # Store config for channel_offsets access during acquisition
        self._focus_map_config = config
        self.focus_map_manager.clear_abort()
        areas = []
        if mExperiment.scanAreas:
            for sa in mExperiment.scanAreas:
                areas.append({
                    "areaId": sa.areaId,
                    "areaName": sa.areaName,
                    "bounds": {
                        "minX": sa.bounds.minX,
                        "maxX": sa.bounds.maxX,
                        "minY": sa.bounds.minY,
                        "maxY": sa.bounds.maxY,
                    },
                })
        else:
            # Fallback: derive bounds from pointList
            if mExperiment.pointList:
                xs = [pt.x for pt in mExperiment.pointList]
                ys = [pt.y for pt in mExperiment.pointList]
                areas.append({
                    "areaId": "default",
                    "areaName": "All Points",
                    "bounds": {
                        "minX": min(xs), "maxX": max(xs),
                        "minY": min(ys), "maxY": max(ys),
                    },
                })

        if not areas:
            self._logger.warning("Focus map: no areas found – skipping")
            return

        # ----------------------------------------------------------
        # Option: reuse a pre-existing manual / global map for all
        # groups by interpolation instead of measuring a new grid.
        # ----------------------------------------------------------
        if config.use_manual_map:
            source_fm = self._find_reusable_manual_map()
            if source_fm is not None:
                self._logger.info(
                    f"Focus map: reusing manual map [{source_fm.group_id}] "
                    f"for {len(areas)} group(s) via interpolation"
                )
                for area in areas:
                    gid = area["areaId"]
                    if self.focus_map_manager.has_fitted_map(gid):
                        self._logger.info(
                            f"Focus map [{gid}]: already fitted – skipping interpolation"
                        )
                        continue
                    try:
                        new_fm = source_fm.interpolate_to_region(
                            group_id=gid,
                            group_name=area["areaName"],
                            bounds=area["bounds"],
                            rows=max(config.rows, 3),
                            cols=max(config.cols, 3),
                            logger=self._logger,
                        )
                        self.focus_map_manager._maps[gid] = new_fm
                        self._logger.info(
                            f"Focus map [{gid}]: interpolated from [{source_fm.group_id}] "
                            f"({new_fm.n_points} pts, fitted={new_fm.is_fitted})"
                        )
                    except Exception as e:
                        self._logger.warning(
                            f"Focus map [{gid}]: interpolation from manual map failed ({e}), "
                            f"falling back to measurement"
                        )
                        self._compute_focus_map_for_group(
                            group_id=gid,
                            group_name=area["areaName"],
                            bounds=area["bounds"],
                            config=config,
                        )
                return
            else:
                self._logger.warning(
                    "Focus map: use_manual_map is enabled but no fitted manual/global map found – "
                    "falling back to per-group measurement"
                )

        if config.fit_by_region:
            # Fit separately per region – skip groups that already have a valid fit
            for area in areas:
                gid = area["areaId"]
                if self.focus_map_manager.has_fitted_map(gid):
                    self._logger.info(
                        f"Focus map [{gid}]: using pre-computed map (already fitted)"
                    )
                    continue
                self._compute_focus_map_for_group(
                    group_id=gid,
                    group_name=area["areaName"],
                    bounds=area["bounds"],
                    config=config,
                )
        else:
            # Fit globally: merge all bounds
            all_min_x = min(a["bounds"]["minX"] for a in areas)
            all_max_x = max(a["bounds"]["maxX"] for a in areas)
            all_min_y = min(a["bounds"]["minY"] for a in areas)
            all_max_y = max(a["bounds"]["maxY"] for a in areas)
            merged = {
                "minX": all_min_x, "maxX": all_max_x,
                "minY": all_min_y, "maxY": all_max_y,
            }
            if self.focus_map_manager.has_fitted_map("global"):
                self._logger.info("Focus map [global]: using pre-computed map (already fitted)")
            else:
                self._compute_focus_map_for_group(
                    group_id="global",
                    group_name="Global Fit",
                    bounds=merged,
                    config=config,
                )

    def _find_reusable_manual_map(self) -> "Optional[FocusMap]":
        """
        Look for a pre-existing fitted focus map that can be reused as
        a global template for all groups.  Preference order:
          1. "manual" (from "Fit from Points" in the UI)
          2. "global" (from a previous global computation)
          3. any other fitted map
        Returns None if nothing suitable is found.
        """
        from imswitch.imcontrol.model.focus_map import FocusMap  # noqa: local import
        for candidate_id in ["manual", "global"]:
            fm = self.focus_map_manager.get(candidate_id)
            if fm is not None and fm.is_fitted:
                return fm
        # Fallback: any fitted map
        for fm in self.focus_map_manager.get_all().values():
            if fm.is_fitted:
                return fm
        return None

    # ================================================================
    # Focus Map API endpoints
    # ================================================================

    @APIExport(requestType="POST")
    def computeFocusMap(self, focusMapConfig: Optional[FocusMapConfig] = None,
                        group_id: Optional[str] = None):
        """
        Compute focus map for one or all scan groups.

        Runs autofocus on a grid of positions within each group's bounds,
        fits a Z surface, and stores the results for use during acquisition.

        Args:
            focusMapConfig: Override config (uses experiment default if None).
                            May include scan_areas from the frontend so that
                            the correct XY bounds are known even before an
                            experiment has been started.
            group_id: If provided, only compute for this group. Otherwise compute for all.

        Returns:
            Dict with focus map results per group
        """
        if focusMapConfig is None:
            focusMapConfig = FocusMapConfig(enabled=True)
        focusMapConfig.af_n_gauss=0  # TODO: Force n_gauss=0 for autofocus during focus mapping to speed it up and avoid fitting issues at low SNR. The main purpose of the focus map is to get a general Z surface, not perfect autofocus results at each point, so this is an acceptable tradeoff. The config parameter is still kept for potential future use if we want to allow more flexible autofocus settings during focus mapping.
        # Store config for channel_offsets access during acquisition
        self._focus_map_config = focusMapConfig

        # Clear any previous abort request
        self.focus_map_manager.clear_abort()

        self._logger.info(
            f"Computing focus map: method={focusMapConfig.method}, "
            f"grid={focusMapConfig.rows}x{focusMapConfig.cols}, "
            f"group_id={group_id}"
        )

        # Determine scan area bounds – prefer scan_areas passed in the config,
        # then fall back to last experiment areas, then to current stage pos.
        results = {}

        if focusMapConfig.scan_areas is not None and len(focusMapConfig.scan_areas) > 0:
            # Caller provided scan areas directly (e.g. from frontend)
            self._last_scan_areas = focusMapConfig.scan_areas
        
        effective_scan_areas = getattr(self, '_last_scan_areas', None)
        if effective_scan_areas is None or len(effective_scan_areas) == 0:
            # Fallback: use current stage position as single area
            pos = self.mStage.getPosition()
            effective_scan_areas = [{
                "areaId": "current",
                "areaName": "Current Position",
                "bounds": {
                    "minX": pos.get("X", 0) - 500,
                    "maxX": pos.get("X", 0) + 500,
                    "minY": pos.get("Y", 0) - 500,
                    "maxY": pos.get("Y", 0) + 500,
                },
            }]

        for area in effective_scan_areas:
            area_id = area.get("areaId", "default")
            area_name = area.get("areaName", area_id)

            if group_id is not None and area_id != group_id:
                continue

            bounds = area.get("bounds", {})
            result = self._compute_focus_map_for_group(
                group_id=area_id,
                group_name=area_name,
                bounds=bounds,
                config=focusMapConfig,
            )
            results[area_id] = result

        return results

    def _compute_focus_map_for_group(self, group_id: str, group_name: str,
                                     bounds: Dict[str, float],
                                     config: FocusMapConfig) -> Dict[str, Any]:
        """
        Compute focus map for a single group by moving stage and running autofocus.

        Args:
            group_id: Identifier for this scan group
            group_name: Human-readable name
            bounds: Dict with minX, maxX, minY, maxY
            config: FocusMapConfig

        Returns:
            FocusMapResult as dict
        """
        fm = self.focus_map_manager.get_or_create(
            group_id=group_id,
            group_name=group_name,
            method=config.method,
            smoothing_factor=config.smoothing_factor,
            z_offset=config.z_offset,
            clamp_enabled=config.clamp_enabled,
            z_min=config.z_min,
            z_max=config.z_max,
        )
        # Always clear points when explicitly (re)computing a focus map.
        # Precomputed maps are protected by the skip-logic in _run_focus_map_phase
        # and computeFocusMap; this function is only reached when we truly want
        # to measure from scratch.
        fm.clear_points()

        # Generate measurement grid
        grid = FocusMap.generate_grid(
            bounds=bounds,
            rows=config.rows,
            cols=config.cols,
            add_margin=config.add_margin,
        )
        self._logger.info(f"Focus map [{group_id}]: measuring {len(grid)} points")

        # Measure each grid point
        for i, (gx, gy) in enumerate(grid):
            # Check abort flag
            if self.focus_map_manager.abort_requested:
                self._logger.warning(f"Focus map [{group_id}]: aborted by user at point {i+1}/{len(grid)}")
                break
            try:
                # Move stage to measurement position
                self.move_stage_xy(posX=gx, posY=gy, relative=False)
                time.sleep(0.1)  # Settle time

                # Run autofocus to find best Z using config parameters
                best_z = self.autofocus(
                    mode=config.af_mode,
                    af_range=config.af_range,
                    af_resolution=config.af_resolution,
                    af_cropsize=config.af_cropsize,
                    af_algorithm=config.af_algorithm,
                    af_settle_time=config.af_settle_time,
                    af_static_offset=config.af_static_offset,
                    af_two_stage=config.af_two_stage,
                    af_n_gauss=config.af_n_gauss,
                    illuminationChannel=config.af_illumination_channel,
                    max_attempts=config.af_max_attempts,
                    target_focus_setpoint=config.af_target_setpoint,
                )

                if best_z is None:
                    # If autofocus failed, read current Z as fallback
                    best_z = self.mStage.getPosition().get("Z", 0)
                    self._logger.warning(
                        f"Focus map [{group_id}]: autofocus failed at ({gx:.1f}, {gy:.1f}), "
                        f"using current Z={best_z:.3f}"
                    )

                fm.add_point(gx, gy, float(best_z))
                self._logger.debug(
                    f"Focus map [{group_id}]: point {i+1}/{len(grid)} "
                    f"at ({gx:.1f}, {gy:.1f}) → Z={best_z:.3f}"
                )
                # settle for a moment 
                time.sleep(0.5)

            except Exception as e:
                self._logger.error(f"Focus map [{group_id}]: failed at ({gx:.1f}, {gy:.1f}): {e}")

        # Fit the surface
        try:
            stats = fm.fit()
            self._logger.info(
                f"Focus map [{group_id}]: fitted {stats.method}, "
                f"MAE={stats.mean_abs_error:.4f}, n={stats.n_points}"
            )
        except ValueError as e:
            self._logger.error(f"Focus map [{group_id}]: fit failed: {e}")
            return fm.to_result().to_dict()

        # Save artifacts if requested
        if config.store_debug_artifacts:
            try:
                save_dir = os.path.join(self.save_dir, "focus_maps")
                fm.save(save_dir)
            except Exception as e:
                self._logger.warning(f"Failed to save focus map artifacts: {e}")

        return fm.to_result().to_dict()

    @APIExport(requestType="GET")
    def getFocusMap(self, group_id: Optional[str] = None):
        """
        Get saved focus maps.

        Args:
            group_id: If provided, get only this group. Otherwise get all.

        Returns:
            Focus map data per group
        """
        if group_id is not None:
            fm = self.focus_map_manager.get(group_id)
            if fm is None:
                raise HTTPException(status_code=404, detail=f"No focus map for group '{group_id}'")
            return fm.to_result().to_dict()
        return self.focus_map_manager.to_dict()

    @APIExport(requestType="POST")
    def getFocusMapPreview(self, group_id: str, resolution: int = 30):
        """
        Get a preview grid for visualization of a focus map.

        Args:
            group_id: Group to preview
            resolution: Grid resolution for preview

        Returns:
            Dict with x, y, z arrays and raw points
        """
        fm = self.focus_map_manager.get(group_id)
        if fm is None or not fm.is_fitted:
            raise HTTPException(status_code=404, detail=f"No fitted focus map for group '{group_id}'")

        return fm.generate_preview_grid(resolution=resolution)

    @APIExport(requestType="POST")
    def clearFocusMap(self, group_id: Optional[str] = None):
        """
        Clear focus map(s).

        Args:
            group_id: If provided, clear only this group. Otherwise clear all.

        Returns:
            Status message
        """
        self.focus_map_manager.clear(group_id)
        target = f"group '{group_id}'" if group_id else "all groups"
        self._logger.info(f"Cleared focus map for {target}")
        return {"status": "cleared", "target": target}

    @APIExport(requestType="POST")
    def interruptFocusMap(self):
        """
        Interrupt an ongoing focus map computation.

        Sets the abort flag so the measurement loop stops at the next grid point.
        The map that was being computed will still try to fit with whatever
        points have been collected so far.

        Returns:
            Status message
        """
        self.focus_map_manager.request_abort()
        self._logger.info("Focus map computation interrupt requested")
        return {"status": "interrupt_requested"}

    @APIExport(requestType="GET")
    def saveFocusMaps(self, path: str = ""):
        """
        Save all current focus maps to disk as JSON files.

        Args:
            path: Directory to save into. Empty string uses default location.

        Returns:
            Dict with saved file info
        """
        if not path:
            path = os.path.join(self.save_dir, "focus_maps")
        os.makedirs(path, exist_ok=True)
        saved = self.focus_map_manager.save_all(path)
        self._logger.info(f"Saved {len(saved)} focus maps to {path}")
        return {"saved_files": saved, "count": len(saved), "path": path}

    @APIExport(requestType="GET")
    def loadFocusMaps(self, path: str = ""):
        """
        Load focus maps from disk, restoring previously saved maps.

        Args:
            path: Directory to load from. Empty string uses default location.

        Returns:
            Dict with loaded map info
        """
        if not path:
            path = os.path.join(self.save_dir, "focus_maps")
        if not os.path.isdir(path):
            return {"loaded_count": 0, "path": path, "error": "Directory not found"}
        count = self.focus_map_manager.load_all(path)
        self._logger.info(f"Loaded {count} focus maps from {path}")
        maps_dict = self.focus_map_manager.to_dict()
        return {"loaded_count": count, "path": path, "maps": maps_dict}

    @APIExport(requestType="POST")
    def computeFocusMapFromPoints(self, request: FocusMapFromPointsRequest):
        """
        Compute a focus map from manually specified XYZ points.

        Instead of running autofocus on a grid, the user provides pre-measured
        reference points (e.g. from clicking on the wellplate viewer and
        recording the current stage Z).

        Args:
            request: FocusMapFromPointsRequest with points and fit config

        Returns:
            FocusMapResult as dict
        """
        points = request.points
        group_id = request.group_id
        group_name = request.group_name
        method = request.method
        smoothing_factor = request.smoothing_factor
        z_offset = request.z_offset
        clamp_enabled = request.clamp_enabled
        z_min = request.z_min
        z_max = request.z_max

        if not points or len(points) < 1:
            return {"error": "At least 1 point is required"}

        self._logger.info(
            f"Computing focus map from {len(points)} manual points, "
            f"method={method}, group_id={group_id}"
        )

        fm = self.focus_map_manager.get_or_create(
            group_id=group_id,
            group_name=group_name,
            method=method,
            smoothing_factor=smoothing_factor,
            z_offset=z_offset,
            clamp_enabled=clamp_enabled,
            z_min=z_min,
            z_max=z_max,
        )
        fm.clear_points()

        # Add all manual points
        for pt in points:
            x = pt.get("x", 0)
            y = pt.get("y", 0)
            z = pt.get("z", 0)
            fm.add_point(float(x), float(y), float(z))

        # Fit the surface
        try:
            stats = fm.fit()
            self._logger.info(
                f"Focus map [{group_id}]: fitted {stats.method} from manual points, "
                f"MAE={stats.mean_abs_error:.4f}, n={stats.n_points}"
            )
        except ValueError as e:
            self._logger.error(f"Focus map [{group_id}]: fit failed: {e}")
            return fm.to_result().to_dict()

        return fm.to_result().to_dict()

    def apply_focus_map_z(self, x: float, y: float, group_id: str = "default",
                          channel: Optional[str] = None) -> Optional[float]:
        """
        Get the focus-mapped Z for a given XY position during acquisition.

        This is called by the normal mode experiment execution to adjust Z
        before acquiring at each tile position.

        Args:
            x: X position
            y: Y position
            group_id: Scan group identifier
            channel: Optional illumination channel name. If provided and
                     channel_offsets are configured in the FocusMapConfig,
                     the per-channel offset will be added to the result.

        Returns:
            Estimated Z position, or None if no map available
        """
        z = self.focus_map_manager.interpolate(x, y, group_id)
        if z is not None and channel:
            # Apply per-channel Z offset if configured
            focus_map_config = getattr(self, '_focus_map_config', None)
            if focus_map_config and focus_map_config.channel_offsets:
                channel_offset = focus_map_config.channel_offsets.get(channel, 0.0)
                if channel_offset != 0.0:
                    self._logger.debug(
                        f"Applying channel offset for '{channel}': {channel_offset} µm"
                    )
                    z += channel_offset
        return z

    # ── Overview Camera Registration Endpoints ─────────────────────────────────

    @APIExport(requestType="GET")
    def getOverviewRegistrationConfig(self, layout_name: str = "Heidstar 4x Histosample"):
        """
        Get overview wizard configuration with slot definitions for a layout.

        Args:
            layout_name: Name of the wellplate layout

        Returns:
            Layout name, slot list with stage corners, corner convention,
            camera availability, and saved status per slide.
        """
        try:
            layout = get_layout_by_name(layout_name)
            layout_dict = layout.dict() if hasattr(layout, 'dict') else layout
            if layout_dict is None: raise ValueError("Layout dict is None")
        except Exception:
            # Fallback: use hardcoded Heidstar layout
            layout_dict = {
                "name": "Heidstar 4x Histosample",
                "unit": "um",
                "width": 127000,
                "height": 84000,
                "wells": [
                    {"x": 18400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide1"},
                    {"x": 48400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide2"},
                    {"x": 78400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide3"},
                    {"x": 108400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide4"},
                ],
            }
        slots = self._overview_registration.get_slot_definitions(layout_dict)
        camera_name = self._overview_camera_name or "overviewcamera"
        status = self._overview_registration.get_status(camera_name, layout_name)

        return {
            "layoutName": layout_dict.get("name", layout_name),
            "cameraName": camera_name,
            "cameraAvailable": self._overview_camera is not None,
            "cornerConvention": "TL,TR,BR,BL",
            "cornerLabels": ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"],
            "slots": [s.dict() for s in slots],
            "status": status.get("slides", {}),
        }

    @APIExport(requestType="POST")
    def snapOverviewImage(self, slot_id: str = "1", camera_name: str = ""):
        """
        Snap a single image from the overview camera for a given slot.

        Args:
            slot_id:     Slide slot index ("1" to "4")
            camera_name: Camera name (auto-detected if empty)

        Returns:
            Snapshot metadata including base64-encoded image.
        """
        cam = self._overview_camera
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"

        if cam is None:
            raise HTTPException(status_code=400, detail="Overview camera not available")

        frame = cam.getLatestFrame()
        if frame is None:
            raise HTTPException(status_code=500, detail="No frame from overview camera")

        # Apply flip settings if available
        try:
            if hasattr(self._setupInfo, 'PixelCalibration'):
                flip = getattr(self._setupInfo.PixelCalibration, 'ObservationCameraFlip', {})
                if isinstance(flip, dict):
                    if flip.get('flipY', False):
                        frame = np.flip(frame, 0)
                    if flip.get('flipX', False):
                        frame = np.flip(frame, 1)
        except Exception:
            pass

        # Get current stage position for traceability
        stage_x, stage_y, stage_z = 0.0, 0.0, 0.0
        try:
            if self.mStage is not None:
                pos = self.mStage.getPosition()
                stage_x = pos.get("X", 0.0)
                stage_y = pos.get("Y", 0.0)
                stage_z = pos.get("Z", 0.0)
        except Exception:
            pass

        # Make contiguous copy
        frame = np.ascontiguousarray(frame)

        # Save snapshot
        meta = self._overview_registration.save_snapshot(
            camera_name=cam_name,
            layout_name="current",
            slot_id=slot_id,
            image=frame,
            stage_x=stage_x,
            stage_y=stage_y,
            stage_z=stage_z,
        )

        # Encode as base64 JPEG for frontend display
        import cv2
        if len(frame.shape) == 2:
            encode_frame = frame
        else:
            encode_frame = frame
        _, jpg_buf = cv2.imencode(".jpg", encode_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        import base64
        b64_image = base64.b64encode(jpg_buf.tobytes()).decode("ascii")

        meta["imageBase64"] = b64_image
        meta["imageMimeType"] = "image/jpeg"
        meta["stagePosition"] = {"x": stage_x, "y": stage_y, "z": stage_z}
        return meta

    @APIExport(requestType="POST")
    def registerOverviewSlide(self, registration_data: dict):
        """
        Save corner picks and compute per-slide registration (homography).

        Args:
            registration_data: dict with keys:
                - cameraName (str)
                - layoutName (str)
                - slotId (str)
                - slotName (str)
                - snapshotId (str)
                - snapshotTimestamp (str)
                - imageWidth (int)
                - imageHeight (int)
                - cornersPx: [{x, y}, ...] – 4 clicked corners in image pixels
                - slotStageCorners: [{x, y}, ...] – 4 target stage corners

        Returns:
            Registration metadata including homography and error metrics.
        """
        try:
            corners_px = [
                PixelPoint(x=c["x"], y=c["y"])
                for c in registration_data["cornersPx"]
            ]
            slot_stage_corners = [
                StagePoint(x=c["x"], y=c["y"])
                for c in registration_data["slotStageCorners"]
            ]

            # Try to load the raw snapshot for warping
            raw_image = None
            snapshot_id = registration_data.get("snapshotId", "")
            cam_name = registration_data.get("cameraName", self._overview_camera_name or "overviewcamera")
            layout_name = registration_data.get("layoutName", "Heidstar 4x Histosample")

            img_path = self._overview_registration.get_snapshot_image_path(
                cam_name, "current", snapshot_id
            )
            if img_path:
                import cv2
                raw_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            reg = self._overview_registration.register_slide(
                camera_name=cam_name,
                layout_name=layout_name,
                slot_id=registration_data["slotId"],
                slot_name=registration_data.get("slotName", f"Slide{registration_data['slotId']}"),
                snapshot_id=snapshot_id,
                snapshot_timestamp=registration_data.get("snapshotTimestamp", ""),
                image_width=registration_data.get("imageWidth", 0),
                image_height=registration_data.get("imageHeight", 0),
                corners_px=corners_px,
                slot_stage_corners=slot_stage_corners,
                raw_image=raw_image,
            )

            return {
                "success": True,
                "slotId": reg.slotId,
                "reprojectionError": reg.reprojectionError,
                "hasOverlayImage": bool(reg.overlayImageRef),
                "cornerOrder": reg.cornerOrder,
                "createdAt": reg.createdAt,
            }
        except Exception as e:
            self._logger.error(f"Registration failed: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

    @APIExport(requestType="GET")
    def getOverviewRegistrationStatus(self, camera_name: str = "", layout_name: str = "Heidstar 4x Histosample"):
        """
        Get completion status for all slide registrations.

        Args:
            camera_name: Camera name (auto-detected if empty)
            layout_name: Layout name

        Returns:
            Per-slide completion status and metadata.
        """
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        return self._overview_registration.get_status(cam_name, layout_name)

    @APIExport(requestType="POST")
    def refreshOverviewSlideImage(self, slot_id: str = "1", camera_name: str = "", layout_name: str = "Heidstar 4x Histosample"):
        """
        Snap a new image and re-warp using existing registration for a slot.
        Does not require new corner picking if registration already exists.

        Args:
            slot_id:     Slide slot index ("1" to "4")
            camera_name: Camera name
            layout_name: Layout name

        Returns:
            Updated overlay metadata.
        """
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        cam = self._overview_camera
        if cam is None:
            raise HTTPException(status_code=400, detail="Overview camera not available")

        frame = cam.getLatestFrame()
        if frame is None:
            raise HTTPException(status_code=500, detail="No frame from overview camera")

        # Apply flips
        try:
            if hasattr(self._setupInfo, 'PixelCalibration'):
                flip = getattr(self._setupInfo.PixelCalibration, 'ObservationCameraFlip', {})
                if isinstance(flip, dict):
                    if flip.get('flipY', False):
                        frame = np.flip(frame, 0)
                    if flip.get('flipX', False):
                        frame = np.flip(frame, 1)
        except Exception:
            pass

        frame = np.ascontiguousarray(frame)

        try:
            result = self._overview_registration.refresh_overlay_image(
                camera_name=cam_name,
                layout_name=layout_name,
                slot_id=slot_id,
                new_image=frame,
            )
            return {"success": True, **result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @APIExport(requestType="GET")
    def getOverviewOverlayData(self, camera_name: str = "", layout_name: str = "Heidstar 4x Histosample"):
        """
        Get all overlay data for the WellSelector canvas rendering.
        Returns base64-encoded overlay images + slot bounds per completed slide.

        Args:
            camera_name: Camera name (auto-detected if empty)
            layout_name: Layout name

        Returns:
            Per-slide overlay images and stage bounds for canvas rendering.
        """
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        return self._overview_registration.get_overlay_data(cam_name, layout_name)


# Copyright (C) 2025 Benedict Diederich
