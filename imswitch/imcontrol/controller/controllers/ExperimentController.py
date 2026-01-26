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

from pydantic import Field

# -----------------------------------------------------------
# Reuse the existing sub-models:
# -----------------------------------------------------------


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

class Experiment(BaseModel):
    # From your old "Experiment" BaseModel:
    name: str
    parameterValue: ParameterValue
    pointList: List[Point] = Field(default_factory=list)

    # NEW: Pre-calculated scan data from frontend
    scanAreas: Optional[List[ScanArea]] = None
    scanMetadata: Optional[ScanMetadata] = None

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
                    isRGB=self.mDetector._isRGB
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
    def acquire_frame(self, channel: str, frameSync: int = 2):
        self._logger.debug(f"Acquiring frame on channel {channel}")

        # ensure we get a fresh frame
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
                  target_focus_setpoint: Optional[float] = None) -> Optional[float]:
        """Perform autofocus using either hardware or software method.

        Args:
            minZ: Minimum Z position for autofocus (software mode only)
            maxZ: Maximum Z position for autofocus (software mode only)
            stepSize: Step size for autofocus scan (software mode only)
            illuminationChannel: Selected illumination channel for autofocus
            mode: "hardware" (fast, one-shot) or "software" (slow, Z-sweep)

        Returns:
            float: Best focus Z position, or None if autofocus failed
        """
        self._logger.debug(
            f"Performing autofocus (mode={mode}) with parameters "
            f"minZ={minZ}, maxZ={maxZ}, stepSize={stepSize}, channel={illuminationChannel}"
        )

        # Route to appropriate autofocus method
        if mode == "hardware":
            return self.autofocus_hardware(target_focus_setpoint=target_focus_setpoint,
                                           max_attempts=max_attempts,
                                           illuminationChannel=illuminationChannel)
        else:
            return self.autofocus_software(
                minZ=minZ,
                maxZ=maxZ,
                stepSize=stepSize,
                illuminationChannel=illuminationChannel
            )

    def autofocus_software(self, minZ: float=0, maxZ: float=0, stepSize: float=0, illuminationChannel: str=""):
        """Perform software-based autofocus using AutofocusController (Z-sweep).

        Args:
            minZ: Minimum Z position for autofocus (not used - uses rangez instead)
            maxZ: Maximum Z position for autofocus (not used - uses rangez instead)
            stepSize: Step size for autofocus scan
            illuminationChannel: Selected illumination channel for autofocus

        Returns:
            float: Best focus Z position, or None if autofocus failed
        """
        self._logger.debug("Performing software autofocus (Z-sweep)... with parameters minZ, maxZ, stepSize, illuminationChannel: %s, %s, %s, %s", minZ, maxZ, stepSize, illuminationChannel)

        # Get the autofocus controller
        autofocusController = self._master.getController('Autofocus')

        if autofocusController is None:
            self._logger.warning("AutofocusController not available - skipping autofocus")
            return None

        # Set illumination if specified
        if illuminationChannel and hasattr(self, '_master') and hasattr(self._master, 'lasersManager'):
            try:
                # Turn on the specified illumination channel for autofocus
                self._logger.debug(f"Setting illumination channel {illuminationChannel} for autofocus")
                # TODO: Set appropriate intensity - this would require getting current intensity or using a default
                # For now, we'll let the autofocus controller handle illumination
            except Exception as e:
                self._logger.warning(f"Failed to set illumination channel {illuminationChannel}: {e}")

        try:
            # Calculate range from min/max
            rangez = abs(maxZ - minZ) / 2.0 if maxZ > minZ else 50.0
            resolutionz = stepSize if stepSize > 0 else 10.0

            # Call autofocus directly - the method is already decorated with @APIExport
            #     def doAutofocusBackground(self, rangez:float=100, resolutionz:float=10, defocusz:float=0, axis:str=gAxis, tSettle:float=0.1, isDebug:bool=False, nGauss:int=7, nCropsize:int=2048, focusAlgorithm:str="LAPE", static_offset:float=0.0, twoStage:bool=False):
            result = autofocusController.doAutofocusBackground(
                rangez=rangez,
                resolutionz=resolutionz,
                defocusz=0,
                axis="Z",
                tSettle =0.1, # TODO: Implement via frontend parameters
                isDebug=False,
                nGauss=7,
                nCropsize=2048,
                focusAlgorithm="LAPE",
                static_offset=0.0,
                twoStage=False
            )

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
                      tPeriod:int=1, nTimes:int=1):
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
                    # Snake pattern
                    if iy % 2 == 1:
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


# Copyright (C) 2025 Benedict Diederich
