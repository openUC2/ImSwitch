from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import dirtools, initLogger, APIExport
import xarray as xr
from mikro_next.api.schema import (
    Image,
    Stage,
    from_array_like,
    create_stage,
    PartialAffineTransformationViewInput,
)
from arkitekt_next import model
from typing import Generator

try:
    from rekuest_next.agents.context import check_cancelled
except ImportError:
    def check_cancelled():
        pass
import os
import datetime
import tifffile as tif
import time
import numpy as np


@model
class Position:
    x: int
    y: int
    z: int


@model
class Point2D:
    x: float
    y: float


# ---------------------------------------------------------------------------
# Well-plate geometry loader
# ---------------------------------------------------------------------------

def _load_well_scan_config(labware_dir_name: str) -> dict:
    """Load well geometry from an openuc2 labware JSON file.

    Walks the openuc2 definitions directory, loads the first *.json found
    in *labware_dir_name*, and returns a dict mapping well_id to
    ``{center_x, center_y, width, height}`` **in µm**.

    The loader handles the mm → µm conversion; this function merely reshapes
    the resulting ``LabwareDefinition`` into the flat format expected by
    ``runWellTileScan``.
    """
    from pathlib import Path
    from imswitch.imcontrol.model.labware.loader import load_labware_from_file

    definitions_root = (
        Path(__file__).parent.parent.parent
        / "model" / "labware" / "definitions" / "openuc2"
    )
    labware_dir = definitions_root / labware_dir_name
    json_files = sorted(labware_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {labware_dir}")

    definition = load_labware_from_file(json_files[0])

    config: dict = {}
    for well_id, well in definition.wells.items():
        g = well.geometry
        if g.shape == "circle":
            w = h = 2.0 * g.radius
        else:
            w = g.width
            h = g.height
        config[well_id] = {
            "center_x": well.x,
            "center_y": well.y,
            "width":    w,
            "height":   h,
        }
    return config


def _load_well_scan_config_safe(labware_dir_name: str, fallback: dict) -> dict:
    """Like ``_load_well_scan_config`` but returns *fallback* on any error."""
    try:
        return _load_well_scan_config(labware_dir_name)
    except Exception as exc:
        import warnings
        warnings.warn(
            f"Could not load labware '{labware_dir_name}': {exc}. "
            "Using built-in fallback?",
            stacklevel=2,
        )
        return fallback


_PLATE_WELL_CONFIGS: dict = {
    "heidstar4": _load_well_scan_config_safe("slide_4x_histosample_heidstar", {}),
    "96well":    _load_well_scan_config_safe("corning_96_wellplate_360ul_flat", {}),
}

# =========================
# Controller
# =========================
class ArkitektController(ImConWidgetController):
    """
    Controller for the Arkitekt widget.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._logger.debug("Initializing")

        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        if len(allDetectorNames) == 0:
            return
        self.mDetector = self._master.detectorsManager[
            self._master.detectorsManager.getAllDeviceNames()[0]
        ]

        if not getattr(self._master, "arkitektManager", None):
            self._logger.warning("ArkitektManager unavailable; controller disabled.")
            return
        if not self._master.arkitektManager.is_enabled():
            self._logger.warning("Arkitekt not enabled; controller disabled.")
            return

        self.arkitekt_app = self._master.arkitektManager.get_arkitekt_app()
        if self.arkitekt_app is None:
            self._logger.warning("Arkitekt app unavailable; controller disabled.")
            return
        self._active_focus_map = None  # set by runWellTileScan; read by runTileScan
        self._pending_well_corner: tuple[float, float] | None = None  # first corner for two-step well definition

        # Per-instance plate config — starts as a copy of module defaults so
        # user-defined bounds don't bleed between sessions or controller instances.
        import copy
        self._plate_well_configs = copy.deepcopy(_PLATE_WELL_CONFIGS)
        self._load_well_overrides()

        self.arkitekt_app.register(self.runTileScan)
        self.arkitekt_app.register(self.defineWellBounds)
        self.arkitekt_app.register(self.saveFirstWellCorner)
        self.arkitekt_app.register(self.saveSecondWellCorner)
        self.arkitekt_app.register(self.previewWell)
        self.arkitekt_app.register(self.runWellTileScan)
        self.arkitekt_app.register(self.goToPosition)
        self.arkitekt_app.register(self.acquireFrame)
        self.arkitekt_app.register(self.getStagePosition)
        self.arkitekt_app.register(self.homeStageAxis)
        self.arkitekt_app.register(self.setLaserState)
        self.arkitekt_app.register(self.moveStage)

        self.arkitekt_app.run_detached()

    def moveToSampleLoadingPosition(
        self, speed: float = 10000, is_blocking: bool = True
    ):
        """Move to sample loading position."""
        positionerNames = self._master.positionersManager.getAllDeviceNames()
        if len(positionerNames) == 0:
            self._logger.warning(
                "No positioners available to move to sample loading position."
            )
            return
        positionerName = positionerNames[0]
        self._logger.debug(
            f"Moving to sample loading position for positioner {positionerName}"
        )
        self._master.positionersManager[positionerName].moveToSampleLoadingPosition(
            speed=speed, is_blocking=is_blocking
        )
        
    
    @APIExport(runOnUIThread=False)
    def getStagePosition(self, positionerName: str | None = None) -> Position:
        """Get current stage position."""
        if positionerName is None:
            positionerNames = self._master.positionersManager.getAllDeviceNames()
            if len(positionerNames) == 0:
                self._logger.warning("No positioners available to get stage position.")
                return None
            positionerName = positionerNames[0]
        mStage = self._master.positionersManager[positionerName]
        currentPositions = mStage.getPosition()
        return Position(
            x=int(currentPositions["X"]),
            y=int(currentPositions["Y"]),
            z=int(currentPositions["Z"]),
        )
    
    @APIExport(runOnUIThread=False)
    def homeStageAxis(self, positionerName: str | None = None, axis: str = "ZXY", is_blocking: bool = True):
        """Home one or more stage axes in given order.
        """
        if positionerName is None:
            positionerNames = self._master.positionersManager.getAllDeviceNames()
            if len(positionerNames) == 0:
                self._logger.warning("No positioners available to home stage axis.")
                return None
            positionerName = positionerNames[0]
        mStage = self._master.positionersManager[positionerName]
        _home_dispatch = {
            "X": mStage.home_x,
            "Y": mStage.home_y,
            "Z": mStage.home_z,
        }
        for ax in axis.upper():
            home_fn = _home_dispatch.get(ax)
            if home_fn is None:
                self._logger.warning(f"Unknown axis '{ax}' – skipping")
                continue
            self._logger.debug(f"Homing axis {ax}")
            home_fn(isBlocking=is_blocking)

        
    @APIExport(runOnUIThread=False) 
    def setLaserState(self, laserName: str, isActive: bool, value: float = 100):
        """Set laser state."""
        if laserName not in self._master.lasersManager.getAllDeviceNames():
            self._logger.warning(f"Laser {laserName} not available to set state.")
            return
        mLaser = self._master.lasersManager[laserName]
        mLaser.setEnabled(isActive)
        mLaser.setValue(value)
        
    @APIExport(runOnUIThread=False)
    def moveStage(self, positionerName: str | None = None, axis: str = "X", distance: float = 100, is_absolute: bool = True, is_blocking: bool = True, speed: float = 10000):
        """Move stage."""
        if positionerName is None:
            positionerNames = self._master.positionersManager.getAllDeviceNames()
            if len(positionerNames) == 0:
                self._logger.warning("No positioners available to move stage.")
                return None
            positionerName = positionerNames[0]
        mStage = self._master.positionersManager[positionerName]
        mStage.move(value=distance, axis=axis, is_absolute=is_absolute, is_blocking=is_blocking, speed=speed)  
    
    
    @APIExport(runOnUIThread=False)
    def acquireFrame(self, frameSync: int = 3) -> Generator[Image, None, None]:
        """Acquire a single frame

        Args:
            frameSync (int): Number of frames to skip to ensure a fresh frame is acquired.
                Default is 3.
        Returns:
            numpy.ndarray: Acquired image frame as a NumPy array.
        """
        # Start acquisition if the detector is not already running
        if not self.mDetector._running:
            self.mDetector.startAcquisition()
            time.sleep(0.3)  # give the camera a moment to deliver its first frame

        timeoutFrameRequest = 3  # seconds
        cTime = time.time()
        lastFrameNumber = -1
        mFrame = None
        currentFrameNumber = -1
        while True:
            mFrame, currentFrameNumber = self.mDetector.getLatestFrame(returnFrameNumber=True)
            if lastFrameNumber == -1:
                lastFrameNumber = currentFrameNumber
            if time.time() - cTime > timeoutFrameRequest:
                if mFrame is None:
                    mFrame = self.mDetector.getLatestFrame(returnFrameNumber=False)
                break
            if currentFrameNumber <= lastFrameNumber + frameSync:
                time.sleep(0.01)
            else:
                break

        if mFrame is None:
            self._logger.error("acquireFrame: detector returned no frame within timeout")
            return

        if len(mFrame.shape) == 2:
            mFrame = np.expand_dims(mFrame, axis=-1)
        image = from_array_like(
            xr.DataArray(
                mFrame,
                dims=["y", "x", "c"],
                attrs={"frame_number": currentFrameNumber},
            ),
            name=f"Frame_{currentFrameNumber}",
        )
        yield image


    @APIExport(runOnUIThread=False)
    def goToPosition(
        self,
        x_micrometer: float,
        y_micrometer: float,
        positionerName: str | None = None,
        speed: float = 10000,
        is_blocking: bool = True,
        t_settle: float = 0.2,
    ) -> None:
        """Move the stage to the specified X,Y position.

        Args:
            x_micrometer (float): Target X position in micrometers.
            y_micrometer (float): Target Y position in micrometers.
            positionerName (str | None): Name of the positioner to use. If None,
                the first available positioner will be used.
            speed (float): Speed of the positioner movement (units per second).
                Default is 10000.
            is_blocking (bool): Whether to wait for movement completion before returning.
                Default is True.
            t_settle (float): Settling time in seconds to wait after movement completes.
                Only used if is_blocking is True. Default is 0.2 seconds.

        Example:
            >>> # Move to position (5000, 3000) micrometers
            >>> goToPosition(x_micrometer=5000, y_micrometer=3000)
            
            >>> # Move with custom speed and non-blocking
            >>> goToPosition(
            ...     x_micrometer=5000,
            ...     y_micrometer=3000,
            ...     speed=5000,
            ...     is_blocking=False
            ... )
        """
        # Get positioner
        if positionerName is None:
            positionerNames = self._master.positionersManager.getAllDeviceNames()
            if len(positionerNames) == 0:
                self._logger.error("No positioners available for positioning")
                return
            positionerName = positionerNames[0]

        mPositioner = self._master.positionersManager[positionerName]

        self._logger.debug(
            f"Moving positioner {positionerName} to ({x_micrometer}, {y_micrometer}) µm"
        )

        # Move to position
        mPositioner.move(
            value=(x_micrometer, y_micrometer),
            axis="XY",
            is_absolute=True,
            is_blocking=is_blocking,
            speed=(speed, speed)
        )

        # Wait for settling if blocking
        if is_blocking:
            time.sleep(t_settle)
            self._logger.debug(
                f"Position reached and settled at ({x_micrometer}, {y_micrometer}) µm"
            )

    @APIExport(runOnUIThread=False)
    def runTileScanInThread(self,
        center_x_micrometer: float | None = None,
        center_y_micrometer: float | None = None,
        range_x_micrometer: float = 100,
        range_y_micrometer: float = 100,
        step_x_micrometer: float | None = None,
        step_y_micrometer: float | None = None,
        overlap_percent: float = 10.0,
        illumination_channel: str | None = None,
        illumination_intensity: float = 100,
        exposure_time: float | None = None,
        gain: float | None = None,
        speed: float = 10000,
        positionerName: str | None = None,
        performAutofocus: bool = False,
        autofocus_range: float = 100,
        autofocus_resolution: float = 10,
        autofocus_illumination_channel: str | None = None,
        objective_id: int | None = None):
        """Run tile scan in a separate thread."""
        import threading

        mThread = threading.Thread(
            target=self.runTileScan,
            kwargs={
                'center_x_micrometer': center_x_micrometer,
                'center_y_micrometer': center_y_micrometer,
                'range_x_micrometer': range_x_micrometer,
                'range_y_micrometer': range_y_micrometer,
                'step_x_micrometer': step_x_micrometer,
                'step_y_micrometer': step_y_micrometer,
                'overlap_percent': overlap_percent,
                'illumination_channel': illumination_channel,
                'illumination_intensity': illumination_intensity,
                'exposure_time': exposure_time,
                'gain': gain,
                'speed': speed,
                'positionerName': positionerName,
                'performAutofocus': performAutofocus,
                'autofocus_range': autofocus_range,
                'autofocus_resolution': autofocus_resolution,
                'autofocus_illumination_channel': autofocus_illumination_channel,
                'objective_id': objective_id
            }
        )
        mThread.start()
        return 1

    def acquire_frame(self, frameSync: int = 3):

        # ensure we get a fresh frame; scale timeout to exposure so long
        # exposures do not prematurely abort
        try:
            exposure_s = self.mDetector.getLatestFrame.__self__._camera.exposure_time / 1e6
        except Exception:
            exposure_s = 0.1
        timeoutFrameRequest = max(1.0, (frameSync + 2) * exposure_s + 0.5)
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

    @APIExport(runOnUIThread=False)
    def runTileScan(
        self,
        center_x_micrometer: float | None = None,
        center_y_micrometer: float | None = None,
        range_x_micrometer: float = 5000,
        range_y_micrometer: float = 5000,
        step_x_micrometer: float | None = None,
        step_y_micrometer: float | None = None,
        overlap_percent: float = 10.0,
        illumination_channel: str | None = "LED",
        illumination_intensity: float = 1024,
        exposure_time: float | None = None,
        gain: float | None = None,
        speed: float = 10000,
        positionerName: str | None = None,
        performAutofocus: bool = False,
        autofocus_range: float = 100,
        autofocus_resolution: float = 10,
        autofocus_illumination_channel: str | None = None,
        objective_id: int | None = None,
        t_settle: float = 0.2,
    ) -> Stage:
        """Run a tile scan with enhanced control over imaging parameters.

        Runs a tile scan by moving the specified positioner in a grid pattern centered
        at the given coordinates, capturing images at each position with specified
        illumination and camera settings, and yielding the images with appropriate
        affine transformations for stitching.

        The step size is automatically calculated based on the current objective's
        field of view and the specified overlap percentage, unless explicitly provided.

        Args:
            center_x_micrometer (float | None): Center position in the X direction (micrometers).
                If None, uses current X position.
            center_y_micrometer (float | None): Center position in the Y direction (micrometers).
                If None, uses current Y position.
            range_x_micrometer (float): Total range to scan in the X direction (micrometers).
            range_y_micrometer (float): Total range to scan in the Y direction (micrometers).
            step_x_micrometer (float | None): Step size in the X direction (micrometers).
                If None, automatically calculated based on objective FOV and overlap.
            step_y_micrometer (float | None): Step size in the Y direction (micrometers).
                If None, automatically calculated based on objective FOV and overlap.
            overlap_percent (float): Percentage of overlap between adjacent tiles (0-100).
                Only used if step_x/y_micrometer are None. Default is 10%.
            illumination_channel (str | None): Name of the illumination source to use.
                If None, uses current illumination settings.
            illumination_intensity (float): Intensity value for the illumination source (0-100).
            exposure_time (float | None): Exposure time in milliseconds. If None, uses current setting.
            gain (float | None): Camera gain value. If None, uses current setting.
            speed (float): Speed of the positioner movement (units per second).
            positionerName (str | None): Name of the positioner to use. If None,
                the first available positioner will be used.
            performAutofocus (bool): Whether to perform autofocus at each tile position.
            autofocus_range (float): Range for autofocus scan in Z direction (micrometers).
            autofocus_resolution (float): Step size for autofocus scan (micrometers).
            autofocus_illumination_channel (str | None): Illumination channel to use for autofocus.
                If None, uses the same as illumination_channel.
            objective_id (int | None): ID of the objective to use (0 or 1).
                If specified, the objective will be moved to this position before scanning
                and magnification will be retrieved from ObjectiveManager. If None, uses current objective.

        Yields:
            Image: Captured image with affine transformation for stitching.

        

        Example:
            >>> # Scan with automatic step size and specific objective
            >>> for image in runTileScan(
            ...     center_x_micrometer=5000,
            ...     center_y_micrometer=5000,
            ...     range_x_micrometer=1000,
            ...     range_y_micrometer=1000,
            ...     overlap_percent=10,  # 10% overlap
            ...     illumination_channel="LED",
            ...     illumination_intensity=50,
            ...     exposure_time=100,
            ...     objective_id=1,  # Switch to objective 1 (0-based indexing)
            ...     performAutofocus=True
            ... ):
            ...     # Process each image
            ...     pass

            >>> # Or specify step size manually
            >>> for image in runTileScan(
            ...     center_x_micrometer=5000,
            ...     center_y_micrometer=5000,
            ...     range_x_micrometer=1000,
            ...     range_y_micrometer=1000,
            ...     step_x_micrometer=200,
            ...     step_y_micrometer=200,
            ...     illumination_channel="LED",
            ...     objective_id=0  # Switch to objective 0
            ... ):
            ...     pass
        """
        # Get objective controller (used for FOV, magnification, and optional switching)
        objective_controller = None
        try:
            objective_controller = self._master.getController('Objective')
        except Exception:
            pass

        # Handle objective switching if specified
        objective_magnification = None
        if objective_id is not None:
            if objective_controller is not None:
                try:
                    self._logger.debug(f"Moving to objective ID: {objective_id}")
                    objective_controller.moveToObjective(objective_id)
                    self._logger.debug(f"Successfully moved to objective ID: {objective_id}")
                except Exception as e:
                    self._logger.error(f"Failed to move to objective ID {objective_id}: {e}")
            else:
                self._logger.warning("ObjectiveController not available, cannot switch objective")

        # Calculate step sizes based on objective FOV if not provided
        if step_x_micrometer is None or step_y_micrometer is None:
            if objective_controller is not None:
                try:
                    fov = objective_controller._getCurrentFOV()
                except Exception:
                    fov = None
                if fov is not None:
                    fov_x, fov_y = fov
                    overlap_factor = 1.0 - (overlap_percent / 100.0)
                    if step_x_micrometer is None:
                        step_x_micrometer = fov_x * overlap_factor
                        self._logger.debug(f"Calculated step_x from FOV: {step_x_micrometer:.2f} µm "
                                         f"(FOV: {fov_x:.2f} µm, overlap: {overlap_percent}%)")
                    if step_y_micrometer is None:
                        step_y_micrometer = fov_y * overlap_factor
                        self._logger.debug(f"Calculated step_y from FOV: {step_y_micrometer:.2f} µm "
                                         f"(FOV: {fov_y:.2f} µm, overlap: {overlap_percent}%)")
                else:
                    self._logger.warning("Could not get FOV from ObjectiveController - no detector dimensions set?")
            else:
                self._logger.warning("ObjectiveController not available for automatic step size calculation")

            # Fallback to default values if still None
            if step_x_micrometer is None:
                step_x_micrometer = 100.0
                self._logger.warning(f"Using default step_x_micrometer: {step_x_micrometer} µm")
            if step_y_micrometer is None:
                step_y_micrometer = 100.0
                self._logger.warning(f"Using default step_y_micrometer: {step_y_micrometer} µm")

        # Get objective magnification after potential switch
        if objective_controller is not None:
            try:
                objective_magnification = objective_controller._getCurrentMagnification()
                if objective_magnification is not None:
                    current_objective_slot = objective_controller.getCurrentObjective()
                    self._logger.debug(
                        f"Using objective slot {current_objective_slot} "
                        f"with magnification: {objective_magnification}x"
                    )
            except Exception:
                pass

        # Get positioner
        if positionerName is None:
            positionerNames = self._master.positionersManager.getAllDeviceNames()
            if len(positionerNames) == 0:
                self._logger.error("No positioners available for tile scan")
                return
            positionerName = positionerNames[0]

        mPositioner = self._master.positionersManager[positionerName]

        # Get current position and use as center if not provided
        current_pos = mPositioner.getPosition()
        if center_x_micrometer is None:
            center_x_micrometer = current_pos.get("X", 0)
            self._logger.debug(f"Using current X position as center: {center_x_micrometer}")
        if center_y_micrometer is None:
            center_y_micrometer = current_pos.get("Y", 0)
            self._logger.debug(f"Using current Y position as center: {center_y_micrometer}")

        # Calculate start positions from center and range
        xStart = center_x_micrometer - range_x_micrometer / 2
        yStart = center_y_micrometer - range_y_micrometer / 2

        # Use the new parameter names internally
        xRange = int(range_x_micrometer)
        yRange = int(range_y_micrometer)
        xStep = int(step_x_micrometer)
        yStep = int(step_y_micrometer)

        self._logger.debug(f"Starting tile scan for positioner {positionerName}")
        self._logger.debug(f"Scan parameters: center=({center_x_micrometer}, {center_y_micrometer}), "
                         f"range=({range_x_micrometer}, {range_y_micrometer}), "
                         f"step=({step_x_micrometer}, {step_y_micrometer})")

        # Set up camera parameters if specified
        if exposure_time is not None and exposure_time > 0:
            self._commChannel.sharedAttrs.sigAttributeSet(
                ['Detector', None, None, "exposureTime"], exposure_time
            )
            self._logger.debug(f"Setting exposure time to {exposure_time}ms")

        if gain is not None and gain >= 0:
            self._commChannel.sharedAttrs.sigAttributeSet(
                ['Detector', None, None, "gain"], gain
            )
            self._logger.debug(f"Setting gain to {gain}")

        # Set up illumination if specified
        original_illumination_state = None
        if illumination_channel is not None:
            try:
                # Store original state to restore later
                laser_manager = self._master.lasersManager
                if illumination_channel in laser_manager.getAllDeviceNames():
                    laser = laser_manager[illumination_channel]
                    original_illumination_state = {
                        'enabled': laser.enabled,
                        'value': laser.power if hasattr(laser, 'power') else 0
                    }
                    # Set illumination
                    laser.setValue(illumination_intensity)
                    if laser.enabled == 0:
                        laser.setEnabled(1)
                    self._logger.debug(f"Set illumination {illumination_channel} to {illumination_intensity}")
            except Exception as e:
                self._logger.warning(f"Failed to set illumination channel {illumination_channel}: {e}")

        # Get autofocus controller if needed
        autofocusController = None
        if performAutofocus:
            autofocusController = self._master.getController('Autofocus')
            if autofocusController is None:
                self._logger.warning("Autofocus requested but AutofocusController not available")
                performAutofocus = False

            # Set autofocus illumination if different from main illumination
            if autofocus_illumination_channel and autofocus_illumination_channel != illumination_channel:
                # TODO: Implement temporary illumination switching for autofocus
                self._logger.debug(f"Using autofocus illumination channel: {autofocus_illumination_channel}")

        # Start camera if not running
        if not self.mDetector._running:
            self.mDetector.startAcquisition()

        # Create stage for stitching metadata
        stage = create_stage(name=f"Tile Scan Stage - {center_x_micrometer},{center_y_micrometer}")

        # Create directory for saving tiles if stage creation failed
        save_dir = None
        metadata_list = []

        # Get positioner (moved earlier to access before calculating center)
        # Already retrieved above when checking center positions

        # Get current position
        current_pos = mPositioner.getPosition()
        current_x = current_pos.get("X", xStart)
        current_y = current_pos.get("Y", yStart)

        self._logger.debug(f"Starting scan from position: ({current_x}, {current_y})")

        # Calculate number of tiles in each direction
        num_tiles_x = int(xRange / xStep) + 1
        num_tiles_y = int(yRange / yStep) + 1
        total_tiles = num_tiles_x * num_tiles_y

        self._logger.info(f"Starting snake scan: {num_tiles_x}x{num_tiles_y} tiles "
                         f"({total_tiles} total), step: ({xStep}, {yStep}) µm, "
                         f"overlap: {overlap_percent}%")

        # Perform scan in snake pattern
        tile_count = 0

        for iy in range(num_tiles_y):
            for ix in range(num_tiles_x):
                check_cancelled()
                # Snake pattern: reverse x direction on odd rows
                if iy % 2 == 1:
                    # Odd row: scan right to left
                    actual_ix = num_tiles_x - 1 - ix
                else:
                    # Even row: scan left to right
                    actual_ix = ix

                # Calculate absolute position
                actual_x = xStart + actual_ix * xStep
                actual_y = yStart + iy * yStep

                # Move to position
                mPositioner.move(
                    value=(actual_x, actual_y),
                    axis="XY",
                    is_absolute=True,
                    is_blocking=True,
                    speed=(speed, speed)
                )
                # Wait for settling
                time.sleep(t_settle)

                # Apply focus-map Z correction (set by runWellTileScan before delegating here)
                if self._active_focus_map is not None and self._active_focus_map.is_fitted:
                    try:
                        z_target = self._active_focus_map.interpolate(actual_x, actual_y)
                        mPositioner.move(value=z_target, axis="Z", is_absolute=True, is_blocking=True)
                    except Exception as _fme:
                        self._logger.warning(
                            f"Focus map Z correction failed at ({actual_x:.1f}, {actual_y:.1f}): {_fme}"
                        )

                # Perform autofocus at this position if requested
                if performAutofocus and autofocusController is not None:
                    try:
                        autofocusController.autoFocus(
                            rangez=autofocus_range,
                            resolutionz=autofocus_resolution,
                            defocusz=0
                        )
                        self._logger.debug(f"Autofocus completed at tile ({actual_ix}, {iy})")
                    except Exception as e:
                        self._logger.error(f"Autofocus failed at tile ({actual_ix}, {iy}): {e}")

                # Capture image
                numpy_array = self.acquire_frame(frameSync=2)

                # Create affine transformation matrix for stitching
                affine_matrix_four_d = [
                    [1, 0, 0, actual_x],
                    [0, 1, 0, actual_y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]

                # Create image with metadata
                actual_id = iy * num_tiles_x + ix
                image_name = f"Tile_{actual_id}_{actual_ix:03d}_{iy:03d}_x{actual_x:.1f}_y{actual_y:.1f}"
                if illumination_channel:
                    image_name += f"_{illumination_channel}"

                # ensure we have a channel axis
                if len(numpy_array.shape) == 2:
                    numpy_array = np.expand_dims(numpy_array, axis=-1)

                if stage is not None:
                    print("Image has the following properties: ", numpy_array.shape)
                    image = from_array_like(
                        xr.DataArray(
                            numpy_array,
                            dims=["y", "x", "c"],
                        ),
                        transformation_views=[
                            PartialAffineTransformationViewInput(
                                affineMatrix=affine_matrix_four_d,
                                stage=stage
                            )
                        ],
                        name=image_name,
                    )
                else:
                    # Save as individual TIF files with JSON metadata

                    # Create metadata dictionary
                    tile_metadata = {
                        "tile_index_x": actual_ix,
                        "tile_index_y": iy,
                        "position_x_um": actual_x,
                        "position_y_um": actual_y,
                        "center_x_um": center_x_micrometer,
                        "center_y_um": center_y_micrometer,
                        "illumination_channel": illumination_channel or "unknown",
                        "illumination_intensity": illumination_intensity,
                        "exposure_time_ms": exposure_time,
                        "gain": gain,
                        "objective_magnification": objective_magnification,
                        "affine_matrix": affine_matrix_four_d,
                        "image_shape": list(numpy_array.shape),
                        "dtype": str(numpy_array.dtype),
                    }

                    # Add to metadata list
                    metadata_list.append(tile_metadata)

                    # Save TIF file
                    tif_filename = f"{image_name}.tif"
                    tif_path = os.path.join(save_dir, tif_filename)
                    tif.imwrite(tif_path, numpy_array)

                    self._logger.debug(f"Saved tile to {tif_path}")

                    # Create a dummy image object for consistency (won't be yielded)
                    image = None

                tile_count += 1
                self._logger.debug(f"Captured tile {tile_count}/{total_tiles} at ({actual_x}, {actual_y})")

        # Save metadata JSON file if we were saving individual TIFs
        if save_dir is not None and metadata_list:
            import json

            # Create comprehensive scan metadata
            scan_metadata = {
                "scan_info": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "center_x_um": center_x_micrometer,
                    "center_y_um": center_y_micrometer,
                    "range_x_um": range_x_micrometer,
                    "range_y_um": range_y_micrometer,
                    "step_x_um": step_x_micrometer,
                    "step_y_um": step_y_micrometer,
                    "overlap_percent": overlap_percent,
                    "num_tiles_x": num_tiles_x,
                    "num_tiles_y": num_tiles_y,
                    "total_tiles": total_tiles,
                    "positioner": positionerName,
                    "illumination_channel": illumination_channel,
                    "illumination_intensity": illumination_intensity,
                    "exposure_time_ms": exposure_time,
                    "gain": gain,
                    "objective_magnification": objective_magnification,
                    "autofocus_enabled": performAutofocus,
                },
                "tiles": metadata_list
            }

            # Save metadata JSON
            metadata_path = os.path.join(save_dir, "scan_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(scan_metadata, f, indent=2)

            self._logger.info(f"Saved scan metadata to {metadata_path}")

        # move back to starting position
        mPositioner.move(
            value=(current_x, current_y),
            axis="XY",
            is_absolute=True,
            is_blocking=False,
            speed=(speed, speed)
        )
        # Restore original illumination state if it was changed
        if original_illumination_state is not None and illumination_channel is not None:
            try:
                laser = self._master.lasersManager[illumination_channel]
                laser.setValue(original_illumination_state['value'])
                laser.setEnabled(original_illumination_state['enabled'])
                self._logger.debug(f"Restored illumination {illumination_channel} to original state")
            except Exception as e:
                self._logger.warning(f"Failed to restore illumination state: {e}")

        self._logger.info(f"Tile scan completed: {tile_count} tiles captured")

        return stage

    # ------------------------------------------------------------------
    # Well-boundary definition and persistence
    # ------------------------------------------------------------------

    def _get_well_overrides_path(self):
        from pathlib import Path
        data_path = dirtools.UserFileDirs.getValidatedDataPath()
        return Path(data_path) / "well_plate_overrides.json"

    def _load_well_overrides(self):
        import json
        path = self._get_well_overrides_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                overrides = json.load(f)
            for plate_type, wells in overrides.items():
                if plate_type not in self._plate_well_configs:
                    self._plate_well_configs[plate_type] = {}
                for well_id, geom in wells.items():
                    self._plate_well_configs[plate_type][well_id] = geom
            self._logger.info(f"Loaded well overrides from {path}")
        except Exception as e:
            self._logger.warning(f"Could not load well overrides: {e}")

    def _save_well_overrides(self):
        import json
        # Save only wells that differ from the module-level defaults.
        overrides: dict = {}
        for plate_type, wells in self._plate_well_configs.items():
            default_wells = _PLATE_WELL_CONFIGS.get(plate_type, {})
            for well_id, geom in wells.items():
                if geom != default_wells.get(well_id):
                    overrides.setdefault(plate_type, {})[well_id] = geom
        path = self._get_well_overrides_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(overrides, f, indent=2)
            self._logger.info(f"Saved well overrides to {path}")
        except Exception as e:
            self._logger.warning(f"Could not save well overrides: {e}")

    @APIExport(runOnUIThread=False)
    def saveFirstWellCorner(self, positionerName: str | None = None) -> Position:
        """Save current stage XY position as first corner of a well rectangle.

        Returns:
            Position: The saved first corner position.
        """
        if positionerName is None:
            names = self._master.positionersManager.getAllDeviceNames()
            if not names:
                self._logger.error("No positioners available to save first well corner.")
                return None
            positionerName = names[0]

        pos = self._master.positionersManager[positionerName].getPosition()
        x, y = pos["X"], pos["Y"]
        self._pending_well_corner = (x, y)
        self._logger.info(f"saveFirstWellCorner: stored ({x:.1f}, {y:.1f}) µm — now move to the opposite corner")
        return Position(x=int(x), y=int(y), z=int(pos.get("Z", 0)))

    @APIExport(runOnUIThread=False)
    def saveSecondWellCorner(
        self,
        well_id: str,
        plate_type: str = "heidstar4",
        positionerName: str | None = None,
    ) -> None:
        """Save current stage XY position as second corner and commit the well bounds.

        Args:
            well_id: Well label to define, e.g. "A1".
            plate_type: Plate type to update. Default "heidstar4".
            positionerName: Positioner name, or None for first available.
        """
        if self._pending_well_corner is None:
            self._logger.error(
                "saveSecondWellCorner: no first corner saved — call saveFirstWellCorner first."
            )
            return

        if positionerName is None:
            names = self._master.positionersManager.getAllDeviceNames()
            if not names:
                self._logger.error("No positioners available to save second well corner.")
                return
            positionerName = names[0]

        pos = self._master.positionersManager[positionerName].getPosition()
        x2, y2 = pos["X"], pos["Y"]
        x1, y1 = self._pending_well_corner
        self._pending_well_corner = None

        min_corner = Point2D(x=min(x1, x2), y=min(y1, y2))
        max_corner = Point2D(x=max(x1, x2), y=max(y1, y2))

        self._logger.info(
            f"saveSecondWellCorner: committing {plate_type}/{well_id} "
            f"corners ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f}) µm"
        )
        self.defineWellBounds(
            well_id=well_id,
            min_corner=min_corner,
            max_corner=max_corner,
            plate_type=plate_type,
        )

    @APIExport(runOnUIThread=False)
    def defineWellBounds(
        self,
        well_id: str,
        min_corner: Point2D,
        max_corner: Point2D,
        plate_type: str = "heidstar4",
    ) -> None:
        """Define a custom rectangular boundary for a well.

        Args:
            well_id: Well label to override, e.g. "A1".
            min_corner: Lower-left corner of the rectangle (x, y in µm).
            max_corner: Upper-right corner of the rectangle (x, y in µm).
            plate_type: Plate type to update. Default "heidstar4".
        """
        if plate_type not in self._plate_well_configs:
            self._logger.error(
                f"Unknown plate type '{plate_type}'. "
                f"Supported: {list(self._plate_well_configs)}"
            )
            return

        well_id_upper = well_id.strip().upper()
        center_x = (min_corner.x + max_corner.x) / 2.0
        center_y = (min_corner.y + max_corner.y) / 2.0
        width    = abs(max_corner.x - min_corner.x)
        height   = abs(max_corner.y - min_corner.y)

        self._plate_well_configs[plate_type][well_id_upper] = {
            "center_x": center_x,
            "center_y": center_y,
            "width":    width,
            "height":   height,
        }
        self._logger.info(
            f"defineWellBounds: {plate_type}/{well_id_upper} → "
            f"center=({center_x:.1f}, {center_y:.1f}) µm, "
            f"size={width:.1f}x{height:.1f} µm"
        )
        self._save_well_overrides()

    # ------------------------------------------------------------------
    # Well-scan helpers
    # ------------------------------------------------------------------

    def _read_current_imaging_settings(self):
        """Return (illumination_channel, illumination_intensity, exposure_time, gain)
        from the current live-view state.  Any value may be None if not readable."""
        illumination_channel = None
        illumination_intensity = None
        try:
            laser_manager = self._master.lasersManager
            for name in laser_manager.getAllDeviceNames():
                laser = laser_manager[name]
                if laser.enabled:
                    illumination_channel = name
                    illumination_intensity = laser.power if hasattr(laser, "power") else None
                    break
        except Exception as e:
            self._logger.warning(f"Could not read laser settings: {e}")

        exposure_time = None
        gain = None
        try:
            params = self.mDetector.parameters if hasattr(self.mDetector, "parameters") else {}
            for key in ("Real exposure time", "Set exposure time", "Exposure time"):
                if key in params:
                    exposure_time = float(params[key].value)
                    break
            for key in ("Gain", "gain"):
                if key in params:
                    gain = float(params[key].value)
                    break
        except Exception as e:
            self._logger.warning(f"Could not read detector settings: {e}")

        return illumination_channel, illumination_intensity, exposure_time, gain

    def _build_focus_map_for_well(
        self,
        autofocusController,
        mPositioner,
        well_bounds: dict,
        grid_rows: int,
        grid_cols: int,
        autofocus_range: float,
        autofocus_resolution: float,
        speed: float,
        t_settle: float,
        autofocus_cropsize: int = 512,
        autofocus_algorithm: str = "LAPE",
        autofocus_settle_time: float = 0.1,
    ):
        """Measure Z at a grid of positions within a well and return a fitted FocusMap.

        Moves the stage in XY to each grid point, runs autofocus to find Z,
        then fits a surface interpolator over all measured points.
        Returns None if fitting fails.
        """
        from imswitch.imcontrol.model.focus_map import FocusMap

        focus_map = FocusMap(group_id="well_scan", method="spline", logger=self._logger)
        grid = FocusMap.generate_grid(bounds=well_bounds, rows=grid_rows, cols=grid_cols)

        self._logger.info(
            f"Focus mapping: {len(grid)} points over "
            f"X=[{well_bounds['minX']:.0f}, {well_bounds['maxX']:.0f}] "
            f"Y=[{well_bounds['minY']:.0f}, {well_bounds['maxY']:.0f}] µm"
        )

        for gx, gy in grid:
            check_cancelled()
            mPositioner.move(
                value=(gx, gy), axis="XY",
                is_absolute=True, is_blocking=True, speed=(speed, speed),
            )
            time.sleep(t_settle)

            autofocusController.autoFocus(
                rangez=autofocus_range,
                resolutionz=autofocus_resolution,
                defocusz=0,
                tSettle=autofocus_settle_time,
                isDebug=False,
                nCropsize=autofocus_cropsize,
                focusAlgorithm=autofocus_algorithm,
            )
            # Wait for the autofocus thread to finish (more reliable than polling)
            af_thread = getattr(autofocusController, '_AutofocusThead', None)
            if af_thread is not None and af_thread.is_alive():
                af_thread.join(timeout=120)
            else:
                # Fallback poll if thread attribute is unavailable
                t0 = time.time()
                while getattr(autofocusController, 'isAutofusRunning', False):
                    check_cancelled()
                    time.sleep(0.1)
                    if time.time() - t0 > 60:
                        self._logger.warning("Autofocus timed out during focus mapping")
                        break

            pos = mPositioner.getPosition()
            z = pos.get("Z", pos.get("z", None))
            if z is None:
                self._logger.warning(f"Focus map: could not read Z at ({gx:.0f}, {gy:.0f}), using 0")
                z = 0.0
            focus_map.add_point(gx, gy, z)
            self._logger.debug(f"Focus map: ({gx:.0f}, {gy:.0f}) → Z={z:.2f} µm")

        if focus_map.n_points == 0:
            self._logger.error("Focus map: no points collected")
            return None

        try:
            focus_map.fit()
            s = focus_map.fit_stats
            self._logger.info(
                f"Focus map fitted: method={s.method}, MAE={s.mean_abs_error:.2f} µm, "
                f"n={s.n_points}"
            )
            return focus_map
        except Exception as e:
            self._logger.error(f"Focus map fitting failed: {e}")
            return None

    @APIExport(runOnUIThread=False)
    def previewWell(
        self,
        well_id: str = "A1",
        plate_type: str = "heidstar4",
        perform_autofocus: bool = True,
        autofocus_range: float = 100,
        autofocus_resolution: float = 10,
        speed: float = 10000,
        t_settle: float = 0.2,
        positionerName: str | None = None,
    ) -> Generator[Image, None, None]:
        """Move to a well center, run autofocus, and capture one frame.

        Args:
            well_id: Well label, e.g. "A1"–"A4" for heidstar4.
            plate_type: Plate type. "heidstar4" (default) or "96well".
            perform_autofocus: Run autofocus before capturing. Default True.
            autofocus_range: Z scan half-range (µm). Default 100.
            autofocus_resolution: Z step size for autofocus (µm). Default 10.
            speed: Stage movement speed (µm/s). Default 10000.
            t_settle: Settling time after stage move (s). Default 0.2.
            positionerName: Positioner name, or None for first available.

        Yields:
            Image: Single frame captured at the well center.
        """
        plate_wells = self._plate_well_configs.get(plate_type)
        if plate_wells is None:
            self._logger.error(
                f"Unknown plate type '{plate_type}'. Supported: {list(_PLATE_WELL_CONFIGS)}"
            )
            return

        well_geom = plate_wells.get(well_id.strip().upper())
        if well_geom is None:
            self._logger.error(
                f"Unknown well '{well_id}' for plate '{plate_type}'. "
                f"Valid wells: {list(plate_wells)}"
            )
            return

        if positionerName is None:
            names = self._master.positionersManager.getAllDeviceNames()
            if not names:
                self._logger.error("No positioners available for well preview")
                return
            positionerName = names[0]
        mPositioner = self._master.positionersManager[positionerName]

        center_x = well_geom["center_x"]
        center_y = well_geom["center_y"]
        self._logger.info(f"Preview: moving to well {well_id} center ({center_x}, {center_y}) µm")

        mPositioner.move(
            value=(center_x, center_y),
            axis="XY",
            is_absolute=True,
            is_blocking=True,
            speed=(speed, speed),
        )
        time.sleep(t_settle)

        if perform_autofocus:
            autofocusController = None
            try:
                autofocusController = self._master.getController("Autofocus")
            except Exception:
                pass

            if autofocusController is not None:
                self._logger.info(f"Preview: running autofocus at well {well_id}...")
                autofocusController.autoFocus(
                    rangez=autofocus_range,
                    resolutionz=autofocus_resolution,
                    defocusz=0,
                    tSettle=t_settle,
                    isDebug=False,
                )
                af_thread = getattr(autofocusController, '_AutofocusThead', None)
                if af_thread is not None and af_thread.is_alive():
                    af_thread.join(timeout=120)
                else:
                    t0 = time.time()
                    while getattr(autofocusController, 'isAutofusRunning', False):
                        check_cancelled()
                        time.sleep(0.1)
                        if time.time() - t0 > 60:
                            self._logger.warning("Autofocus timed out during well preview")
                            break
            else:
                self._logger.warning("AutofocusController not available – skipping autofocus")

        self._logger.info(f"Preview: capturing image at well {well_id}")
        yield from self.acquireFrame()

    @APIExport(runOnUIThread=False)
    def runWellTileScan(
        self,
        well_id: str = "A1",
        plate_type: str = "heidstar4",
        illumination_channel: str | None = None,
        illumination_intensity: float | None = None,
        exposure_time: float | None = None,
        gain: float | None = None,
        overlap_percent: float = 20.0,
        focus_map_grid_rows: int = 3,
        focus_map_grid_cols: int = 3,
        autofocus_range: float = 100,
        autofocus_resolution: float = 5,
        autofocus_cropsize: int = 512,
        autofocus_algorithm: str = "LAPE",
        autofocus_settle_time: float = 0.1,
        objective_id: int | None = None,
        speed: float = 20000,
        t_settle: float = 0.2,
        positionerName: str | None = None,
    ) -> Generator[Image, None, None]:
        """Scan an entire well with focus mapping.

        Args:
            well_id: Well label, e.g. "A1"-"A4" for heidstar4, or "A1"-"H12" for 96well.
            plate_type: Plate geometry. "heidstar4" (default) or "96well".
            illumination_channel: Laser/LED name. None → first active channel in live view.
            illumination_intensity: Source intensity. None → reads from live view.
            exposure_time: Camera exposure in ms. None → reads from current detector setting.
            gain: Camera gain. None → reads from current detector setting.
            overlap_percent: Tile overlap (0-100). Default 20 %.
            focus_map_grid_rows: Autofocus rows for focus map. Default 3.
            focus_map_grid_cols: Autofocus columns for focus map. Default 3.
            autofocus_range: Z scan half-range per autofocus step (µm). Default 100.
            autofocus_resolution: Z step size for autofocus (µm). Default 5.
            autofocus_cropsize: Crop size (px) for the focus quality metric. Default 512.
            autofocus_algorithm: Focus metric algorithm ("LAPE", "GLVA", "JPEG"). Default "LAPE".
            autofocus_settle_time: Settle time (s) between Z steps during autofocus. Default 0.1.
            objective_id: Objective slot to activate before scanning, or None.
            speed: Stage speed (µm/s). Default 40000.
            t_settle: Settling time after each XY move (s). Default 0.2.
            positionerName: Positioner name, or None for first available.

        Yields:
            Image: Tile images with affine transformation metadata for stitching.
        """
        # --- Validate well ID -----------------------------------------------
        plate_wells = self._plate_well_configs.get(plate_type)
        if plate_wells is None:
            self._logger.error(
                f"Unknown plate type '{plate_type}'. Supported: {list(_PLATE_WELL_CONFIGS)}"
            )
            return

        well_geom = plate_wells.get(well_id.strip().upper())
        if well_geom is None:
            self._logger.error(
                f"Unknown well '{well_id}' for plate '{plate_type}'. "
                f"Valid wells: {list(plate_wells)}"
            )
            return

        center_x = well_geom["center_x"]
        center_y = well_geom["center_y"]
        width    = well_geom["width"]
        height   = well_geom["height"]

        # --- Resolve positioner ---------------------------------------------
        if positionerName is None:
            names = self._master.positionersManager.getAllDeviceNames()
            if not names:
                self._logger.error("No positioners available for well tile scan")
                return
            positionerName = names[0]
        mPositioner = self._master.positionersManager[positionerName]

        # --- Read live-view imaging settings where not supplied -------------
        live_ch, live_intensity, live_exposure, live_gain = self._read_current_imaging_settings()
        if illumination_channel is None:
            illumination_channel = live_ch
        if illumination_intensity is None:
            illumination_intensity = live_intensity if live_intensity is not None else 1023.0
        if exposure_time is None:
            exposure_time = live_exposure
        if gain is None:
            gain = live_gain

        self._logger.info(
            f"Well tile scan: well={well_id}, plate={plate_type}, "
            f"center=({center_x}, {center_y}) µm, scan={width}×{height} µm | "
            f"illum={illumination_channel} @ {illumination_intensity}, "
            f"exp={exposure_time} ms, gain={gain}"
        )

        # --- Build focus map ------------------------------------------------
        self._active_focus_map = None
        autofocusController = None
        try:
            autofocusController = self._master.getController("Autofocus")
        except Exception:
            pass

        if autofocusController is not None:
            well_bounds = {
                "minX": center_x - width  / 2,
                "maxX": center_x + width  / 2,
                "minY": center_y - height / 2,
                "maxY": center_y + height / 2,
            }
            self._logger.info(
                f"Building focus map ({focus_map_grid_rows}×{focus_map_grid_cols} grid) "
                f"for well {well_id}..."
            )
            self._active_focus_map = self._build_focus_map_for_well(
                autofocusController=autofocusController,
                mPositioner=mPositioner,
                well_bounds=well_bounds,
                grid_rows=focus_map_grid_rows,
                grid_cols=focus_map_grid_cols,
                autofocus_range=autofocus_range,
                autofocus_resolution=autofocus_resolution,
                speed=speed,
                t_settle=t_settle,
                autofocus_cropsize=autofocus_cropsize,
                autofocus_algorithm=autofocus_algorithm,
                autofocus_settle_time=autofocus_settle_time,
            )
            if self._active_focus_map is None:
                self._logger.warning("Focus map unavailable – continuing without Z correction")
        else:
            self._logger.warning("AutofocusController not available – scanning without focus map")

        # --- Run tile scan (focus map Z correction applied inside) ----------
        try:
            stage = self.runTileScan(
                center_x_micrometer=center_x,
                center_y_micrometer=center_y,
                range_x_micrometer=width,
                range_y_micrometer=height,
                overlap_percent=overlap_percent,
                illumination_channel=illumination_channel,
                illumination_intensity=illumination_intensity,
                exposure_time=exposure_time,
                gain=gain,
                speed=speed,
                positionerName=positionerName,
                performAutofocus=False,
                objective_id=objective_id,
                t_settle=t_settle,
            )
        finally:
            self._active_focus_map = None
        
        return stage

    @APIExport(runOnUIThread=False)
    def deconvolve(self) -> int:
        """Trigger deconvolution via Arkitekt."""
        # grab an image
        frame = self.mDetector.getLatestFrame()  # X,Y,C, uint8 numpy array
        numpy_array = list(frame)[0]

        # Deconvolve using Arkitekt
        deconvolved_image = self._master.arkitektManager.upload_and_deconvolve_image(
            numpy_array
        )
        # QUESTION: Is this a synchronous call? Do we need to wait for the result?
        # The result that came back was none

        if deconvolved_image is not None:
            print("Image deconvolution successful!")
            return 2
        else:
            print("Deconvolution failed, returning original image")
            return 1
