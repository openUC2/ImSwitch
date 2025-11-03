from imswitch.imcontrol.model.managers.ArkitektManager import set_global_context_locally
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import dirtools, initLogger, APIExport
import xarray as xr
from arkitekt_next import register, easy, progress
from mikro_next.api.schema import (
    Image,
    from_array_like,
    create_stage,
    PartialAffineTransformationViewInput,
)
from typing import Generator


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

        self.arkitekt_app = self._master.arkitektManager.get_arkitekt_app()
        self.arkitekt_app.register(self.moveToSampleLoadingPosition)
        self.arkitekt_app.register(self.runTileScan)
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
    def runTileScan(
        self,
        center_x_micrometer: float = 0,
        center_y_micrometer: float = 0,
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
        objective_magnification: float | None = None,
    ) -> Generator[Image, None, None]:
        """Run a tile scan with enhanced control over imaging parameters.

        Runs a tile scan by moving the specified positioner in a grid pattern centered
        at the given coordinates, capturing images at each position with specified 
        illumination and camera settings, and yielding the images with appropriate
        affine transformations for stitching.
        
        The step size is automatically calculated based on the current objective's
        field of view and the specified overlap percentage, unless explicitly provided.

        Args:
            center_x_micrometer (float): Center position in the X direction (micrometers).
            center_y_micrometer (float): Center position in the Y direction (micrometers).
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
            objective_magnification (float | None): Magnification of the objective lens.
                If None, uses the current objective magnification from ObjectiveManager.

        Yields:
            Image: Captured image with affine transformation for stitching.

        Example:
            >>> # Scan with automatic step size based on objective FOV
            >>> for image in runTileScan(
            ...     center_x_micrometer=5000, 
            ...     center_y_micrometer=5000,
            ...     range_x_micrometer=1000,
            ...     range_y_micrometer=1000,
            ...     overlap_percent=10,  # 10% overlap
            ...     illumination_channel="LED",
            ...     illumination_intensity=50,
            ...     exposure_time=100,
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
            ...     illumination_channel="LED"
            ... ):
            ...     pass
        """
        # Get objective manager for FOV calculation
        objective_manager = None
        if hasattr(self._master, 'objectiveManager'):
            objective_manager = self._master.objectiveManager
        
        # Calculate step sizes based on objective FOV if not provided
        if step_x_micrometer is None or step_y_micrometer is None:
            if objective_manager is not None:
                fov = objective_manager.getCurrentFOV()
                if fov is not None:
                    fov_x, fov_y = fov
                    # Calculate step size with overlap
                    # step = FOV * (1 - overlap/100)
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
                    self._logger.warning("Could not get FOV from ObjectiveManager - no detector dimensions set?")
            else:
                self._logger.warning("ObjectiveManager not available for automatic step size calculation")
            
            # Fallback to default values if still None
            if step_x_micrometer is None:
                step_x_micrometer = 100.0
                self._logger.warning(f"Using default step_x_micrometer: {step_x_micrometer} µm")
            if step_y_micrometer is None:
                step_y_micrometer = 100.0
                self._logger.warning(f"Using default step_y_micrometer: {step_y_micrometer} µm")
        
        # Get objective magnification if not provided
        if objective_magnification is None and objective_manager is not None:
            objective_magnification = objective_manager.getCurrentMagnification()
            if objective_magnification is not None:
                self._logger.debug(f"Using current objective magnification: {objective_magnification}x")
        
        # Calculate start positions from center and range
        xStart = center_x_micrometer - range_x_micrometer / 2
        yStart = center_y_micrometer - range_y_micrometer / 2
        
        # Use the new parameter names internally
        xRange = int(range_x_micrometer)
        yRange = int(range_y_micrometer)
        xStep = int(step_x_micrometer)
        yStep = int(step_y_micrometer)
        
        # Get positioner
        if positionerName is None:
            positionerNames = self._master.positionersManager.getAllDeviceNames()
            if len(positionerNames) == 0:
                self._logger.error("No positioners available for tile scan")
                return
            positionerName = positionerNames[0]
        
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
        try:
            stage = create_stage(name=f"Tile Scan Stage - {center_x_micrometer},{center_y_micrometer}")
        except Exception as e:
            self._logger.error(f"Failed to create stage for tile scan: {e}")
            stage = None 
            
        # Get positioner
        mPositioner = self._master.positionersManager[positionerName]
        
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
                    is_blocking=True
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
                numpy_array = self.mDetector.getLatestFrame()
                
                # Create affine transformation matrix for stitching
                affine_matrix_four_d = [
                    [1, 0, 0, actual_x],
                    [0, 1, 0, actual_y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
                
                # Create image with metadata
                image_name = f"Tile_{actual_ix:03d}_{iy:03d}_x{actual_x:.1f}_y{actual_y:.1f}"
                if illumination_channel:
                    image_name += f"_{illumination_channel}"
                
                if stage is not None:

                    image = from_array_like(
                        xr.DataArray(
                            numpy_array,
                            dims=["y", "x"],
                            attrs={
                                "tile_x": actual_ix,
                                "tile_y": iy,
                                "position_x_um": actual_x,
                                "position_y_um": actual_y,
                                "illumination_channel": illumination_channel or "unknown",
                                "illumination_intensity": illumination_intensity,
                                "exposure_time_ms": exposure_time,
                                "gain": gain,
                                "objective_magnification": objective_magnification,
                            }
                        ),
                        transformation_views=[
                            PartialAffineTransformationViewInput(
                                affineMatrix=affine_matrix_four_d,
                                stage=stage
                            )
                        ],
                        name=image_name,
                    )
                
                tile_count += 1
                self._logger.debug(f"Captured tile {tile_count}/{total_tiles} at ({actual_x}, {actual_y})")
                
                if stage is not None:
                    yield image
        
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
