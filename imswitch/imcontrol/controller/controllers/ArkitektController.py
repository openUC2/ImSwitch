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
import os
import datetime
import tifffile as tif
import time


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
    ) -> Generator[Image, None, None]:
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
        # Get objective manager for FOV calculation
        objective_manager = None
        if hasattr(self._master, 'objectiveManager'):
            objective_manager = self._master.objectiveManager
        
        # Handle objective switching if specified
        objective_magnification = None
        if objective_id is not None:
            # Get objective controller for moving the objective
            objective_controller = None
            try:
                objective_controller = self._master.getController('Objective')
                if objective_controller is not None:
                    self._logger.debug(f"Moving to objective ID: {objective_id}")
                    objective_controller.moveToObjective(objective_id)  # This is a blocking operation
                    self._logger.debug(f"Successfully moved to objective ID: {objective_id}")
                else:
                    self._logger.warning("ObjectiveController not available, cannot switch objective")
            except Exception as e:
                self._logger.error(f"Failed to move to objective ID {objective_id}: {e}")
        
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
        
        # Get objective magnification from manager after potential switch
        if objective_manager is not None:
            objective_magnification = objective_manager.getCurrentMagnification()
            if objective_magnification is not None:
                current_objective_slot = objective_manager.getCurrentObjective()
                self._logger.debug(f"Using objective slot {current_objective_slot} with magnification: {objective_magnification}x")
        
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
        try:
            stage = create_stage(name=f"Tile Scan Stage - {center_x_micrometer},{center_y_micrometer}")
        except Exception as e:
            self._logger.error(f"Failed to create stage for tile scan: {e}")
            stage = None
        
        # Create directory for saving tiles if stage creation failed
        save_dir = None
        metadata_list = []
        if stage is None:
            # Get data storage path from ImSwitch config
            data_path = dirtools.UserFileDirs.Data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            scan_name = f"tilescan_{timestamp}_cx{center_x_micrometer:.0f}_cy{center_y_micrometer:.0f}"
            save_dir = os.path.join(data_path, scan_name)
            os.makedirs(save_dir, exist_ok=True)
            self._logger.info(f"Saving tiles to: {save_dir}")
            
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
                # Wait for settling
                time.sleep(t_settle)
                
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
                
                if stage is not None and image is not None:
                    yield image
        
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
            is_blocking=False
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
