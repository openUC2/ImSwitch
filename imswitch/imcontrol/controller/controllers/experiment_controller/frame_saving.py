"""Frame -> OME writer dispatch, the fast-scan writer loop, OMERO/writer config endpoints.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
import time
from fastapi import HTTPException
import numpy as np
from typing import Optional, Dict, Any


from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowContext
from imswitch.imcommon.model import dirtools, APIExport

from imswitch.imcontrol.model.io import OMEWriter, OMEWriterConfig, OMEFileStorePaths


class FrameSavingMixin:
    """Frame -> OME writer dispatch, the fast-scan writer loop, OMERO/writer config endpoints."""

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

    @APIExport()
    def getLastScanAsOMEZARR(self):
        """ Returns the last OME-Zarr folder as a zipped file for download. """
        try:
            return self.getOmeZarrUrl()
        except Exception as e:
            self._logger.error(f"Error while getting last scan as OME-Zarr: {e}")
            raise HTTPException(status_code=500, detail="Error while getting last scan as OME-Zarr.")

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

        # Colour (RGB) detectors deliver (H, W, 3) arrays. Preserve the colour
        # for RGB cameras (the OME writer now carries a trailing samples axis);
        # only collapse to grayscale for non-RGB detectors that nonetheless
        # hand us a 3-channel frame.
        if img.ndim == 3 and img.shape[2] == 3 and not getattr(self, 'isRGB', False):
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(img.dtype) # TODO: Is this common sense @Franzili?

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

        # Environmental (I2C) sensor snapshot alongside the frame, when an I2C
        # sensor controller is configured. Throttled hardware read + optional
        # sidecar-CSV logging (one row per fresh read). No-op otherwise.
        try:
            i2c_reading = self._read_i2c_snapshot()
            if i2c_reading and i2c_reading.get("ok"):
                for _k in ("temperature_c", "humidity_pct", "lux", "ch0_full", "ch1_ir"):
                    _v = i2c_reading.get(_k)
                    if _v is not None:
                        ome_metadata[f"i2c_{_k}"] = _v
                _seq = getattr(self, "_i2c_read_seq", 0)
                if _seq != getattr(self, "_i2c_logged_seq", -1):
                    self._i2c_logged_seq = _seq
                    self._log_i2c_row(i2c_reading, ome_metadata.get("time_index", 0))
        except Exception:
            pass

        # Resolve the writer for this tile *before* we hand work to
        # the background saver — pulling from ``context`` on the
        # worker thread would be a lifetime hazard.
        file_writers = context.get_object("file_writers")
        if file_writers is None or position_center_index >= len(file_writers):
            self._logger.error(
                f"No OME writer found for tile index {position_center_index}"
            )
            metadata["frame_saved"] = False
            metadata.pop("result", None)
            return
        ome_writer = file_writers[position_center_index]

        # Copy the frame so the camera SDK ring buffer can be reused
        # while the background thread writes. Skipping this is the
        # textbook way to end up with torn frames in the output stack.
        # ~5–10 ms for 9 MP on a Pi 5 — much cheaper than the inline
        # write it replaces.
        try:
            img_copy = img.copy()
        except Exception as copy_err:
            self._logger.error(f"Frame copy failed: {copy_err}")
            metadata["frame_saved"] = False
            metadata.pop("result", None)
            return

        # The on_success callback runs on the saver thread. Keep it
        # cheap — signal emission is already thread-safe (see
        # noqt.SignalInstance.emit which run_coroutine_threadsafe's
        # the broadcast onto the shared event loop).
        def _on_success(chunk_info):
            if ome_writer.store:
                data_path = dirtools.UserFileDirs.getValidatedDataPath()
                self.setOmeZarrUrl(
                    ome_writer.store.split(data_path)[-1]
                )
            if chunk_info and "rel_chunk" in chunk_info:
                self.sigUpdateOMEZarrStore.emit({
                    "event": "zarr_chunk",
                    "path": chunk_info["rel_chunk"],
                    "zarr": str(self.getOmeZarrUrl()),
                })

        def _on_error(err):
            self._logger.error(f"Error saving OME frame: {err}")

        # Queue the write. ``submit`` BLOCKS when the queue is full,
        # which is intentional: it lets a slow disk back-pressure the
        # acquisition loop rather than silently dropping frames.
        # maxsize=32 (see __init__) caps peak RAM at ~32 × frame_size.
        self._frame_saver.submit(
            ome_writer.write_frame,
            img_copy,
            ome_metadata,
            on_success=_on_success,
            on_error=_on_error,
        )

        # The frame is in the saver's queue or already on disk by the
        # time this method returns; either way it's "saved" from the
        # workflow's perspective. (If the actual disk write later
        # fails, ``_on_error`` logs it.)
        metadata["frame_saved"] = True

        # Release the workflow's reference to the frame. We already
        # made our own copy above; without this pop the array stays
        # alive in WorkflowContext.data for every step, causing
        # unbounded RAM growth during long wellplate experiments.
        metadata.pop("result", None)

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
            logger=self._logger,
            isRGB=getattr(self, 'isRGB', False),
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
