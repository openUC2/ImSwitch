"""
Normal mode implementation for ExperimentController.

This module handles experiment execution where Python controls each step
of the scanning process using workflow steps for precise control.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np

from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowStep
from imswitch.imcontrol.model.io import OMEWriter, OMEROConnectionParams
from .experiment_mode_base import ExperimentModeBase


class ExperimentNormalMode(ExperimentModeBase):
    """
    Normal mode experiment execution.
    
    In normal mode, Python controls each step: moving stage, setting illumination,
    acquiring frames, and saving data. This provides maximum flexibility and
    precise control over the experiment sequence.
    """

    def execute_experiment(self,
                         snake_tiles: List[List[Dict]],
                         illumination_intensities: List[float],
                         illumination_sources: List[str],
                         isRGB: bool = False,
                         shared_file_writers: Optional[List[OMEWriter]] = None,
                         omero_connection_params: Optional[OMEROConnectionParams] = None,
                         shared_omero_key: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Execute experiment in normal mode.
        
        Args:
            snake_tiles: List of tiles containing scan points
            illumination_intensities: List of illumination values
            illumination_sources: List of illumination source names
            isRGB: Whether images are RGB
            shared_file_writers: Optional shared writers from previous timepoints (for timelapse).
                               If provided, reuses these writers instead of creating new ones.
            omero_connection_params: Optional OMERO connection parameters for streaming upload.
            shared_omero_key: Optional key for shared OMERO uploader (for timelapse).
            **kwargs: Additional parameters including z_positions, exposures, gains, etc.
            
        Returns:
            Dictionary with execution results including workflow steps and file writers
        """
        self._logger.debug("Normal mode is enabled. Creating workflow steps for precise control.")

        # Extract parameters
        z_positions = kwargs.get('z_positions', [0])
        exposures = kwargs.get('exposures', [100])
        gains = kwargs.get('gains', [1])
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
        autofocus_max_attempts = kwargs.get('autofocus_max_attempts', 2)
        autofocus_target_focus_setpoint = kwargs.get('autofocus_target_focus_setpoint', None)
        initial_z_position = kwargs.get('initial_z_position', None)
        t_period = kwargs.get('t_period', 1)
        # Timing parameters from GUI (used for settle/exposure waits)
        t_pre_s = kwargs.get('t_pre_s', 0.09)  # Pre-exposure settle time in seconds
        t_post_s = kwargs.get('t_post_s', 0.05)  # Post-exposure time in seconds
        # New parameters for multi-timepoint support
        n_times = kwargs.get('n_times', 1)  # Total number of time points

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
            )
        # Create workflow steps for each tile
        for position_center_index, tiles in enumerate(snake_tiles):
            step_id = self._create_tile_workflow_steps(
                tiles, position_center_index, step_id, workflow_steps,
                z_positions, initial_z_position, illumination_sources, illumination_intensities,
                exposures, gains, t, is_auto_focus, autofocus_min,
                autofocus_max, autofocus_step_size, autofocus_illumination_channel,
                autofocus_mode, autofocus_max_attempts, autofocus_target_focus_setpoint, n_times,
                t_pre_s=t_pre_s, t_post_s=t_post_s
            )

        # Add finalization steps
        step_id = self._add_finalization_steps(
            workflow_steps, step_id, snake_tiles, illumination_sources,
            illumination_intensities, t_period, t, n_times
        )

        # Add step to set LED status to idle when done
        workflow_steps.append(WorkflowStep(
            step_id=step_id,
            name="Set LED status to idle",
            main_func=self.controller.set_led_status,
            main_params={"status": "idle"})
        )
        step_id += 1

        # Save experiment protocol to JSON
        protocol_data = {
            "experiment_name": exp_name,
            "experiment_mode": "normal",
            "directory": dir_path,
            "filename": m_file_name,
            "timepoint": t,
            "total_timepoints": n_times,
            "tile_count": len(snake_tiles),
            "step_count": step_id,
            "snake_tiles": snake_tiles,
            "z_positions": z_positions,
            "illumination_sources": illumination_sources,
            "illumination_intensities": illumination_intensities,
            "exposures": exposures,
            "gains": gains,
            "autofocus": {
                "enabled": is_auto_focus,
                "min": autofocus_min,
                "max": autofocus_max,
                "step_size": autofocus_step_size,
                "channel": autofocus_illumination_channel,
                "mode": autofocus_mode,
                "max_attempts": autofocus_max_attempts,
                "target_focus_setpoint": autofocus_target_focus_setpoint
            },
            "workflow_steps": [self._serialize_workflow_step(step) for step in workflow_steps]
        }
        
        # Create protocol file path
        protocol_file_path = os.path.join(dir_path, f"{m_file_name}_t{t:04d}")
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
                          n_times: int = 1) -> List[OMEWriter]:

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
        for position_center_index, tiles in enumerate(snake_tiles):
            experiment_name = f"{t}_{exp_name}_{position_center_index}"
            m_file_path = os.path.join(
                dir_path,
                m_file_name + str(position_center_index) + "_" + experiment_name + "_" + ".ome.tif"
            )
            self._logger.debug(f"OME-TIFF path: {m_file_path}")

            # Create file paths with shared individual_tiffs directory
            file_paths = self.create_ome_file_paths(m_file_path.replace(".ome.tif", ""), shared_individual_tiffs_dir)

            # Calculate tile and grid parameters
            tile_shape = (self.controller.mDetector._shape[-1], self.controller.mDetector._shape[-2])
            grid_shape, grid_geometry = self.calculate_grid_parameters(tiles)

            # Create writer configuration
            n_channels = sum(np.array(illumination_intensities) > 0)
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
                n_channels=n_channels
            )

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
            )
            file_writers.append(ome_writer)

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
                                  t_post_s: float = 0.05) -> int:
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

        # Turn on illumination once at the beginning if only one source
        active_sources_count = sum(np.array(illumination_intensities) > 0)
        is_first_tile = (position_center_index == 0)

        '''
        if active_sources_count == 1 and is_first_tile:
            for illu_index, illu_source in enumerate(illumination_sources):
                illu_intensity = illumination_intensities[illu_index] if illu_index < len(illumination_intensities) else 0
                if illu_intensity > 0:
                    workflow_steps.append(WorkflowStep(
                        name="Turn on single illumination source for entire scan",
                        step_id=step_id,
                        main_func=self.controller.set_laser_power,
                        main_params={"power": illu_intensity, "channel": illu_source},
                        post_funcs=[self.controller.wait_time],
                        post_params={"seconds": 0.05},
                    ))
                    step_id += 1
                    break  # Only one active source
        '''
        # Iterate over positions in the tile
        for m_index, m_point in enumerate(tiles):
            try:
                name = f"Move to point {m_point['iterator']}"
            except Exception:
                name = f"Move to point {m_point['x']}, {m_point['y']}"

            # Move to XY position
            workflow_steps.append(WorkflowStep(
                name=name,
                step_id=step_id,
                main_func=self.controller.move_stage_xy,
                main_params={"posX": m_point["x"], "posY": m_point["y"], "relative": False},
            ))
            step_id += 1
            
            # Perform autofocus if enabled
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
                        "mode": autofocus_mode
                    },
                ))
                step_id += 1

            # Iterate over Z positions
            for index_z, i_z in enumerate(z_positions):
                # Move to Z position if we have more than one Z position
                if (len(z_positions) > 1 or (len(z_positions) == 1 and m_index == 0)) and (i_z != 0 and len(z_positions) != 1): # TODO: The latter case is just to ensure that we don't have false values coming from the hardware
                    workflow_steps.append(WorkflowStep(
                        name="Move to Z position",
                        step_id=step_id,
                        main_func=self.controller.move_stage_z,
                        main_params={"posZ": i_z, "relative": False},
                        pre_funcs=[self.controller.wait_time],
                        pre_params={"seconds": t_pre_s},
                    ))
                    step_id += 1

                # Iterate over illumination sources
                for illu_index, illu_source in enumerate(illumination_sources):
                    illu_intensity = illumination_intensities[illu_index] if illu_index < len(illumination_intensities) else 0
                    if illu_intensity <= 0:
                        continue

                    # Turn on illumination - use tPre as settle time after activation
                    workflow_steps.append(WorkflowStep(
                        name="Turn on illumination",
                        step_id=step_id,
                        main_func=self.controller.set_laser_power,
                        main_params={"power": illu_intensity, "channel": illu_source},
                        post_funcs=[self.controller.wait_time],
                        post_params={"seconds": t_pre_s},
                    ))
                    step_id += 1

                    # Acquire frame
                    exposure_time = exposures[illu_index] if illu_index < len(exposures) else exposures[0]
                    gain = gains[illu_index] if illu_index < len(gains) else gains[0]

                    # In single TIFF mode, all positions within a timepoint use the same writer
                    # The writer index corresponds to the timepoint for timelapse sequences
                    is_single_tiff_mode = getattr(self.controller, '_ome_write_single_tiff', False)
                    writer_index = t if is_single_tiff_mode else position_center_index

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
                            "position_center_index": writer_index,  # Use writer_index instead
                            "runningNumber": step_id,
                            "illuminationChannel": illu_source,
                            "illuminationValue": illu_intensity,
                            "z_index": index_z,
                            "channel_index": illu_index,
                        },
                    ))
                    step_id += 1

                    # Turn off illumination only if multiple sources (for switching between them)
                    workflow_steps.append(WorkflowStep(
                        name="Turn off illumination",
                        step_id=step_id,
                        main_func=self.controller.set_laser_power,
                        main_params={"power": 0, "channel": illu_source},
                    ))
                    step_id += 1

            # Move back to the current Z position after processing all points in the tile
            if len(z_positions) > 1 :
                workflow_steps.append(WorkflowStep(
                    name="Move back to current Z position",
                    step_id=step_id,
                    main_func=self.controller.move_stage_z,
                    main_params={"posZ": initial_z_position, "relative": False},
                    pre_funcs=[self.controller.wait_time],
                    pre_params={"seconds": t_pre_s},
                ))
                step_id += 1



        # Finalize OME writer for this tile (skip in single TIFF mode since we only have one writer)
        # Always finalize tile writers since each timepoint creates its own writers
        is_single_tiff_mode = getattr(self.controller, '_ome_write_single_tiff', False)
        should_finalize_tile = not is_single_tiff_mode

        if should_finalize_tile:
            workflow_steps.append(WorkflowStep(
                name=f"Finalize OME writer for tile {position_center_index} (timepoint {t})",
                step_id=step_id,
                main_func=self.controller.dummy_main_func,
                main_params={},
                post_funcs=[self.controller.finalize_tile_ome_writer],
                post_params={"tile_index": position_center_index},
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
                              t: int, n_times: int) -> int:
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
        # Always finalize OME writers since each timepoint creates its own writers
        workflow_steps.append(WorkflowStep(
            name=f"Finalize OME writers (timepoint {t})",
            step_id=step_id,
            main_func=self.controller.dummy_main_func,
            main_params={},
            post_funcs=[self.controller.finalize_current_ome_writer],
            post_params={"time_index": t}
        ))
        step_id += 1

        # Turn off all illuminations
        for illu_index, illu_source in enumerate(illumination_sources):
            illu_intensity = illumination_intensities[illu_index] if illu_index < len(illumination_intensities) else 0
            if illu_intensity <= 0:
                continue

            workflow_steps.append(WorkflowStep(
                name="Turn off illumination",
                step_id=step_id,
                main_func=self.controller.set_laser_power,
                main_params={"power": 0, "channel": illu_source},
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

        return step_id
