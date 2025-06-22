"""
Normal mode implementation for ExperimentController.

This module handles experiment execution where Python controls each step
of the scanning process using workflow steps for precise control.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np

from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowStep
from .experiment_mode_base import ExperimentModeBase
from .ome_writer import OMEWriter


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
                         **kwargs) -> Dict[str, Any]:
        """
        Execute experiment in normal mode.
        
        Args:
            snake_tiles: List of tiles containing scan points
            illumination_intensities: List of illumination values
            illumination_sources: List of illumination source names
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
        t_period = kwargs.get('t_period', 1)
        # New parameters for multi-timepoint support
        n_times = kwargs.get('n_times', 1)  # Total number of time points
        
        # Initialize workflow components
        workflow_steps = []
        file_writers = []
        step_id = 0
        
        # Set up OME writers for each tile - create new writers for each timepoint
        file_writers = self._setup_ome_writers(
            snake_tiles, t, exp_name, dir_path, m_file_name, 
            z_positions, illumination_intensities
        )
        
        # Create workflow steps for each tile
        for position_center_index, tiles in enumerate(snake_tiles):
            step_id = self._create_tile_workflow_steps(
                tiles, position_center_index, step_id, workflow_steps,
                z_positions, illumination_sources, illumination_intensities,
                exposures, gains, t, is_auto_focus, autofocus_min, 
                autofocus_max, autofocus_step_size, n_times
            )
        
        # Add finalization steps
        step_id = self._add_finalization_steps(
            workflow_steps, step_id, snake_tiles, illumination_sources, 
            illumination_intensities, t_period, t
        )
        
        return {
            "status": "workflow_created",
            "mode": "normal",
            "workflow_steps": workflow_steps,
            "file_writers": file_writers,
            "step_count": step_id
        }
    
    def _setup_ome_writers(self,
                          snake_tiles: List[List[Dict]],
                          t: int,
                          exp_name: str,
                          dir_path: str,
                          m_file_name: str,
                          z_positions: List[float],
                          illumination_intensities: List[float]) -> List[OMEWriter]:
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
            
        Returns:
            List of OMEWriter instances
        """
        file_writers = []
        
        # Check if single TIFF writing is enabled (single tile scan mode)
        is_single_tiff_mode = getattr(self.controller, '_ome_write_single_tiff', False)
        
        if is_single_tiff_mode:
            # Create a single OME writer for all tiles in single TIFF mode
            experiment_name = f"{t}_{exp_name}"
            m_file_path = os.path.join(dir_path, f"{m_file_name}_{experiment_name}.ome.tif")
            self._logger.debug(f"Single TIFF mode - OME-TIFF path: {m_file_path}")
            
            # Create file paths
            file_paths = self.create_ome_file_paths(m_file_path.replace(".ome.tif", ""))
            
            # Calculate combined tile and grid parameters for all positions
            all_tiles = [tile for tiles in snake_tiles for tile in tiles]  # Flatten all tiles
            tile_shape = (self.controller.mDetector._shape[-1], self.controller.mDetector._shape[-2])
            grid_shape, grid_geometry = self.calculate_grid_parameters(all_tiles)
            
            # Create writer configuration for single TIFF mode
            n_channels = sum(np.array(illumination_intensities) > 0)
            writer_config = self.create_writer_config(
                write_tiff=False,  # Disable individual TIFF files
                write_zarr=self.controller._ome_write_zarr,
                write_stitched_tiff=False,  # Disable stitched TIFF
                write_tiff_single=True,  # Enable single TIFF writing
                min_period=0.1,
                n_time_points=1,
                n_z_planes=len(z_positions),
                n_channels=n_channels
            )
            
            # Create single OME writer for all positions
            ome_writer = OMEWriter(
                file_paths=file_paths,
                tile_shape=tile_shape,
                grid_shape=grid_shape,
                grid_geometry=grid_geometry,
                config=writer_config,
                logger=self._logger
            )
            file_writers.append(ome_writer)
            
        else:
            # Original behavior: create separate writers for each tile position
            for position_center_index, tiles in enumerate(snake_tiles):
                experiment_name = f"{t}_{exp_name}_{position_center_index}"
                m_file_path = os.path.join(
                    dir_path, 
                    m_file_name + str(position_center_index) + "_" + experiment_name + "_" + ".ome.tif"
                )
                self._logger.debug(f"OME-TIFF path: {m_file_path}")
                
                # Create file paths
                file_paths = self.create_ome_file_paths(m_file_path.replace(".ome.tif", ""))
                
                # Calculate tile and grid parameters
                tile_shape = (self.controller.mDetector._shape[-1], self.controller.mDetector._shape[-2])
                grid_shape, grid_geometry = self.calculate_grid_parameters(tiles)
                
                # Create writer configuration
                n_channels = sum(np.array(illumination_intensities) > 0)
                writer_config = self.create_writer_config(
                    write_tiff=self.controller._ome_write_tiff,
                    write_zarr=self.controller._ome_write_zarr,
                    write_stitched_tiff=self.controller._ome_write_stitched_tiff,
                    write_tiff_single=False,  # Disable single TIFF for multi-tile mode
                    min_period=0.1,  # Faster for normal mode
                    n_time_points=1,
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
                    logger=self._logger
                )
                file_writers.append(ome_writer)
        
        return file_writers
    
    def _create_tile_workflow_steps(self,
                                  tiles: List[Dict],
                                  position_center_index: int,
                                  step_id: int,
                                  workflow_steps: List[WorkflowStep],
                                  z_positions: List[float],
                                  illumination_sources: List[str],
                                  illumination_intensities: List[float],
                                  exposures: List[float],
                                  gains: List[float],
                                  t: int,
                                  is_auto_focus: bool,
                                  autofocus_min: float,
                                  autofocus_max: float,
                                  autofocus_step_size: float,
                                  n_times: int) -> int:
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
            
        Returns:
            Updated step ID
        """
        # Get scan range information
        initial_z_position = self.controller.mStage.getPosition()["Z"]
        min_x, max_x, min_y, max_y, _, _ = self.compute_scan_ranges([tiles])
        m_pixel_size = self.controller.detectorPixelSize[-1] if hasattr(self.controller, 'detectorPixelSize') else 1.0
        
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

            # Iterate over Z positions
            for index_z, i_z in enumerate(z_positions):
                # Move to Z position if we have more than one Z position
                if len(z_positions) > 1 or (len(z_positions) == 1 and m_index == 0):
                    workflow_steps.append(WorkflowStep(
                        name="Move to Z position",
                        step_id=step_id,
                        main_func=self.controller.move_stage_z,
                        main_params={"posZ": i_z, "relative": False},
                        pre_funcs=[self.controller.wait_time],
                        pre_params={"seconds": 0.1},
                    ))
                    step_id += 1

                # Iterate over illumination sources
                for illu_index, illu_source in enumerate(illumination_sources):
                    illu_intensity = illumination_intensities[illu_index] if illu_index < len(illumination_intensities) else 0
                    if illu_intensity <= 0:
                        continue

                    # Turn on illumination
                    if sum(np.array(illumination_intensities) > 0) > 1 or m_index == 0:
                        workflow_steps.append(WorkflowStep(
                            name="Turn on illumination",
                            step_id=step_id,
                            main_func=self.controller.set_laser_power,
                            main_params={"power": illu_intensity, "channel": illu_source},
                            post_funcs=[self.controller.wait_time],
                            post_params={"seconds": 0.05},
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

                    # Turn off illumination if multiple sources
                    if len(illumination_intensities) > 1 and sum(np.array(illumination_intensities) > 0) > 1:
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
                pre_params={"seconds": 0.1},
            ))
            step_id += 1

        # Perform autofocus if enabled
        if is_auto_focus:
            workflow_steps.append(WorkflowStep(
                name="Autofocus",
                step_id=step_id,
                main_func=self.controller.autofocus,
                main_params={"minZ": autofocus_min, "maxZ": autofocus_max, "stepSize": autofocus_step_size},
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
                              snake_tiles: List[List[Dict]],
                              illumination_sources: List[str],
                              illumination_intensities: List[float],
                              t_period: float,
                              t: int) -> int:
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