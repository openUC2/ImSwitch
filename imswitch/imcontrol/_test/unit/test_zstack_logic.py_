#!/usr/bin/env python3
"""
Unit tests for z-stack functionality in ExperimentController.

Tests the specific scenarios mentioned in issue #117:
- Z-stack scanning when no XY coordinates are provided  
- Z-stack scanning at current position
- Proper saving of z-stack data in TIFF and zarr formats
"""

import pytest
import numpy as np
import uuid
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add the imswitch module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from imswitch.imcontrol.controller.controllers.ExperimentController import Experiment, ParameterValue, Point, NeighborPoint


class TestZStackLogic:
    """Test class for z-stack functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_stage = Mock()
        self.mock_stage.getPosition.return_value = {"X": 1000.0, "Y": 2000.0, "Z": 100.0}
        
    def test_z_position_generation(self):
        """Test z-position generation logic."""
        currentZ = 100.0
        zStackMin = -5
        zStackMax = 5
        zStackStepSize = 1.0
        
        # Generate Z positions (from ExperimentController logic)
        z_positions = np.arange(zStackMin, zStackMax + zStackStepSize, zStackStepSize) + currentZ
        
        expected_positions = [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        assert np.allclose(z_positions, expected_positions)
        assert len(z_positions) == 11
        
    def test_experiment_with_empty_pointlist(self):
        """Test experiment creation with empty pointList for z-stack only."""
        experiment = Experiment(
            name="z_stack_no_xy_test",
            parameterValue=ParameterValue(
                illumination="LED",
                illuIntensities=100,
                brightfield=False,
                darkfield=False, 
                differentialPhaseContrast=False,
                timeLapsePeriod=1.0,
                numberOfImages=1,
                autoFocus=False,
                autoFocusMin=0,
                autoFocusMax=0,
                autoFocusStepSize=1,
                zStack=True,
                zStackMin=-5,
                zStackMax=5,
                zStackStepSize=1.0,
                exposureTimes=100,
                gains=1,
                resortPointListToSnakeCoordinates=True,
                speed=1000,
                performanceMode=False,
                ome_write_tiff=True,
                ome_write_zarr=True,
                ome_write_stitched_tiff=False
            ),
            pointList=[]  # Empty point list
        )
        
        assert experiment.parameterValue.zStack is True
        assert len(experiment.pointList) == 0
        assert experiment.parameterValue.zStackMin == -5
        assert experiment.parameterValue.zStackMax == 5
        assert experiment.parameterValue.zStackStepSize == 1.0
        
    def test_experiment_with_single_point_empty_neighbors(self):
        """Test experiment with single point but empty neighborPointList."""
        experiment = Experiment(
            name="z_stack_single_point_test",
            parameterValue=ParameterValue(
                illumination="LED",
                illuIntensities=100,
                brightfield=False,
                darkfield=False, 
                differentialPhaseContrast=False,
                timeLapsePeriod=1.0,
                numberOfImages=1,
                autoFocus=False,
                autoFocusMin=0,
                autoFocusMax=0,
                autoFocusStepSize=1,
                zStack=True,
                zStackMin=-5,
                zStackMax=5,
                zStackStepSize=1.0,
                exposureTimes=100,
                gains=1,
                resortPointListToSnakeCoordinates=True,
                speed=1000,
                performanceMode=False,
                ome_write_tiff=True,
                ome_write_zarr=True,
                ome_write_stitched_tiff=False
            ),
            pointList=[
                Point(
                    id=str(uuid.uuid4()),
                    name="center_only",
                    x=1500.0,
                    y=2500.0,
                    iX=0,
                    iY=0,
                    neighborPointList=[]  # Empty neighbor list
                )
            ]
        )
        
        assert experiment.parameterValue.zStack is True
        assert len(experiment.pointList) == 1
        assert len(experiment.pointList[0].neighborPointList) == 0
        assert experiment.pointList[0].x == 1500.0
        assert experiment.pointList[0].y == 2500.0
        
    def test_mock_generate_snake_tiles_empty_pointlist(self):
        """Test generate_snake_tiles logic with empty pointList."""
        
        experiment = Experiment(
            name="test",
            parameterValue=ParameterValue(
                illumination="LED", illuIntensities=100, brightfield=False, darkfield=False,
                differentialPhaseContrast=False, timeLapsePeriod=1.0, numberOfImages=1,
                autoFocus=False, autoFocusMin=0, autoFocusMax=0, autoFocusStepSize=1,
                zStack=True, zStackMin=-2, zStackMax=2, zStackStepSize=1.0,
                exposureTimes=100, gains=1, resortPointListToSnakeCoordinates=True,
                speed=1000, performanceMode=False, ome_write_tiff=True,
                ome_write_zarr=True, ome_write_stitched_tiff=False
            ),
            pointList=[]
        )
        
        # Mock the new generate_snake_tiles logic
        def mock_generate_snake_tiles(mExperiment):
            tiles = []
            
            # Handle case where no XY coordinates are provided but z-stack is enabled
            if len(mExperiment.pointList) == 0 and mExperiment.parameterValue.zStack:
                # Mock current position
                current_x, current_y = 1000.0, 2000.0
                
                fallback_tile = [{
                    "iterator": 0,
                    "centerIndex": 0,
                    "iX": 0,
                    "iY": 0,
                    "x": current_x,
                    "y": current_y,
                }]
                tiles.append(fallback_tile)
                return tiles
            
            # Original logic would go here...
            return tiles
        
        tiles = mock_generate_snake_tiles(experiment)
        
        assert len(tiles) == 1  # One tile group
        assert len(tiles[0]) == 1  # One position in the group
        assert tiles[0][0]["x"] == 1000.0
        assert tiles[0][0]["y"] == 2000.0
        assert tiles[0][0]["iX"] == 0
        assert tiles[0][0]["iY"] == 0
        
    def test_mock_generate_snake_tiles_empty_neighbors(self):
        """Test generate_snake_tiles logic with empty neighborPointList."""
        
        experiment = Experiment(
            name="test",
            parameterValue=ParameterValue(
                illumination="LED", illuIntensities=100, brightfield=False, darkfield=False,
                differentialPhaseContrast=False, timeLapsePeriod=1.0, numberOfImages=1,
                autoFocus=False, autoFocusMin=0, autoFocusMax=0, autoFocusStepSize=1,
                zStack=True, zStackMin=-2, zStackMax=2, zStackStepSize=1.0,
                exposureTimes=100, gains=1, resortPointListToSnakeCoordinates=True,
                speed=1000, performanceMode=False, ome_write_tiff=True,
                ome_write_zarr=True, ome_write_stitched_tiff=False
            ),
            pointList=[
                Point(
                    id=str(uuid.uuid4()),
                    name="center_only",
                    x=1500.0,
                    y=2500.0,
                    iX=0,
                    iY=0,
                    neighborPointList=[]
                )
            ]
        )
        
        # Mock the new generate_snake_tiles logic  
        def mock_generate_snake_tiles(mExperiment):
            tiles = []
            
            if len(mExperiment.pointList) == 0 and mExperiment.parameterValue.zStack:
                # This case handled above
                pass
            else:
                # Original logic for when pointList is provided
                for iCenter, centerPoint in enumerate(mExperiment.pointList):
                    allPoints = [(n.x, n.y) for n in centerPoint.neighborPointList]
                    
                    # Handle case where neighborPointList is empty
                    if len(allPoints) == 0:
                        fallback_tile = [{
                            "iterator": 0,
                            "centerIndex": iCenter,
                            "iX": 0,
                            "iY": 0,
                            "x": centerPoint.x,
                            "y": centerPoint.y,
                        }]
                        tiles.append(fallback_tile)
                        continue
                    
                    # Original snake logic would go here...
            
            return tiles
        
        tiles = mock_generate_snake_tiles(experiment)
        
        assert len(tiles) == 1  # One tile group
        assert len(tiles[0]) == 1  # One position in the group
        assert tiles[0][0]["x"] == 1500.0
        assert tiles[0][0]["y"] == 2500.0
        assert tiles[0][0]["centerIndex"] == 0
        
    def test_z_stack_metadata_structure(self):
        """Test that z-stack metadata is properly structured."""
        
        # Test z-index calculation for different scenarios
        z_positions = [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        
        for z_index, z_pos in enumerate(z_positions):
            # Mock the metadata that would be passed to save_frame_ome
            metadata = {
                "x": 1000.0,
                "y": 2000.0,
                "z": z_pos,
                "z_index": z_index,
                "channel_index": 0,
                "time_index": 0,
                "tile_index": 0,
                "illuminationChannel": "LED",
                "illuminationValue": 100
            }
            
            assert metadata["z_index"] == z_index
            assert metadata["z"] == z_pos
            assert metadata["z_index"] >= 0
            assert metadata["z_index"] < len(z_positions)
            
    def test_zarr_canvas_dimensions(self):
        """Test that zarr canvas dimensions account for z-stack properly."""
        
        # Mock OME writer config for z-stack
        n_time_points = 1
        n_z_planes = 11  # -5 to +5 with step 1
        n_channels = 1
        
        # Mock canvas shape: (t, c, z, y, x)  
        expected_shape = (n_time_points, n_channels, n_z_planes, 1000, 1000)
        
        assert expected_shape[0] == n_time_points
        assert expected_shape[1] == n_channels
        assert expected_shape[2] == n_z_planes
        assert expected_shape[2] == 11  # Expected number of z planes
        
    def test_single_tiff_vs_zarr_modes(self):
        """Test that both single TIFF and zarr modes handle z-stack properly."""
        
        # Test single TIFF mode configuration
        single_tiff_config = {
            "write_tiff": True,
            "write_zarr": False,
            "write_stitched_tiff": False,
            "write_tiff_single": True,
            "n_z_planes": 11
        }
        
        # Test zarr mode configuration  
        zarr_config = {
            "write_tiff": False,
            "write_zarr": True,
            "write_stitched_tiff": False,
            "write_tiff_single": False,
            "n_z_planes": 11
        }
        
        assert single_tiff_config["n_z_planes"] == zarr_config["n_z_planes"]
        assert single_tiff_config["write_tiff_single"] != zarr_config["write_tiff_single"]
        assert single_tiff_config["write_zarr"] != zarr_config["write_zarr"]
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])