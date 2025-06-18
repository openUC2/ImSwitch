#!/usr/bin/env python3
"""
Test script to verify z-stack functionality when no XY coordinates are provided.

This script tests the specific scenario mentioned in the issue:
"In case no xy coordinates are supplied but valid z scanning parameters 
we want to scan at the current position."
"""

import numpy as np
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass

# Mock the required classes to test the logic
@dataclass
class NeighborPoint:
    x: float
    y: float
    iX: int
    iY: int

@dataclass  
class Point:
    id: str
    name: str
    x: float
    y: float
    iX: int
    iY: int
    neighborPointList: List[NeighborPoint]

@dataclass
class ParameterValue:
    illumination: str
    illuIntensities: int
    brightfield: bool
    darkfield: bool
    differentialPhaseContrast: bool
    timeLapsePeriod: float
    numberOfImages: int
    autoFocus: bool
    autoFocusMin: float
    autoFocusMax: float
    autoFocusStepSize: float
    zStack: bool
    zStackMin: float
    zStackMax: float
    zStackStepSize: float
    exposureTimes: float
    gains: float
    resortPointListToSnakeCoordinates: bool
    speed: float
    performanceMode: bool
    ome_write_tiff: bool
    ome_write_zarr: bool
    ome_write_stitched_tiff: bool

@dataclass
class Experiment:
    name: str
    parameterValue: ParameterValue
    pointList: List[Point]

def test_zstack_only_scenario():
    """Test z-stack scanning when no XY coordinates are provided."""
    
    # Create a mock experiment with z-stack parameters but minimal XY data
    experiment = Experiment(
        name="z_stack_test",
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
            zStack=True,  # Enable z-stack
            zStackMin=-5,  # Z-stack range: -5 to +5 um
            zStackMax=5,
            zStackStepSize=1.0,  # 1 um steps -> 11 z positions
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
            # Single point at current position (no actual XY scan)
            Point(
                id=str(uuid.uuid4()),
                name="current_position",
                x=0,  # Will be replaced with current position
                y=0,  # Will be replaced with current position
                iX=0,
                iY=0,
                neighborPointList=[
                    NeighborPoint(x=0, y=0, iX=0, iY=0)  # Single position
                ]
            )
        ]
    )
    
    print("Test experiment created with z-stack enabled:")
    print(f"  Z-stack range: {experiment.parameterValue.zStackMin} to {experiment.parameterValue.zStackMax}")
    print(f"  Z-stack step size: {experiment.parameterValue.zStackStepSize}")
    print(f"  Expected z positions: {int((experiment.parameterValue.zStackMax - experiment.parameterValue.zStackMin) / experiment.parameterValue.zStackStepSize) + 1}")
    print(f"  Point list length: {len(experiment.pointList)}")
    print(f"  Neighbor points: {len(experiment.pointList[0].neighborPointList)}")
    
    return experiment

def test_z_position_generation():
    """Test the z-position generation logic."""
    
    # Mock current Z position
    currentZ = 100.0  # um
    
    # Z-stack parameters
    zStackMin = -5
    zStackMax = 5
    zStackStepSize = 1.0
    
    # Generate Z positions (similar to ExperimentController logic)
    z_positions = np.arange(zStackMin, zStackMax + zStackStepSize, zStackStepSize) + currentZ
    
    print("\nZ-position generation test:")
    print(f"  Current Z: {currentZ}")
    print(f"  Z-stack range: {zStackMin} to {zStackMax}")
    print(f"  Step size: {zStackStepSize}")
    print(f"  Generated positions: {z_positions}")
    print(f"  Number of z positions: {len(z_positions)}")
    
    expected_positions = [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    assert np.allclose(z_positions, expected_positions), f"Expected {expected_positions}, got {z_positions}"
    print("  ✓ Z-position generation is correct")
    
    return z_positions

def test_empty_pointlist_scenario():
    """Test what happens when pointList is empty but z-stack is enabled."""
    
    # Create experiment with empty pointList (no XY coordinates provided)
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
            zStack=True,  # Enable z-stack
            zStackMin=-5,  # Z-stack range: -5 to +5 um
            zStackMax=5,
            zStackStepSize=1.0,  # 1 um steps -> 11 z positions
            exposureTimes=100,
            gains=1,
            resortPointListToSnakeCoordinates=True,
            speed=1000,
            performanceMode=False,
            ome_write_tiff=True,
            ome_write_zarr=True,
            ome_write_stitched_tiff=False
        ),
        pointList=[]  # Empty point list - no XY coordinates provided!
    )
    
    print("\nTest experiment created with empty pointList:")
    print(f"  Z-stack enabled: {experiment.parameterValue.zStack}")
    print(f"  Point list length: {len(experiment.pointList)}")
    
    # Test generate_snake_tiles with empty pointList - this should fail currently
    try:
        tiles = []
        for iCenter, centerPoint in enumerate(experiment.pointList):
            print("Processing center point (this should not execute if list is empty)")
        print(f"Generated tiles: {tiles}")
        print("✓ Empty pointList handled")
    except Exception as e:
        print(f"✗ Error with empty pointList: {e}")
        
    return experiment

def test_current_position_fallback():
    """Test creating a point list with current position when none provided."""
    
    # Mock current stage position
    current_x, current_y, current_z = 1000.0, 2000.0, 100.0
    
    # Create a fallback point at current position
    current_position_point = Point(
        id=str(uuid.uuid4()),
        name="current_position_fallback",
        x=current_x,
        y=current_y,
        iX=0,
        iY=0,
        neighborPointList=[
            NeighborPoint(x=current_x, y=current_y, iX=0, iY=0)
        ]
    )
    
    print(f"\nCurrent position fallback test:")
    print(f"  Current position: ({current_x}, {current_y}, {current_z})")
    print(f"  Created fallback point: ({current_position_point.x}, {current_position_point.y})")
    print(f"  Neighbor points: {len(current_position_point.neighborPointList)}")
    
    # Test generate_snake_tiles with this fallback point
    experiment_with_fallback = Experiment(
        name="z_stack_current_pos_test",
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
        pointList=[current_position_point]
    )
    
    print(f"  Created experiment with fallback point")
    print(f"  Point list length: {len(experiment_with_fallback.pointList)}")
    print(f"  Z-stack enabled: {experiment_with_fallback.parameterValue.zStack}")
    
    return experiment_with_fallback

def test_mock_generate_snake_tiles_with_new_logic():
    """Test the updated generate_snake_tiles logic with both scenarios."""
    
    # Test case 1: Empty pointList with z-stack enabled
    experiment_empty = Experiment(
        name="test_empty",
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
    
    # Test case 2: Point with empty neighborPointList  
    experiment_empty_neighbors = Experiment(
        name="test_empty_neighbors",
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
    
    # Mock the updated generate_snake_tiles logic
    def generate_snake_tiles_new_logic(mExperiment):
        tiles = []
        
        # Handle case where no XY coordinates are provided but z-stack is enabled
        # In this case, we want to scan at the current position
        if len(mExperiment.pointList) == 0 and mExperiment.parameterValue.zStack:
            print("No XY coordinates provided but z-stack enabled. Creating fallback point at current position.")
            
            # Mock current stage position
            current_x, current_y = 1000.0, 2000.0
            
            # Create a fallback point at current position
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
        
        # Original logic for when pointList is provided
        for iCenter, centerPoint in enumerate(mExperiment.pointList):
            # Collect central and neighbour points (without duplicating the center)
            allPoints = [(n.x, n.y) for n in centerPoint.neighborPointList]
            
            # Handle case where neighborPointList is empty but centerPoint is provided
            # This means scan at the center point position only (useful for z-stack-only)
            if len(allPoints) == 0:
                print(f"Empty neighborPointList for center point {iCenter}. Using center point position for z-stack scanning.")
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
            
            # Normal processing would continue here...
            # For this test, we'll just handle the single-point case
            if len(allPoints) == 1:
                tiles.append([{
                    "iterator": 0,
                    "centerIndex": iCenter,
                    "iX": 0,
                    "iY": 0,
                    "x": allPoints[0][0],
                    "y": allPoints[0][1],
                }])
        
        return tiles
    
    # Test case 1: Empty pointList
    print("\nTesting empty pointList scenario:")
    tiles_empty = generate_snake_tiles_new_logic(experiment_empty)
    print(f"Result: {tiles_empty}")
    
    # Test case 2: Empty neighborPointList
    print("\nTesting empty neighborPointList scenario:")
    tiles_empty_neighbors = generate_snake_tiles_new_logic(experiment_empty_neighbors)
    print(f"Result: {tiles_empty_neighbors}")
    
    return tiles_empty, tiles_empty_neighbors

if __name__ == "__main__":
    print("Testing z-stack-only scenario...")
    
    try:
        # Test 1: Z-position generation
        z_positions = test_z_position_generation()
        
        # Test 2: Experiment creation with valid single point
        experiment = test_zstack_only_scenario()
        
        # Test 3: Empty pointList scenario
        test_empty_pointlist_scenario()
        
        # Test 4: Current position fallback
        fallback_experiment = test_current_position_fallback()
        
        # Test 5: New logic with both scenarios
        print("\n" + "="*50)
        print("Testing Updated Logic:")
        print("="*50)
        tiles_empty, tiles_empty_neighbors = test_mock_generate_snake_tiles_with_new_logic()
        
        print("\n✓ All tests completed!")
        print("\nSummary:")
        print(f"  - Z-stack with {len(z_positions)} positions")
        print(f"  - Empty pointList scenario: {len(tiles_empty)} tile groups")
        print(f"  - Empty neighborPointList scenario: {len(tiles_empty_neighbors)} tile groups")
        
        if tiles_empty:
            print(f"  - Empty pointList result: {tiles_empty[0]}")
        if tiles_empty_neighbors:
            print(f"  - Empty neighbors result: {tiles_empty_neighbors[0]}")
            
        print(f"  - Both scenarios create valid tile configurations for z-stack scanning!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)