"""
Protocol Manager for Experiment Controller

Handles the structuring of experiment protocols based on user input,
including snake pattern generation and scan range calculations.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from imswitch.imcommon.model import initLogger


class ProtocolManager:
    """Manages experiment protocol structuring and parameter validation."""

    def __init__(self):
        self._logger = initLogger(self)

    def get_num_xy_steps(self, pointList: List) -> Tuple[int, int]:
        """
        Calculate number of X/Y steps needed for the point list.

        Args:
            pointList: List of neighbor points with iX, iY coordinates

        Returns:
            Tuple of (num_x_steps, num_y_steps)
        """
        if len(pointList) == 0:
            return 1, 1
            
        all_iX = [point.iX for point in pointList]
        all_iY = [point.iY for point in pointList]
        
        min_iX, max_iX = min(all_iX), max(all_iX)
        min_iY, max_iY = min(all_iY), max(all_iY)
        
        num_x_steps = (max_iX - min_iX) + 1
        num_y_steps = (max_iY - min_iY) + 1
        
        return num_x_steps, num_y_steps

    def generate_snake_tiles(self, mExperiment) -> List[List[Dict[str, Any]]]:
        """
        Generate snake pattern tiles from experiment points.
        
        Args:
            mExperiment: Experiment object with pointList
            
        Returns:
            List of tiles with snake-pattern coordinate dictionaries
        """
        tiles = []
        for iCenter, centerPoint in enumerate(mExperiment.pointList):
            # Collect central and neighbour points
            allPoints = [(n.x, n.y) for n in centerPoint.neighborPointList]
            # Sort by y then by x (raster order)
            allPoints.sort(key=lambda coords: (coords[1], coords[0]))

            num_x_steps, num_y_steps = self.get_num_xy_steps(
                centerPoint.neighborPointList
            )
            allPointsSnake = [0] * (num_x_steps * num_y_steps)
            iTile = 0
            
            for iY in range(num_y_steps):
                for iX in range(num_x_steps):
                    if iY % 2 == 1 and num_x_steps != 1:
                        mIdex = iY * num_x_steps + num_x_steps - 1 - iX
                    else:
                        mIdex = iTile
                        
                    if (len(allPointsSnake) <= mIdex or 
                        len(allPoints) <= iTile):
                        allPointsSnake[mIdex] = None
                        continue
                        
                    allPointsSnake[mIdex] = {
                        "iterator": iTile,
                        "centerIndex": iCenter,
                        "iX": iX,
                        "iY": iY,
                        "x": allPoints[iTile][0],
                        "y": allPoints[iTile][1],
                    }
                    iTile += 1
            tiles.append(allPointsSnake)
            
        return tiles

    def compute_scan_ranges(self, snake_tiles: List[List[Dict[str, Any]]]) -> Tuple[float, float, float, float, float, float]:
        """
        Compute scan ranges from snake tiles.
        
        Args:
            snake_tiles: List of tiles with coordinate dictionaries
            
        Returns:
            Tuple of (minX, maxX, minY, maxY, diffX, diffY)
        """
        # Flatten all point dictionaries from all tiles
        all_points = [pt for tile in snake_tiles for pt in tile]
        
        minX = min(pt["x"] for pt in all_points)
        maxX = max(pt["x"] for pt in all_points)
        minY = min(pt["y"] for pt in all_points)
        maxY = max(pt["y"] for pt in all_points)
        
        # Compute step between two adjacent points in X/Y
        unique_x = np.unique([pt["x"] for pt in all_points])
        unique_y = np.unique([pt["y"] for pt in all_points])
        
        diffX = np.diff(unique_x).min() if len(unique_x) > 1 else 0
        diffY = np.diff(unique_y).min() if len(unique_y) > 1 else 0
        
        return minX, maxX, minY, maxY, diffX, diffY

    def validate_experiment_parameters(self, mExperiment) -> Dict[str, Any]:
        """
        Validate and normalize experiment parameters.
        
        Args:
            mExperiment: Experiment object to validate
            
        Returns:
            Dictionary of validated parameters
        """
        p = mExperiment.parameterValue
        
        # Normalize list parameters
        illumination_sources = p.illumination
        if not isinstance(illumination_sources, list):
            illumination_sources = [illumination_sources]
            
        illumination_intensities = p.illuIntensities  
        if not isinstance(illumination_intensities, list):
            illumination_intensities = [illumination_intensities]
            
        gains = p.gains
        if not isinstance(gains, list):
            gains = [gains]
            
        exposures = p.exposureTimes
        if not isinstance(exposures, list):
            exposures = [exposures]
            
        # Adjust list lengths to match illumination sources
        if len(gains) != len(illumination_sources):
            gains = [-1] * len(illumination_sources)
        if len(exposures) != len(illumination_sources):
            exposures = [exposures[0]] * len(illumination_sources)
            
        # Calculate Z-steps
        z_steps = 1
        if p.zStack:
            z_steps = int(
                (p.zStackMax - p.zStackMin) // p.zStackStepSize
            ) + 1
            
        return {
            'illumination_sources': illumination_sources,
            'illumination_intensities': illumination_intensities,
            'gains': gains,
            'exposures': exposures,
            'z_steps': z_steps,
            'performance_mode': p.performanceMode,
            'is_z_stack': p.zStack,
            'is_autofocus': p.autoFocus,
            'is_brightfield': p.brightfield,
            'is_darkfield': p.darkfield,
            'is_dpc': p.differentialPhaseContrast,
            'n_times': p.numberOfImages,
            't_period': p.timeLapsePeriod,
            'z_stack_min': p.zStackMin,
            'z_stack_max': p.zStackMax,
            'z_stack_step': p.zStackStepSize,
            'autofocus_min': p.autoFocusMin,
            'autofocus_max': p.autoFocusMax,
            'autofocus_step': p.autoFocusStepSize,
            'speed': p.speed if p.speed > 0 else None
        }