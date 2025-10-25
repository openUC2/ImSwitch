# Focus Map Implementation for Global Z-Offset Correction

## Issue Summary

Implement a focus map functionality that computes global Z-offsets based on XY coordinates to compensate for sample tilt and stage non-planarity. This feature would be integrated into a new `FocusLockManager` and exposed to the `PositionerManager`/`ESP32StageManager` to automatically apply focus corrections during XY movements.

## Feature Description

### Core Functionality
The focus map system would:
1. **Calibrate focus at multiple XY positions** using autofocus or manual focus lock
2. **Generate a 3-point planar interpolation map** from calibration coordinates
3. **Automatically apply Z-offsets** during stage movements based on XY position
4. **Integrate with existing focus lock system** for seamless operation

### Use Cases
- **Sample tilt compensation**: Automatically adjust focus when scanning across tilted samples
- **Stage non-planarity correction**: Compensate for mechanical imperfections in the stage
- **Large area imaging**: Maintain focus consistency across wide field scans
- **Multi-position experiments**: Ensure consistent focus across different sample regions

## Technical Requirements

### 1. New FocusLockManager Class
Create a dedicated manager class to handle focus map operations:

```python
class FocusLockManager:
    def __init__(self, focus_lock_controller, positioner_manager):
        self.focus_lock_controller = focus_lock_controller
        self.positioner_manager = positioner_manager
        self.focus_map_coords = []  # List of (x, y, z) tuples
        self.use_focus_map = False
        self.focus_map_enabled = False
        
    # Core focus map methods (see implementation details below)
```

### 2. Core Methods Implementation

#### Focus Map Generation
```python
def gen_focus_map(self, coord1: Tuple[float, float], 
                  coord2: Tuple[float, float], 
                  coord3: Tuple[float, float]) -> bool:
    """
    Navigate to 3 coordinates and generate focus map by autofocusing.
    
    Args:
        coord1-3: Tuples of (x,y) values in mm
        
    Returns:
        bool: Success status
        
    Raises:
        ValueError: If coordinates are collinear
    """
```

#### Focus Map Validation
```python
def set_focus_map_use(self, enable: bool) -> bool:
    """
    Enable/disable focus map usage with validation.
    
    Args:
        enable: Whether to enable focus map
        
    Returns:
        bool: Whether focus map is now active
    """
```

#### Z-Offset Calculation
```python
def calculate_z_offset(self, x: float, y: float) -> float:
    """
    Calculate Z-offset for given XY coordinate using planar interpolation.
    
    Args:
        x, y: Coordinates in mm
        
    Returns:
        float: Z-offset in mm to apply
    """
```

### 3. Integration Points

#### PositionerManager Integration
Modify the positioner manager to automatically apply focus corrections:

```python
# In PositionerManager or ESP32StageManager
def move_with_focus_correction(self, x: float, y: float, z: float = None):
    """Move with automatic focus map correction."""
    if self.focus_manager and self.focus_manager.use_focus_map:
        z_offset = self.focus_manager.calculate_z_offset(x, y)
        if z is not None:
            z += z_offset
        else:
            # Apply offset to current Z
            current_z = self.getPosition()["Z"]
            z = current_z + z_offset
    
    # Perform the move
    self.move(x=x, y=y, z=z)
```

#### API Integration
```python
# New API endpoints
@APIExport
def generateFocusMap(self, coord1, coord2, coord3) -> Dict[str, Any]

@APIExport  
def addCurrentPositionToFocusMap(self) -> Dict[str, Any]

@APIExport
def setFocusMapEnabled(self, enabled: bool) -> bool

@APIExport
def getFocusMapData(self) -> Dict[str, Any]

@APIExport
def clearFocusMap() -> bool
```

## Detailed Implementation Plan

### 1. Focus Map Data Structure

```python
@dataclass
class FocusMapData:
    """Focus map calibration data."""
    coordinates: List[Tuple[float, float, float]]  # (x, y, z) in mm
    plane_coefficients: Optional[Tuple[float, float, float]] = None  # ax + by + c = z
    is_valid: bool = False
    timestamp: float = 0.0
    reference_position: Optional[Tuple[float, float, float]] = None
    
    def calculate_z_for_xy(self, x: float, y: float) -> float:
        """Calculate Z coordinate for given X,Y using plane equation."""
        if not self.is_valid or self.plane_coefficients is None:
            return 0.0
        a, b, c = self.plane_coefficients
        return a * x + b * y + c
        
    def get_z_offset(self, x: float, y: float) -> float:
        """Get Z offset relative to reference position."""
        if not self.reference_position:
            return 0.0
        predicted_z = self.calculate_z_for_xy(x, y)
        reference_z = self.reference_position[2]
        return predicted_z - reference_z
```

### 2. Planar Interpolation Algorithm

```python
def _calculate_plane_coefficients(self, coords: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Calculate plane coefficients from 3 points using least squares.
    Plane equation: z = ax + by + c
    
    Args:
        coords: List of (x, y, z) coordinates
        
    Returns:
        Tuple of (a, b, c) coefficients
        
    Raises:
        ValueError: If points are collinear
    """
    if len(coords) < 3:
        raise ValueError("Need at least 3 points for plane fitting")
        
    # Extract coordinates
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    z_coords = [coord[2] for coord in coords]
    
    # Check for collinearity
    x1, y1, z1 = coords[0]
    x2, y2, z2 = coords[1]
    x3, y3, z3 = coords[2]
    
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(det) < 1e-10:
        raise ValueError("Points are collinear - cannot fit plane")
    
    # Solve using least squares: [x y 1] * [a b c]^T = z
    import numpy as np
    A = np.column_stack([x_coords, y_coords, np.ones(len(coords))])
    coefficients = np.linalg.lstsq(A, z_coords, rcond=None)[0]
    
    return tuple(coefficients)
```

### 3. Autofocus Integration

```python
def _autofocus_at_position(self, x: float, y: float) -> float:
    """
    Move to position and perform autofocus to get optimal Z.
    
    Args:
        x, y: Target coordinates in mm
        
    Returns:
        float: Focused Z position in mm
    """
    # Move to XY position
    self.positioner_manager.move(x=x, y=y, is_blocking=True)
    
    # Enable focus lock measurement
    self.focus_lock_controller.startFocusMeasurement()
    
    # Wait for stable measurement
    time.sleep(1.0)
    
    # Perform focus optimization (could use various methods)
    if hasattr(self.focus_lock_controller, 'runAutofocus'):
        # Use dedicated autofocus if available
        result = self.focus_lock_controller.runAutofocus()
        focused_z = result.get('optimal_z')
    else:
        # Use focus lock calibration as autofocus
        current_z = self.positioner_manager.getPosition()["Z"]
        calib_result = self.focus_lock_controller.runFocusCalibration(
            from_position=current_z - 5.0,
            to_position=current_z + 5.0,
            num_steps=21
        )
        # Find Z position with maximum focus value
        focus_data = calib_result['focus_data']
        position_data = calib_result['position_data']
        max_focus_idx = np.argmax(focus_data)
        focused_z = position_data[max_focus_idx]
        
        # Move to optimal position
        self.positioner_manager.move(z=focused_z, is_blocking=True)
    
    return focused_z
```

### 4. Manager Class Structure

```python
class FocusLockManager:
    """
    Manager for focus map functionality and global focus corrections.
    Integrates with FocusLockController and PositionerManager.
    """
    
    def __init__(self, focus_lock_controller, positioner_manager):
        self.focus_lock_controller = focus_lock_controller
        self.positioner_manager = positioner_manager
        self.focus_map_data = FocusMapData()
        self._logger = initLogger(self)
        
    @APIExport(runOnUIThread=True)
    def gen_focus_map(self, coord1: Tuple[float, float], 
                      coord2: Tuple[float, float], 
                      coord3: Tuple[float, float]) -> Dict[str, Any]:
        """Generate focus map from 3 coordinates."""
        
    @APIExport(runOnUIThread=True)
    def add_current_coords_to_focus_map(self) -> Dict[str, Any]:
        """Add current position to focus map after autofocusing."""
        
    @APIExport(runOnUIThread=True)
    def set_focus_map_use(self, enable: bool) -> bool:
        """Enable/disable focus map usage."""
        
    @APIExport(runOnUIThread=True)
    def clear_focus_map(self) -> bool:
        """Clear current focus map."""
        
    @APIExport(runOnUIThread=True)
    def get_focus_map_data(self) -> Dict[str, Any]:
        """Get current focus map data."""
        
    def calculate_z_offset(self, x: float, y: float) -> float:
        """Calculate Z offset for given XY coordinates."""
        
    def is_focus_map_active(self) -> bool:
        """Check if focus map is active and valid."""
```

## Integration with Existing Systems

### 1. PositionerManager Wrapper
```python
# In PositionerManager
def set_focus_manager(self, focus_manager: 'FocusLockManager'):
    """Set focus manager for automatic focus correction."""
    self.focus_manager = focus_manager

def move(self, x=None, y=None, z=None, **kwargs):
    """Enhanced move method with focus correction."""
    # Apply focus correction if enabled
    if (self.focus_manager and 
        self.focus_manager.is_focus_map_active() and 
        x is not None and y is not None):
        
        z_offset = self.focus_manager.calculate_z_offset(x, y)
        if z is not None:
            z += z_offset
        else:
            current_z = self.getPosition()["Z"]
            z = current_z + z_offset
            
        self._logger.debug(f"Applied focus map offset: {z_offset:.3f} mm at ({x:.2f}, {y:.2f})")
    
    # Call original move method
    return self._original_move(x=x, y=y, z=z, **kwargs)
```

### 2. Setup Configuration
```python
# In setup configuration
focusMap:
  enabled: true
  autofocus_range_mm: 10.0  # Range for autofocus scan
  autofocus_steps: 21       # Steps for autofocus
  validation_threshold: 0.95 # R² threshold for plane fitting
  max_offset_mm: 5.0        # Maximum allowed offset
```

## Expected Benefits

1. **Automated Focus Maintenance**: Eliminates manual focus adjustments during large scans
2. **Improved Image Quality**: Consistent focus across entire sample area
3. **Time Savings**: Reduces need for frequent manual refocusing
4. **Robustness**: Compensates for mechanical imperfections and sample preparation variations
5. **Scalability**: Works with any number of calibration points (minimum 3)

## Testing Strategy

### 1. Validation Tests
- [ ] Test with known tilted samples
- [ ] Verify mathematical accuracy of plane fitting
- [ ] Test edge cases (collinear points, extreme tilts)
- [ ] Performance testing with different calibration point counts

### 2. Integration Tests  
- [ ] Test with different positioner types
- [ ] Verify compatibility with existing focus lock functionality
- [ ] Test API endpoints and parameter validation
- [ ] Long-term stability testing

### 3. User Experience Tests
- [ ] Ease of calibration workflow
- [ ] Clear error messages and validation feedback
- [ ] Performance impact on movement speed
- [ ] Accuracy across different magnifications

## Implementation Priority

1. **HIGH**: Core focus map data structures and algorithms
2. **HIGH**: Basic 3-point calibration and plane fitting
3. **HIGH**: Integration with PositionerManager for automatic offset application  
4. **MEDIUM**: Autofocus integration for calibration
5. **MEDIUM**: API endpoints and configuration management
6. **LOW**: Advanced features (more than 3 points, non-planar interpolation)

## Related Files

- `imswitch/imcontrol/controller/controllers/FocusLockController.py`
- `imswitch/imcontrol/model/managers/positioners/PositionerManager.py`  
- `imswitch/imcontrol/model/managers/positioners/ESP32StageManager.py`
- Setup configuration files

---

**Labels**: `enhancement`, `focus-lock`, `focus-map`, `stage-control`, `autofocus`  
**Milestone**: Focus Management v2.0  
**Priority**: High  
**Estimated Effort**: 2-3 weeks
