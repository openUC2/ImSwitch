# STORM Controller Renewal

This document describes the renewal of the STORM controller in ImSwitch, addressing issue #126.

## Overview

The STORM controller has been renewed to remove Qt widget dependencies and modernize the architecture with enhanced microeye integration, better parameter management, and improved data storage.

## Key Changes

### 1. Removed Widget Dependencies
- **Before**: Controller relied on Qt widgets for parameter management and UI updates
- **After**: Controller operates independently without widget dependencies
- **Impact**: Enables headless operation and better API-driven workflows

### 2. Enhanced Parameter Management with Pydantic
- **Added**: `storm_models.py` with pydantic models for type-safe parameter validation
- **Models**:
  - `STORMProcessingParameters`: microeye processing configuration
  - `STORMAcquisitionParameters`: acquisition settings
  - `STORMReconstructionResult`: processing results
  - `STORMStatusResponse`: status information

### 3. Modernized microEye Integration
- **Before**: Qt-threaded worker with signal-based communication
- **After**: Modern async processing with background threads
- **Improvements**:
  - Better error handling
  - Enhanced frame processing pipeline
  - Improved localization storage

### 4. Enhanced Data Storage
- **Pattern**: Follows experimentcontroller data organization
- **Structure**:
  ```
  UserFileDirs.Data/STORMController/YYYYMMDD_HHMMSS/session_id/
  ├── raw_frames/
  ├── reconstructed_frames/
  ├── localizations/
  └── final_reconstruction.tif
  ```

### 5. New REST API Endpoints

#### Parameter Management
- `setSTORMProcessingParameters(**kwargs)`: Set processing parameters with validation
- `getSTORMReconstructionStatus()`: Get comprehensive status

#### Local Processing Control
- `startSTORMReconstructionLocal(**kwargs)`: Start local microeye processing
- `stopSTORMReconstructionLocal()`: Stop and finalize local processing
- `getLastReconstructedImagePath()`: Get path to last reconstructed image
- `triggerSingleFrameReconstruction()`: Process single frame

### 6. Signal Updates for Frontend
- `sigFrameAcquired(int)`: Frame number acquired
- `sigFrameProcessed(STORMReconstructionResult)`: Processing result
- `sigAcquisitionStateChanged(bool)`: Acquisition state
- `sigProcessingStateChanged(bool)`: Processing state  
- `sigErrorOccurred(str)`: Error notifications

## API Usage Examples

### Basic Local Processing
```python
# Set processing parameters
result = controller.setSTORMProcessingParameters(
    threshold=0.3,
    fit_roi_size=15,
    fitting_method='2D_Phasor_CPU'
)

# Start local reconstruction
result = controller.startSTORMReconstructionLocal(
    session_id='my_session',
    process_locally=True,
    save_enabled=True
)

# Stop and get results
result = controller.stopSTORMReconstructionLocal()
final_path = controller.getLastReconstructedImagePath()
```

### Parameter Validation
```python
# Pydantic validates parameter types and ranges
try:
    controller.setSTORMProcessingParameters(
        threshold=1.5,  # Error: > 1.0
        fit_roi_size=6   # Error: < 7  
    )
except ValidationError as e:
    print(f"Invalid parameters: {e}")
```

### Status Monitoring
```python
status = controller.getSTORMReconstructionStatus()
print(f"Active: {status['acquisition_active']}")
print(f"Frames processed: {status['frames_processed']}")
print(f"Last reconstruction: {status['last_reconstruction_path']}")
```

## Asynchronous Behavior

### Arkitekt (Remote Processing)
- **Preserved**: All existing Arkitekt functionality
- **Enhanced**: Better integration with renewed parameter system
- **Usage**: Start/stop frame acquisition into zarr stores with signal updates

### microEye (Local Processing)  
- **New**: Enhanced local processing pipeline
- **Features**:
  - Fast frame acquisition using `getChunk()`
  - Background processing with modern threading
  - Local storage of raw and reconstructed data
  - Real-time localization and accumulation
  - Signal updates for frontend consumption

## Backward Compatibility

### Legacy API Support
- `setSTORMParameters()`: Marked deprecated, redirects to new method
- `triggerSTORMReconstruction()`: Preserved for compatibility
- Existing Arkitekt functions: Fully preserved

### Migration Path
1. **Immediate**: Use new API endpoints for enhanced functionality
2. **Gradual**: Replace legacy parameter setting with pydantic models
3. **Future**: Remove deprecated methods in next major version

## Files Added/Modified

### New Files
- `imswitch/imcontrol/model/storm_models.py`: Pydantic parameter models
- `examples/storm_controller_renewal_examples.py`: Usage examples

### Modified Files
- `imswitch/imcontrol/controller/controllers/STORMReconController.py`: 
  - Removed widget dependencies
  - Added new API endpoints
  - Modernized worker class
  - Enhanced microeye integration

### Updated Tests
- `imswitch/imcontrol/_test/unit/test_storm_recon_controller.py`:
  - Tests for new functionality
  - Widget-independent test cases
  - Parameter validation tests

## Benefits

1. **Headless Operation**: Controller works without Qt widgets
2. **Type Safety**: Pydantic models ensure parameter validation
3. **Better Data Organization**: Structured storage following best practices
4. **Enhanced Processing**: Improved microeye integration with better error handling
5. **Modern Architecture**: Async processing without Qt threading dependencies
6. **API-Driven**: Full functionality available via REST endpoints
7. **Frontend Integration**: Rich signal updates for UI consumption

## Future Enhancements

1. **Real-time Visualization**: Stream reconstructed frames to frontend
2. **Advanced Analytics**: Built-in localization analysis tools
3. **Cloud Integration**: Enhanced Arkitekt workflows
4. **Performance Optimization**: GPU acceleration for processing
5. **Multi-channel Support**: Simultaneous multi-color STORM

## Testing

Run the renewed controller tests:
```bash
python -m unittest imswitch.imcontrol._test.unit.test_storm_recon_controller.TestSTORMReconControllerRenewed
```

Run the example script:
```bash
python examples/storm_controller_renewal_examples.py
```

## Dependencies

### Required
- `pydantic>=2.0`: Parameter validation models
- `tifffile`: Image saving
- `numpy`: Array processing

### Optional
- `microeye`: Local STORM processing (when available)
- `arkitekt-next`: Remote processing (when available)
- `zarr`: OME-Zarr storage support

## Migration Checklist

- [ ] Update parameter setting to use new pydantic models
- [ ] Replace widget-dependent code with new API endpoints  
- [ ] Update frontend to consume new signal updates
- [ ] Test local processing workflows
- [ ] Verify Arkitekt integration still works
- [ ] Update documentation and examples
- [ ] Remove deprecated code in future version