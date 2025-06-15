# STORMReconController API Usage Examples

The reorganized STORMReconController provides a backend/client API for fast STORM frame acquisition and analysis. Here are usage examples:

## Basic Fast Acquisition

```python
# Start fast STORM acquisition with basic parameters
result = controller.startFastSTORMAcquisition(
    session_id="experiment_001",
    exposure_time=10.0  # ms
)

if result['success']:
    print(f"Started session: {result['session_id']}")
    
    # Get frames using the async generator
    for frame_data in controller.getSTORMFrameGenerator(num_frames=1000):
        if 'error' in frame_data:
            print(f"Error: {frame_data['error']}")
            break
            
        # Access raw frame data
        raw_frame = frame_data['raw_frame']
        
        # Access processed frame (if microEye is available)
        processed_frame = frame_data['processed_frame']
        
        # Access localization parameters
        localizations = frame_data['localization_params']
        
        # Access metadata
        metadata = frame_data['metadata']
        print(f"Frame {metadata['frame_number']} at {metadata['timestamp']}")
        
    # Stop acquisition
    stop_result = controller.stopFastSTORMAcquisition()
    print(f"Stopped: {stop_result['message']}")
```

## Acquisition with Cropping

```python
# Start acquisition with cropping to reduce data size
result = controller.startFastSTORMAcquisition(
    session_id="cropped_experiment",
    crop_x=100,         # Top-left X coordinate
    crop_y=100,         # Top-left Y coordinate  
    crop_width=512,     # Width of crop region
    crop_height=512,    # Height of crop region
    exposure_time=5.0
)

print(f"Cropping: {result['cropping']}")

# Frames will now be cropped to the specified region
for frame_data in controller.getSTORMFrameGenerator(num_frames=500):
    cropped_frame = frame_data['raw_frame']
    print(f"Cropped frame shape: {cropped_frame.shape}")
    # Should be (512, 512) instead of full sensor size
```

## Acquisition with OME-Zarr Saving

```python
import os
from datetime import datetime

# Create save path
save_dir = "/path/to/data"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"storm_data_{timestamp}.zarr")

# Start acquisition with automatic OME-Zarr saving
result = controller.startFastSTORMAcquisition(
    session_id="saved_experiment",
    save_path=save_path,
    save_format="omezarr",
    crop_x=50, crop_y=50,
    crop_width=400, crop_height=400
)

if result['success']:
    print(f"Saving to: {result['save_path']}")
    
    # Frames are automatically saved as acquired
    for frame_data in controller.getSTORMFrameGenerator(num_frames=2000):
        # Just process metadata - frames are saved automatically
        metadata = frame_data['metadata']
        if metadata['frame_number'] % 100 == 0:
            print(f"Saved frame {metadata['frame_number']}")
    
    # Stop and finalize saving
    controller.stopFastSTORMAcquisition()
    print(f"Data saved to {save_path}")
```

## Status Monitoring

```python
# Check current status
status = controller.getSTORMStatus()
print(f"Acquisition active: {status['acquisition_active']}")
print(f"Current session: {status['session_id']}")
print(f"MicroEye available: {status['microeye_available']}")
print(f"Cropping params: {status['cropping_params']}")

# Configure STORM processing parameters  
params = controller.setSTORMParameters(
    threshold=0.3,      # Detection threshold
    roi_size=15,        # ROI size for fitting
    update_rate=2       # Update rate for live processing
)
print(f"Current parameters: {params}")
```

## Integration with External Systems

```python
# Example: Integration with Arkitekt-style workflow
async def storm_acquisition_workflow():
    """Example async workflow for STORM acquisition."""
    
    # Configure acquisition
    session_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start acquisition
    result = controller.startFastSTORMAcquisition(
        session_id=session_id,
        crop_x=200, crop_y=200,
        crop_width=300, crop_height=300,
        save_path=f"/data/{session_id}.zarr",
        save_format="omezarr"
    )
    
    if not result['success']:
        raise Exception(f"Failed to start acquisition: {result['message']}")
    
    try:
        # Process frames in real-time
        frame_count = 0
        localization_count = 0
        
        for frame_data in controller.getSTORMFrameGenerator(num_frames=5000):
            if 'error' in frame_data:
                print(f"Acquisition error: {frame_data['error']}")
                break
                
            frame_count += 1
            
            # Count localizations if available
            if frame_data['localization_params'] is not None:
                localization_count += len(frame_data['localization_params'])
            
            # Report progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, {localization_count} localizations")
                
        print(f"Acquisition complete: {frame_count} frames, {localization_count} total localizations")
        
    finally:
        # Ensure acquisition is stopped
        controller.stopFastSTORMAcquisition()

# For synchronous usage
# asyncio.run(storm_acquisition_workflow())
```

## Error Handling

```python
try:
    # Start acquisition
    result = controller.startFastSTORMAcquisition(session_id="error_test")
    
    if not result['success']:
        print(f"Failed to start: {result['message']}")
        return
    
    # Process frames with error handling
    for frame_data in controller.getSTORMFrameGenerator(num_frames=1000, timeout=5.0):
        if 'error' in frame_data:
            print(f"Frame error at {frame_data['timestamp']}: {frame_data['error']}")
            break
            
        # Process frame...
        
except Exception as e:
    print(f"Unexpected error: {e}")
    
finally:
    # Always ensure acquisition is stopped
    if controller.getSTORMStatus()['acquisition_active']:
        controller.stopFastSTORMAcquisition()
```

## Legacy Compatibility

The original STORMReconController functionality is preserved:

```python
# Original trigger reconstruction method still works
controller.triggerSTORMReconstruction()

# Original GUI integration still works
controller.setShowSTORMRecon(enabled=True)
controller.changeRate(updateRate=5)

# Worker-based processing still available
# (automatically used when microEye is available)
```