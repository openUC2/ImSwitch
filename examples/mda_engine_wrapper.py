#!/usr/bin/env python3
"""
MDA Engine Wrapper for ImSwitch

This example demonstrates how to use useq-schema's MDASequence with ImSwitch's
hardware abstraction layer (HAL), similar to how pymmcore-plus uses it.

The pattern follows:
1. Define hardware control methods that match useq-schema's expectations
2. Create an engine class that executes MDAEvents using ImSwitch's HAL
3. Use useq-schema's MDASequence to generate events
4. Execute the sequence with ImSwitch hardware managers

This approach allows you to:
- Use the same MDA protocol as pymmcore-plus
- Leverage ImSwitch's hardware abstraction layer
- Share experiment protocols between different systems
- Benefit from useq-schema's validation and features

References:
- https://github.com/ianhi/raman-mda-engine
- https://github.com/ddd42-star/opto-loop-sim
- https://github.com/pymmcore-plus/useq-schema
"""

import time
from typing import Any, Callable, Dict, Optional, Protocol
from dataclasses import dataclass
import numpy as np

try:
    from useq import MDASequence, MDAEvent
    HAS_USEQ = True
except ImportError:
    HAS_USEQ = False
    print("Warning: useq-schema not installed. Install with: pip install useq-schema")


@dataclass
class ImSwitchHardwareState:
    """Tracks the current state of ImSwitch hardware."""
    current_channel: Optional[str] = None
    current_exposure: Optional[float] = None
    current_z_pos: Optional[float] = None
    current_xy_pos: Optional[tuple] = None
    detector_running: bool = False


class ImSwitchMDAEngine:
    """
    MDA Engine that executes useq-schema MDASequence using ImSwitch's HAL.
    
    This engine bridges useq-schema's standardized acquisition protocol with
    ImSwitch's hardware managers, similar to how pymmcore-plus integrates with
    Micro-Manager.
    
    Example usage:
        # In your ImSwitch controller:
        engine = ImSwitchMDAEngine(
            detector_manager=self._master.detectorsManager,
            positioners_manager=self._master.positionersManager,
            lasers_manager=self._master.lasersManager
        )
        
        # Define an MDA sequence
        from useq import MDASequence, Channel, ZRangeAround
        
        sequence = MDASequence(
            channels=[
                Channel(config="DAPI", exposure=50.0),
                Channel(config="FITC", exposure=100.0)
            ],
            z_plan=ZRangeAround(range=10.0, step=2.0),
            time_plan={"interval": 30.0, "loops": 5}
        )
        
        # Run the sequence
        engine.run(sequence, output_path="/path/to/save")
    """
    
    def __init__(
        self,
        detector_manager: Any,
        positioners_manager: Any,
        lasers_manager: Any,
        logger: Optional[Any] = None
    ):
        """
        Initialize the MDA engine with ImSwitch managers.
        
        Args:
            detector_manager: ImSwitch DetectorsManager instance
            positioners_manager: ImSwitch PositionersManager instance
            lasers_manager: ImSwitch LasersManager instance
            logger: Optional logger instance
        """
        self.detector_manager = detector_manager
        self.positioners_manager = positioners_manager
        self.lasers_manager = lasers_manager
        self.logger = logger
        
        self.state = ImSwitchHardwareState()
        self._hooks_before_event = []
        self._hooks_after_event = []
        self._is_running = False
        
    def register_hook_before_event(self, func: Callable[[MDAEvent], None]):
        """Register a callback to run before each event."""
        self._hooks_before_event.append(func)
        
    def register_hook_after_event(self, func: Callable[[MDAEvent, np.ndarray], None]):
        """Register a callback to run after each event (receives image data)."""
        self._hooks_after_event.append(func)
        
    def run(
        self,
        sequence: MDASequence,
        output_path: Optional[str] = None,
        blocking: bool = True
    ) -> None:
        """
        Execute an MDA sequence using ImSwitch hardware.
        
        Args:
            sequence: useq-schema MDASequence to execute
            output_path: Optional path to save acquired images
            blocking: If True, block until sequence completes
        """
        if not HAS_USEQ:
            raise RuntimeError("useq-schema not installed")
            
        self._is_running = True
        
        # Ensure detector is running
        if not self.state.detector_running:
            self._start_detector()
            
        try:
            # Execute each event in the sequence
            for event in sequence:
                if not self._is_running:
                    break
                    
                # Run pre-event hooks
                for hook in self._hooks_before_event:
                    hook(event)
                    
                # Execute the event
                image = self._execute_event(event)
                
                # Run post-event hooks
                for hook in self._hooks_after_event:
                    hook(event, image)
                    
                # Handle saving if output path specified
                if output_path and image is not None:
                    self._save_image(image, event, output_path)
                    
        finally:
            self._is_running = False
            self._cleanup()
            
    def _execute_event(self, event: MDAEvent) -> Optional[np.ndarray]:
        """
        Execute a single MDAEvent.
        
        This is where the useq-schema event is translated to ImSwitch
        hardware commands.
        """
        # Move XY stage if position specified
        if event.x_pos is not None or event.y_pos is not None:
            self._move_xy(event.x_pos or 0, event.y_pos or 0)
            
        # Move Z stage if position specified
        if event.z_pos is not None:
            self._move_z(event.z_pos)
            
        # Setup channel (illumination + exposure)
        if event.channel is not None:
            self._setup_channel(event.channel, event.exposure)
            
        # Wait for any specified minimum start time
        if event.min_start_time is not None:
            # Calculate wait time based on sequence start
            # This would need sequence-level timing tracking
            pass
            
        # Acquire image
        image = self._acquire_image()
        
        # Cleanup illumination
        if event.channel is not None:
            self._cleanup_channel(event.channel)
            
        return image
        
    def _move_xy(self, x: float, y: float) -> None:
        """Move XY stage to absolute position."""
        if self.positioners_manager is not None:
            # Get the stage positioner (adapt to your setup)
            stage = self.positioners_manager.getAllDeviceNames()[0]
            
            # Move to position (API depends on your PositionersManager implementation)
            # This is a simplified example - adapt to your actual API
            try:
                self.positioners_manager[stage].move(x, "X", absolute=True)
                self.positioners_manager[stage].move(y, "Y", absolute=True)
                self.state.current_xy_pos = (x, y)
                
                if self.logger:
                    self.logger.debug(f"Moved stage to ({x}, {y})")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error moving XY stage: {e}")
                    
    def _move_z(self, z: float) -> None:
        """Move Z stage to absolute position."""
        if self.positioners_manager is not None:
            stage = self.positioners_manager.getAllDeviceNames()[0]
            
            try:
                self.positioners_manager[stage].move(z, "Z", absolute=True)
                self.state.current_z_pos = z
                
                if self.logger:
                    self.logger.debug(f"Moved Z stage to {z}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error moving Z stage: {e}")
                    
    def _setup_channel(self, channel: Any, exposure: Optional[float] = None) -> None:
        """
        Setup illumination and exposure for a channel.
        
        Args:
            channel: useq Channel object with config and optional exposure
            exposure: Override exposure time in milliseconds
        """
        channel_name = channel.config
        exposure_time = exposure or channel.exposure or 100.0
        
        # Set up illumination
        if self.lasers_manager is not None and channel_name in self.lasers_manager.getAllDeviceNames():
            try:
                # Turn on laser/LED at appropriate power
                # Get power from channel config if available
                power = getattr(channel, 'power', 100.0)
                self.lasers_manager[channel_name].setValue(power)
                self.lasers_manager[channel_name].setEnabled(True)
                
                if self.logger:
                    self.logger.debug(f"Enabled {channel_name} at power {power}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error setting up channel {channel_name}: {e}")
                    
        # Set detector exposure
        if self.detector_manager is not None:
            try:
                detector = self.detector_manager.getAllDeviceNames()[0]
                # Set exposure time (API depends on your detector)
                # This is simplified - adapt to your actual detector API
                if hasattr(self.detector_manager[detector], 'setExposure'):
                    self.detector_manager[detector].setExposure(exposure_time)
                    
                self.state.current_exposure = exposure_time
                
                if self.logger:
                    self.logger.debug(f"Set exposure to {exposure_time} ms")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error setting exposure: {e}")
                    
        self.state.current_channel = channel_name
        
    def _cleanup_channel(self, channel: Any) -> None:
        """Turn off illumination after acquisition."""
        channel_name = channel.config
        
        if self.lasers_manager is not None and channel_name in self.lasers_manager.getAllDeviceNames():
            try:
                self.lasers_manager[channel_name].setEnabled(False)
                
                if self.logger:
                    self.logger.debug(f"Disabled {channel_name}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error cleaning up channel: {e}")
                    
    def _acquire_image(self) -> Optional[np.ndarray]:
        """Acquire a single image from the detector."""
        if self.detector_manager is None:
            return None
            
        try:
            detector = self.detector_manager.getAllDeviceNames()[0]
            # Get latest frame (API depends on your detector)
            image = self.detector_manager[detector].getLatestFrame()
            
            if self.logger:
                self.logger.debug(f"Acquired image: shape={image.shape if image is not None else None}")
                
            return image
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error acquiring image: {e}")
            return None
            
    def _start_detector(self) -> None:
        """Start the detector if not already running."""
        if self.detector_manager is not None:
            try:
                detector = self.detector_manager.getAllDeviceNames()[0]
                if hasattr(self.detector_manager[detector], 'startAcquisition'):
                    self.detector_manager[detector].startAcquisition()
                    self.state.detector_running = True
                    
                    if self.logger:
                        self.logger.info("Started detector acquisition")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error starting detector: {e}")
                    
    def _cleanup(self) -> None:
        """Cleanup after sequence completion."""
        # Turn off all lasers
        if self.lasers_manager is not None:
            for laser_name in self.lasers_manager.getAllDeviceNames():
                try:
                    self.lasers_manager[laser_name].setEnabled(False)
                except:
                    pass
                    
    def _save_image(self, image: np.ndarray, event: MDAEvent, output_path: str) -> None:
        """
        Save acquired image with metadata.
        
        This is a simplified example - you would typically use a proper
        imaging format like OME-TIFF with full metadata.
        """
        import os
        import tifffile
        
        # Create filename from event metadata
        t_idx = event.index.get('t', 0)
        c_idx = event.index.get('c', 0)
        z_idx = event.index.get('z', 0)
        p_idx = event.index.get('p', 0)
        
        filename = f"t{t_idx:04d}_p{p_idx:04d}_c{c_idx:02d}_z{z_idx:04d}.tif"
        filepath = os.path.join(output_path, filename)
        
        # Save with basic metadata
        metadata = {
            'event_index': dict(event.index),
            'channel': event.channel.config if event.channel else None,
            'exposure': event.exposure,
            'z_pos': event.z_pos,
            'x_pos': event.x_pos,
            'y_pos': event.y_pos,
        }
        
        os.makedirs(output_path, exist_ok=True)
        tifffile.imwrite(filepath, image, metadata=metadata)
        
        if self.logger:
            self.logger.debug(f"Saved image to {filepath}")
            
    def stop(self) -> None:
        """Stop the currently running sequence."""
        self._is_running = False


def example_usage_with_imswitch():
    """
    Example showing how to use the MDA engine within ImSwitch.
    
    This would typically be called from an ImSwitch controller or API endpoint.
    """
    # This example assumes you have access to ImSwitch managers
    # In practice, you'd get these from your controller's _master object
    
    if not HAS_USEQ:
        print("Please install useq-schema: pip install useq-schema")
        return
        
    from useq import MDASequence, Channel, ZRangeAround, TIntervalLoops
    
    print("=== ImSwitch MDA Engine Example ===\n")
    
    # Mock managers for demonstration
    # In real usage, you'd pass actual ImSwitch managers:
    # engine = ImSwitchMDAEngine(
    #     detector_manager=self._master.detectorsManager,
    #     positioners_manager=self._master.positionersManager,
    #     lasers_manager=self._master.lasersManager,
    #     logger=self._logger
    # )
    
    # Create a multi-dimensional sequence
    sequence = MDASequence(
        channels=[
            Channel(config="DAPI", exposure=50.0),
            Channel(config="FITC", exposure=100.0),
        ],
        z_plan=ZRangeAround(range=10.0, step=2.0),  # 10μm range, 2μm steps
        time_plan=TIntervalLoops(interval=30.0, loops=5),  # 5 timepoints, 30s interval
    )
    
    print(f"Created MDA sequence with:")
    print(f"  - {len(list(sequence))} total events")
    print(f"  - Channels: DAPI, FITC")
    print(f"  - Z-stack: 10μm range, 2μm steps")
    print(f"  - Time-lapse: 5 timepoints, 30s interval")
    print()
    
    # Add custom hooks
    def before_event_hook(event: MDAEvent):
        print(f"  About to execute event: {dict(event.index)}")
        
    def after_event_hook(event: MDAEvent, image: np.ndarray):
        shape_str = f"{image.shape}" if image is not None else "None"
        print(f"  Completed event, image shape: {shape_str}")
    
    # In real usage:
    # engine.register_hook_before_event(before_event_hook)
    # engine.register_hook_after_event(after_event_hook)
    # engine.run(sequence, output_path="/path/to/save")
    
    print("To use with real hardware, instantiate ImSwitchMDAEngine with")
    print("your ImSwitch managers and call engine.run(sequence)")


if __name__ == "__main__":
    example_usage_with_imswitch()
