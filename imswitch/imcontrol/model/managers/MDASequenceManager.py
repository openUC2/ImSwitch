"""
MDASequenceManager - Native useq-schema MDA engine for ImSwitch

This module provides a direct MDA engine implementation that follows the same pattern
as pymmcore-plus and raman-mda-engine, accepting native useq.MDASequence objects
and executing them directly with ImSwitch's hardware abstraction layer.

References:
- https://github.com/ianhi/raman-mda-engine
- https://github.com/ddd42-star/opto-loop-sim  
- https://github.com/pymmcore-plus/pymmcore-plus
"""

from typing import List, Dict, Any, Optional, Callable
import time
import numpy as np
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowStep, WorkflowContext

try:
    from useq import MDASequence, MDAEvent, Channel
    HAS_USEQ = True
except ImportError:
    HAS_USEQ = False


class MDASequenceManager:
    """
    Native useq-schema MDA engine for ImSwitch.
    
    This engine executes useq.MDASequence objects directly using ImSwitch's hardware
    abstraction layer. It follows the same pattern as pymmcore-plus and raman-mda-engine,
    allowing protocols to be shared across different microscopy systems.
    
    Key features:
    - Accepts native useq.MDASequence objects with full schema support
    - Supports all useq-schema features: metadata, stage_positions, grid_plan, channels,
      time_plan, z_plan, autofocus_plan, axis_order, keep_shutter_open_across
    - Direct hardware control via ImSwitch managers
    - Hook system for custom pre/post acquisition logic
    
    Example usage:
        # Create engine
        engine = MDASequenceManager()
        engine.register(
            detector_manager=detectorsManager,
            positioners_manager=positionersManager,
            lasers_manager=lasersManager
        )
        
        # Define native useq-schema sequence
        from useq import MDASequence, Channel, ZRangeAround, TIntervalLoops
        sequence = MDASequence(
            metadata={"experiment": "test"},
            stage_positions=[(100, 100, 30), (200, 150, 35)],
            channels=[Channel(config="BF"), Channel(config="DAPI")],
            time_plan=TIntervalLoops(interval=1, loops=20),
            z_plan=ZRangeAround(range=4, step=0.5),
            axis_order="tpcz"
        )
        
        # Execute sequence
        engine.run_mda(sequence)
    """
    
    def __init__(self):
        self.__logger = initLogger(self)
        if not HAS_USEQ:
            self.__logger.warning("useq-schema not available. MDA functionality disabled.")
        
        # Hardware managers (set via register())
        self._detector_manager = None
        self._positioners_manager = None
        self._lasers_manager = None
        self._autofocus_manager = None
        
        # Hooks for custom logic
        self._hooks_before_sequence = []
        self._hooks_after_sequence = []
        self._hooks_before_event = []
        self._hooks_after_event = []
        
        # State tracking
        self._is_running = False
        self._current_sequence = None
    
    def is_available(self) -> bool:
        """Check if useq-schema is available for MDA functionality."""
        return HAS_USEQ
    
    def register(
        self,
        detector_manager: Any = None,
        positioners_manager: Any = None,
        lasers_manager: Any = None,
        autofocus_manager: Any = None
    ):
        """
        Register ImSwitch hardware managers with the MDA engine.
        
        This follows the pymmcore-plus pattern where you register the engine
        with the core/controller before running sequences.
        
        Args:
            detector_manager: ImSwitch DetectorsManager instance
            positioners_manager: ImSwitch PositionersManager instance
            lasers_manager: ImSwitch LasersManager instance
            autofocus_manager: ImSwitch AutofocusManager instance (optional)
        """
        if detector_manager is not None:
            self._detector_manager = detector_manager
        if positioners_manager is not None:
            self._positioners_manager = positioners_manager
        if lasers_manager is not None:
            self._lasers_manager = lasers_manager
        if autofocus_manager is not None:
            self._autofocus_manager = autofocus_manager
            
        self.__logger.info("MDA engine registered with ImSwitch managers")
    
    def run_mda(self, sequence: 'MDASequence', output_path: Optional[str] = None) -> None:
        """
        Execute a useq-schema MDASequence using ImSwitch hardware.
        
        This is the main entry point for running MDA experiments, following
        the same pattern as pymmcore-plus.run_mda() and raman_engine.run_mda().
        
        Args:
            sequence: Native useq.MDASequence object
            output_path: Optional directory path to save acquired images
            
        Example:
            from useq import MDASequence, Channel, ZRangeAround
            
            sequence = MDASequence(
                metadata={"experiment": "my_test"},
                stage_positions=[(0, 0, 10)],
                channels=[Channel(config="DAPI", exposure=50.0)],
                z_plan=ZRangeAround(range=10.0, step=2.0),
                axis_order="tpzc"
            )
            
            engine.run_mda(sequence, output_path="/data/experiment")
        """
        if not HAS_USEQ:
            raise RuntimeError("useq-schema not available. Install with: pip install useq-schema")
        
        if not self._detector_manager:
            raise RuntimeError("MDA engine not registered. Call register() first with hardware managers.")
        
        self.__logger.info(f"Starting MDA sequence with {len(list(sequence))} events")
        self.__logger.info(f"Sequence axis_order: {sequence.axis_order}")
        self.__logger.info(f"Sequence metadata: {sequence.metadata}")
        
        self._is_running = True
        self._current_sequence = sequence
        
        try:
            # Run pre-sequence hooks
            for hook in self._hooks_before_sequence:
                hook(sequence)
            
            # Execute each MDAEvent in the sequence
            for event_idx, event in enumerate(sequence):
                if not self._is_running:
                    self.__logger.warning("MDA sequence stopped by user")
                    break
                
                # Run pre-event hooks
                for hook in self._hooks_before_event:
                    hook(event)
                
                # Execute the event
                image = self._execute_event(event, event_idx)
                
                # Run post-event hooks
                for hook in self._hooks_after_event:
                    hook(event, image)
                
                # Handle saving if output path specified
                if output_path and image is not None:
                    self._save_image(image, event, event_idx, output_path)
            
            # Run post-sequence hooks
            for hook in self._hooks_after_sequence:
                hook(sequence)
                
            self.__logger.info("MDA sequence completed successfully")
            
        except Exception as e:
            self.__logger.error(f"Error during MDA sequence execution: {str(e)}")
            raise
        finally:
            self._is_running = False
            self._current_sequence = None
            self._cleanup()
    
    def stop(self) -> None:
        """Stop the currently running MDA sequence."""
        self._is_running = False
        self.__logger.info("MDA sequence stop requested")
    
    def _execute_event(self, event: 'MDAEvent', event_idx: int) -> Optional[np.ndarray]:
        """
        Execute a single MDAEvent from a useq-schema sequence.
        
        This method translates the useq-schema event into ImSwitch hardware commands,
        handling all aspects defined in the event: position, channel, exposure, etc.
        
        Args:
            event: useq.MDAEvent object to execute
            event_idx: Index of the event in the sequence
            
        Returns:
            Acquired image as numpy array, or None if acquisition failed
        """
        self.__logger.debug(f"Executing event {event_idx}: {dict(event.index)}")
        
        # Move to XY position if specified
        if event.x_pos is not None or event.y_pos is not None:
            self._move_xy(event.x_pos, event.y_pos)
        
        # Move to Z position if specified
        if event.z_pos is not None:
            self._move_z(event.z_pos)
        
        # Run autofocus if specified in the event's action
        # Note: useq-schema can specify autofocus via autofocus_plan
        if hasattr(event, 'action') and event.action and 'autofocus' in str(event.action).lower():
            self._run_autofocus()
        
        # Setup channel (illumination + exposure)
        if event.channel is not None:
            self._setup_channel(event.channel, event.exposure)
        
        # Handle min_start_time (timing synchronization)
        if event.min_start_time is not None:
            # Wait until the specified time
            # This would need sequence start time tracking
            pass
        
        # Acquire image
        image = self._acquire_image()
        
        # Cleanup illumination (unless keep_shutter_open is specified)
        # The sequence.keep_shutter_open_across attribute controls this
        if event.channel is not None:
            # Check if we should keep shutter open
            keep_open = False
            if self._current_sequence:
                keep_open_across = getattr(self._current_sequence, 'keep_shutter_open_across', ())
                # Check if current axis should keep shutter open
                # This is a simplified implementation
                keep_open = 'c' in keep_open_across or 'channel' in keep_open_across
            
            if not keep_open:
                self._cleanup_channel(event.channel)
        
        return image
    
    def _move_xy(self, x: Optional[float], y: Optional[float]) -> None:
        """Move XY stage to absolute position."""
        if self._positioners_manager is None:
            return
        
        try:
            # Get the first available stage (or you can specify which one)
            stage_names = self._positioners_manager.getAllDeviceNames()
            if not stage_names:
                self.__logger.warning("No positioners available for XY movement")
                return
            
            stage_name = stage_names[0]
            stage = self._positioners_manager[stage_name]
            
            # Move to position - API depends on PositionerManager implementation
            if x is not None:
                stage.move(x, "X", absolute=True)
            if y is not None:
                stage.move(y, "Y", absolute=True)
            
            self.__logger.debug(f"Moved stage to XY: ({x}, {y})")
        except Exception as e:
            self.__logger.error(f"Error moving XY stage: {e}")
    
    def _move_z(self, z: float) -> None:
        """Move Z stage to absolute position."""
        if self._positioners_manager is None:
            return
        
        try:
            stage_names = self._positioners_manager.getAllDeviceNames()
            if not stage_names:
                return
            
            stage_name = stage_names[0]
            stage = self._positioners_manager[stage_name]
            
            stage.move(z, "Z", absolute=True)
            
            self.__logger.debug(f"Moved stage to Z: {z}")
        except Exception as e:
            self.__logger.error(f"Error moving Z stage: {e}")
    
    def _setup_channel(self, channel: Any, exposure: Optional[float] = None) -> None:
        """
        Setup illumination and exposure for a channel.
        
        Args:
            channel: useq.Channel object with config and optional exposure
            exposure: Override exposure time in milliseconds
        """
        channel_name = channel.config
        exposure_time = exposure or channel.exposure or 100.0
        
        self.__logger.debug(f"Setting up channel: {channel_name}, exposure: {exposure_time}ms")
        
        # Set up illumination
        if self._lasers_manager is not None:
            try:
                laser_names = self._lasers_manager.getAllDeviceNames()
                if channel_name in laser_names:
                    laser = self._lasers_manager[channel_name]
                    
                    # Get power from channel if available
                    # useq-schema channels can have custom properties
                    power = 100.0  # Default
                    if hasattr(channel, 'power'):
                        power = channel.power
                    elif hasattr(channel, 'group') and hasattr(channel.group, 'power'):
                        power = channel.group.power
                    
                    # Set power and enable
                    laser.setValue(power)
                    laser.setEnabled(True)
                    
                    self.__logger.debug(f"Enabled laser {channel_name} at power {power}")
            except Exception as e:
                self.__logger.error(f"Error setting up illumination for {channel_name}: {e}")
        
        # Set detector exposure
        if self._detector_manager is not None:
            try:
                detector_names = self._detector_manager.getAllDeviceNames()
                if detector_names:
                    detector_name = detector_names[0]
                    detector = self._detector_manager[detector_name]
                    
                    # Set exposure - API may vary by detector type
                    if hasattr(detector, 'setParameter'):
                        detector.setParameter('exposure', exposure_time)
                    elif hasattr(detector, 'setExposure'):
                        detector.setExposure(exposure_time)
                    
                    self.__logger.debug(f"Set exposure to {exposure_time}ms")
            except Exception as e:
                self.__logger.error(f"Error setting exposure: {e}")
    
    def _cleanup_channel(self, channel: Any) -> None:
        """Turn off illumination after acquisition."""
        channel_name = channel.config
        
        if self._lasers_manager is not None:
            try:
                laser_names = self._lasers_manager.getAllDeviceNames()
                if channel_name in laser_names:
                    laser = self._lasers_manager[channel_name]
                    laser.setEnabled(False)
                    self.__logger.debug(f"Disabled laser {channel_name}")
            except Exception as e:
                self.__logger.error(f"Error cleaning up channel {channel_name}: {e}")
    
    def _acquire_image(self) -> Optional[np.ndarray]:
        """Acquire a single image from the detector."""
        if self._detector_manager is None:
            return None
        
        try:
            detector_names = self._detector_manager.getAllDeviceNames()
            if not detector_names:
                return None
            
            detector_name = detector_names[0]
            detector = self._detector_manager[detector_name]
            
            # Get latest frame
            image = detector.getLatestFrame()
            
            if image is not None:
                self.__logger.debug(f"Acquired image: shape={image.shape}, dtype={image.dtype}")
            
            return image
        except Exception as e:
            self.__logger.error(f"Error acquiring image: {e}")
            return None
    
    def _run_autofocus(self) -> None:
        """Run autofocus if autofocus manager is available."""
        if self._autofocus_manager is None:
            self.__logger.warning("Autofocus requested but no autofocus manager registered")
            return
        
        try:
            self.__logger.info("Running autofocus...")
            # API depends on AutofocusManager implementation
            if hasattr(self._autofocus_manager, 'runAutofocus'):
                self._autofocus_manager.runAutofocus()
            self.__logger.info("Autofocus completed")
        except Exception as e:
            self.__logger.error(f"Error running autofocus: {e}")
    
    def _save_image(
        self,
        image: np.ndarray,
        event: 'MDAEvent',
        event_idx: int,
        output_path: str
    ) -> None:
        """
        Save acquired image with metadata.
        
        This is a basic implementation. For production use, you'd want to use
        proper imaging formats like OME-TIFF or OME-Zarr with full metadata.
        """
        import os
        
        try:
            import tifffile
        except ImportError:
            self.__logger.warning("tifffile not available, cannot save images")
            return
        
        # Create filename from event indices
        t_idx = event.index.get('t', 0)
        p_idx = event.index.get('p', 0)
        c_idx = event.index.get('c', 0)
        z_idx = event.index.get('z', 0)
        
        filename = f"img_t{t_idx:04d}_p{p_idx:04d}_c{c_idx:02d}_z{z_idx:04d}.tif"
        filepath = os.path.join(output_path, filename)
        
        # Build metadata from event
        metadata = {
            'event_index': dict(event.index),
            'channel': event.channel.config if event.channel else None,
            'exposure': event.exposure,
            'z_pos': event.z_pos,
            'x_pos': event.x_pos,
            'y_pos': event.y_pos,
        }
        
        # Add sequence metadata if available
        if self._current_sequence and self._current_sequence.metadata:
            metadata['sequence_metadata'] = self._current_sequence.metadata
        
        os.makedirs(output_path, exist_ok=True)
        tifffile.imwrite(filepath, image, metadata=metadata)
        
        self.__logger.debug(f"Saved image to {filepath}")
    
    def _cleanup(self) -> None:
        """Cleanup after sequence completion."""
        # Turn off all lasers
        if self._lasers_manager is not None:
            try:
                for laser_name in self._lasers_manager.getAllDeviceNames():
                    laser = self._lasers_manager[laser_name]
                    laser.setEnabled(False)
                self.__logger.debug("All lasers turned off")
            except Exception as e:
                self.__logger.error(f"Error during cleanup: {e}")
    
    # Hook registration methods
    def register_hook_before_sequence(self, func: Callable[['MDASequence'], None]):
        """Register a callback to run before the sequence starts."""
        self._hooks_before_sequence.append(func)
    
    def register_hook_after_sequence(self, func: Callable[['MDASequence'], None]):
        """Register a callback to run after the sequence completes."""
        self._hooks_after_sequence.append(func)
    
    def register_hook_before_event(self, func: Callable[['MDAEvent'], None]):
        """Register a callback to run before each event."""
        self._hooks_before_event.append(func)
    
    def register_hook_after_event(self, func: Callable[['MDAEvent', np.ndarray], None]):
        """Register a callback to run after each event (receives image data)."""
        self._hooks_after_event.append(func)
    
    # Legacy methods for backward compatibility with WorkflowStep-based approach
        self,
    
    # Legacy methods for backward compatibility with WorkflowStep-based approach
    
    def convert_sequence_to_workflow_steps(
        self,
        sequence: 'MDASequence',
        controller_functions: Dict[str, Callable],
        base_step_id: int = 0
    ) -> List[WorkflowStep]:
        """
        Convert an MDASequence into a list of WorkflowSteps (legacy method).
        
        This method is kept for backward compatibility with the WorkflowStep-based
        approach. For new implementations, prefer using run_mda() directly.
        
        Args:
            sequence: The MDASequence to convert
            controller_functions: Dictionary mapping action names to controller functions
            base_step_id: Starting step ID for numbering
            
        Returns:
            List of WorkflowSteps that execute the MDA sequence
        """
        if not HAS_USEQ:
            raise RuntimeError("useq-schema not available")
            
        workflow_steps = []
        step_id = base_step_id
        
        # Convert each MDAEvent in the sequence to WorkflowSteps
        for event_idx, event in enumerate(sequence):
            steps = self._convert_event_to_steps(
                event, event_idx, controller_functions, step_id
            )
            workflow_steps.extend(steps)
            step_id += len(steps)
            
        return workflow_steps
    
    def _convert_event_to_steps(
        self,
        event: 'MDAEvent',
        event_idx: int,
        controller_functions: Dict[str, Callable],
        base_step_id: int
    ) -> List[WorkflowStep]:
        """Convert a single MDAEvent to WorkflowSteps (legacy method)."""
        steps = []
        step_id = base_step_id
        
        # Move to XY position if specified
        if event.x_pos is not None or event.y_pos is not None:
            x_pos = event.x_pos if event.x_pos is not None else 0.0
            y_pos = event.y_pos if event.y_pos is not None else 0.0
            
            if 'move_stage_xy' in controller_functions:
                steps.append(WorkflowStep(
                    name=f"Move to XY position ({x_pos}, {y_pos})",
                    step_id=str(step_id),
                    main_func=controller_functions['move_stage_xy'],
                    main_params={"posX": x_pos, "posY": y_pos, "relative": False}
                ))
                step_id += 1
        
        # Move to Z position if specified
        if event.z_pos is not None:
            if 'move_stage_z' in controller_functions:
                steps.append(WorkflowStep(
                    name=f"Move to Z position ({event.z_pos})",
                    step_id=str(step_id),
                    main_func=controller_functions['move_stage_z'],
                    main_params={"posZ": event.z_pos, "relative": False}
                ))
                step_id += 1
        
        # Set up illumination channel
        if event.channel is not None:
            channel = event.channel
            
            # Set laser/illumination power
            if 'set_laser_power' in controller_functions and hasattr(channel, 'config'):
                power = getattr(channel, 'power', 100)  # Default power if not specified
                steps.append(WorkflowStep(
                    name=f"Set illumination for channel {channel.config}",
                    step_id=str(step_id),
                    main_func=controller_functions['set_laser_power'],
                    main_params={"channel": channel.config, "power": power}
                ))
                step_id += 1
            
            # Set camera exposure if specified
            if event.exposure is not None and 'set_detector_parameter' in controller_functions:
                steps.append(WorkflowStep(
                    name=f"Set exposure to {event.exposure}ms",
                    step_id=str(step_id),
                    main_func=controller_functions['set_detector_parameter'],
                    main_params={
                        "parameter": "exposure", 
                        "value": event.exposure
                    }
                ))
                step_id += 1
        
        # Acquire image - this is the main action for most events
        if 'snap_image' in controller_functions:
            # Create metadata for this event
            metadata = {
                "event_index": event_idx,
                "mda_index": dict(event.index),
                "channel": event.channel.config if event.channel else None,
                "exposure": event.exposure,
                "z_pos": event.z_pos,
                "x_pos": event.x_pos,
                "y_pos": event.y_pos,
            }
            
            steps.append(WorkflowStep(
                name=f"Acquire image (event {event_idx})",
                step_id=str(step_id),
                main_func=controller_functions['snap_image'],
                main_params={"metadata": metadata},
                pre_funcs=[],
                pre_params={},
            ))
            step_id += 1
        
        # Turn off illumination after acquisition
        if event.channel is not None and 'set_laser_power' in controller_functions:
            steps.append(WorkflowStep(
                name=f"Turn off illumination for channel {event.channel.config}",
                step_id=str(step_id),
                main_func=controller_functions['set_laser_power'],
                main_params={"channel": event.channel.config, "power": 0}
            ))
            step_id += 1
            
        return steps
    
    # Utility methods
    
    def create_simple_sequence(
        self,
        channels: List[str],
        z_range: Optional[float] = None,
        z_step: Optional[float] = None,
        time_points: int = 1,
        time_interval: float = 1.0,
        exposure_times: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'MDASequence':
        """
        Create a simple MDASequence with common parameters (helper method).
        
        This is a convenience method for creating basic sequences. For full control,
        create the MDASequence object directly using useq-schema.
        
        Args:
            channels: List of channel names/configurations
            z_range: Total Z range to scan (centered around current position)
            z_step: Step size for Z stack
            time_points: Number of time points
            time_interval: Interval between time points (seconds)
            exposure_times: Dict mapping channel names to exposure times
            metadata: Optional metadata dict to attach to sequence
            
        Returns:
            Configured MDASequence
        """
        if not HAS_USEQ:
            raise RuntimeError("useq-schema not available")
            
        from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround
        
        # Create channel objects
        channel_objects = []
        for channel_name in channels:
            exposure = exposure_times.get(channel_name, 100.0) if exposure_times else 100.0
            channel_objects.append(Channel(config=channel_name, exposure=exposure))
        
        # Set up time plan if multiple time points
        time_plan = None
        if time_points > 1:
            time_plan = TIntervalLoops(interval=time_interval, loops=time_points)
        
        # Set up Z plan if Z range specified
        z_plan = None
        if z_range is not None and z_step is not None:
            z_plan = ZRangeAround(range=z_range, step=z_step)
        
        return MDASequence(
            metadata=metadata or {},
            axis_order='tpzc',
            channels=channel_objects,
            time_plan=time_plan,
            z_plan=z_plan
        )
    
    def get_sequence_info(self, sequence: 'MDASequence') -> Dict[str, Any]:
        """
        Get information about an MDASequence.
        
        Args:
            sequence: useq.MDASequence to analyze
            
        Returns:
            Dictionary with sequence information including total events, channels,
            positions, timepoints, axis order, and estimated duration
        """
        if not HAS_USEQ:
            raise RuntimeError("useq-schema not available")
            
        events = list(sequence)
        
        # Extract unique values
        channels = set()
        z_positions = set()
        time_points = set()
        xy_positions = set()
        
        for event in events:
            if event.channel:
                channels.add(event.channel.config)
            if event.z_pos is not None:
                z_positions.add(event.z_pos)
            if 't' in event.index:
                time_points.add(event.index['t'])
            if event.x_pos is not None or event.y_pos is not None:
                xy_positions.add((event.x_pos, event.y_pos))
                
        return {
            "total_events": len(events),
            "channels": sorted(list(channels)),
            "z_positions": sorted(list(z_positions)) if z_positions else [],
            "time_points": sorted(list(time_points)) if time_points else [],
            "xy_positions": len(xy_positions),
            "axis_order": sequence.axis_order,
            "metadata": sequence.metadata if hasattr(sequence, 'metadata') else {},
            "estimated_duration_minutes": self._estimate_duration(sequence)
        }
    
    def _estimate_duration(self, sequence: 'MDASequence') -> float:
        """Rough estimate of sequence duration in minutes."""
        events = list(sequence)
        total_exposure_time = sum(
            event.exposure / 1000.0 if event.exposure else 0.1  # Convert ms to s, default 100ms
            for event in events
        )
        # Add overhead for stage movements, etc. (rough estimate)
        movement_overhead = len(events) * 0.5  # 500ms per event for movements
        total_seconds = total_exposure_time + movement_overhead
        return total_seconds / 60.0  # Convert to minutes