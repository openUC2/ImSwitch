"""
MDASequenceManager - Bridge between useq-schema MDASequence and ImSwitch WorkflowSteps
"""

from typing import List, Dict, Any, Optional, Callable
import time
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowStep, WorkflowContext

try:
    from useq import MDASequence, MDAEvent, Channel
    HAS_USEQ = True
except ImportError:
    HAS_USEQ = False


class MDASequenceManager:
    """
    Manager to convert useq-schema MDASequence into ImSwitch WorkflowSteps.
    
    This enables the use of the standard useq-schema for defining microscope
    acquisition protocols while maintaining compatibility with ImSwitch's
    existing workflow system.
    """
    
    def __init__(self):
        self.__logger = initLogger(self)
        if not HAS_USEQ:
            self.__logger.warning("useq-schema not available. MDA functionality disabled.")
    
    def is_available(self) -> bool:
        """Check if useq-schema is available for MDA functionality."""
        return HAS_USEQ
    
    def convert_sequence_to_workflow_steps(
        self,
        sequence: 'MDASequence',
        controller_functions: Dict[str, Callable],
        base_step_id: int = 0
    ) -> List[WorkflowStep]:
        """
        Convert an MDASequence into a list of WorkflowSteps.
        
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
        """Convert a single MDAEvent to WorkflowSteps."""
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
                # Assume channel config maps to laser name and we set a default power
                # In a real implementation, this would be more sophisticated
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
    
    def create_simple_sequence(
        self,
        channels: List[str],
        z_range: Optional[float] = None,
        z_step: Optional[float] = None,
        time_points: int = 1,
        time_interval: float = 1.0,
        exposure_times: Optional[Dict[str, float]] = None
    ) -> 'MDASequence':
        """
        Create a simple MDASequence with common parameters.
        
        Args:
            channels: List of channel names/configurations
            z_range: Total Z range to scan (centered around current position)
            z_step: Step size for Z stack
            time_points: Number of time points
            time_interval: Interval between time points (seconds)
            exposure_times: Dict mapping channel names to exposure times
            
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
            axis_order='tpzc',
            channels=channel_objects,
            time_plan=time_plan,
            z_plan=z_plan
        )
    
    def get_sequence_info(self, sequence: 'MDASequence') -> Dict[str, Any]:
        """Get information about an MDASequence."""
        if not HAS_USEQ:
            raise RuntimeError("useq-schema not available")
            
        events = list(sequence)
        
        # Extract unique values
        channels = set()
        z_positions = set()
        time_points = set()
        
        for event in events:
            if event.channel:
                channels.add(event.channel.config)
            if event.z_pos is not None:
                z_positions.add(event.z_pos)
            if 't' in event.index:
                time_points.add(event.index['t'])
                
        return {
            "total_events": len(events),
            "channels": sorted(list(channels)),
            "z_positions": sorted(list(z_positions)) if z_positions else [],
            "time_points": sorted(list(time_points)) if time_points else [],
            "axis_order": sequence.axis_order,
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