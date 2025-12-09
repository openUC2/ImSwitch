"""
Example: Integrating MDA Engine Pattern into ImSwitch Controller

This example shows how to add MDA engine functionality to an existing
ImSwitch controller, enabling direct useq-schema protocol execution.

This follows the same pattern as:
- pymmcore-plus (Micro-Manager)
- raman-mda-engine (Raman spectroscopy)
- opto-loop-sim (Optogenetics)

Usage in your controller:
    from examples.mda_engine_integration import add_mda_engine_to_controller
    
    class MyController:
        def __init__(self, master):
            self._master = master
            # Add MDA engine capability
            add_mda_engine_to_controller(self)
            
        def run_custom_experiment(self):
            # Now you can use self.mda_engine
            sequence = create_my_sequence()
            self.mda_engine.run(sequence)
"""

from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    from imswitch.imcontrol.controller.controllers.ExperimentController import ExperimentController

try:
    from useq import MDASequence, Channel, ZRangeAround, TIntervalLoops, GridRowsColumns
    HAS_USEQ = True
except ImportError:
    HAS_USEQ = False


def add_mda_engine_to_controller(controller: 'ExperimentController'):
    """
    Add MDA engine functionality to an ExperimentController instance.
    
    This adds:
    - controller.mda_engine: Direct hardware control engine
    - controller.run_mda_with_engine(): Method to run MDA sequences
    - controller.create_mda_with_hooks(): Helper to create sequences with hooks
    
    Args:
        controller: ExperimentController instance
    """
    if not HAS_USEQ:
        controller._logger.warning("useq-schema not available, MDA engine not added")
        return
        
    from examples.mda_engine_wrapper import ImSwitchMDAEngine
    
    # Create engine with controller's managers
    controller.mda_engine = ImSwitchMDAEngine(
        detector_manager=controller._master.detectorsManager,
        positioners_manager=controller._master.positionersManager,
        lasers_manager=controller._master.lasersManager,
        logger=controller._logger
    )
    
    # Add convenience method
    def run_mda_with_engine(sequence: MDASequence, output_dir: str = None):
        """Run an MDA sequence using the direct engine."""
        if output_dir is None:
            from datetime import datetime
            from imswitch.imcommon.model import dirtools
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                dirtools.UserFileDirs.Data,
                'MDAEngine',
                timestamp
            )
        
        controller._logger.info(f"Running MDA sequence with engine, saving to {output_dir}")
        controller.mda_engine.run(sequence, output_path=output_dir)
        return output_dir
        
    controller.run_mda_with_engine = run_mda_with_engine
    
    # Add hook builder
    def create_mda_with_hooks(
        channels,
        z_range=None,
        z_step=None,
        time_points=1,
        time_interval=1.0,
        grid_rows=None,
        grid_cols=None,
        fov_size=None,
        before_event_hook=None,
        after_event_hook=None
    ):
        """
        Helper to create MDA sequence with hooks.
        
        Example:
            def my_autofocus(event):
                if event.index.get('z', 0) == 0:
                    controller.autofocus_manager.runAutofocus()
                    
            sequence = controller.create_mda_with_hooks(
                channels=["DAPI", "FITC"],
                z_range=10.0,
                z_step=2.0,
                before_event_hook=my_autofocus
            )
            controller.run_mda_with_engine(sequence)
        """
        # Build channel objects
        channel_objs = [Channel(config=ch, exposure=100.0) for ch in channels]
        
        # Build z plan
        z_plan = None
        if z_range and z_step:
            z_plan = ZRangeAround(range=z_range, step=z_step)
            
        # Build time plan
        time_plan = None
        if time_points > 1:
            time_plan = TIntervalLoops(interval=time_interval, loops=time_points)
            
        # Build grid plan
        grid_plan = None
        if grid_rows and grid_cols and fov_size:
            from useq import RelativePosition
            grid_plan = GridRowsColumns(
                rows=grid_rows,
                columns=grid_cols,
                relative_to=RelativePosition.CENTER,
                fov_width=fov_size,
                fov_height=fov_size
            )
        
        # Create sequence
        sequence = MDASequence(
            channels=channel_objs,
            z_plan=z_plan,
            time_plan=time_plan,
            grid_plan=grid_plan
        )
        
        # Register hooks
        if before_event_hook:
            controller.mda_engine.register_hook_before_event(before_event_hook)
        if after_event_hook:
            controller.mda_engine.register_hook_after_event(after_event_hook)
            
        return sequence
        
    controller.create_mda_with_hooks = create_mda_with_hooks
    
    controller._logger.info("MDA engine functionality added to controller")


# Example usage patterns
def example_1_simple_acquisition(controller):
    """Example 1: Simple Z-stack with two channels."""
    if not HAS_USEQ:
        return
        
    sequence = MDASequence(
        channels=[
            Channel(config="DAPI", exposure=50.0),
            Channel(config="FITC", exposure=100.0)
        ],
        z_plan=ZRangeAround(range=10.0, step=2.0)
    )
    
    output_dir = controller.run_mda_with_engine(sequence)
    print(f"Acquisition complete, saved to {output_dir}")


def example_2_with_autofocus(controller):
    """Example 2: Time-lapse with autofocus at each timepoint."""
    if not HAS_USEQ:
        return
        
    def autofocus_hook(event):
        """Run autofocus at start of each timepoint."""
        if event.index.get('c', 0) == 0 and event.index.get('z', 0) == 0:
            controller._logger.info(f"Running autofocus for timepoint {event.index.get('t', 0)}")
            # Uncomment if autofocus manager available:
            # controller._master.autofocusManager.runAutofocus()
    
    sequence = controller.create_mda_with_hooks(
        channels=["Brightfield"],
        time_points=10,
        time_interval=60.0,  # Every minute
        before_event_hook=autofocus_hook
    )
    
    controller.run_mda_with_engine(sequence)


def example_3_multi_position_timelapse(controller):
    """Example 3: Multi-position time-lapse (grid pattern)."""
    if not HAS_USEQ:
        return
        
    # Track images for quick analysis
    image_stats = []
    
    def analyze_hook(event, image):
        """Quick analysis after each acquisition."""
        if image is not None:
            stats = {
                'timepoint': event.index.get('t', 0),
                'position': event.index.get('p', 0),
                'channel': event.channel.config if event.channel else None,
                'mean': float(image.mean()),
                'std': float(image.std()),
                'min': float(image.min()),
                'max': float(image.max())
            }
            image_stats.append(stats)
            controller._logger.debug(f"Image stats: {stats}")
    
    sequence = controller.create_mda_with_hooks(
        channels=["DAPI", "FITC"],
        grid_rows=3,
        grid_cols=3,
        fov_size=100.0,  # 100μm field of view
        time_points=5,
        time_interval=300.0,  # Every 5 minutes
        after_event_hook=analyze_hook
    )
    
    output_dir = controller.run_mda_with_engine(sequence)
    
    # Save statistics
    import json
    stats_file = os.path.join(output_dir, 'image_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(image_stats, f, indent=2)
    
    print(f"Acquired {len(image_stats)} images")
    print(f"Statistics saved to {stats_file}")


def example_4_shared_protocol():
    """
    Example 4: Protocol that works with both ImSwitch and pymmcore-plus.
    
    This demonstrates the key advantage: the same MDASequence definition
    can be executed on different systems.
    """
    if not HAS_USEQ:
        return
        
    # Define experiment (software-agnostic!)
    experiment = MDASequence(
        channels=[
            Channel(config="DAPI", exposure=50.0),
            Channel(config="FITC", exposure=100.0)
        ],
        z_plan=ZRangeAround(range=10.0, step=2.0),
        time_plan=TIntervalLoops(interval=60.0, loops=10)
    )
    
    # Save protocol to file (can be shared with other systems)
    protocol_file = "/tmp/shared_protocol.json"
    import json
    with open(protocol_file, 'w') as f:
        # Note: useq-schema objects can be serialized to JSON
        json.dump(experiment.dict(), f, indent=2)
    
    print(f"Protocol saved to {protocol_file}")
    print("This protocol can be loaded and executed on:")
    print("  - ImSwitch (using ImSwitchMDAEngine)")
    print("  - pymmcore-plus (using PMCEngine)")
    print("  - Any other useq-schema compatible system")
    
    # To execute on ImSwitch:
    # controller.run_mda_with_engine(experiment)
    
    # To execute on pymmcore-plus:
    # from pymmcore_plus import CMMCorePlus
    # from pymmcore_plus.mda import mda_engine
    # mmc = CMMCorePlus()
    # mda_engine.run_mda(mmc, experiment)


if __name__ == "__main__":
    print("MDA Engine Integration Examples")
    print("================================\n")
    print("This module provides examples of using the MDA engine pattern")
    print("with ImSwitch controllers.\n")
    print("To use in your controller:")
    print("  from examples.mda_engine_integration import add_mda_engine_to_controller")
    print("  add_mda_engine_to_controller(self)")
    print("\nSee the example functions for usage patterns:")
    print("  - example_1_simple_acquisition")
    print("  - example_2_with_autofocus")
    print("  - example_3_multi_position_timelapse")
    print("  - example_4_shared_protocol")
