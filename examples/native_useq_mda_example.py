#!/usr/bin/env python3
"""
Native useq-schema MDA Example for ImSwitch

This example demonstrates how to use native useq.MDASequence objects with
ImSwitch, following the same pattern as pymmcore-plus and raman-mda-engine.

This shows the EXACT pattern requested by @beniroquai - using native useq-schema
objects directly without any wrapper/simplified API.

Based on: https://github.com/ianhi/raman-mda-engine/blob/main/examples/with-notebook.ipynb
"""


try:
    from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround, AbsolutePosition
    HAS_USEQ = True
except ImportError:
    print("Error: useq-schema not installed")
    print("Install with: pip install useq-schema")
    HAS_USEQ = False
    exit(1)


def example_raman_style_protocol():
    """
    Example following the EXACT pattern from raman-mda-engine.
    
    This creates a native useq.MDASequence with all the features:
    - metadata
    - stage_positions (as AbsolutePosition tuples)
    - channels
    - time_plan
    - z_plan
    - axis_order
    """
    print("=== Example 1: Raman-Style MDA Protocol ===\n")

    # Define metadata (can be any dict)
    metadata = {
        "raman": {
            "z": "center",
            "channel": "BF",
        },
        "experiment_type": "multi-position-timelapse",
        "user": "researcher_name"
    }

    # Create the MDA sequence with native useq-schema objects
    mda = MDASequence(
        metadata=metadata,
        stage_positions=[
            AbsolutePosition(x=100.0, y=100.0, z=30.0),
            AbsolutePosition(x=200.0, y=150.0, z=35.0)
        ],
        channels=[
            Channel(config="BF"),      # Brightfield
            Channel(config="DAPI")     # DAPI fluorescence
        ],
        time_plan=TIntervalLoops(interval=1, loops=20),  # 20 timepoints, 1 second apart
        z_plan=ZRangeAround(range=4.0, step=0.5),         # 4µm range, 0.5µm steps
        axis_order="tpcz",  # time, position, channel, z
    )

    print("MDA Sequence created:")
    print(f"  axis_order: {mda.axis_order}")
    print(f"  shape: {mda.shape}")
    print(f"  total events: {len(list(mda))}")
    print(f"  metadata: {mda.metadata}")
    print()

    # Show the native useq-schema structure (like in the raman example)
    print("Native useq-schema __dict__:")
    for key, value in mda.__dict__.items():
        if key.startswith('_'):
            continue
        print(f"  '{key}' = {value}")
    print()

    return mda


def example_imswitch_execution():
    """
    Example showing how to execute with ImSwitch's MDA engine.
    
    This follows the pattern:
    1. Create native useq.MDASequence
    2. Get ImSwitch controller/engine
    3. Register engine with hardware managers
    4. Run the sequence with engine.run_mda(sequence)
    """
    print("=== Example 2: Executing with ImSwitch ===\n")

    # Step 1: Create native useq-schema sequence
    mda = MDASequence(
        metadata={"experiment": "test_run"},
        stage_positions=[
            AbsolutePosition(x=0.0, y=0.0, z=10.0),
        ],
        channels=[
            Channel(config="DAPI", exposure=50.0),
            Channel(config="FITC", exposure=100.0)
        ],
        z_plan=ZRangeAround(range=10.0, step=2.0),
        time_plan=TIntervalLoops(interval=30.0, loops=5),
        axis_order="tpcz"
    )

    print("Sequence configuration:")
    print(f"  Total events: {len(list(mda))}")
    print(f"  Axis order: {mda.axis_order}")
    print()

    # Step 2 & 3: In real usage, you would:
    #   from imswitch.imcontrol.model.managers.MDASequenceManager import MDASequenceManager
    #
    #   # Get the engine
    #   engine = MDASequenceManager()
    #
    #   # Register with ImSwitch managers (like registering with CMMCorePlus)
    #   engine.register(
    #       detector_manager=controller._master.detectorsManager,
    #       positioners_manager=controller._master.positionersManager,
    #       lasers_manager=controller._master.lasersManager,
    #       autofocus_manager=controller._master.autofocusManager  # optional
    #   )
    #
    #   # Step 4: Run the sequence
    #   engine.run_mda(mda, output_path="/data/experiment")

    print("To execute with ImSwitch:")
    print("  1. engine = MDASequenceManager()")
    print("  2. engine.register(detector_mgr, pos_mgr, laser_mgr)")
    print("  3. engine.run_mda(mda)")
    print()

    return mda


def example_with_hooks():
    """
    Example showing how to add custom hooks for autofocus, analysis, etc.
    
    This is similar to how you'd use pymmcore-plus's event system.
    """
    print("=== Example 3: Using Hooks for Custom Logic ===\n")

    mda = MDASequence(
        channels=[Channel(config="Brightfield")],
        stage_positions=[
            AbsolutePosition(x=i*100.0, y=i*100.0, z=10.0)
            for i in range(3)
        ],
        time_plan=TIntervalLoops(interval=60.0, loops=10),
        axis_order="tpzc"
    )

    print("Sequence with hooks:")
    print(f"  3 positions × 10 timepoints = {len(list(mda))} events")
    print()

    # Define custom hooks
    def before_event_hook(event):
        """Called before each acquisition event."""
        print(f"  → About to acquire: {dict(event.index)}")

        # Example: Run autofocus at first channel of each position
        if event.index.get('c', 0) == 0:
            print(f"    Running autofocus at position {event.index.get('p', 0)}")

    def after_event_hook(event, image):
        """Called after each acquisition (receives image)."""
        if image is not None:
            mean_val = image.mean() if hasattr(image, 'mean') else 0
            print(f"  ← Acquired image, mean intensity: {mean_val:.1f}")

    # In real usage:
    #   engine.register_hook_before_event(before_event_hook)
    #   engine.register_hook_after_event(after_event_hook)
    #   engine.run_mda(mda)

    print("Custom hooks allow:")
    print("  - Autofocus before certain acquisitions")
    print("  - Real-time image analysis")
    print("  - Drift correction")
    print("  - Custom logging")
    print("  - Hardware adjustments during acquisition")
    print()


def example_grid_positions():
    """
    Example showing grid-based multi-position acquisition.
    
    useq-schema supports sophisticated position patterns.
    """
    print("=== Example 4: Grid-Based Multi-Position ===\n")

    try:
        from useq import GridRowsColumns

        # Use GridRowsColumns without RelativePosition for simplicity
        mda = MDASequence(
            metadata={"scan_type": "grid"},
            channels=[Channel(config="Brightfield", exposure=10.0)],
            grid_plan=GridRowsColumns(
                rows=3,
                columns=3,
                fov_width=100.0,   # µm
                fov_height=100.0,
            ),
            axis_order="pzc"
        )

        print("Grid acquisition:")
        print(f"  3×3 grid = {len(list(mda))} positions")
        print("  FOV: 100×100 µm")
        print()

        # Show first few positions
        print("First few positions:")
        for i, event in enumerate(list(mda)[:5]):
            print(f"  Position {i}: x={event.x_pos:.1f}, y={event.y_pos:.1f}")
        print()

    except (ImportError, Exception) as e:
        print(f"Grid example skipped: {e}")
        print()


def example_protocol_sharing():
    """
    Example showing how protocols can be shared between systems.
    
    The same MDASequence works with:
    - ImSwitch (using MDASequenceManager)
    - pymmcore-plus (using CMMCorePlus)
    - raman-mda-engine
    - Any other useq-schema compatible system
    """
    print("=== Example 5: Protocol Sharing Across Systems ===\n")

    # Create a standard protocol
    protocol = MDASequence(
        metadata={
            "protocol_name": "StandardImaging_v1",
            "created_by": "lab_member",
            "microscope": "any"  # Works on any system!
        },
        channels=[
            Channel(config="DAPI", exposure=50.0),
            Channel(config="FITC", exposure=100.0),
            Channel(config="TRITC", exposure=150.0)
        ],
        z_plan=ZRangeAround(range=20.0, step=2.0),
        time_plan=TIntervalLoops(interval=300.0, loops=12),  # Every 5 min, 12 times
        axis_order="tzcg"
    )

    print("This SAME protocol can be used on:")
    print()

    print("1. ImSwitch:")
    print("   engine = MDASequenceManager()")
    print("   engine.register(...)")
    print("   engine.run_mda(protocol)")
    print()

    print("2. pymmcore-plus:")
    print("   from pymmcore_plus import CMMCorePlus")
    print("   from pymmcore_plus.mda import mda_engine")
    print("   core = CMMCorePlus()")
    print("   mda_engine.run_mda(core, protocol)")
    print()

    print("3. Save/load as JSON:")
    print("   import json")
    print("   with open('protocol.json', 'w') as f:")
    print("       json.dump(protocol.dict(), f)")
    print("   # Load and use on different system")
    print()

    print("✓ Protocol standardization across microscopy systems!")
    print()


def main():
    """Run all examples."""
    if not HAS_USEQ:
        return

    print("=" * 70)
    print("Native useq-schema MDA Examples for ImSwitch")
    print("=" * 70)
    print()
    print("This demonstrates using native useq.MDASequence objects directly,")
    print("following the EXACT pattern from pymmcore-plus and raman-mda-engine.")
    print()

    # Run examples
    example_raman_style_protocol()
    example_imswitch_execution()
    example_with_hooks()
    example_grid_positions()
    example_protocol_sharing()

    print("=" * 70)
    print("Key Points:")
    print("  • Use NATIVE useq.MDASequence objects (not simplified wrappers)")
    print("  • Support ALL useq-schema features (metadata, positions, etc.)")
    print("  • Register engine before running: engine.register(...)")
    print("  • Execute with: engine.run_mda(sequence)")
    print("  • Protocols are PORTABLE across different microscopy systems")
    print("=" * 70)


if __name__ == "__main__":
    main()
