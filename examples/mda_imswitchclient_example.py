#!/usr/bin/env python3
"""
Example: Using Native useq-schema MDA with imswitchclient

This example demonstrates how to create and execute native useq-schema MDASequence
protocols from a Jupyter notebook or Python script, sending them to ImSwitch via
the REST API using the imswitchclient library.

This follows the pattern requested by @beniroquai - formulate protocols outside
ImSwitch runtime and send them via REST API for execution.

Requirements:
    pip install useq-schema requests imswitchclient
"""

import requests
from typing import Dict, Any
import json

# If using imswitchclient (when available):
# from imswitchclient import ImSwitchClient

try:
    from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround, AbsolutePosition
    HAS_USEQ = True
except ImportError:
    print("Error: useq-schema not installed")
    print("Install with: pip install useq-schema")
    HAS_USEQ = False


class MDAClient:
    """
    Client for sending native useq-schema MDA sequences to ImSwitch via REST API.
    
    This can be used from Jupyter notebooks or Python scripts to execute
    MDA protocols on a running ImSwitch instance.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the MDA client.
        
        Args:
            base_url: Base URL of the ImSwitch REST API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/experimentcontroller"
    
    def check_mda_available(self) -> Dict[str, Any]:
        """Check if MDA functionality is available on the server."""
        try:
            response = requests.get(f"{self.api_base}/get_mda_capabilities")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error checking MDA capabilities: {e}")
            return {}
    
    def run_mda_sequence(self, sequence: 'MDASequence') -> Dict[str, Any]:
        """
        Execute a native useq-schema MDASequence on ImSwitch.
        
        Args:
            sequence: Native useq.MDASequence object
            
        Returns:
            Response dict with execution status and info
        """
        # Convert MDASequence to dict/JSON
        # useq-schema objects have a .model_dump() method for serialization
        sequence_dict = sequence.model_dump() if hasattr(sequence, 'model_dump') else sequence.dict()
        
        print(f"Sending MDA sequence with {len(list(sequence))} events...")
        print(f"Axis order: {sequence.axis_order}")
        
        try:
            response = requests.post(
                f"{self.api_base}/run_native_mda_sequence",
                json=sequence_dict,
                timeout=10  # Connection timeout, not execution timeout
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✓ MDA sequence started successfully")
            print(f"  Status: {result.get('status')}")
            print(f"  Save directory: {result.get('save_directory')}")
            print(f"  Estimated duration: {result.get('estimated_duration_minutes', 0):.1f} minutes")
            
            return result
        except requests.exceptions.RequestException as e:
            print(f"✗ Error sending MDA sequence: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Response: {e.response.text}")
            raise
    
    def get_sequence_info(self, sequence: 'MDASequence') -> Dict[str, Any]:
        """
        Get information about a sequence without executing it.
        
        This is useful for previewing what the sequence will do.
        """
        sequence_dict = sequence.model_dump() if hasattr(sequence, 'model_dump') else sequence.dict()
        
        # For now, we compute this locally
        # Could also add a server endpoint for this
        events = list(sequence)
        return {
            "total_events": len(events),
            "axis_order": sequence.axis_order,
            "metadata": sequence.metadata if hasattr(sequence, 'metadata') else {}
        }


def example_1_simple_xyz_timelapse():
    """
    Example 1: Simple XYZ time-lapse scan.
    
    This creates a multi-position, Z-stack time-lapse as requested by @beniroquai.
    """
    if not HAS_USEQ:
        return
    
    print("=" * 70)
    print("Example 1: XYZ Time-Lapse Scan")
    print("=" * 70)
    print()
    
    # Define the MDA sequence
    sequence = MDASequence(
        metadata={
            "experiment": "xyz_timelapse",
            "description": "Multi-position Z-stack time-lapse",
            "user": "researcher"
        },
        stage_positions=[
            AbsolutePosition(x=100.0, y=100.0, z=30.0),
            AbsolutePosition(x=200.0, y=150.0, z=35.0),
            AbsolutePosition(x=150.0, y=200.0, z=32.0)
        ],
        channels=[
            Channel(config="Brightfield", exposure=10.0)
        ],
        z_plan=ZRangeAround(range=10.0, step=2.0),  # 10µm range, 2µm steps
        time_plan=TIntervalLoops(interval=60.0, loops=10),  # 10 timepoints, 1 min apart
        axis_order="tpzc"  # time, position, z, channel
    )
    
    print(f"Created MDA sequence:")
    print(f"  3 positions × 6 Z-slices × 10 timepoints = {len(list(sequence))} events")
    print(f"  Axis order: {sequence.axis_order}")
    print(f"  Metadata: {sequence.metadata}")
    print()
    
    # Send to ImSwitch
    client = MDAClient(base_url="http://localhost:8000")
    
    # Check if MDA is available
    caps = client.check_mda_available()
    if not caps.get('mda_available'):
        print("MDA functionality not available on server")
        return
    
    print(f"Server capabilities:")
    print(f"  Available channels: {caps.get('available_channels')}")
    print(f"  Stage available: {caps.get('stage_available')}")
    print()
    
    # Execute the sequence
    # result = client.run_mda_sequence(sequence)
    print("To execute, uncomment: result = client.run_mda_sequence(sequence)")
    print()


def example_2_multi_channel_zstack():
    """
    Example 2: Multi-channel Z-stack at multiple positions.
    """
    if not HAS_USEQ:
        return
    
    print("=" * 70)
    print("Example 2: Multi-Channel Z-Stack at Multiple Positions")
    print("=" * 70)
    print()
    
    sequence = MDASequence(
        metadata={
            "experiment": "multi_channel_zstack",
            "sample": "cells_sample_01"
        },
        stage_positions=[
            AbsolutePosition(x=0.0, y=0.0, z=10.0),
            AbsolutePosition(x=100.0, y=0.0, z=10.0)
        ],
        channels=[
            Channel(config="DAPI", exposure=50.0),
            Channel(config="FITC", exposure=100.0),
            Channel(config="TRITC", exposure=150.0)
        ],
        z_plan=ZRangeAround(range=20.0, step=2.0),  # 20µm range, 2µm steps
        axis_order="pczg"  # position, channel, z
    )
    
    print(f"Created MDA sequence:")
    print(f"  2 positions × 3 channels × 11 Z-slices = {len(list(sequence))} events")
    print(f"  Axis order: {sequence.axis_order}")
    print()
    
    # Show the sequence as JSON (what gets sent to the API)
    print("Sequence as JSON (first 500 chars):")
    sequence_json = json.dumps(
        sequence.model_dump() if hasattr(sequence, 'model_dump') else sequence.dict(), 
        indent=2
    )
    print(sequence_json[:500] + "...")
    print()
    
    # To execute:
    # client = MDAClient()
    # result = client.run_mda_sequence(sequence)


def example_3_timelapse_with_autofocus():
    """
    Example 3: Time-lapse with autofocus metadata.
    
    Note: Autofocus execution would need to be handled by ImSwitch's
    autofocus hooks in the MDASequenceManager.
    """
    if not HAS_USEQ:
        return
    
    print("=" * 70)
    print("Example 3: Time-Lapse with Autofocus Metadata")
    print("=" * 70)
    print()
    
    sequence = MDASequence(
        metadata={
            "experiment": "timelapse_autofocus",
            "autofocus": {
                "enabled": True,
                "frequency": "every_position",  # Run at each position
                "method": "software"
            }
        },
        stage_positions=[
            AbsolutePosition(x=i*100.0, y=i*50.0, z=10.0)
            for i in range(4)
        ],
        channels=[
            Channel(config="Brightfield", exposure=10.0)
        ],
        time_plan=TIntervalLoops(interval=300.0, loops=20),  # 20 timepoints, 5 min apart
        axis_order="tpc"
    )
    
    print(f"Created MDA sequence:")
    print(f"  4 positions × 20 timepoints = {len(list(sequence))} events")
    print(f"  Autofocus metadata: {sequence.metadata.get('autofocus')}")
    print()


def example_4_integration_with_imswitchclient():
    """
    Example 4: Integration with imswitchclient library (when available).
    
    This shows how the MDA functionality could be integrated into the
    imswitchclient library's experimentController module.
    """
    print("=" * 70)
    print("Example 4: imswitchclient Integration Pattern")
    print("=" * 70)
    print()
    
    print("Future imswitchclient API (proposed):")
    print()
    print("```python")
    print("from imswitchclient import ImSwitchClient")
    print("from useq import MDASequence, Channel, ZRangeAround")
    print()
    print("# Connect to ImSwitch")
    print("client = ImSwitchClient('localhost', 8000)")
    print()
    print("# Create MDA sequence")
    print("sequence = MDASequence(")
    print("    channels=[Channel(config='DAPI', exposure=50.0)],")
    print("    z_plan=ZRangeAround(range=10.0, step=2.0)")
    print(")")
    print()
    print("# Execute via experimentController")
    print("result = client.experimentController.run_mda_sequence(sequence)")
    print("print(f'MDA started: {result}')") 
    print("```")
    print()


def example_5_raw_requests():
    """
    Example 5: Using raw requests library (no imswitchclient needed).
    """
    if not HAS_USEQ:
        return
    
    print("=" * 70)
    print("Example 5: Using Raw Requests (No imswitchclient Required)")
    print("=" * 70)
    print()
    
    # Create sequence
    sequence = MDASequence(
        metadata={"experiment": "test"},
        channels=[Channel(config="Brightfield", exposure=10.0)],
        z_plan=ZRangeAround(range=5.0, step=1.0),
        axis_order="zc"
    )
    
    # Convert to dict
    sequence_dict = sequence.model_dump() if hasattr(sequence, 'model_dump') else sequence.dict()
    
    print("Using plain requests library:")
    print()
    print("```python")
    print("import requests")
    print("from useq import MDASequence, Channel, ZRangeAround")
    print()
    print("# Create sequence")
    print("sequence = MDASequence(...)")
    print()
    print("# Send to ImSwitch")
    print("response = requests.post(")
    print("    'http://localhost:8000/api/experimentcontroller/run_native_mda_sequence',")
    print("    json=sequence.dict()")
    print(")")
    print("print(response.json())")
    print("```")
    print()
    print(f"Example payload (first 300 chars):")
    print(json.dumps(sequence_dict, indent=2)[:300] + "...")
    print()


def main():
    """Run all examples."""
    print()
    print("=" * 70)
    print("Native useq-schema MDA with imswitchclient Examples")
    print("=" * 70)
    print()
    print("These examples show how to create MDA sequences in a Jupyter notebook")
    print("or Python script and execute them on ImSwitch via REST API.")
    print()
    
    if not HAS_USEQ:
        return
    
    # Run examples
    example_1_simple_xyz_timelapse()
    example_2_multi_channel_zstack()
    example_3_timelapse_with_autofocus()
    example_4_integration_with_imswitchclient()
    example_5_raw_requests()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("✓ Create MDA sequences with native useq-schema objects")
    print("✓ Send to ImSwitch via REST API endpoint: /run_native_mda_sequence")
    print("✓ Works from Jupyter notebooks, Python scripts, or any HTTP client")
    print("✓ Can be integrated into imswitchclient library")
    print("✓ Protocol-compatible with pymmcore-plus and other useq systems")
    print()


if __name__ == "__main__":
    main()
