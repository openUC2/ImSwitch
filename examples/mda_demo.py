#!/usr/bin/env python3
"""
MDA Integration Demo for ImSwitch

This script demonstrates how to use the new MDA (Multi-Dimensional Acquisition)
functionality in ImSwitch using the useq-schema standard.

Example usage:
    python mda_demo.py --server http://localhost:8000 --demo simple
    python mda_demo.py --server http://localhost:8000 --demo timelapse  
    python mda_demo.py --server http://localhost:8000 --demo full
"""

import requests
import json
import time
import argparse
from typing import Dict, Any

class MDADemo:
    """Demonstration class for MDA functionality in ImSwitch."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.api_base = f"{self.server_url}/ExperimentController"
        
    def check_capabilities(self) -> Dict[str, Any]:
        """Check if MDA functionality is available."""
        try:
            response = requests.get(f"{self.api_base}/get_mda_capabilities")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error checking MDA capabilities: {e}")
            return {}
    
    def preview_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Preview an experiment without running it."""
        try:
            response = requests.post(
                f"{self.api_base}/get_mda_sequence_info",
                json=experiment
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error previewing experiment: {e}")
            return {}
    
    def start_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Start an MDA experiment."""
        try:
            response = requests.post(
                f"{self.api_base}/start_mda_experiment", 
                json=experiment
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error starting experiment: {e}")
            return {}
    
    def demo_simple_zstack(self):
        """Demonstrate a simple Z-stack with two channels."""
        print("=== Simple Z-Stack Demo ===")
        
        experiment = {
            "channels": [
                {"name": "635", "exposure": 50.0, "power": 100.0},
                {"name": "488", "exposure": 100.0, "power": 80.0}
            ],
            "z_range": 10.0,  # 10 µm total range
            "z_step": 2.0,    # 2 µm steps  
            "time_points": 1,
            "experiment_name": "Simple_ZStack_Demo"
        }
        
        print("Experiment configuration:")
        print(json.dumps(experiment, indent=2))
        
        # Preview the experiment
        print("\nPreviewing experiment...")
        preview = self.preview_experiment(experiment)
        if preview:
            print(f"Total events: {preview.get('total_events', 'N/A')}")
            print(f"Channels: {preview.get('channels', 'N/A')}")
            print(f"Z positions: {preview.get('z_positions', 'N/A')}")
            print(f"Estimated duration: {preview.get('estimated_duration_minutes', 'N/A'):.1f} minutes")
        
        # Uncomment to actually start the experiment
        print("\nStarting experiment...")
        result = self.start_experiment(experiment)
        print(f"Result: {result}")
        
    def demo_timelapse(self):
        """Demonstrate a time-lapse experiment."""
        print("=== Time-Lapse Demo ===")
        
        experiment = {
            "channels": [
                {"name": "Brightfield", "exposure": 10.0, "power": 50.0}
            ],
            "time_points": 10,     # 10 time points
            "time_interval": 30.0, # Every 30 seconds
            "experiment_name": "Timelapse_Demo"
        }
        
        print("Experiment configuration:")
        print(json.dumps(experiment, indent=2))
        
        # Preview the experiment
        print("\nPreviewing experiment...")
        preview = self.preview_experiment(experiment)
        if preview:
            print(f"Total events: {preview.get('total_events', 'N/A')}")
            print(f"Time points: {preview.get('time_points', 'N/A')}")
            print(f"Estimated duration: {preview.get('estimated_duration_minutes', 'N/A'):.1f} minutes")
    
    def demo_full_mda(self):
        """Demonstrate a full multi-dimensional experiment."""
        print("=== Full Multi-Dimensional Acquisition Demo ===")
        
        experiment = {
            "channels": [
                {"name": "DAPI", "exposure": 50.0, "power": 100.0},
                {"name": "FITC", "exposure": 100.0, "power": 80.0}, 
                {"name": "TRITC", "exposure": 150.0, "power": 90.0}
            ],
            "z_range": 20.0,       # 20 µm Z range
            "z_step": 2.0,         # 2 µm steps
            "time_points": 5,      # 5 time points
            "time_interval": 300.0, # Every 5 minutes
            "experiment_name": "Full_MDA_Demo"
        }
        
        print("Experiment configuration:")
        print(json.dumps(experiment, indent=2))
        
        # Preview the experiment  
        print("\nPreviewing experiment...")
        preview = self.preview_experiment(experiment)
        if preview:
            print(f"Total events: {preview.get('total_events', 'N/A')}")
            print(f"Channels: {preview.get('channels', 'N/A')}")
            print(f"Z positions: {preview.get('z_positions', 'N/A')}")
            print(f"Time points: {preview.get('time_points', 'N/A')}")
            print(f"Estimated duration: {preview.get('estimated_duration_minutes', 'N/A'):.1f} minutes")

def main():

    server = "http://localhost:8001"
    demo_model = "simple" # simple, timelapse, full
    demo = MDADemo(server)
    
    # Check if MDA is available
    print("Checking MDA capabilities...")
    caps = demo.check_capabilities()
    
    if not caps.get('mda_available'):
        print("❌ MDA functionality is not available")
        print("Make sure useq-schema is installed and ImSwitch is running")
        return
    
    print("✅ MDA functionality is available")
    print(f"Available channels: {caps.get('available_channels', [])}")
    print(f"Stage available: {caps.get('stage_available', False)}")
    print()
    
    # Run the selected demo
    if demo_model == 'simple':
        demo.demo_simple_zstack()
    elif demo_model == 'timelapse':
        demo.demo_timelapse()
    elif demo_model == 'full':
        demo.demo_full_mda()
    
    print("\n=== Demo Complete ===")
    print("To actually run experiments, uncomment the start_experiment() calls")
    print("and make sure your ImSwitch setup is configured with appropriate hardware.")

if __name__ == "__main__":
    main()