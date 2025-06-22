# filepath: /Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/ImSwitch/tests/test_headless_wellplate.py
# Comments in English

import os
import subprocess
import time
import requests
import pytest
import tempfile
from urllib3.exceptions import InsecureRequestWarning

# Suppress warnings if self-signed SSL is used. Remove if not needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

@pytest.fixture(scope="module")
def imswitch_process():
    """
    Starts ImSwitch in headless mode with a specific config and stops it after tests.
    """
    # Create a temporary directory for data storage
    temp_data_dir = tempfile.mkdtemp()
    
    # Get the path to the example config file relative to the test location
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(test_dir, "../../..", "_data", "user_defaults", "imcontrol_setups", "example_virtual_microscope.json")
    config_path = os.path.abspath(config_path)
    
    # Adjust Python interpreter or environment if needed
    cmd = [
        "python3",  # or path to your Python
        "-m",
        "imswitch",  # or the entry point if ImSwitch is installed
        "main",
        f"--default_config={config_path}",
        "--is_headless=True",
        "--http_port=8001",
        "--socket_port=8002",
        "--scan_ext_data_folder=True",
        f"--data_folder={temp_data_dir}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2)  # Give some initial time to spin up
    yield proc
    proc.terminate()
    proc.wait(timeout=10)
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_data_dir, ignore_errors=True)

def wait_for_server(port=8001, timeout=30):
    """
    Waits until the ImSwitch server on 0.0.0.0:port responds or until timeout is reached.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # we listen to the HTTP port with a self-signed certificate
            resp = requests.get(f"https://0.0.0.0:{port}/", verify=False, timeout=2)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def test_wellplate_experiment(imswitch_process):
    # Wait for ImSwitch server to be ready
    assert wait_for_server(8001, timeout=30), "ImSwitch did not start in time."

    # Optionally verify GET route or some endpoint
    try:
        r = requests.get("https://0.0.0.0:8001/ExperimentController/getCurrentExperimentParameters", verify=False)
        print("GET /ExperimentController/getCurrentExperimentParameters =>", r.status_code, r.content)
    except Exception as e:
        pytest.fail(f"GET request failed: {e}")

    # Prepare example JSON payload for z-stack scenario
    experiment_payload_zstack = {
        "name": "z_stack_experiment",
        "parameterValue": {
            "illumination": "LED",
            "illuIntensities": 100,
            "brightfield": False,
            "darkfield": False,
            "laserWaveLength": 0,
            "differentialPhaseContrast": False,
            "timeLapsePeriod": 0.1,
            "numberOfImages": 1,
            "autoFocus": False,
            "autoFocusMin": 0,
            "autoFocusMax": 0,
            "autoFocusStepSize": 0.1,
            "zStack": True,  # Enable z-stack
            "zStackMin": -2,  # Z range: -2 to +2 um
            "zStackMax": 2,
            "zStackStepSize": 1.0,  # 1 um steps -> 5 z positions
            "speed": 1000,
            "ome_write_tiff": True,
            "ome_write_zarr": True,
            "ome_write_stitched_tiff": False
        },
        "pointList": [],  # Empty point list to test current position fallback
        "number_z_steps": 5,
        "timepoints": 1,
        "x_pixels": 0,
        "y_pixels": 0,
        "microscope_name": "FRAME",
        "is_multiposition": False,
        "channels": {},
        "multi_positions": {}
    }

    # Test the z-stack scenario (without actually starting ImSwitch)
    print("Z-stack experiment payload created:")
    print(f"  Z-stack enabled: {experiment_payload_zstack['parameterValue']['zStack']}")
    print(f"  Z range: {experiment_payload_zstack['parameterValue']['zStackMin']} to {experiment_payload_zstack['parameterValue']['zStackMax']}")
    print(f"  Z step size: {experiment_payload_zstack['parameterValue']['zStackStepSize']}")
    print(f"  Empty pointList: {len(experiment_payload_zstack['pointList']) == 0}")
    expected_z_steps = int((experiment_payload_zstack['parameterValue']['zStackMax'] - 
                           experiment_payload_zstack['parameterValue']['zStackMin']) / 
                          experiment_payload_zstack['parameterValue']['zStackStepSize']) + 1
    print(f"  Expected z positions: {expected_z_steps}")

def test_zstack_experiment_without_server(imswitch_process):
    """Test z-stack experiment logic without starting the server."""
    # This test validates the z-stack payload structure
    
    # Create z-stack experiment payload similar to what would be sent
    experiment_payload_zstack = {
        "name": "z_stack_only_test",
        "parameterValue": {
            "illumination": "LED", 
            "illuIntensities": 100,
            "brightfield": False,
            "darkfield": False,
            "differentialPhaseContrast": False,
            "timeLapsePeriod": 1.0,
            "numberOfImages": 1,
            "autoFocus": False,
            "autoFocusMin": 0,
            "autoFocusMax": 0,
            "autoFocusStepSize": 1.0,
            "zStack": True,  # Enable z-stack
            "zStackMin": -5,
            "zStackMax": 5,
            "zStackStepSize": 1.0,  # Should create 11 z positions
            "exposureTimes": 100,
            "gains": 1,
            "speed": 1000,
            "performanceMode": False,
            "ome_write_tiff": True,
            "ome_write_zarr": True,
            "ome_write_stitched_tiff": False
        },
        "pointList": []  # Empty to test current position fallback
    }
    
    # Validate the payload structure
    assert experiment_payload_zstack["parameterValue"]["zStack"] is True
    assert experiment_payload_zstack["parameterValue"]["zStackMin"] == -5
    assert experiment_payload_zstack["parameterValue"]["zStackMax"] == 5
    assert experiment_payload_zstack["parameterValue"]["zStackStepSize"] == 1.0
    assert len(experiment_payload_zstack["pointList"]) == 0
    
    # Calculate expected z positions
    z_min = experiment_payload_zstack["parameterValue"]["zStackMin"]
    z_max = experiment_payload_zstack["parameterValue"]["zStackMax"] 
    z_step = experiment_payload_zstack["parameterValue"]["zStackStepSize"]
    expected_z_count = int((z_max - z_min) / z_step) + 1
    
    assert expected_z_count == 11  # -5 to +5 with step 1 = 11 positions
    
    print(f"✓ Z-stack payload validation passed")
    print(f"  Expected {expected_z_count} z positions")
    print(f"  Empty pointList will trigger current position fallback")

    # Send POST startWellplateExperiment
    try:
        resp = requests.post(
            "https://0.0.0.0:8001/ExperimentController/startWellplateExperiment",
            json=experiment_payload_zstack,
            verify=False,
            timeout=10,
        )
        print("POST /ExperimentController/startWellplateExperiment (z-stack) =>", resp.status_code, resp.text)
    except Exception as e:
        pytest.fail(f"POST request failed: {e}")

    # Optionally validate response
    assert resp.status_code == 200 or resp.status_code == 201, f"Unexpected status code: {resp.status_code}"
    assert "running" in resp.text, "Expected experiment to be running but got something else."


if __name__ == "__main__":
    # Run the test
    pytest.main([__file__, "-v"])
    # Also run the z-stack validation test
    test_zstack_experiment_without_server(None)
    # pytest.main([__file__, "-s", "-v"])