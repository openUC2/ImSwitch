# filepath: /Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/ImSwitch/tests/test_headless_wellplate.py
# Comments in English

import os
import subprocess
import time
import requests
import pytest
from urllib3.exceptions import InsecureRequestWarning

# Suppress warnings if self-signed SSL is used. Remove if not needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

@pytest.fixture(scope="module")
def imswitch_process():
    """
    Starts ImSwitch in headless mode with a specific config and stops it after tests.
    """
    # Adjust Python interpreter or environment if needed
    cmd = [
        "python3",  # or path to your Python
        "-m",
        "imswitch",  # or the entry point if ImSwitch is installed
        "main",
        "--default_config=/Users/bene/ImSwitchConfig/imcontrol_setups/example_virtual_microscope.json",
        "--is_headless=True",
        "--http_port=8001",
        "--socket_port=8002",
        "--scan_ext_data_folder=True",
        "--data_folder=~/Downloads",
        "--ext_drive_mount=/Volumes"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2)  # Give some initial time to spin up
    yield proc
    proc.terminate()
    proc.wait(timeout=10)

def wait_for_server(port=8001, timeout=30):
    """
    Waits until the ImSwitch server on localhost:port responds or until timeout is reached.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/", verify=False, timeout=2)
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
        r = requests.get("http://localhost:8001/ExperimentController/getCurrentExperimentParameters", verify=False)
        print("GET /ExperimentController/getCurrentExperimentParameters =>", r.status_code, r.content)
    except Exception as e:
        pytest.fail(f"GET request failed: {e}")

    # Prepare example JSON payload
    experiment_payload = {
        "name": "experiment",
        "parameterValue": {
            "illumination": "Brightfield",
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
            "zStack": False,
            "zStackMin": 0,
            "zStackMax": 0,
            "zStackStepSize": 0.1,
            "speed": 0,
        },
        "pointList": [
            {
                "id": "8e00159d-b08b-4691-a761-db749a7fa833",
                "name": "",
                "x": 70312.04719048041,
                "y": 41886.72251515574,
                "neighborPointList": [
                    {"x": 70312.04719048041, "y": 41886.72251515574, "iX": 0, "iY": 0},
                    {"x": 110312.04719048041, "y": 41886.72251515574, "iX": 1, "iY": 0},
                    {"x": 30312.047190480414, "y": 41886.72251515574, "iX": -1, "iY": 0},
                    {"x": 70312.04719048041, "y": 61886.72251515574, "iX": 0, "iY": 1},
                    {"x": 70312.04719048041, "y": 21886.722515155743, "iX": 0, "iY": -1},
                    {"x": 110312.04719048041, "y": 61886.72251515574, "iX": 1, "iY": 1},
                    {"x": 110312.04719048041, "y": 21886.722515155743, "iX": 1, "iY": -1},
                    {"x": 30312.047190480414, "y": 61886.72251515574, "iX": -1, "iY": 1},
                    {"x": 30312.047190480414, "y": 21886.722515155743, "iX": -1, "iY": -1},
                ],
            }
        ],
        "number_z_steps": 0,
        "timepoints": 1,
        "x_pixels": 0,
        "y_pixels": 0,
        "microscope_name": "FRAME",
        "is_multiposition": False,
        "channels": {},
        "multi_positions": {}
    }

    # Send POST startWellplateExperiment
    try:
        resp = requests.post(
            "http://localhost:8001/ExperimentController/startWellplateExperiment",
            json=experiment_payload,
            verify=False,
            timeout=10,
        )
        print("POST /ExperimentController/startWellplateExperiment =>", resp.status_code, resp.text)
    except Exception as e:
        pytest.fail(f"POST request failed: {e}")

    # Optionally validate response
    assert resp.status_code == 200 or resp.status_code == 201, f"Unexpected status code: {resp.status_code}"
    assert "running" in resp.text, "Expected experiment to be running but got something else."


if __name__ == "__main__":
    # Run the test
    pytest.main([__file__, "-v"])
    # pytest.main([__file__, "-s", "-v"])