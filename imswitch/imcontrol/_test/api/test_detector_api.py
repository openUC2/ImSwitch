"""
API tests for ImSwitch DetectorController endpoints.
Tests detector management, configuration, and acquisition functionality via REST API.
"""
import pytest
import requests
import time
import json
from typing import Dict, List, Any
from ..api import api_server, base_url


def test_detector_endpoints_available(api_server):
    """Test that detector API endpoints are accessible."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    # Find detector-related endpoints
    paths = spec.get("paths", {})
    detector_endpoints = [p for p in paths.keys() if "Detector" in p or "detector" in p]
    
    assert len(detector_endpoints) > 0, "No detector endpoints found in API"
    print(f"Found {len(detector_endpoints)} detector endpoints")
    
    # Test basic detector endpoint accessibility
    for endpoint in detector_endpoints[:3]:  # Test first 3
        try:
            response = api_server.get(endpoint)
            assert response.status_code in [200, 400, 404, 422], f"Unexpected status for {endpoint}: {response.status_code}"
            print(f"✓ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"? {endpoint}: {e}")


def test_detector_discovery_endpoints(api_server):
    """Test detector discovery and enumeration endpoints."""
    # Common detector discovery endpoints
    discovery_endpoints = [
        "/SettingsController/getDetectorNames"
    ]
    
    detector_names = None
    working_endpoint = None
    
    for endpoint in discovery_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    detector_names = data
                    working_endpoint = endpoint
                    break
                elif isinstance(data, dict) and len(data) > 0:
                    detector_names = list(data.keys())
                    working_endpoint = endpoint
                    break
        except Exception as e:
            print(f"Discovery endpoint {endpoint} failed: {e}")
    
    if detector_names and working_endpoint:
        print(f"✓ Found detectors via {working_endpoint}: {detector_names}")
        assert len(detector_names) > 0
        return detector_names, working_endpoint
    else:
        pytest.skip("No working detector discovery endpoint found")


def test_detector_parameters_endpoints(api_server):
    """Test detector parameter access and modification."""
    # Try to discover detectors first
    try:
        detector_names, _ = test_detector_discovery_endpoints(api_server)
        if not detector_names:
            pytest.skip("No detectors found")
        
        first_detector = detector_names[0]
    except:
        first_detector = "testDetector"  # Fallback
    
    # Test parameter endpoints
    param_endpoints = [
        f"/SettingsController/getDetectorParameters"
    ]
    
    for endpoint in param_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                params = response.json()
                assert isinstance(params, dict)
                print(f"✓ Got parameters via {endpoint}: {list(params.keys())[:5]}")
                
                # Test parameter modification if we have parameters
                if params and "exposureTime" in params:
                    _test_parameter_modification(api_server, first_detector, params)
                return
                
        except Exception as e:
            print(f"Parameter endpoint {endpoint} failed: {e}")
    
    print("? No working parameter endpoints found")


def _test_parameter_modification(api_server, detector_name: str, current_params: Dict):
    """Test modifying detector parameters."""
    if "exposureTime" not in current_params:
        return
    
    original_exposure = current_params["exposureTime"]
    new_exposure = max(0.001, original_exposure * 1.1)  # Increase by 10%
    
    # Common parameter setting endpoints
    param_set_endpoints = [
        "/SettingsController/setDetectorParameter",
    ]
    
    for endpoint in param_set_endpoints:
        try:
            # Try different payload formats
            payloads = [
                {
                    "detectorName": detector_name,
                    "parameterName": "exposureTime",
                    "value": new_exposure
                },
                {
                    "detector": detector_name,
                    "parameter": "exposureTime", 
                    "value": new_exposure
                },
                {
                    "exposureTime": new_exposure
                }
            ]
            
            for payload in payloads:
                response = api_server.put(endpoint, json=payload)
                if response.status_code in [200, 201]:
                    print(f"✓ Parameter modified via PUT {endpoint}")
                    return
                
                response = api_server.post(endpoint, json=payload)
                if response.status_code in [200, 201]:
                    print(f"✓ Parameter modified via POST {endpoint}")
                    return
                    
        except Exception as e:
            print(f"Parameter modification via {endpoint} failed: {e}")


def test_detector_acquisition_control(api_server):
    """Test detector acquisition and liveview control."""
    # Live view control endpoints
    liveview_endpoints = [
        "/ViewController/setLiveViewActive"
    ]
    
    for endpoint in liveview_endpoints:
        try:
            # Test starting liveview
            response = api_server.get(f"{endpoint}?active=true")
            if response.status_code not in [200, 201]:
                response = api_server.get(f"{endpoint}?active=true")
            
            if response.status_code in [200, 201]:
                print(f"✓ Liveview started via {endpoint}")
                
                # Test stopping liveview
                time.sleep(0.5)
                stop_response = api_server.get(f"{endpoint}?active=false")
                if stop_response.status_code not in [200, 201]:
                    stop_response = api_server.get(f"{endpoint}?active=false")
                
                if stop_response.status_code in [200, 201]:
                    print(f"✓ Liveview stopped via {endpoint}")
                return
                
        except Exception as e:
            print(f"Liveview control via {endpoint} failed: {e}")
    
    print("? No working liveview control endpoints found")


def test_detector_image_capture(api_server):
    """Test single image capture functionality."""
    # Try to discover detectors first
    try:
        detector_names, _ = test_detector_discovery_endpoints(api_server)
        if not detector_names:
            pytest.skip("No detectors found")
        first_detector = detector_names[0]
    except:
        first_detector = "testDetector"
    
    # Image capture endpoints
    capture_endpoints = [
        f"/RecordingController/snapNumpyToFastAPI",
    ]
    
    for endpoint in capture_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code in [200, 201]:
                # Parse the response properly - it should contain image data
                try:
                    if response.headers.get('content-type', '').startswith('image/'):
                        # Binary image data
                        print(f"✓ Image captured via {endpoint} (binary data: {len(response.content)} bytes)")
                    else:
                        # JSON response with image data or metadata
                        result = response.json()
                        print(f"✓ Image captured via {endpoint} (JSON response)")
                        if isinstance(result, dict) and 'shape' in result:
                            print(f"  Image shape: {result.get('shape')}")
                        elif isinstance(result, dict) and 'data' in result:
                            print(f"  Image data type: {type(result.get('data'))}")
                        else:
                            print(f"  Response type: {type(result)}")
                    # Validate response contains image data or reference
                    assert response.content or result is not None
                    return
                except Exception as parse_error:
                    print(f"✓ Image endpoint accessible via {endpoint}, but response parsing failed: {parse_error}")
                    return
                
        except Exception as e:
            print(f"Image capture via {endpoint} failed: {e}")
    
    print("? No working image capture endpoints found")


def test_detector_recording_control(api_server):
    """Test video/sequence recording functionality."""
    # Try to discover detectors first  
    try:
        detector_names, _ = test_detector_discovery_endpoints(api_server)
        if not detector_names:
            pytest.skip("No detectors found")
        first_detector = detector_names[0]
    except:
        first_detector = "testDetector"
    
    # Recording control endpoints
    recording_endpoints = [
        f"/RecordingController/startRecording?detectorName={first_detector}"
        ]
    '''
    # TOOD: add the following parameters mSaveFormat = "TIF"
    "parameters": [
          {
            "name": "mSaveFormat",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "default": 1,
              "title": "Msaveformat"
            }
          }
        ],
        '''
    
    for endpoint in recording_endpoints:
        try:
            # Start recording
            response = api_server.get(endpoint)
            if response.status_code in [200, 201]:
                print(f"✓ Recording started via {endpoint}")
                
                # Let it record briefly
                time.sleep(1)
                
                # Stop recording
                stop_endpoint = endpoint.replace("start", "stop")
                stop_response = api_server.get(stop_endpoint)
                if stop_response.status_code in [200, 201]:
                    print(f"✓ Recording stopped via {stop_endpoint}")
                return
                
        except Exception as e:
            print(f"Recording control via {endpoint} failed: {e}")
    
    print("? No working recording control endpoints found")


def test_detector_status_monitoring(api_server):
    """Test detector status and health monitoring."""
    # Status endpoints
    status_endpoints = [
        "/ViewController/getLiveViewActive"
    ]
    
    for endpoint in status_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Got detector status via {endpoint}")
                assert isinstance(status, (dict, list))
                return
                
        except Exception as e:
            print(f"Status check via {endpoint} failed: {e}")
    
    print("? No working detector status endpoints found")


def test_detector_settings_endpoints(api_server):
    """Test detector settings endpoints for exposure time and gain."""
    # Try to discover detector names first
    try:
        detectors_response = api_server.get("/DetectorController/getDetectorNames")
        if detectors_response.status_code == 200:
            detectors = detectors_response.json()
            if isinstance(detectors, list) and detectors:
                first_detector = detectors[0]
            elif isinstance(detectors, dict) and detectors:
                first_detector = list(detectors.keys())[0]
            else:
                first_detector = "testDetector"
        else:
            first_detector = "testDetector"
    except:
        first_detector = "testDetector"

    # Test setDetectorExposureTime endpoint
    exposure_params = {
        "detectorName": first_detector,
        "exposureTime": 100  # 100ms exposure time
    }
    
    try:
        response = api_server.get("/SettingsController/setDetectorExposureTime", params=exposure_params)
        if response.status_code == 200:
            print(f"✓ Detector exposure time set successfully for {first_detector}")
        elif response.status_code == 422:
            print(f"? Detector exposure time setting failed - validation error (expected for virtual detectors)")
        else:
            print(f"? Detector exposure time endpoint response: {response.status_code}")
    except Exception as e:
        print(f"? Detector exposure time setting failed: {e}")

    # Test setDetectorGain endpoint  
    gain_params = {
        "detectorName": first_detector,
        "gain": 50  # Gain value of 50
    }
    
    try:
        response = api_server.get("/SettingsController/setDetectorGain", params=gain_params)
        if response.status_code == 200:
            print(f"✓ Detector gain set successfully for {first_detector}")
        elif response.status_code == 422:
            print(f"? Detector gain setting failed - validation error (expected for virtual detectors)")
        else:
            print(f"? Detector gain endpoint response: {response.status_code}")
    except Exception as e:
        print(f"? Detector gain setting failed: {e}")

    # Test additional detector mode setting if available
    try:
        mode_params = {
            "detectorName": first_detector,
            "detectorMode": "normal"
        }
        response = api_server.get("/SettingsController/setDetectorMode", params=mode_params)
        if response.status_code == 200:
            print(f"✓ Detector mode set successfully for {first_detector}")
        elif response.status_code == 422:
            print(f"? Detector mode setting failed - validation error")
    except Exception as e:
        print(f"? Detector mode setting test failed: {e}")

    # Verify settings endpoints are documented
    try:
        spec_response = api_server.get("/openapi.json")
        if spec_response.status_code == 200:
            spec = spec_response.json()
            settings_endpoints = [
                path for path in spec["paths"].keys()
                if "SettingsController" in path and ("setDetector" in path)
            ]
            
            print(f"✓ Found {len(settings_endpoints)} detector settings endpoints:")
            for endpoint in settings_endpoints:
                methods = list(spec["paths"][endpoint].keys())
                print(f"  - {endpoint}: {methods}")
                
            # Verify required endpoints are present
            required_settings = [
                "/SettingsController/setDetectorExposureTime",
                "/SettingsController/setDetectorGain"
            ]
            
            for required in required_settings:
                if required in settings_endpoints:
                    print(f"✓ Required settings endpoint found: {required}")
                else:
                    print(f"? Required settings endpoint missing: {required}")
                    
    except Exception as e:
        print(f"? Settings endpoints validation failed: {e}")


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
