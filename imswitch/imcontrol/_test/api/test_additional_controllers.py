"""
API tests for ImSwitch additional controller endpoints.
Tests laser control, scanning, recording, and other microscopy-specific functionality.
"""
import pytest
import requests
import time
import json
from typing import Dict, List, Any
from ..api import api_server, base_url


def test_laser_controller_endpoints(api_server):
    """Test laser controller API endpoints."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    
    # Find laser-related endpoints
    paths = spec.get("paths", {})
    laser_endpoints = [p for p in paths.keys() if "Laser" in p or "laser" in p]
    
    if not laser_endpoints:
        pytest.skip("No laser endpoints found in API")
    
    print(f"Found {len(laser_endpoints)} laser endpoints")
    
    # Test laser discovery
    discovery_endpoints = [
        "/LaserController/getLaserNames",
    ]
    
    '''TODO: Change to
        "/LaserController/getLaserNames": {
      "get": {
        "summary": "Getlasernames",
        "description": "Returns the device names of all lasers. These device names can be\npassed to other laser-related functions.",
        "operationId": "getLaserNames_LaserController_getLaserNames_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "items": { "type": "string" },
                  "type": "array",
                  "title": "Response Getlasernames Lasercontroller Getlasernames Get"
                }
              }
            }
          }
        }
      }
    },

'''
    
    for endpoint in discovery_endpoints:
        try:
            response = api_server.get(endpoint)
            if response.status_code == 200:
                lasers = response.json()
                print(f"✓ Found lasers via {endpoint}: {lasers}")
                if lasers:
                    test_laser_control(api_server, lasers)
                return
        except Exception as e:
            print(f"Laser discovery via {endpoint} failed: {e}")


def test_laser_control(api_server, lasers):
    """Test laser power and state control."""
    if isinstance(lasers, dict):
        first_laser = list(lasers.keys())[0]
    elif isinstance(lasers, list):
        first_laser = lasers[0]
    else:
        return
    '''TODO: Change to 
    
        "/LaserController/setLaserActive": {
      "get": {
        "summary": "Setlaseractive",
        "description": "Sets whether the specified laser is powered on.",
        "operationId": "setLaserActive_LaserController_setLaserActive_get",
        "parameters": [
          {
            "name": "laserName",
            "in": "query",
            "required": true,
            "schema": { "type": "string", "title": "Lasername" }
          },
          {
            "name": "active",
            "in": "query",
            "required": true,
            "schema": { "type": "boolean", "title": "Active" }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/HTTPValidationError" }
              }
            }
          }
        }
      }
    },
    "/LaserController/setLaserValue": {
      "get": {
        "summary": "Setlaservalue",
        "description": "Sets the value of the specified laser, in the units that the laser\nuses.",
        "operationId": "setLaserValue_LaserController_setLaserValue_get",
        "parameters": [
          {
            "name": "laserName",
            "in": "query",
            "required": true,
            "schema": { "type": "string", "title": "Lasername" }
          },
          {
            "name": "value",
            "in": "query",
            "required": true,
            "schema": {
              "anyOf": [{ "type": "integer" }, { "type": "number" }],
              "title": "Value"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/HTTPValidationError" }
              }
            }
          }
        }
      }
    },
first set enable, then set value
'''
    # Test laser power control
    power_endpoints = [
        f"/LaserController/setLaserActive",
        
    ]
    
    for endpoint in power_endpoints:
        try:
            if "?" in endpoint:
                response = api_server.post(endpoint)
            else:
                response = api_server.put(endpoint, json={"power": 50})
            
            if response.status_code in [200, 201]:
                print(f"✓ Laser power set via {endpoint}")
                break
        except Exception as e:
            print(f"Laser power control via {endpoint} failed: {e}")
    
    # Test laser enable/disable
    enable_endpoints = [
        f"/LaserController/setLaserEnabled?laserName={first_laser}&enabled=true",
        f"/LaserController/enable?laser={first_laser}",
        f"/lasers/{first_laser}/enable"
    ]
    
    for endpoint in enable_endpoints:
        try:
            response = api_server.post(endpoint)
            if response.status_code in [200, 201]:
                print(f"✓ Laser enabled via {endpoint}")
                # Disable after test
                disable_endpoint = endpoint.replace("enable", "disable").replace("enabled=true", "enabled=false")
                api_server.post(disable_endpoint)
                break
        except Exception as e:
            print(f"Laser enable via {endpoint} failed: {e}")


def test_video_streaming(api_server):
    """Test video streaming functionality."""
    streaming_endpoints = [
        "/RecordingController/video_feeder"
    ]
    
    for endpoint in streaming_endpoints:
        try:
            params = {"startStream": True}
            response = api_server.get(endpoint, params=params)
            if response.status_code in [200, 400, 501]:  # Various acceptable responses
                print(f"✓ Video streaming endpoint accessible: {endpoint} ({response.status_code})")
                
                # Test stopping stream
                params = {"startStream": False}
                response = api_server.get(endpoint, params=params)
                if response.status_code in [200, 400, 501]:
                    print(f"✓ Video streaming stop via {endpoint} ({response.status_code})")
                break
        except Exception as e:
            print(f"Video streaming via {endpoint} failed: {e}")



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