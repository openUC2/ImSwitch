"""
Controller for step-by-step acceptance testing of microscope functionality.
Tests motion, lighting, camera, and autofocus features with user confirmation.
"""

from typing import Dict, List, Optional
import time
import json

from imswitch import IS_HEADLESS
from imswitch.imcommon.model import APIExport
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import initLogger


class AcceptanceTestController(ImConWidgetController):
    """
    Controller for acceptance testing of microscope hardware and software.
    Provides step-by-step validation workflow with user confirmation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        
        # Store test results
        self.test_results = {
            "motion": {},
            "lighting": {},
            "camera": {},
            "autofocus": {},
            "timestamp": None
        }
        
        # Test state
        self.current_test_step = None
        self.test_in_progress = False

    # ==================== Motion Tests ====================
    
    @APIExport(runOnUIThread=True)
    def homeAxisX(self, positionerName: str = None) -> Dict:
        """
        Home X axis and return status.
        
        Args:
            positionerName: Name of the positioner (optional, uses default if None)
            
        Returns:
            Dict with status and message
        """
        try:
            if positionerName is None:
                positionerName = self._master.positionersManager.getAllDeviceNames()[0]
            
            positioner = self._master.positionersManager[positionerName]
            
            # Execute homing for X axis
            if hasattr(positioner, 'home'):
                positioner.home_x()
                self.__logger.info(f"Homing X axis for {positionerName}")
                return {
                    "status": "success",
                    "message": "X axis homing initiated",
                    "axis": "X",
                    "positioner": positionerName
                }
            else:
                return {
                    "status": "error",
                    "message": "Positioner does not support homing",
                    "axis": "X",
                    "positioner": positionerName
                }
        except Exception as e:
            self.__logger.error(f"Error homing X axis: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "axis": "X"
            }

    @APIExport(runOnUIThread=True)
    def homeAxisY(self, positionerName: str = None) -> Dict:
        """
        Home Y axis and return status.
        
        Args:
            positionerName: Name of the positioner (optional, uses default if None)
            
        Returns:
            Dict with status and message
        """
        try:
            if positionerName is None:
                positionerName = self._master.positionersManager.getAllDeviceNames()[0]
            
            positioner = self._master.positionersManager[positionerName]
            
            # Execute homing for Y axis
            if hasattr(positioner, 'home'):
                positioner.home_y()
                self.__logger.info(f"Homing Y axis for {positionerName}")
                return {
                    "status": "success",
                    "message": "Y axis homing initiated",
                    "axis": "Y",
                    "positioner": positionerName
                }
            else:
                return {
                    "status": "error",
                    "message": "Positioner does not support homing",
                    "axis": "Y",
                    "positioner": positionerName
                }
        except Exception as e:
            self.__logger.error(f"Error homing Y axis: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "axis": "Y"
            }

    @APIExport(runOnUIThread=True)
    def homeAxisZ(self, positionerName: str = None) -> Dict:
        """
        Home Z axis and return status.
        
        Args:
            positionerName: Name of the positioner (optional, uses default if None)
            
        Returns:
            Dict with status and message
        """
        try:
            if positionerName is None:
                positionerName = self._master.positionersManager.getAllDeviceNames()[0]
            
            positioner = self._master.positionersManager[positionerName]
            
            # Execute homing for Z axis
            if hasattr(positioner, 'home'):
                positioner.home_z()
                self.__logger.info(f"Homing Z axis for {positionerName}")
                return {
                    "status": "success",
                    "message": "Z axis homing initiated",
                    "axis": "Z",
                    "positioner": positionerName
                }
            else:
                return {
                    "status": "error",
                    "message": "Positioner does not support homing",
                    "axis": "Z",
                    "positioner": positionerName
                }
        except Exception as e:
            self.__logger.error(f"Error homing Z axis: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "axis": "Z"
            }

    @APIExport(runOnUIThread=True)
    def homeAxisA(self, positionerName: str = None) -> Dict:
        """
        Home A axis and return status.
        
        Args:
            positionerName: Name of the positioner (optional, uses default if None)
        Returns:
            Dict with status and message
        """
        try:
            if positionerName is None:
                positionerName = self._master.positionersManager.getAllDeviceNames()[0]
            
            positioner = self._master.positionersManager[positionerName]
            
            # Execute homing for A axis
            if hasattr(positioner, 'home'):
                positioner.home_a()
                self.__logger.info(f"Homing A axis for {positionerName}")
                return {
                    "status": "success",
                    "message": "A axis homing initiated",
                    "axis": "A",
                    "positioner": positionerName
                }
            else:
                return {
                    "status": "error",
                    "message": "Positioner does not support homing",
                    "axis": "A",
                    "positioner": positionerName
                }
        except Exception as e:
            self.__logger.error(f"Error homing A axis: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "axis": "A"
            }


    @APIExport(runOnUIThread=True)
    def moveToTestPosition(self, positionerName: str = None, x: float = 10000, y: float = 10000, z: float = -1000, a: float = 1000) -> Dict:
        """
        Move stage to a predefined test position.
        
        Args:
            positionerName: Name of the positioner (optional)
            x: X position in micrometers (default: 2000)
            y: Y position in micrometers (default: 2000)
            z: Z position in micrometers (default: 2000)
            
        Returns:
            Dict with status and position info
        """
        try:
            speed = 5000
            if positionerName is None:
                positionerName = self._master.positionersManager.getAllDeviceNames()[0]
            
            self.positioner = self._master.positionersManager[positionerName]
            
            # Move to absolute position for X, Y, Z (test all three axes)
            if 'X' in self.positioner.axes:
                self.positioner.move(value=x, axis='X', speed=speed, is_absolute=False, is_blocking=True)
            if 'Y' in self.positioner.axes:
                self.positioner.move(value=y, axis='Y', speed=speed, is_absolute=False, is_blocking=True)
            if 'Z' in self.positioner.axes:
                self.positioner.move(value=z/-10, axis='Z', speed=speed, is_absolute=False, is_blocking=True)
            if 'A' in self.positioner.axes:
                self.positioner.move(value=a, axis='A', speed=speed, is_absolute=True, is_blocking=True)
                
            self.__logger.info(f"Moving to test position X:{x}, Y:{y}, Z:{z}")
            
            return {
                "status": "success",
                "message": "Moving to test position",
                "target_position": {"x": x, "y": y, "z": z},
                "positioner": positionerName
            }
        except Exception as e:
            self.__logger.error(f"Error moving to test position: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    @APIExport(runOnUIThread=True)
    def moveXPlus(self, positionerName: str = None, distance: float = 1000) -> Dict:
        """
        Move stage in positive X direction.
        
        Args:
            positionerName: Name of the positioner (optional)
            distance: Distance to move in micrometers (default: 1000)
            
        Returns:
            Dict with status and movement info
        """
        try:
            if positionerName is None:
                positionerName = self._master.positionersManager.getAllDeviceNames()[0]
            
            positioner = self._master.positionersManager[positionerName]
            positioner.move(value=distance, axis='X', is_absolute=False, is_blocking=False)
            
            self.__logger.info(f"Moving +X by {distance} µm")
            
            return {
                "status": "success",
                "message": f"Moving +{distance} µm in X direction",
                "axis": "X",
                "direction": "plus",
                "distance": distance,
                "positioner": positionerName
            }
        except Exception as e:
            self.__logger.error(f"Error moving X+: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    @APIExport(runOnUIThread=True)
    def moveXMinus(self, positionerName: str = None, distance: float = 1000) -> Dict:
        """
        Move stage in negative X direction.
        
        Args:
            positionerName: Name of the positioner (optional)
            distance: Distance to move in micrometers (default: 1000)
            
        Returns:
            Dict with status and movement info
        """
        try:
            if positionerName is None:
                positionerName = self._master.positionersManager.getAllDeviceNames()[0]
            
            positioner = self._master.positionersManager[positionerName]
            positioner.move(value=-distance, axis='X', is_absolute=False, is_blocking=False)
            
            self.__logger.info(f"Moving -X by {distance} µm")
            
            return {
                "status": "success",
                "message": f"Moving -{distance} µm in X direction",
                "axis": "X",
                "direction": "minus",
                "distance": distance,
                "positioner": positionerName
            }
        except Exception as e:
            self.__logger.error(f"Error moving X-: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    # ==================== Lighting Tests ====================
    
    @APIExport(runOnUIThread=True)
    def getAvailableLightSources(self) -> Dict:
        """
        Get list of available light sources/lasers.
        
        Returns:
            Dict with list of available light sources
        """
        try:
            laser_names = self._master.lasersManager.getAllDeviceNames()
            
            sources = []
            for name in laser_names:
                laser = self._master.lasersManager[name]
                sources.append({
                    "name": name,
                    "enabled": int(laser.enabled if hasattr(laser, 'enabled') else False)
                })
            
            return {
                "status": "success",
                "light_sources": sources,
                "count": len(sources)
            }
        except Exception as e:
            self.__logger.error(f"Error getting light sources: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "light_sources": []
            }

    @APIExport(runOnUIThread=True)
    def turnOnLight(self, laserName: str) -> Dict:
        """
        Turn on a specific light source.
        
        Args:
            laserName: Name of the laser/light source
            
        Returns:
            Dict with status
        """
        try:
            laser = self._master.lasersManager[laserName]
            # Turn on
            laser.setEnabled(True)
            laser.setValue(255)  # Set intensity
            time.sleep(0.5)  # Allow time to see the light ON
            # Turn off
            laser.setEnabled(False)
            
            self.__logger.info(f"Toggled light on/off: {laserName}")
            
            return {
                "status": "success",
                "message": f"Light source {laserName} toggled ON then OFF",
                "laser": laserName,
                "action": "toggle"
            }
        except Exception as e:
            self.__logger.error(f"Error turning on light {laserName}: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "laser": laserName
            }

    @APIExport(runOnUIThread=True)
    def setLaserActiveAcceptanceTest(self, laser: str, active: bool, intensity: int = 255) -> Dict:
        """
        Set a laser/light source active or inactive.
        
        Args:
            laser: Name of the laser/light source
            active: True to turn on, False to turn off
            intensity: Intensity value (0-255) when turning on
            
        Returns:
            Dict with status
        """
        try:
            laserDevice = self._master.lasersManager[laser]
            
            if active:
                laserDevice.setEnabled(True)
                laserDevice.setValue(intensity)
                action = "enabled"
            else:
                laserDevice.setEnabled(False)
                action = "disabled"
            
            self.__logger.info(f"Set laser {laser} active={active}")
            
            return {
                "status": "success",
                "message": f"Light source {laser} {action}",
                "laser": laser,
                "active": active
            }
        except Exception as e:
            self.__logger.error(f"Error setting laser active: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "laser": laser
            }

    @APIExport(runOnUIThread=True)
    def turnOffLight(self, laserName: str) -> Dict:
        """
        Turn off a specific light source.
        
        Args:
            laserName: Name of the laser/light source
            
        Returns:
            Dict with status
        """
        try:
            laser = self._master.lasersManager[laserName]
            laser.setEnabled(False)
            
            self.__logger.info(f"Turned off light: {laserName}")
            
            return {
                "status": "success",
                "message": f"Light source {laserName} turned OFF",
                "laser": laserName,
                "enabled": False
            }
        except Exception as e:
            self.__logger.error(f"Error turning off light {laserName}: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "laser": laserName
            }

    # ==================== Camera/Detector Tests ====================
    
    @APIExport(runOnUIThread=True)
    def getCameraInfo(self) -> Dict:
        """
        Get information about available cameras and their specifications.
        
        Returns:
            Dict with camera information
        """
        try:
            detector_names = self._master.detectorsManager.getAllDeviceNames()
            
            cameras = []
            for name in detector_names:
                detector = self._master.detectorsManager[name]
                
                # Get camera specifications
                camera_info = {
                    "name": name,
                    "model": detector.model if hasattr(detector, 'model') else "Unknown",
                }
                
                # Get shape if available
                if hasattr(detector, 'parameters') and 'image_width' in detector.parameters:
                    camera_info["width_pixels"] = detector.parameters['image_width']
                    camera_info["height_pixels"] = detector.parameters['image_height']
                elif hasattr(detector, 'shape'):
                    camera_info["width_pixels"] = detector.shape[1]
                    camera_info["height_pixels"] = detector.shape[0]
                
                # Get pixel size if available
                if hasattr(detector, 'pixelSizeUm'):
                    camera_info["pixel_size_um"] = detector.pixelSizeUm
                
                cameras.append(camera_info)
            
            return {
                "status": "success",
                "cameras": cameras,
                "count": len(cameras)
            }
        except Exception as e:
            self.__logger.error(f"Error getting camera info: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "cameras": []
            }

    @APIExport(runOnUIThread=True)
    def getCurrentExposureAndGain(self, detectorName: str = None) -> Dict:
        """
        Get current exposure time and gain settings.
        
        Args:
            detectorName: Name of the detector (optional)
            
        Returns:
            Dict with exposure and gain values
        """
        try:
            if detectorName is None:
                detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
            
            detector = self._master.detectorsManager[detectorName]
            
            exposure = None
            gain = None
            
            # Try to get exposure time
            if hasattr(detector, 'getParameter'):
                try:
                    exposure = detector.getParameter('Exposure')
                except:
                    pass
                try:
                    gain = detector.getParameter('Gain')
                except:
                    pass
            
            return {
                "status": "success",
                "detector": detectorName,
                "exposure_ms": exposure,
                "gain": gain
            }
        except Exception as e:
            self.__logger.error(f"Error getting exposure/gain: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    # ==================== Autofocus Tests ====================
    
    @APIExport(runOnUIThread=True)
    def runAutofocus(self, maxZ:int=100, minZ:int=-100, stepSize:float=10.0, illuminationChannel=None) -> Dict:
        """
        Run software-based autofocus.
        
        Returns:
            Dict with autofocus result
        """
        try:
            
            if illuminationChannel is None:
                # pick one that sounds like "LED"
                self.laserNames = self._master.lasersManager.getAllDeviceNames()
                for name in self.laserNames:
                    if "led" in name.lower():
                        illuminationChannel = name
                        break
            # get autofocus controller
            autofocusController = self._master.getController('Autofocus')

            if autofocusController is None:
                self._logger.warning("AutofocusController not available - skipping autofocus")
                return None

            # Calculate range from min/max
            rangez = abs(maxZ - minZ) / 2.0 if maxZ > minZ else 50.0
            resolutionz = stepSize if stepSize > 0 else 10.0

            # Call autofocus directly - the method is already decorated with @APIExport
            self.__logger.info("Autofocus started")
            result = autofocusController.doAutofocusBackground(
                rangez=rangez,
                resolutionz=resolutionz,
                defocusz=0,
                axis="Z",
                tSettle =0.1, # TODO: Implement via frontend parameters
                isDebug=False,
                nGauss=7,
                nCropsize=2048,
                focusAlgorithm="LAPE",
                static_offset=0.0,
                twoStage=False
            )

            self._logger.debug(f"Autofocus completed successfully")
            return {"result": result}

        except Exception as e:
            self.__logger.error(f"Error running autofocus: {str(e)}")

            return {
                "status": "success",
                "message": "Autofocus initiated"
            }
        except Exception as e:
            self.__logger.error(f"Error running autofocus: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    @APIExport(runOnUIThread=True)
    def getAutofocusStatus(self) -> Dict:
        """
        Get current autofocus status.
        
        Returns:
            Dict with autofocus status
        """
        try:
            if not hasattr(self._master, 'autofocusController'):
                return {
                    "status": "unavailable",
                    "message": "Autofocus not available"
                }
            
            # This is a placeholder - actual implementation depends on autofocus controller
            return {
                "status": "available",
                "message": "Autofocus ready"
            }
        except Exception as e:
            self.__logger.error(f"Error getting autofocus status: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    # ==================== Test Recording & Reporting ====================
    
    @APIExport(runOnUIThread=True)
    def recordTestResult(self, category: str, test_name: str, passed: bool, notes: str = "") -> Dict:
        """
        Record a test result.
        
        Args:
            category: Test category (motion, lighting, camera, autofocus)
            test_name: Name of the specific test
            passed: Whether the test passed
            notes: Optional notes from user
            
        Returns:
            Dict with confirmation
        """
        try:
            if category not in self.test_results:
                self.test_results[category] = {}
            
            self.test_results[category][test_name] = {
                "passed": passed,
                "notes": notes,
                "timestamp": time.time()
            }
            
            self.__logger.info(f"Recorded test result: {category}/{test_name} = {'PASS' if passed else 'FAIL'}")
            
            return {
                "status": "success",
                "message": "Test result recorded",
                "category": category,
                "test_name": test_name,
                "passed": passed
            }
        except Exception as e:
            self.__logger.error(f"Error recording test result: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    @APIExport(runOnUIThread=True)
    def getTestReport(self) -> Dict:
        """
        Get complete test report with all results.
        
        Returns:
            Dict with all test results
        """
        try:
            # Calculate summary statistics
            total_tests = 0
            passed_tests = 0
            
            for category, tests in self.test_results.items():
                if isinstance(tests, dict):
                    for test_name, result in tests.items():
                        if isinstance(result, dict) and 'passed' in result:
                            total_tests += 1
                            if result['passed']:
                                passed_tests += 1
            
            return {
                "status": "success",
                "test_results": self.test_results,
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
                }
            }
        except Exception as e:
            self.__logger.error(f"Error getting test report: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    @APIExport(runOnUIThread=True)
    def resetTestResults(self) -> Dict:
        """
        Reset all test results to start a new test session.
        
        Returns:
            Dict with confirmation
        """
        try:
            self.test_results = {
                "motion": {},
                "lighting": {},
                "camera": {},
                "autofocus": {},
                "timestamp": time.time()
            }
            
            self.__logger.info("Test results reset")
            
            return {
                "status": "success",
                "message": "Test results have been reset"
            }
        except Exception as e:
            self.__logger.error(f"Error resetting test results: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
