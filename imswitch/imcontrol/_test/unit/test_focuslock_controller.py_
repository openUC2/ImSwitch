"""
Unit tests for FocusLockController refactoring.

Tests the new dataclass-based parameter management, API methods,
and separation of concerns between measurement and locking.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

from imswitch.imcontrol.controller.controllers.FocusLockController import (
    FocusLockController,
    FocusLockParams,
    PIControllerParams, 
    CalibrationParams,
    FocusLockState,
)


class TestFocusLockDataclasses:
    """Test the dataclasses used for parameter management."""
    
    def test_focus_lock_params_creation(self):
        """Test FocusLockParams dataclass creation and conversion."""
        params = FocusLockParams(
            focus_metric="astigmatism",
            gaussian_sigma=15.0,
            background_threshold=50.0,
            update_freq=20.0,
        )
        
        assert params.focus_metric == "astigmatism"
        assert params.gaussian_sigma == 15.0
        assert params.background_threshold == 50.0
        assert params.update_freq == 20.0
        
        # Test dictionary conversion
        param_dict = params.to_dict()
        assert param_dict["focus_metric"] == "astigmatism"
        assert param_dict["gaussian_sigma"] == 15.0
        
    def test_pi_controller_params_creation(self):
        """Test PIControllerParams dataclass creation and conversion."""
        params = PIControllerParams(
            kp=1.5,
            ki=0.8,
            safety_distance_limit=10.0,
        )
        
        assert params.kp == 1.5
        assert params.ki == 0.8
        assert params.safety_distance_limit == 10.0
        
        param_dict = params.to_dict()
        assert param_dict["kp"] == 1.5
        assert param_dict["safety_distance_limit"] == 10.0
        
    def test_calibration_params_creation(self):
        """Test CalibrationParams dataclass creation and conversion."""
        params = CalibrationParams(
            from_position=45.0,
            to_position=55.0,
            num_steps=30,
            settle_time=1.0,
        )
        
        assert params.from_position == 45.0
        assert params.to_position == 55.0
        assert params.num_steps == 30
        assert params.settle_time == 1.0
        
        param_dict = params.to_dict()
        assert param_dict["num_steps"] == 30
        assert param_dict["settle_time"] == 1.0
        
    def test_focus_lock_state_creation(self):
        """Test FocusLockState dataclass creation and conversion."""
        state = FocusLockState(
            is_measuring=True,
            is_locked=False,
            current_focus_value=123.45,
        )
        
        assert state.is_measuring is True
        assert state.is_locked is False
        assert state.current_focus_value == 123.45
        
        state_dict = state.to_dict()
        assert state_dict["is_measuring"] is True
        assert state_dict["current_focus_value"] == 123.45


class TestFocusLockControllerParameterManagement:
    """Test parameter management functionality."""
    
    @pytest.fixture
    def mock_setup_info(self):
        """Create mock setup info for testing."""
        setup_info = Mock()
        setup_info.focusLock = Mock()
        setup_info.focusLock.camera = "test_camera"
        setup_info.focusLock.positioner = "test_positioner"
        setup_info.focusLock.updateFreq = 15
        setup_info.focusLock.piKp = 2.0
        setup_info.focusLock.piKi = 1.0
        setup_info.focusLock.focusLockMetric = "JPG"
        setup_info.focusLock.cropCenter = None
        setup_info.focusLock.cropSize = None
        return setup_info
    
    @pytest.fixture
    def mock_master(self):
        """Create mock master controller."""
        master = Mock()
        master.detectorsManager = {"test_camera": Mock()}
        master.positionersManager = {"test_positioner": Mock()}
        master.detectorsManager["test_camera"].startAcquisition = Mock()
        master.positionersManager["test_positioner"].get_abs = Mock(return_value=50.0)
        return master
    
    @pytest.fixture
    def mock_comm_channel(self):
        """Create mock communication channel."""
        comm_channel = Mock()
        comm_channel.sigUpdateImage = Mock()
        comm_channel.sigUpdateImage.connect = Mock()
        return comm_channel
    
    @patch('imswitch.imcontrol.controller.controllers.FocusLockController.IS_HEADLESS', True)
    def test_parameter_initialization(self, mock_setup_info, mock_master, mock_comm_channel):
        """Test that parameters are properly initialized from setup info."""
        with patch('imswitch.imcontrol.controller.controllers.FocusLockController.ProcessDataThread'), \
             patch('imswitch.imcontrol.controller.controllers.FocusLockController.FocusCalibThread'):
            
            controller = FocusLockController(
                setupInfo=mock_setup_info,
                commChannel=mock_comm_channel,
                master=mock_master,
            )
            
            # Check that dataclass parameters are initialized correctly
            assert controller._focus_params.focus_metric == "JPG"
            assert controller._focus_params.update_freq == 15
            assert controller._pi_params.kp == 2.0
            assert controller._pi_params.ki == 1.0
            
            # Check legacy parameters for backward compatibility
            assert controller.kp == 2.0
            assert controller.ki == 1.0
    
    @patch('imswitch.imcontrol.controller.controllers.FocusLockController.IS_HEADLESS', True)
    def test_get_focus_lock_params_api(self, mock_setup_info, mock_master, mock_comm_channel):
        """Test the getFocusLockParams API method."""
        with patch('imswitch.imcontrol.controller.controllers.FocusLockController.ProcessDataThread'), \
             patch('imswitch.imcontrol.controller.controllers.FocusLockController.FocusCalibThread'):
            
            controller = FocusLockController(
                setupInfo=mock_setup_info,
                commChannel=mock_comm_channel, 
                master=mock_master,
            )
            
            params = controller.getFocusLockParams()
            
            assert isinstance(params, dict)
            assert params["focus_metric"] == "JPG"
            assert params["update_freq"] == 15
            assert "gaussian_sigma" in params
            assert "background_threshold" in params
    
    @patch('imswitch.imcontrol.controller.controllers.FocusLockController.IS_HEADLESS', True)
    def test_set_focus_lock_params_api(self, mock_setup_info, mock_master, mock_comm_channel):
        """Test the setFocusLockParams API method."""
        with patch('imswitch.imcontrol.controller.controllers.FocusLockController.ProcessDataThread') as mock_thread, \
             patch('imswitch.imcontrol.controller.controllers.FocusLockController.FocusCalibThread'):
            
            mock_process_thread = Mock()
            mock_thread.return_value = mock_process_thread
            
            controller = FocusLockController(
                setupInfo=mock_setup_info,
                commChannel=mock_comm_channel,
                master=mock_master,
            )
            
            # Test updating parameters
            updated_params = controller.setFocusLockParams(
                focus_metric="astigmatism",
                gaussian_sigma=20.0,
                background_threshold=60.0,
            )
            
            assert updated_params["focus_metric"] == "astigmatism"
            assert updated_params["gaussian_sigma"] == 20.0
            assert updated_params["background_threshold"] == 60.0
            
            # Check that legacy attributes are updated
            assert controller.gaussianSigma == 20.0
            assert controller.backgroundThreshold == 60.0
            
            # Check that process thread is notified
            mock_process_thread.setFocusLockMetric.assert_called_with("astigmatism")
    
    @patch('imswitch.imcontrol.controller.controllers.FocusLockController.IS_HEADLESS', True)
    def test_get_set_pi_controller_params(self, mock_setup_info, mock_master, mock_comm_channel):
        """Test PI controller parameter management."""
        with patch('imswitch.imcontrol.controller.controllers.FocusLockController.ProcessDataThread'), \
             patch('imswitch.imcontrol.controller.controllers.FocusLockController.FocusCalibThread'):
            
            controller = FocusLockController(
                setupInfo=mock_setup_info,
                commChannel=mock_comm_channel,
                master=mock_master,
            )
            
            # Test getting parameters
            params = controller.getPIControllerParams()
            assert params["kp"] == 2.0
            assert params["ki"] == 1.0
            
            # Test setting parameters
            updated_params = controller.setPIControllerParams(
                kp=3.5,
                ki=1.8,
                safety_distance_limit=8.0,
            )
            
            assert updated_params["kp"] == 3.5
            assert updated_params["ki"] == 1.8
            assert updated_params["safety_distance_limit"] == 8.0
            
            # Check legacy attributes
            assert controller.kp == 3.5
            assert controller.ki == 1.8


class TestFocusLockMeasurementControl:
    """Test focus measurement control functionality."""
    
    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller for testing."""
        controller = Mock()
        controller._state = FocusLockState()
        controller._emitStateChangedSignal = Mock()
        controller._logger = Mock()
        controller.unlockFocus = Mock()
        return controller
    
    def test_start_focus_measurement(self, mock_controller):
        """Test starting focus measurements."""
        # Import the actual methods to test
        from imswitch.imcontrol.controller.controllers.FocusLockController import FocusLockController
        
        # Bind the method to our mock controller
        start_method = FocusLockController.startFocusMeasurement.__get__(mock_controller)
        
        # Test starting measurement
        result = start_method()
        
        assert result is True
        assert mock_controller._state.is_measuring is True
        mock_controller._emitStateChangedSignal.assert_called_once()
        
        # Test starting when already measuring  
        mock_controller._emitStateChangedSignal.reset_mock()
        result = start_method()
        
        assert result is False
        mock_controller._emitStateChangedSignal.assert_not_called()
    
    def test_stop_focus_measurement(self, mock_controller):
        """Test stopping focus measurements."""
        from imswitch.imcontrol.controller.controllers.FocusLockController import FocusLockController
        
        # Set initial state
        mock_controller._state.is_measuring = True
        
        # Bind the method to our mock controller
        stop_method = FocusLockController.stopFocusMeasurement.__get__(mock_controller)
        
        # Test stopping measurement
        result = stop_method()
        
        assert result is True
        assert mock_controller._state.is_measuring is False
        mock_controller.unlockFocus.assert_called_once()
        mock_controller._emitStateChangedSignal.assert_called_once()


class TestCalibrationFunctionality:
    """Test focus calibration functionality."""
    
    def test_calibration_params_management(self):
        """Test calibration parameter management."""
        # This would test the calibration parameter API methods
        params = CalibrationParams(
            from_position=40.0,
            to_position=60.0,
            num_steps=25,
            settle_time=0.8,
        )
        
        param_dict = params.to_dict()
        assert param_dict["from_position"] == 40.0
        assert param_dict["to_position"] == 60.0
        assert param_dict["num_steps"] == 25
        assert param_dict["settle_time"] == 0.8


if __name__ == "__main__":
    # Simple test runner if running this file directly
    import sys
    
    # Test dataclasses
    print("Testing dataclasses...")
    test_dataclasses = TestFocusLockDataclasses()
    test_dataclasses.test_focus_lock_params_creation()
    test_dataclasses.test_pi_controller_params_creation()
    test_dataclasses.test_calibration_params_creation()
    test_dataclasses.test_focus_lock_state_creation()
    print("✓ Dataclass tests passed")
    
    # Test calibration params
    print("Testing calibration functionality...")
    test_calib = TestCalibrationFunctionality()
    test_calib.test_calibration_params_management()
    print("✓ Calibration tests passed")
    
    print("All tests passed!")