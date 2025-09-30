import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os

# Mock the microEye imports since they might not be available
with patch.dict('sys.modules', {
    'microEye': Mock(),
    'microEye.Filters': Mock(),
    'microEye.fitting.fit': Mock(),
    'microEye.fitting.results': Mock()
}):
    from imswitch.imcontrol.controller.controllers.STORMReconController import STORMReconController


class TestSTORMReconControllerRenewed(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the master controller and its managers
        self.mock_master = Mock()
        self.mock_detectors_manager = Mock()
        self.mock_detector = Mock()
        
        # Configure mock detector
        self.mock_detector.getLatestFrame.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
        # Mock getChunk to return array with batch dimension
        self.mock_detector.getChunk.return_value = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint16)
        self.mock_detector.startAcquisition = Mock()
        self.mock_detector.stopAcquisition = Mock()
        self.mock_detector.crop = Mock()
        
        # Configure detectors manager
        self.mock_detectors_manager.getAllDeviceNames.return_value = ['TestDetector']
        self.mock_detectors_manager.__getitem__.return_value = self.mock_detector
        self.mock_master.detectorsManager = self.mock_detectors_manager
        
        # Mock communication channel - no widget needed
        self.mock_comm_channel = Mock()
        
        # Create controller without widget dependency
        self.controller = None
        
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.HAS_STORM_MODELS', False)
    def test_initialization_without_microeye(self):
        """Test controller initialization without microeye."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None  # No widget
        )
        
        self.assertIsNotNone(controller)
        self.assertFalse(controller._acquisition_active)
        self.assertIsNone(controller._current_session_id)
        self.assertEqual(controller._frame_count, 0)

    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', True)
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.HAS_STORM_MODELS', True)
    def test_initialization_with_microeye(self):
        """Test controller initialization with microeye."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        self.assertIsNotNone(controller)
        self.assertTrue(hasattr(controller, 'imageComputationWorker'))
        self.assertTrue(hasattr(controller, '_processing_params'))

    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.HAS_STORM_MODELS', False)
    def test_set_storm_processing_parameters(self):
        """Test setting STORM processing parameters."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        # Test parameter setting without pydantic models
        result = controller.setSTORMProcessingParameters(
            threshold=0.5,
            fit_roi_size=15,
            update_rate=5
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['parameters']['threshold'], 0.5)

    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', True)
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.HAS_STORM_MODELS', False)
    def test_start_local_storm_reconstruction(self):
        """Test starting local STORM reconstruction."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        # Mock the worker
        controller.imageComputationWorker = Mock()
        controller.imageComputationWorker.setActive = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = controller.startSTORMReconstructionLocal(
                session_id="test_local_session",
                processing={'threshold': 0.3}
            )
            
            self.assertTrue(result['success'])
            self.assertEqual(result['session_id'], 'test_local_session')
            
            # Verify worker was activated
            controller.imageComputationWorker.setActive.assert_called_with(True)

    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_get_storm_reconstruction_status(self):
        """Test getting STORM reconstruction status."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        status = controller.getSTORMReconstructionStatus()
        
        self.assertIn('acquisition_active', status)
        self.assertIn('local_processing_active', status)
        self.assertIn('last_reconstruction_path', status)
        self.assertIn('microeye_worker_available', status)
        self.assertFalse(status['acquisition_active'])

    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_get_last_reconstructed_image_path(self):
        """Test getting last reconstructed image path."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        # Initially should be None
        path = controller.getLastReconstructedImagePath()
        self.assertIsNone(path)
        
        # Set a path and test retrieval
        test_path = "/test/path/reconstruction.tif"
        controller._last_reconstruction_path = test_path
        path = controller.getLastReconstructedImagePath()
        self.assertEqual(path, test_path)

    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', True)
    def test_enhanced_frame_processing(self):
        """Test enhanced frame processing functionality."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        # Setup session directory
        controller._current_session_id = "test_session"
        controller._setupLocalDataDirectory()
        
        # Mock worker
        controller.imageComputationWorker = Mock()
        controller.imageComputationWorker.reconSTORMFrame.return_value = (
            np.ones((100, 100), dtype=np.float32),  # reconstructed frame
            np.array([[50, 50, 100, 1000]])  # localizations
        )
        
        # Test frame processing
        test_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
        result_path = controller._enhancedProcessFrame(test_frame)
        
        self.assertIsNotNone(result_path)
        self.assertTrue(result_path.endswith('.tif'))

    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', True)
    def test_worker_functionality(self):
        """Test the modernized worker functionality."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        # Test worker initialization
        worker = controller.STORMReconImageComputationWorker()
        
        self.assertEqual(worker.threshold, 0.2)
        self.assertEqual(worker.fit_roi_size, 13)
        self.assertFalse(worker.active)
        
        # Test parameter setting
        worker.setThreshold(0.5)
        worker.setFitRoiSize(15)
        worker.setActive(True)
        
        self.assertEqual(worker.threshold, 0.5)
        self.assertEqual(worker.fit_roi_size, 15)
        self.assertTrue(worker.active)

    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_backward_compatibility(self):
        """Test that legacy API methods still work."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        # Test legacy setSTORMParameters method
        with patch.object(controller, '_logger') as mock_logger:
            result = controller.setSTORMParameters(
                threshold=0.4,
                roi_size=11,
                update_rate=8
            )
            
            # Should warn about deprecation
            mock_logger.warning.assert_called()
            self.assertTrue(result['success'])

    def test_data_directory_setup(self):
        """Test local data directory setup."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            None
        )
        
        controller._current_session_id = "test_session"
        controller._setupLocalDataDirectory()
        
        self.assertIsNotNone(controller._session_directory)
        self.assertTrue(str(controller._session_directory).endswith('test_session'))


if __name__ == '__main__':
    unittest.main()