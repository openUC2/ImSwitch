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


class TestSTORMReconController(unittest.TestCase):
    
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
        
        # Mock communication channel and widget
        self.mock_comm_channel = Mock()
        self.mock_widget = Mock()
        
        # Mock the widget signals
        self.mock_widget.sigShowToggled = Mock()
        self.mock_widget.sigShowToggled.connect = Mock()
        self.mock_widget.sigUpdateRateChanged = Mock()
        self.mock_widget.sigUpdateRateChanged.connect = Mock()
        self.mock_widget.sigSliderValueChanged = Mock()
        self.mock_widget.sigSliderValueChanged.connect = Mock()
        
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_initialization(self):
        """Test controller initialization."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            self.mock_widget
        )
        
        self.assertIsNotNone(controller)
        self.assertEqual(controller.threshold, 0.2)
        self.assertEqual(controller.updateRate, 0)
        self.assertFalse(controller._acquisition_active)
        self.assertIsNone(controller._current_session_id)
    
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_start_fast_storm_acquisition(self):
        """Test starting fast STORM acquisition."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            self.mock_widget
        )
        
        # Test starting acquisition without saving
        result = controller.startFastSTORMAcquisition(
            session_id="test_session",
            crop_x=10, crop_y=10, crop_width=50, crop_height=50
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['session_id'], 'test_session')
        self.assertTrue(controller._acquisition_active)
        self.assertIsNotNone(controller._cropping_params)
        self.assertEqual(controller._cropping_params['x'], 10)
        self.assertEqual(controller._cropping_params['width'], 50)
        self.assertFalse(result.get('direct_saving', False))
        
        # Verify detector methods were called
        self.mock_detector.crop.assert_called_once_with(10, 10, 50, 50)
        self.mock_detector.startAcquisition.assert_called_once()
    
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.IS_ARKITEKT', False)
    def test_start_fast_storm_acquisition_with_saving(self):
        """Test starting fast STORM acquisition with direct saving."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            self.mock_widget
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_storm.tiff")
            
            # Test starting acquisition with saving (should trigger direct saving mode)
            result = controller.startFastSTORMAcquisition(
                session_id="test_session_save",
                save_path=save_path,
                save_format="tiff"
            )
            
            self.assertTrue(result['success'])
            self.assertEqual(result['session_id'], 'test_session_save')
            self.assertTrue(controller._acquisition_active)
            self.assertTrue(result.get('direct_saving', False))  # Should be in direct saving mode
            self.assertEqual(controller._save_path, save_path)
            self.assertEqual(controller._save_format, "tiff")
            
            # Stop acquisition to clean up
            controller.stopFastSTORMAcquisition()
    
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_stop_fast_storm_acquisition(self):
        """Test stopping fast STORM acquisition."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            self.mock_widget
        )
        
        # Start acquisition first
        controller.startFastSTORMAcquisition(session_id="test_session")
        
        # Stop acquisition
        result = controller.stopFastSTORMAcquisition()
        
        self.assertTrue(result['success'])
        self.assertEqual(result['session_id'], 'test_session')
        self.assertFalse(controller._acquisition_active)
        self.assertIsNone(controller._current_session_id)
        
        # Verify detector stop was called
        self.mock_detector.stopAcquisition.assert_called_once()
    
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_get_storm_status(self):
        """Test getting STORM status."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            self.mock_widget
        )
        
        # Test status when inactive
        status = controller.getSTORMStatus()
        self.assertFalse(status['acquisition_active'])
        self.assertIsNone(status['session_id'])
        self.assertFalse(status['microeye_available'])
        self.assertFalse(status['direct_saving_mode'])
        self.assertEqual(status['frames_saved'], 0)
        
        # Start acquisition and test status
        controller.startFastSTORMAcquisition(session_id="test_session")
        status = controller.getSTORMStatus()
        self.assertTrue(status['acquisition_active'])
        self.assertEqual(status['session_id'], 'test_session')
    
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_set_storm_parameters(self):
        """Test setting STORM parameters."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            self.mock_widget
        )
        
        # Set parameters
        result = controller.setSTORMParameters(
            threshold=0.5,
            update_rate=5
        )
        
        self.assertEqual(controller.threshold, 0.5)
        self.assertEqual(controller.updateRate, 5)
        self.assertEqual(result['threshold'], 0.5)
        self.assertEqual(result['update_rate'], 5)
    
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False)
    def test_frame_generator(self):
        """Test the frame generator functionality."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            self.mock_widget
        )
        
        # Start acquisition
        controller.startFastSTORMAcquisition(session_id="test_session")
        
        # Test generator
        generator = controller.getSTORMFrameGenerator(num_frames=3, timeout=1.0)
        
        # Get a few frames
        frames = []
        for i, frame_data in enumerate(generator):
            frames.append(frame_data)
            if i >= 2:  # Get 3 frames
                break
        
        self.assertEqual(len(frames), 3)
        
        # Check frame structure
        for frame_data in frames:
            self.assertIn('raw_frame', frame_data)
            self.assertIn('metadata', frame_data)
            self.assertIsInstance(frame_data['raw_frame'], np.ndarray)
            self.assertEqual(frame_data['metadata']['session_id'], 'test_session')
    
    @patch('imswitch.imcontrol.controller.controllers.STORMReconController.isMicroEye', False) 
    def test_acquisition_with_cropping(self):
        """Test acquisition with cropping parameters."""
        controller = STORMReconController(
            self.mock_master, 
            self.mock_comm_channel, 
            self.mock_widget
        )
        
        # Start acquisition with cropping
        result = controller.startFastSTORMAcquisition(
            session_id="crop_test",
            crop_x=20, crop_y=30, crop_width=40, crop_height=35
        )
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['cropping'])
        
        # Test that cropping parameters are stored correctly
        expected_crop = {'x': 20, 'y': 30, 'width': 40, 'height': 35}
        self.assertEqual(controller._cropping_params, expected_crop)
        
        # Verify detector crop method was called with correct parameters
        self.mock_detector.crop.assert_called_once_with(20, 30, 40, 35)


if __name__ == '__main__':
    unittest.main()