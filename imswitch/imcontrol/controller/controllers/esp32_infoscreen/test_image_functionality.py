#!/usr/bin/env python3
"""
Tests for image display functionality
"""

import unittest
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from uc2_serial_controller import UC2SerialController

class TestImageFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.controller = UC2SerialController()
        self.controller.is_connected = True
        self.controller.serial_conn = Mock()
        
        # Mock the _send_message method to capture sent messages
        self.sent_messages = []
        def mock_send_message(message):
            self.sent_messages.append(message)
            return True
        self.controller._send_message = mock_send_message
    
    def test_send_image_numpy_array_rgb(self):
        """Test sending RGB numpy array image"""
        # Create test image: 10x10 red square
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # All red
        
        success = self.controller.send_image(test_image, "Test Red")
        
        self.assertTrue(success)
        self.assertEqual(len(self.sent_messages), 1)
        
        message = self.sent_messages[0]
        self.assertEqual(message["type"], "display_image_command")
        self.assertEqual(message["data"]["tab_name"], "Test Red")
        self.assertEqual(message["data"]["width"], 10)
        self.assertEqual(message["data"]["height"], 10)
        self.assertEqual(message["data"]["format"], "rgb565")
        self.assertIn("image_data", message["data"])
    
    def test_send_image_grayscale_conversion(self):
        """Test conversion of grayscale image to RGB"""
        # Create grayscale test image
        test_image = np.ones((5, 5), dtype=np.uint8) * 128  # Gray
        
        success = self.controller.send_image(test_image, "Gray Test")
        
        self.assertTrue(success)
        message = self.sent_messages[0]
        self.assertEqual(message["data"]["width"], 5)
        self.assertEqual(message["data"]["height"], 5)
    
    def test_send_image_rgba_conversion(self):
        """Test conversion of RGBA image to RGB"""
        # Create RGBA test image (with alpha channel)
        test_image = np.ones((3, 3, 4), dtype=np.uint8) * 100
        test_image[:, :, 3] = 255  # Full alpha
        
        success = self.controller.send_image(test_image, "RGBA Test")
        
        self.assertTrue(success)
        message = self.sent_messages[0]
        # Should be converted to RGB format
        self.assertEqual(message["data"]["width"], 3)
        self.assertEqual(message["data"]["height"], 3)
    
    def test_send_image_float_conversion(self):
        """Test conversion of float image to uint8"""
        # Create float image (0.0 to 1.0 range)
        test_image = np.ones((4, 4, 3), dtype=np.float32) * 0.5  # Mid gray
        
        success = self.controller.send_image(test_image, "Float Test")
        
        self.assertTrue(success)
        # Should be converted internally
    
    def test_send_image_resize_large(self):
        """Test automatic resizing of large images"""
        # Create image larger than default max size
        large_image = np.zeros((300, 400, 3), dtype=np.uint8)
        
        with patch.object(self.controller.logger, 'info') as mock_log:
            success = self.controller.send_image(large_image, "Large")
            
            self.assertTrue(success)
            # Should log the resize operation
            mock_log.assert_called()
            # Check that resize happened
            message = self.sent_messages[0]
            self.assertLessEqual(message["data"]["width"], 240)
            self.assertLessEqual(message["data"]["height"], 160)
    
    def test_send_image_tab_name_truncation(self):
        """Test tab name truncation for long names"""
        test_image = np.ones((5, 5, 3), dtype=np.uint8)
        long_name = "This is a very long tab name that should be truncated"
        
        success = self.controller.send_image(test_image, long_name)
        
        self.assertTrue(success)
        message = self.sent_messages[0]
        self.assertLessEqual(len(message["data"]["tab_name"]), 15)
    
    @patch('os.path.exists')
    def test_send_image_file_path(self, mock_exists):
        """Test sending image from file path"""
        mock_exists.return_value = True
        
        # Mock PIL Image inside the method where it's imported
        with patch('PIL.Image') as mock_pil_image:
            mock_img = Mock()
            mock_img.convert.return_value = mock_img
            mock_pil_image.open.return_value = mock_img
            
            # Mock numpy.array to return test data
            with patch('numpy.array', return_value=np.ones((10, 10, 3), dtype=np.uint8)):
                success = self.controller.send_image("/fake/path/image.jpg", "File Test")
                
                # Should attempt to open file
                mock_pil_image.open.assert_called_with("/fake/path/image.jpg")
                self.assertTrue(success)
    
    def test_send_image_invalid_input(self):
        """Test error handling for invalid image inputs"""
        # Test with invalid type
        success = self.controller.send_image("not_a_real_file.jpg", "Invalid")
        self.assertFalse(success)
        
        # Test with wrong shape
        invalid_image = np.ones((10, 10, 5), dtype=np.uint8)  # 5 channels
        success = self.controller.send_image(invalid_image, "Invalid Shape")
        self.assertFalse(success)
    
    def test_rgb565_conversion(self):
        """Test RGB888 to RGB565 conversion"""
        # Test with known values
        test_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        
        rgb565_data = self.controller._convert_to_rgb565(test_image)
        
        # Should get 6 bytes (3 pixels * 2 bytes per pixel)
        self.assertEqual(len(rgb565_data), 6)
        
        # Test red pixel (should be 0xF800 in RGB565)
        red_pixel = (rgb565_data[1] << 8) | rgb565_data[0]  # Little endian
        self.assertEqual(red_pixel & 0xF800, 0xF800)  # Red bits set
        
        # Test green pixel (should be 0x07E0 in RGB565)
        green_pixel = (rgb565_data[3] << 8) | rgb565_data[2]  # Little endian
        self.assertEqual(green_pixel & 0x07E0, 0x07E0)  # Green bits set
        
        # Test blue pixel (should be 0x001F in RGB565)
        blue_pixel = (rgb565_data[5] << 8) | rgb565_data[4]  # Little endian
        self.assertEqual(blue_pixel & 0x001F, 0x001F)  # Blue bits set
    
    def test_send_image_on_snap_callback(self):
        """Test automatic image sending on snap button press"""
        # Create mock image source
        test_image = np.ones((20, 20, 3), dtype=np.uint8) * 100
        image_source = Mock(return_value=test_image)
        
        # Setup the callback
        self.controller.send_image_on_snap(image_source)
        
        # Simulate snap button press
        self.controller._notify_callbacks('snap_image_command', {})
        
        # Should have called the image source and sent image
        image_source.assert_called_once()
        self.assertEqual(len(self.sent_messages), 1)
        
        message = self.sent_messages[0]
        self.assertEqual(message["type"], "display_image_command")
        self.assertIn("Snap", message["data"]["tab_name"])
    
    def test_send_image_on_snap_with_none_return(self):
        """Test snap callback when image source returns None"""
        image_source = Mock(return_value=None)
        
        with patch.object(self.controller.logger, 'warning') as mock_log:
            self.controller.send_image_on_snap(image_source)
            self.controller._notify_callbacks('snap_image_command', {})
            
            # Should log warning and not send message
            mock_log.assert_called()
            self.assertEqual(len(self.sent_messages), 0)
    
    def test_send_image_on_snap_with_exception(self):
        """Test snap callback error handling"""
        image_source = Mock(side_effect=Exception("Camera error"))
        
        with patch.object(self.controller.logger, 'error') as mock_log:
            self.controller.send_image_on_snap(image_source)
            self.controller._notify_callbacks('snap_image_command', {})
            
            # Should log error and not crash
            mock_log.assert_called()
            self.assertEqual(len(self.sent_messages), 0)
    
    def test_send_image_without_pil(self):
        """Test behavior when PIL is not available"""
        test_image = np.ones((5, 5, 3), dtype=np.uint8)
        
        # Mock the import inside the method to raise ImportError
        original_import = __builtins__.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'PIL.Image':
                raise ImportError("No module named 'PIL'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with patch.object(self.controller.logger, 'error') as mock_log:
                success = self.controller.send_image(test_image, "No PIL")
                
                self.assertFalse(success)
                mock_log.assert_called()

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)