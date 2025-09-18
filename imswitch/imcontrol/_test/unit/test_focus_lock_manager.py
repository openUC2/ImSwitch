"""
Unit tests for FocusLockManager focus map functionality.
"""

import unittest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from imswitch.imcontrol.model.managers.FocusLockManager import FocusLockManager


class TestFocusLockManager(unittest.TestCase):
    """Test cases for FocusLockManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Mock dirtools.UserFileDirs.Root to use our test directory
        self.mock_root_patcher = patch('imswitch.imcontrol.model.managers.FocusLockManager.dirtools.UserFileDirs.Root', self.test_dir)
        self.mock_root_patcher.start()
        
        # Mock logger
        self.mock_logger_patcher = patch('imswitch.imcontrol.model.managers.FocusLockManager.initLogger')
        self.mock_logger = self.mock_logger_patcher.start()
        self.mock_logger.return_value = MagicMock()
        
        # Create manager instance
        self.manager = FocusLockManager("test_profile")
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_root_patcher.stop()
        self.mock_logger_patcher.stop()
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager._profile_name, "test_profile")
        self.assertIsNone(self.manager._focus_map)
        self.assertIsInstance(self.manager._focus_params, dict)
    
    def test_add_point(self):
        """Test adding focus points."""
        # Add first point
        self.manager.add_point(0.0, 0.0, 10.0)
        
        self.assertIsNotNone(self.manager._focus_map)
        self.assertEqual(len(self.manager._focus_map["points"]), 1)
        
        point = self.manager._focus_map["points"][0]
        self.assertEqual(point["x_um"], 0.0)
        self.assertEqual(point["y_um"], 0.0)
        self.assertEqual(point["z_um"], 10.0)
        
        # Add more points
        self.manager.add_point(100.0, 0.0, 10.5)
        self.manager.add_point(0.0, 100.0, 9.5)
        
        self.assertEqual(len(self.manager._focus_map["points"]), 3)
    
    def test_plane_fitting(self):
        """Test plane fitting algorithm."""
        # Add three non-collinear points
        self.manager.add_point(0.0, 0.0, 10.0)
        self.manager.add_point(100.0, 0.0, 10.5)
        self.manager.add_point(0.0, 100.0, 9.5)
        
        # Fit plane
        result = self.manager.fit("plane")
        
        self.assertIn("plane", result)
        coeffs = result["plane"]
        self.assertIn("a", coeffs)
        self.assertIn("b", coeffs)
        self.assertIn("c", coeffs)
        
        # Check coefficients are reasonable
        a, b, c = coeffs["a"], coeffs["b"], coeffs["c"]
        
        # Verify plane equation for input points
        z1 = a * 0.0 + b * 0.0 + c
        z2 = a * 100.0 + b * 0.0 + c
        z3 = a * 0.0 + b * 100.0 + c
        
        # Should be close to original Z values
        self.assertAlmostEqual(z1, 10.0, places=6)
        self.assertAlmostEqual(z2, 10.5, places=6)
        self.assertAlmostEqual(z3, 9.5, places=6)
    
    def test_plane_fitting_collinear_points(self):
        """Test plane fitting with collinear points (should fail)."""
        # Add three collinear points
        self.manager.add_point(0.0, 0.0, 10.0)
        self.manager.add_point(10.0, 10.0, 10.0)
        self.manager.add_point(20.0, 20.0, 10.0)
        
        # Should raise error for collinear points
        with self.assertRaises(ValueError):
            self.manager.fit("plane")
    
    def test_insufficient_points(self):
        """Test fitting with insufficient points."""
        # Add only two points
        self.manager.add_point(0.0, 0.0, 10.0)
        self.manager.add_point(100.0, 0.0, 10.5)
        
        # Should raise error for insufficient points
        with self.assertRaises(ValueError):
            self.manager.fit("plane")
    
    def test_get_z_offset(self):
        """Test Z offset calculation."""
        # Add points and fit plane
        self.manager.add_point(0.0, 0.0, 10.0)
        self.manager.add_point(100.0, 0.0, 10.5)
        self.manager.add_point(0.0, 100.0, 9.5)
        self.manager.fit("plane")
        
        # Test offset calculation
        offset1 = self.manager.get_z_offset(0.0, 0.0)
        self.assertAlmostEqual(offset1, 10.0, places=6)
        
        offset2 = self.manager.get_z_offset(100.0, 0.0)
        self.assertAlmostEqual(offset2, 10.5, places=6)
        
        offset3 = self.manager.get_z_offset(0.0, 100.0)
        self.assertAlmostEqual(offset3, 9.5, places=6)
        
        # Test interpolation at intermediate point
        offset_mid = self.manager.get_z_offset(50.0, 50.0)
        self.assertIsInstance(offset_mid, float)
    
    def test_no_map_z_offset(self):
        """Test Z offset with no map returns zero."""
        offset = self.manager.get_z_offset(10.0, 20.0)
        self.assertEqual(offset, 0.0)
    
    def test_channel_offsets(self):
        """Test channel-specific Z offsets."""
        # Set channel offsets
        self.manager.set_channel_offset("DAPI", 0.0)
        self.manager.set_channel_offset("FITC", 0.8)
        self.manager.set_channel_offset("TRITC", 1.2)
        
        # Test retrieval
        self.assertEqual(self.manager.get_channel_offset("DAPI"), 0.0)
        self.assertEqual(self.manager.get_channel_offset("FITC"), 0.8)
        self.assertEqual(self.manager.get_channel_offset("TRITC"), 1.2)
        self.assertEqual(self.manager.get_channel_offset("unknown"), 0.0)
    
    def test_map_persistence(self):
        """Test saving and loading focus maps."""
        # Add points and fit
        self.manager.add_point(0.0, 0.0, 10.0)
        self.manager.add_point(100.0, 0.0, 10.5)
        self.manager.add_point(0.0, 100.0, 9.5)
        self.manager.fit("plane")
        
        # Save map
        self.manager.save_map()
        
        # Create new manager and load
        new_manager = FocusLockManager("test_profile")
        
        # Should have loaded the map
        self.assertIsNotNone(new_manager._focus_map)
        self.assertEqual(len(new_manager._focus_map["points"]), 3)
        self.assertIn("fit", new_manager._focus_map)
        
        # Test offset calculation works
        offset = new_manager.get_z_offset(0.0, 0.0)
        self.assertAlmostEqual(offset, 10.0, places=6)
    
    def test_clear_map(self):
        """Test clearing focus map."""
        # Add points
        self.manager.add_point(0.0, 0.0, 10.0)
        self.manager.add_point(100.0, 0.0, 10.5)
        
        self.assertIsNotNone(self.manager._focus_map)
        
        # Clear map
        self.manager.clear_map()
        
        self.assertIsNone(self.manager._focus_map)
        self.assertFalse(self.manager.is_map_active())
    
    def test_map_stats(self):
        """Test focus map statistics."""
        # No map initially
        stats = self.manager.get_map_stats()
        self.assertFalse(stats["active"])
        self.assertEqual(stats["point_count"], 0)
        
        # Add points and fit
        self.manager.add_point(0.0, 0.0, 10.0)
        self.manager.add_point(100.0, 0.0, 12.0)
        self.manager.add_point(0.0, 100.0, 8.0)
        self.manager.fit("plane")
        
        stats = self.manager.get_map_stats()
        self.assertTrue(stats["active"])
        self.assertEqual(stats["point_count"], 3)
        self.assertEqual(stats["method"], "plane")
        self.assertEqual(stats["z_range_um"], 4.0)  # 12.0 - 8.0
        self.assertIn("created_at", stats)
        self.assertIn("updated_at", stats)
    
    def test_parameter_management(self):
        """Test focus parameter get/set."""
        # Get default params
        params = self.manager.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn("settle_band_um", params)
        
        # Set new params
        new_params = {
            "settle_band_um": 2.0,
            "settle_timeout_ms": 2000
        }
        self.manager.set_params(new_params)
        
        # Verify updates
        updated_params = self.manager.get_params()
        self.assertEqual(updated_params["settle_band_um"], 2.0)
        self.assertEqual(updated_params["settle_timeout_ms"], 2000)


if __name__ == '__main__':
    unittest.main()