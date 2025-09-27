"""
Unit tests for FocusLockConfig configuration system.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from imswitch.imcontrol.model.managers.FocusLockConfig import FocusLockConfig, validate_focus_config


class TestFocusLockConfig(unittest.TestCase):
    """Test cases for FocusLockConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Mock dirtools.UserFileDirs.Root to use our test directory
        self.mock_root_patcher = patch('imswitch.imcontrol.model.managers.FocusLockConfig.dirtools.UserFileDirs.Root', self.test_dir)
        self.mock_root_patcher.start()
        
        # Mock logger
        self.mock_logger_patcher = patch('imswitch.imcontrol.model.managers.FocusLockConfig.initLogger')
        self.mock_logger = self.mock_logger_patcher.start()
        self.mock_logger.return_value = MagicMock()
        
        # Create config instance
        self.config = FocusLockConfig("test_profile")
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_root_patcher.stop()
        self.mock_logger_patcher.stop()
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        # Should have loaded defaults
        focuslock_config = self.config.get("focuslock")
        self.assertIsInstance(focuslock_config, dict)
        self.assertIn("enabled", focuslock_config)
        self.assertIn("settle_band_um", focuslock_config)
        self.assertIn("watchdog", focuslock_config)
        
        experiment_config = self.config.get("experiment")
        self.assertIsInstance(experiment_config, dict)
        self.assertIn("use_focus_lock_live", experiment_config)
        self.assertIn("channel_z_offsets", experiment_config)
    
    def test_config_persistence(self):
        """Test saving and loading configuration."""
        # Modify some values
        self.config.set("focuslock", "settle_band_um", 2.5)
        self.config.set("experiment", "use_focus_lock_live", False)
        self.config.save_config()
        
        # Create new config instance
        config2 = FocusLockConfig("test_profile")
        
        # Should have loaded modified values
        self.assertEqual(config2.get("focuslock", "settle_band_um"), 2.5)
        self.assertEqual(config2.get("experiment", "use_focus_lock_live"), False)
    
    def test_section_updates(self):
        """Test updating entire sections."""
        updates = {
            "settle_band_um": 3.0,
            "settle_timeout_ms": 2500
        }
        
        self.config.update_section("focuslock", updates)
        
        self.assertEqual(self.config.get("focuslock", "settle_band_um"), 3.0)
        self.assertEqual(self.config.get("focuslock", "settle_timeout_ms"), 2500)
    
    def test_convenience_methods(self):
        """Test convenience methods for common operations."""
        # Test focus map enabled check
        self.config.set("experiment", "apply_focus_map", True)
        self.config.set("focusmap", "use_focus_map", True)
        self.assertTrue(self.config.is_focus_map_enabled())
        
        self.config.set("experiment", "apply_focus_map", False)
        self.assertFalse(self.config.is_focus_map_enabled())
        
        # Test focus lock live enabled
        self.config.set("experiment", "use_focus_lock_live", True)
        self.assertTrue(self.config.is_focus_lock_live_enabled())
        
        # Test channel offsets
        offsets = self.config.get_channel_offsets()
        self.assertIsInstance(offsets, dict)
        
        self.config.set_channel_offset("test_channel", 1.5)
        self.assertEqual(self.config.get_channel_offsets()["test_channel"], 1.5)
        
        # Test Z move order
        z_order = self.config.get_z_move_order()
        self.assertIn(z_order, ["Z_first", "Z_last"])
    
    def test_focuslock_params_format(self):
        """Test getting parameters in FocusLockManager format."""
        params = self.config.get_focuslock_params()
        
        required_keys = ["lock_enabled", "z_ref_um", "settle_band_um", 
                        "settle_timeout_ms", "settle_window_ms", "watchdog"]
        
        for key in required_keys:
            self.assertIn(key, params)
        
        self.assertIsInstance(params["watchdog"], dict)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            "focuslock": {
                "settle_band_um": 1.0,
                "settle_timeout_ms": 1500,
                "watchdog": {"max_abs_error_um": 5.0}
            },
            "experiment": {
                "z_move_order": "Z_first",
                "channel_z_offsets": {"DAPI": 0.0, "FITC": 0.8}
            }
        }
        
        errors = validate_focus_config(valid_config)
        self.assertEqual(len(errors), 0)
        
        # Invalid config
        invalid_config = {
            "focuslock": {
                "settle_band_um": -1.0,  # Should be positive
                "settle_timeout_ms": "invalid",  # Should be int
            },
            "experiment": {
                "z_move_order": "invalid",  # Should be Z_first or Z_last
                "channel_z_offsets": "not_a_dict"  # Should be dict
            }
        }
        
        errors = validate_focus_config(invalid_config)
        self.assertGreater(len(errors), 0)
        self.assertIn("focuslock.settle_band_um", errors)
        self.assertIn("experiment.z_move_order", errors)
    
    def test_config_merging(self):
        """Test that partial configs are merged with defaults."""
        # Create partial config file
        partial_config = {
            "focuslock": {
                "settle_band_um": 2.0
                # Missing other focuslock settings
            }
            # Missing experiment section entirely
        }
        
        config_file = Path(self.test_dir) / "focus" / "focus_config_merge_test.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(partial_config, f)
        
        # Load config - should have merged with defaults
        config = FocusLockConfig("merge_test")
        
        # Should have our custom value
        self.assertEqual(config.get("focuslock", "settle_band_um"), 2.0)
        
        # Should have default values for missing keys
        self.assertIn("settle_timeout_ms", config.get("focuslock"))
        self.assertIn("experiment", config.to_dict())
    
    def test_example_config_creation(self):
        """Test creating example configuration file."""
        example_path = Path(self.test_dir) / "example_config.json"
        
        FocusLockConfig.create_example_config(example_path)
        
        self.assertTrue(example_path.exists())
        
        # Load and verify structure
        with open(example_path, 'r') as f:
            example_config = json.load(f)
        
        self.assertIn("focuslock", example_config)
        self.assertIn("focusmap", example_config)
        self.assertIn("experiment", example_config)
        self.assertIn("_comments", example_config)


if __name__ == '__main__':
    unittest.main()