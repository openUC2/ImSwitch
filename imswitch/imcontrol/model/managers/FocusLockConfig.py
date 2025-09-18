"""
Configuration support for focus lock and focus map functionality.
Provides default configurations and loading/saving utilities.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
from imswitch.imcommon.model import dirtools, initLogger


class FocusLockConfig:
    """Configuration management for focus lock and focus map settings."""
    
    DEFAULT_CONFIG = {
        "focuslock": {
            "enabled": True,
            "settle_band_um": 1.0,
            "settle_timeout_ms": 1500,
            "settle_window_ms": 200,
            "watchdog": {
                "max_abs_error_um": 5.0,
                "max_time_without_settle_ms": 5000,
                "action": "abort"
            }
        },
        "focusmap": {
            "method": "plane",
            "use_focus_map": True,
            "auto_save": True,
            "interpolation_cache_size": 100
        },
        "experiment": {
            "use_focus_lock_live": True,
            "apply_focus_map": True,
            "z_move_order": "Z_first",  # "Z_first" or "Z_last"
            "channel_z_offsets": {
                "DAPI": 0.0,
                "FITC": 0.8,
                "TRITC": 1.2,
                "Cy5": 1.5
            }
        }
    }
    
    def __init__(self, profile_name: str = "default"):
        """
        Initialize focus lock configuration.
        
        Args:
            profile_name: Configuration profile name
        """
        self._logger = initLogger(self)
        self._profile_name = profile_name
        
        # Setup config paths
        self._config_dir = Path(dirtools.UserFileDirs.Root) / "focus"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config_file = self._config_dir / f"focus_config_{profile_name}.json"
        
        # Load configuration
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        if not self._config_file.exists():
            self._logger.info(f"No focus config found, using defaults")
            return self.DEFAULT_CONFIG.copy()
        
        try:
            with open(self._config_file, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            config = self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
            self._logger.info(f"Loaded focus config from {self._config_file}")
            return config
            
        except Exception as e:
            self._logger.error(f"Failed to load focus config: {e}, using defaults")
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with defaults."""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self._config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            self._logger.info(f"Saved focus config to {self._config_file}")
        except Exception as e:
            self._logger.error(f"Failed to save focus config: {e}")
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section name
            key: Optional key within section
            
        Returns:
            Configuration value or section
        """
        if section not in self._config:
            self._logger.warning(f"Config section '{section}' not found")
            return {}
        
        if key is None:
            return self._config[section].copy()
        
        return self._config[section].get(key)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section: Configuration section name
            key: Key within section
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section][key] = value
        self._logger.debug(f"Set config {section}.{key} = {value}")
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Update multiple values in a configuration section.
        
        Args:
            section: Configuration section name
            updates: Dictionary of key-value pairs to update
        """
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section].update(updates)
        self._logger.debug(f"Updated config section {section}: {updates}")
    
    def get_focuslock_params(self) -> Dict[str, Any]:
        """Get focus lock parameters in FocusLockManager format."""
        focuslock_config = self.get("focuslock")
        
        return {
            "lock_enabled": focuslock_config.get("enabled", True),
            "z_ref_um": 0.0,  # Set dynamically
            "settle_band_um": focuslock_config.get("settle_band_um", 1.0),
            "settle_timeout_ms": focuslock_config.get("settle_timeout_ms", 1500),
            "settle_window_ms": focuslock_config.get("settle_window_ms", 200),
            "watchdog": focuslock_config.get("watchdog", {})
        }
    
    def get_experiment_params(self) -> Dict[str, Any]:
        """Get experiment parameters."""
        return self.get("experiment")
    
    def get_focusmap_params(self) -> Dict[str, Any]:
        """Get focus map parameters."""
        return self.get("focusmap")
    
    def is_focus_map_enabled(self) -> bool:
        """Check if focus map is enabled."""
        return self.get("experiment", "apply_focus_map") and self.get("focusmap", "use_focus_map")
    
    def is_focus_lock_live_enabled(self) -> bool:
        """Check if live focus lock is enabled."""
        return self.get("experiment", "use_focus_lock_live")
    
    def get_channel_offsets(self) -> Dict[str, float]:
        """Get channel Z offsets."""
        return self.get("experiment", "channel_z_offsets") or {}
    
    def set_channel_offset(self, channel: str, offset_um: float) -> None:
        """Set Z offset for a specific channel."""
        offsets = self.get_channel_offsets()
        offsets[channel] = float(offset_um)
        self.set("experiment", "channel_z_offsets", offsets)
    
    def get_z_move_order(self) -> str:
        """Get Z move order preference."""
        return self.get("experiment", "z_move_order") or "Z_first"
    
    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return self._config.copy()
    
    @classmethod
    def create_example_config(cls, config_path: Path) -> None:
        """Create an example configuration file."""
        config = cls.DEFAULT_CONFIG.copy()
        
        # Add some comments as a special key
        config["_comments"] = {
            "focuslock.settle_band_um": "Focus error tolerance in micrometers",
            "focuslock.settle_timeout_ms": "Maximum time to wait for focus settle",
            "focuslock.watchdog.max_abs_error_um": "Focus error threshold for experiment abort",
            "experiment.use_focus_lock_live": "Enable continuous focus lock during experiments",
            "experiment.apply_focus_map": "Apply Z corrections from focus map",
            "experiment.z_move_order": "Z_first: move Z before XY, Z_last: move Z after XY",
            "experiment.channel_z_offsets": "Per-channel Z offsets in micrometers"
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Failed to create example config: {e}")


# Configuration validation utilities
def validate_focus_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate focus configuration and return errors.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Dictionary of validation errors (empty if valid)
    """
    errors = {}
    
    # Validate focuslock section
    if "focuslock" in config:
        fl_config = config["focuslock"]
        
        if "settle_band_um" in fl_config:
            if not isinstance(fl_config["settle_band_um"], (int, float)) or fl_config["settle_band_um"] <= 0:
                errors["focuslock.settle_band_um"] = "Must be a positive number"
        
        if "settle_timeout_ms" in fl_config:
            if not isinstance(fl_config["settle_timeout_ms"], int) or fl_config["settle_timeout_ms"] <= 0:
                errors["focuslock.settle_timeout_ms"] = "Must be a positive integer"
        
        if "watchdog" in fl_config:
            watchdog = fl_config["watchdog"]
            if "max_abs_error_um" in watchdog:
                if not isinstance(watchdog["max_abs_error_um"], (int, float)) or watchdog["max_abs_error_um"] <= 0:
                    errors["focuslock.watchdog.max_abs_error_um"] = "Must be a positive number"
    
    # Validate experiment section
    if "experiment" in config:
        exp_config = config["experiment"]
        
        if "z_move_order" in exp_config:
            if exp_config["z_move_order"] not in ["Z_first", "Z_last"]:
                errors["experiment.z_move_order"] = "Must be 'Z_first' or 'Z_last'"
        
        if "channel_z_offsets" in exp_config:
            offsets = exp_config["channel_z_offsets"]
            if not isinstance(offsets, dict):
                errors["experiment.channel_z_offsets"] = "Must be a dictionary"
            else:
                for channel, offset in offsets.items():
                    if not isinstance(offset, (int, float)):
                        errors[f"experiment.channel_z_offsets.{channel}"] = "Must be a number"
    
    return errors