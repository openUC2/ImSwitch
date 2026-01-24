"""
LepmonOS Configuration Management Module

This module handles configuration reading/writing for the Lepmon moth trap,
mirroring the json_read_write.py functionality from LepmonOS_update.

Configuration structure matches Lepmon_config.json:
- general: serielnumber, project_name, usb_drive, current_folder, language
- locality: country, province, Kreis
- capture_mode: minutes_to_sunset, minutes_to_sunrise, flash, interval, dusk_treshold
- image_quality: brightness settings, gamma correction
- GPS: latitude, longitude
- software: version, date
- powermode: supply type, heater
"""

import os
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


# Default configuration matching LepmonOS structure
DEFAULT_CONFIG = {
    "general": {
        "serielnumber": "SN000001",
        "project_name": "Lepmon#",
        "usb_drive": "/mnt/usb_drive",
        "current_folder": "",
        "current_log": "",
        "language": "de",
        "expected_images": 0,
        "Control_Night": False,
        "Control_End": False,
    },
    "locality": {
        "country": "Germany",
        "province": "TH",
        "Kreis": "JEN",
        "do_not_delete_path": "",
    },
    "capture_mode": {
        "minutes_to_sunset": 15,
        "minutes_to_sunrise": 60,
        "flash": 0.25,
        "interval": 2,  # minutes between captures
        "dusk_treshold": 90,  # lux threshold for darkness
        "error_code": 0,
        "initial_exposure": 150,  # ms
        "initial_gain": 7,
        "current_exposure": 150,
        "current_gain": 7,
        "Heizung": False,  # heater active
    },
    "image_quality": {
        "black_sanity_level": 0.025,
        "brightness_tolerance": 8,
        "brightness_reference": 170,
        "minimal_gain": 5,
        "maximal_gain": 15,
        "step_gain": 0.5,
        "maximal_exposure": 170,
        "minimal_exposure": 100,
        "step_exposure": 5,
        "focus_threshold": 225.0,
        "gamma_correction": True,
        "gamma_value": 1.5,
    },
    "GPS": {
        "latitude": 50.9271,
        "Pol": "N",
        "longitude": 11.5892,
        "Block": "E",
    },
    "software": {
        "version": "2.1.5",
        "date": "2025-01-16",
    },
    "powermode": {
        "supply": "Netz",  # "Netz" or "Solar"
        "Heizung": True,
    },
}


@dataclass
class CaptureSettings:
    """Settings for image capture workflow"""
    minutes_to_sunset: int = 15
    minutes_to_sunrise: int = 60
    flash_duration: float = 0.25  # seconds for LED dimming
    interval: int = 2  # minutes between captures
    dusk_threshold: int = 90  # lux
    initial_exposure: int = 150  # ms
    initial_gain: float = 7.0
    current_exposure: int = 150
    current_gain: float = 7.0
    heater_enabled: bool = False


@dataclass
class ImageQualitySettings:
    """Settings for image quality control"""
    black_sanity_level: float = 0.025
    brightness_tolerance: int = 8
    brightness_reference: int = 170
    minimal_gain: float = 5.0
    maximal_gain: float = 15.0
    step_gain: float = 0.5
    maximal_exposure: int = 170
    minimal_exposure: int = 100
    step_exposure: int = 5
    focus_threshold: float = 225.0
    gamma_correction: bool = True
    gamma_value: float = 1.5


@dataclass
class GPSCoordinates:
    """GPS coordinate storage"""
    latitude: float = 50.9271
    longitude: float = 11.5892
    pol: str = "N"  # N or S
    block: str = "E"  # E or W


class LepmonConfig:
    """
    Configuration manager for Lepmon system.
    
    Provides methods to read/write configuration values,
    mirroring get_value_from_section() and write_value_to_section()
    from LepmonOS json_read_write.py.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to JSON config file. If None, uses default config in memory.
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()
        
        # Dataclass instances for typed access
        self.capture = CaptureSettings()
        self.image_quality = ImageQualitySettings()
        self.gps = GPSCoordinates()
        
        self._sync_dataclasses()
    
    def _load_config(self):
        """Load configuration from file or use defaults"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config from {self.config_path}: {e}")
                self._config = DEFAULT_CONFIG.copy()
        else:
            self._config = DEFAULT_CONFIG.copy()
    
    def _save_config(self):
        """Save configuration to file if path is set"""
        if self.config_path:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self._config, f, indent=4)
            except IOError as e:
                print(f"Error saving config to {self.config_path}: {e}")
    
    def _sync_dataclasses(self):
        """Sync dataclass instances with config dict"""
        # Capture settings
        cm = self._config.get("capture_mode", {})
        self.capture.minutes_to_sunset = cm.get("minutes_to_sunset", 15)
        self.capture.minutes_to_sunrise = cm.get("minutes_to_sunrise", 60)
        self.capture.flash_duration = cm.get("flash", 0.25)
        self.capture.interval = cm.get("interval", 2)
        self.capture.dusk_threshold = cm.get("dusk_treshold", 90)
        self.capture.initial_exposure = cm.get("initial_exposure", 150)
        self.capture.initial_gain = cm.get("initial_gain", 7)
        self.capture.current_exposure = cm.get("current_exposure", 150)
        self.capture.current_gain = cm.get("current_gain", 7)
        self.capture.heater_enabled = cm.get("Heizung", False)
        
        # Image quality settings
        iq = self._config.get("image_quality", {})
        self.image_quality.black_sanity_level = iq.get("black_sanity_level", 0.025)
        self.image_quality.brightness_tolerance = iq.get("brightness_tolerance", 8)
        self.image_quality.brightness_reference = iq.get("brightness_reference", 170)
        self.image_quality.minimal_gain = iq.get("minimal_gain", 5)
        self.image_quality.maximal_gain = iq.get("maximal_gain", 15)
        self.image_quality.step_gain = iq.get("step_gain", 0.5)
        self.image_quality.maximal_exposure = iq.get("maximal_exposure", 170)
        self.image_quality.minimal_exposure = iq.get("minimal_exposure", 100)
        self.image_quality.step_exposure = iq.get("step_exposure", 5)
        self.image_quality.focus_threshold = iq.get("focus_threshold", 225.0)
        self.image_quality.gamma_correction = iq.get("gamma_correction", True)
        self.image_quality.gamma_value = iq.get("gamma_value", 1.5)
        
        # GPS coordinates
        gps = self._config.get("GPS", {})
        self.gps.latitude = gps.get("latitude", 50.9271)
        self.gps.longitude = gps.get("longitude", 11.5892)
        self.gps.pol = gps.get("Pol", "N")
        self.gps.block = gps.get("Block", "E")
    
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a value from the configuration.
        
        Equivalent to LepmonOS get_value_from_section().
        
        Args:
            section: Configuration section name (e.g., "capture_mode")
            key: Key within the section
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(section, {}).get(key, default)
    
    def set_value(self, section: str, key: str, value: Any, save: bool = True):
        """
        Set a value in the configuration.
        
        Equivalent to LepmonOS write_value_to_section().
        
        Args:
            section: Configuration section name
            key: Key within the section
            value: Value to set
            save: Whether to save to file immediately
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        
        # Sync dataclasses
        self._sync_dataclasses()
        
        if save:
            self._save_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section"""
        return self._config.get(section, {}).copy()
    
    def set_section(self, section: str, data: Dict[str, Any], save: bool = True):
        """Set an entire configuration section"""
        self._config[section] = data
        self._sync_dataclasses()
        if save:
            self._save_config()
    
    # Convenience properties for common values
    @property
    def serial_number(self) -> str:
        return self.get_value("general", "serielnumber", "SN000001")
    
    @property
    def project_name(self) -> str:
        return self.get_value("general", "project_name", "Lepmon#")
    
    @property
    def language(self) -> str:
        return self.get_value("general", "language", "de")
    
    @property
    def usb_drive_path(self) -> str:
        return self.get_value("general", "usb_drive", "/mnt/usb_drive")
    
    @property
    def current_folder(self) -> str:
        return self.get_value("general", "current_folder", "")
    
    @property
    def software_version(self) -> str:
        return self.get_value("software", "version", "2.1.5")
    
    @property
    def power_mode(self) -> str:
        return self.get_value("powermode", "supply", "Netz")
    
    @property
    def coordinates(self) -> tuple:
        """Returns (latitude, longitude)"""
        return (self.gps.latitude, self.gps.longitude)
    
    def get_lepmon_code(self) -> tuple:
        """
        Get Lepmon identification code components.
        
        Equivalent to LepmonOS get_Lepmon_code().
        
        Returns:
            Tuple of (project_name, province, kreis, sensor_id)
        """
        project_name = self.project_name
        province = self.get_value("locality", "province", "XX")
        kreis = self.get_value("locality", "Kreis", "XXX")
        sensor_id = self.serial_number
        return (project_name, province, kreis, sensor_id)
    
    def create_image_code(self) -> str:
        """
        Create image filename code.
        
        Equivalent to code generation in LepmonOS snap_image().
        
        Returns:
            Formatted image code string
        """
        project_name, province, kreis, sensor_id = self.get_lepmon_code()
        now = datetime.now()
        code = f"{project_name}{sensor_id}_{province}_{kreis}_{now.strftime('%Y-%m-%d')}_T_{now.strftime('%H%M')}"
        return code
    
    def create_folder_path(self) -> str:
        """
        Create folder path for current capture session.
        
        Returns:
            Full path for capture folder
        """
        usb_path = self.usb_drive_path
        code = self.create_image_code()
        return os.path.join(usb_path, code)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export full configuration as dictionary"""
        return self._config.copy()
    
    def to_frontend_config(self) -> Dict[str, Any]:
        """
        Export configuration in format expected by frontend.
        
        Returns:
            Dict with keys matching Redux state structure
        """
        return {
            "exposureTime": self.capture.current_exposure,
            "gain": self.capture.current_gain,
            "timelapsePeriod": self.capture.interval * 60,  # convert to seconds
            "storagePath": self.usb_drive_path,
            "experimentName": self.project_name,
            "isRecordVideo": False,
            "fileFormat": "JPG",
            "frameRate": 1,
            "delayTimeAfterRestart": self.capture.flash_duration,
            "duskThreshold": self.capture.dusk_threshold,
            "minutesToSunset": self.capture.minutes_to_sunset,
            "minutesToSunrise": self.capture.minutes_to_sunrise,
            "gammaCorrection": self.image_quality.gamma_correction,
            "gammaValue": self.image_quality.gamma_value,
            "latitude": self.gps.latitude,
            "longitude": self.gps.longitude,
            "serialNumber": self.serial_number,
            "softwareVersion": self.software_version,
            "powerMode": self.power_mode,
            "heaterEnabled": self.capture.heater_enabled,
        }
