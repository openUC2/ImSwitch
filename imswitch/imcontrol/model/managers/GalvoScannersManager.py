"""
Manager for multiple galvo scanners.

This module provides the GalvoScannersManager which manages all galvo scanner
sub-managers in the system.
"""

from .MultiManager import MultiManager


class GalvoScannersManager(MultiManager):
    """
    GalvoScannersManager interface for dealing with GalvoScannerManagers.
    
    It is a MultiManager for galvo scanners, providing a unified interface
    to manage multiple galvo scanner devices.
    
    Usage in setup JSON:
    {
        "galvoScanners": {
            "ESP32Galvo": {
                "managerName": "ESP32GalvoScannerManager",
                "managerProperties": {
                    "rs232device": "ESP32",
                    "nx": 256,
                    "ny": 256
                }
            }
        },
        "availableWidgets": ["GalvoScanner", ...]
    }
    """

    def __init__(self, galvoScannerInfos, **lowLevelManagers):
        """
        Initialize the galvo scanners manager.
        
        Args:
            galvoScannerInfos: Dictionary of galvo scanner configurations
            **lowLevelManagers: Low-level managers (rs232sManager, etc.)
        """
        super().__init__(galvoScannerInfos, 'galvoscanners', **lowLevelManagers)

    def start_all_scans(self, **kwargs):
        """Start scanning on all galvo scanners."""
        return self.execOnAll(lambda m: m.start_scan(**kwargs))

    def stop_all_scans(self):
        """Stop scanning on all galvo scanners."""
        return self.execOnAll(lambda m: m.stop_scan())

    def get_all_status(self):
        """Get status of all galvo scanners."""
        return self.execOnAll(lambda m: m.get_status())


# Copyright (C) 2020-2025 ImSwitch developers
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
