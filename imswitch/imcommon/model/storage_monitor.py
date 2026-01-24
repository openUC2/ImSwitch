"""
Storage Monitor - Background monitoring for external USB drives.

This module provides functionality to monitor external storage drives for
mount/unmount events and notify registered callbacks.
"""

import threading
import time
from typing import List, Optional, Callable, Dict, Any
from imswitch.imcommon.model import initLogger
from imswitch.imcommon.model.storage_paths import scan_external_drives


class StorageMonitor:
    """
    Background monitor for external storage drives.
    
    Monitors configured mount paths for drive mount/unmount events
    and triggers callbacks when changes are detected.
    """

    def __init__(self, mount_paths: Optional[List[str]] = None, poll_interval: int = 5):
        """
        Initialize storage monitor.
        
        Args:
            mount_paths: List of paths to monitor (e.g., ['/media', '/Volumes'])
            poll_interval: Seconds between scans (default: 5)
        """
        self._logger = initLogger('StorageMonitor')
        self._mount_paths = mount_paths or []
        self._poll_interval = poll_interval
        self._running = False
        self._thread = None
        self._callbacks = []
        self._last_drives = {}  # {path: drive_info}

    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Add callback function to be called on storage changes.
        
        Args:
            callback: Function with signature (event_type: str, drive_info: Dict)
                     event_type is either "mounted" or "unmounted"
        """
        self._callbacks.append(callback)
        self._logger.debug(f"Added callback: {callback.__name__}")

    def remove_callback(self, callback: Callable):
        """Remove callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            self._logger.debug(f"Removed callback: {callback.__name__}")

    def start(self):
        """Start background monitoring thread."""
        if self._running:
            self._logger.warning("Storage monitor already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._logger.info(f"Storage monitor started (poll_interval={self._poll_interval}s, paths={self._mount_paths})")

    def stop(self):
        """Stop background monitoring thread."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=self._poll_interval + 1)
        self._logger.info("Storage monitor stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        # Initial scan to populate baseline
        try:
            initial_drives = scan_external_drives(self._mount_paths)
            self._last_drives = {drive['path']: drive for drive in initial_drives}
            self._logger.debug(f"Initial scan found {len(self._last_drives)} drives")
        except Exception as e:
            self._logger.error(f"Error in initial drive scan: {e}")

        while self._running: # @ethanjli I could not find any callback mechanism to notify the ImSwitch server about changes in the mounted drives. So I implemented a polling mechanism here. Not sure if this is the best way to do it in terms of resource usage.
            try:
                # Scan for current drives
                current_drives = scan_external_drives(self._mount_paths)
                current_paths = {drive['path']: drive for drive in current_drives}

                # Detect new drives (mounted)
                for path, drive_info in current_paths.items():
                    if path not in self._last_drives:
                        self._logger.info(f"Drive mounted: {path}")
                        self._notify_callbacks("mounted", drive_info)

                # Detect removed drives (unmounted)
                for path, drive_info in self._last_drives.items():
                    if path not in current_paths:
                        self._logger.info(f"Drive unmounted: {path}")
                        self._notify_callbacks("unmounted", drive_info)

                # Update baseline
                self._last_drives = current_paths

            except Exception as e:
                self._logger.error(f"Error in storage monitor loop: {e}")

            # Wait for next poll interval
            time.sleep(self._poll_interval)

    def _notify_callbacks(self, event_type: str, drive_info: Dict[str, Any]):
        """Notify all registered callbacks of storage change."""
        for callback in self._callbacks:
            try:
                callback(event_type, drive_info)
            except Exception as e:
                self._logger.error(f"Error in callback {callback.__name__}: {e}")


# Global monitor instance (singleton pattern)
_monitor_instance: Optional[StorageMonitor] = None
_monitor_lock = threading.Lock()


def get_storage_monitor() -> Optional[StorageMonitor]:
    """
    Get the global storage monitor instance.
    
    Returns:
        StorageMonitor instance or None if not started
    """
    return _monitor_instance


def start_storage_monitoring(mount_paths: Optional[List[str]] = None,
                             poll_interval: int = 5) -> StorageMonitor:
    """
    Start global storage monitoring.
    
    Args:
        mount_paths: List of paths to monitor (e.g., ['/media', '/Volumes'])
        poll_interval: Seconds between scans (default: 5)
        
    Returns:
        StorageMonitor instance
    """
    global _monitor_instance

    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = StorageMonitor(mount_paths=mount_paths, poll_interval=poll_interval)
            _monitor_instance.start()
        else:
            logger = initLogger('storage_monitor')
            logger.warning("Storage monitor already started, returning existing instance")

    return _monitor_instance


def stop_storage_monitoring():
    """Stop global storage monitoring."""
    global _monitor_instance

    with _monitor_lock:
        if _monitor_instance is not None:
            _monitor_instance.stop()
            _monitor_instance = None


# Copyright (C) 2020-2024 ImSwitch developers
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
