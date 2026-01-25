"""
Session management for acquisition data.

Provides session directory layout, metadata storage, and multi-instance
access coordination via file locking.
"""

import json
import os
import time
import uuid
import fcntl
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """
    Session-level metadata for an acquisition.
    
    Contains all information needed to describe an acquisition session,
    from user-provided metadata to acquisition parameters.
    """
    # Required identifiers
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    base_path: str = ""
    
    # Timestamps
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # User-provided metadata
    project: Optional[str] = None
    experiment: Optional[str] = None
    sample: Optional[str] = None
    user: Optional[str] = None
    description: Optional[str] = None
    
    # Acquisition parameters
    n_time_points: int = 1
    n_z_planes: int = 1
    n_channels: int = 1
    time_interval_s: Optional[float] = None
    z_step_um: Optional[float] = None
    
    # Grid/mosaic parameters
    n_positions: int = 1
    grid_shape: Optional[tuple] = None  # (nx, ny)
    
    # Status
    status: str = "initialized"  # initialized, acquiring, completed, error
    frames_written: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'base_path': self.base_path,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'project': self.project,
            'experiment': self.experiment,
            'sample': self.sample,
            'user': self.user,
            'description': self.description,
            'n_time_points': self.n_time_points,
            'n_z_planes': self.n_z_planes,
            'n_channels': self.n_channels,
            'time_interval_s': self.time_interval_s,
            'z_step_um': self.z_step_um,
            'n_positions': self.n_positions,
            'grid_shape': self.grid_shape,
            'status': self.status,
            'frames_written': self.frames_written,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionInfo':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SessionFileLock:
    """
    File-based lock for session directory.
    
    Ensures single-writer, multiple-reader access to session data.
    """
    
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._lock_fd = None
    
    def acquire_writer(self) -> bool:
        """Acquire exclusive write lock."""
        try:
            self._lock_fd = open(self.lock_path, 'w')
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fd.write(f"writer_pid={os.getpid()}\ntime={time.time()}\n")
            self._lock_fd.flush()
            return True
        except (IOError, OSError) as e:
            logger.warning(f"Failed to acquire write lock: {e}")
            if self._lock_fd:
                self._lock_fd.close()
                self._lock_fd = None
            return False
    
    def release(self):
        """Release the lock."""
        if self._lock_fd:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
            except:
                pass
            self._lock_fd = None
    
    def is_locked(self) -> bool:
        """Check if session is locked by another writer."""
        if not os.path.exists(self.lock_path):
            return False
        
        try:
            with open(self.lock_path, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                return False
        except (IOError, OSError):
            return True
    
    def __enter__(self):
        self.acquire_writer()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class SessionManager:
    """
    Manages session directory layout and access.
    
    Session Directory Structure:
    ```
    session_YYYYMMDD_HHMMSS_<uuid>/
        session.json           # Session info + hub snapshot
        .session.lock          # Write lock file
        detectors.json         # Detector contexts
        data.ome.zarr/         # OME-Zarr data (live NGFF store)
        *.ome.tiff             # OME-TIFF files (optional)
        frames/                # Individual frame metadata (optional)
    ```
    """
    
    SESSION_JSON = "session.json"
    DETECTORS_JSON = "detectors.json"
    LOCK_FILE = ".session.lock"
    ZARR_DIR = "data.ome.zarr"
    
    def __init__(self, base_dir: str):
        """
        Initialize session manager.
        
        Args:
            base_dir: Root directory for all sessions
        """
        self.base_dir = Path(base_dir)
        self._current_session: Optional[SessionInfo] = None
        self._current_lock: Optional[SessionFileLock] = None
    
    def create_session(self, session_info: SessionInfo, 
                      hub_snapshot: Optional[Dict[str, Any]] = None,
                      detector_contexts: Optional[Dict[str, Dict[str, Any]]] = None) -> Path:
        """
        Create a new session directory with initial metadata.
        
        Args:
            session_info: Session information
            hub_snapshot: Optional MetadataHub snapshot
            detector_contexts: Optional detector context dicts
            
        Returns:
            Path to session directory
        """
        # Generate session directory name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir_name = f"session_{timestamp}_{session_info.session_id[:8]}"
        session_dir = self.base_dir / session_dir_name
        
        # Create directory
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Update session info with path
        session_info.base_path = str(session_dir)
        session_info.status = "initialized"
        
        # Acquire write lock
        lock_path = session_dir / self.LOCK_FILE
        self._current_lock = SessionFileLock(str(lock_path))
        if not self._current_lock.acquire_writer():
            raise RuntimeError(f"Could not acquire write lock for session {session_info.session_id}")
        
        # Write session.json
        session_data = {
            'session_info': session_info.to_dict(),
            'hub_snapshot': hub_snapshot or {},
            'created_at': time.time(),
        }
        with open(session_dir / self.SESSION_JSON, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Write detectors.json
        if detector_contexts:
            with open(session_dir / self.DETECTORS_JSON, 'w') as f:
                json.dump(detector_contexts, f, indent=2)
        
        self._current_session = session_info
        logger.info(f"Created session directory: {session_dir}")
        
        return session_dir
    
    def finalize_session(self, session_info: Optional[SessionInfo] = None):
        """
        Finalize session and update metadata.
        
        Args:
            session_info: Updated session info (or use current)
        """
        info = session_info or self._current_session
        if not info or not info.base_path:
            return
        
        session_dir = Path(info.base_path)
        
        # Update session info
        info.end_time = time.time()
        info.status = "completed"
        
        # Read existing session.json and update
        session_json_path = session_dir / self.SESSION_JSON
        if session_json_path.exists():
            with open(session_json_path, 'r') as f:
                session_data = json.load(f)
            session_data['session_info'] = info.to_dict()
            session_data['finalized_at'] = time.time()
            with open(session_json_path, 'w') as f:
                json.dump(session_data, f, indent=2)
        
        # Release lock
        if self._current_lock:
            self._current_lock.release()
            self._current_lock = None
        
        self._current_session = None
        logger.info(f"Finalized session: {info.session_id}")
    
    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List available sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summaries
        """
        sessions = []
        
        if not self.base_dir.exists():
            return sessions
        
        # Find all session directories
        for session_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not session_dir.is_dir() or not session_dir.name.startswith("session_"):
                continue
            
            session_json_path = session_dir / self.SESSION_JSON
            if not session_json_path.exists():
                continue
            
            try:
                with open(session_json_path, 'r') as f:
                    data = json.load(f)
                
                info = data.get('session_info', {})
                lock_path = session_dir / self.LOCK_FILE
                lock = SessionFileLock(str(lock_path))
                
                sessions.append({
                    'session_id': info.get('session_id'),
                    'path': str(session_dir),
                    'status': info.get('status'),
                    'start_time': info.get('start_time'),
                    'end_time': info.get('end_time'),
                    'project': info.get('project'),
                    'frames_written': info.get('frames_written', 0),
                    'is_active': lock.is_locked(),
                })
                
                if len(sessions) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Error reading session {session_dir}: {e}")
                continue
        
        return sessions
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full session data by ID.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Full session data or None
        """
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_json_path = session_dir / self.SESSION_JSON
            if not session_json_path.exists():
                continue
            
            try:
                with open(session_json_path, 'r') as f:
                    data = json.load(f)
                
                if data.get('session_info', {}).get('session_id') == session_id:
                    return data
                    
            except Exception as e:
                continue
        
        return None
    
    def get_zarr_path(self, session_dir: Path) -> Path:
        """Get path to OME-Zarr store for a session."""
        return session_dir / self.ZARR_DIR


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
