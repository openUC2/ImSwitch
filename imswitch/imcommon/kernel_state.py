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

"""
Shared state for embedded Jupyter kernel information.
This module provides a way to share kernel connection details between 
the embedded kernel and notebook server.
"""

import os
import glob
import threading

# Global state for embedded kernel
_kernel_state = {
    'connection_file': None,
    'kernel_id': None,
    'is_running': False,
    'lock': threading.Lock()
}

def get_embedded_kernel_connection_file():
    """
    Get the connection file path for the embedded kernel.
    
    Returns:
        str or None: Path to the kernel connection file if available
    """
    with _kernel_state['lock']:
        if _kernel_state['connection_file'] and os.path.exists(_kernel_state['connection_file']):
            return _kernel_state['connection_file']
    
    # Fallback: search for most recent kernel file
    return find_latest_kernel_connection_file()

def set_embedded_kernel_connection_file(connection_file):
    """
    Set the connection file path for the embedded kernel.
    
    Args:
        connection_file (str): Path to the kernel connection file
    """
    with _kernel_state['lock']:
        _kernel_state['connection_file'] = connection_file
        _kernel_state['is_running'] = True

def get_embedded_kernel_id():
    """
    Get the kernel ID for the embedded kernel.
    
    Returns:
        str or None: Kernel ID if available
    """
    with _kernel_state['lock']:
        return _kernel_state['kernel_id']

def set_embedded_kernel_id(kernel_id):
    """
    Set the kernel ID for the embedded kernel.
    
    Args:
        kernel_id (str): Kernel ID
    """
    with _kernel_state['lock']:
        _kernel_state['kernel_id'] = kernel_id

def is_embedded_kernel_running():
    """
    Check if the embedded kernel is running.
    
    Returns:
        bool: True if kernel is running
    """
    with _kernel_state['lock']:
        return _kernel_state['is_running']

def set_embedded_kernel_running(is_running):
    """
    Set the running state of the embedded kernel.
    
    Args:
        is_running (bool): Whether the kernel is running
    """
    with _kernel_state['lock']:
        _kernel_state['is_running'] = is_running

def find_latest_kernel_connection_file():
    """
    Find the most recently created kernel connection file.
    
    Returns:
        str or None: Path to the most recent kernel connection file
    """
    try:
        import jupyter_core.paths
        runtime_dir = jupyter_core.paths.jupyter_runtime_dir()
    except ImportError:
        # Fallback locations
        runtime_dirs = [
            os.path.expanduser('~/.local/share/jupyter/runtime'),
            os.path.expanduser('~/.jupyter'),
            '/tmp'
        ]
        runtime_dir = None
        for d in runtime_dirs:
            if os.path.exists(d):
                runtime_dir = d
                break
    
    if not runtime_dir or not os.path.exists(runtime_dir):
        return None
    
    # Find all kernel files
    kernel_files = glob.glob(os.path.join(runtime_dir, 'kernel-*.json'))
    
    if not kernel_files:
        return None
    
    # Return the most recently modified one
    return max(kernel_files, key=os.path.getmtime)