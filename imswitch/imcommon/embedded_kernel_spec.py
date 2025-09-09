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
Custom kernel spec for connecting to embedded ImSwitch kernel.
"""

import os
import json
import tempfile
import sys
import shutil

# Import kernel_state functions directly to avoid dependency issues
def get_embedded_kernel_connection_file():
    """Find the most recently created kernel connection file."""
    try:
        import jupyter_core.paths
        runtime_dir = jupyter_core.paths.jupyter_runtime_dir()
    except ImportError:
        # Fallback locations for different platforms
        runtime_dirs = [
            os.path.expanduser('~/.local/share/jupyter/runtime'),
            os.path.expanduser('~/Library/Jupyter/runtime'),  # macOS
            os.path.expanduser('~/.jupyter/runtime'),
            '/tmp'
        ]
        runtime_dir = None
        for d in runtime_dirs:
            if os.path.exists(d):
                runtime_dir = d
                break
    
    if not runtime_dir or not os.path.exists(runtime_dir):
        return None
    
    import glob
    kernel_files = glob.glob(os.path.join(runtime_dir, 'kernel-*.json'))
    
    if not kernel_files:
        return None
    
    # Return the most recently modified one
    return max(kernel_files, key=os.path.getmtime)

def is_embedded_kernel_running():
    """Check if there's a recent kernel connection file (simple heuristic)."""
    connection_file = get_embedded_kernel_connection_file()
    if not connection_file:
        return False
    
    # Check if file is recent (within last 30 minutes)
    import time
    file_age = time.time() - os.path.getmtime(connection_file)
    return file_age < 1800  # 30 minutes

def create_imswitch_kernel_spec():
    """
    Create a custom kernel spec that connects to the running ImSwitch embedded kernel.
    
    Returns:
        str: Path to kernel spec directory or None if no embedded kernel
    """
    connection_file = get_embedded_kernel_connection_file()
    if not connection_file or not os.path.exists(connection_file):
        return None
    
    # Get the directory where ImSwitch is installed
    imswitch_dir = os.path.dirname(os.path.dirname(__file__))
    proxy_script = os.path.join(imswitch_dir, "..", "imswitch_kernel_proxy.py")
    
    # Make sure proxy script exists
    if not os.path.exists(proxy_script):
        print(f"Warning: Proxy script not found at {proxy_script}")
        return None
    
    # Create a temporary kernel spec directory
    kernel_spec_dir = tempfile.mkdtemp(prefix="imswitch_kernel_")
    
    # Create kernel.json with actual connection file path
    kernel_json = {
        "argv": [
            sys.executable,
            proxy_script,
            "--connection-file", connection_file
        ],
        "display_name": "ImSwitch (Live Connection)",
        "language": "python",
        "metadata": {
            "debugger": True,
            "description": "Python kernel connected to running ImSwitch instance with live access to managers"
        }
    }
    
    kernel_json_path = os.path.join(kernel_spec_dir, "kernel.json")
    with open(kernel_json_path, 'w') as f:
        json.dump(kernel_json, f, indent=2)
    
    print(f"Created kernel spec at {kernel_spec_dir}")
    print(f"Connection file: {connection_file}")
    
    return kernel_spec_dir

def install_imswitch_kernel_spec():
    """
    Install the ImSwitch kernel spec if an embedded kernel is running.
    
    Returns:
        bool: True if kernel spec was installed successfully
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        
        kernel_spec_dir = create_imswitch_kernel_spec()
        if not kernel_spec_dir:
            print("No embedded kernel found to create spec for")
            return False
        
        # Install the kernel spec
        ksm = KernelSpecManager()
        ksm.install_kernel_spec(kernel_spec_dir, "imswitch_embedded", user=True, replace=True)
        
        print(f"✅ Installed ImSwitch embedded kernel spec")
        print(f"The kernel 'ImSwitch (Live Connection)' should now appear in Jupyter")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install ImSwitch kernel spec: {e}")
        return False

def remove_imswitch_kernel_spec():
    """
    Remove the ImSwitch kernel spec.
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        
        ksm = KernelSpecManager()
        ksm.remove_kernel_spec("imswitch_embedded")
        print("✅ Removed ImSwitch embedded kernel spec")
        return True
        
    except Exception as e:
        print(f"❌ Failed to remove ImSwitch kernel spec: {e}")
        return False

def ensure_kernel_spec_available():
    """
    Ensure the ImSwitch kernel spec is available if an embedded kernel is running.
    This should be called when starting the notebook server.
    """
    connection_file = get_embedded_kernel_connection_file()
    if connection_file and os.path.exists(connection_file):
        return install_imswitch_kernel_spec()
    return False