# Install test mocks for missing dependencies when running tests
import sys
if 'pytest' in sys.modules or any('pytest' in arg for arg in sys.argv):
    try:
        from . import _test_mocks
    except ImportError:
        pass

# Import the new configuration system
from .config import get_config

# used to be, but actions will replace this with the current release TAG -> >2.1.0
__version__ = "2.1.56"
__httpport__ = 8001
__socketport__ = 8002
__ssl__ = True
__jupyter_port__ = 8888
jupyternotebookurl = ""
__argparse__ = None

'''
These are LEGACY flags for backward compatibility.
New code should use the configuration system: from imswitch.config import get_config
'''
IS_HEADLESS = True  # Default to GUI mode
DEFAULT_SETUP_FILE = None
DEFAULT_CONFIG_PATH = None
DEFAULT_DATA_PATH = None
SOCKET_STREAM = True           # Stream Images via socket ?
SCAN_EXT_DATA_FOLDER = False  # Scan external data folder for new data ?
EXT_DRIVE_MOUNT = None

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
