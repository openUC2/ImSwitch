import os
import subprocess
import sys
from imswitch.config import get_config
import imswitch
def openFolderInOS(folderPath):
    """ Open a folder in the OS's default file browser. """
    try:
        if sys.platform == 'darwin':
            subprocess.check_call(['open', folderPath])
        elif sys.platform == 'linux':
            subprocess.check_call(['xdg-open', folderPath])
        elif sys.platform == 'win32':
            os.startfile(folderPath)
    except FileNotFoundError or subprocess.CalledProcessError as err:
        raise OSToolsError(err)


def restartSoftware(module='imswitch', forceConfigFile=False):
    """ Restarts the software. """
    config = get_config()
    
    if config.is_headless:
        # we read the args from the config and restart the software using the same arguments
        # we need to add the module name to the arguments

        '''
        in docker:
        params+=" --http-port ${HTTP_PORT:-8001}"
        params+=" --socket-port ${SOCKET_PORT:-8002}"
        params+=" --config-folder ${CONFIG_PATH:-None}"
        params+=" --config-file ${CONFIG_FILE:-None}"
        params+=" --ext-data-folder ${DATA_PATH:-None}"
        params+=" --scan-ext-drive-mount $(SCAN_EXT_DATA_FOLDER:-false)"
        python3 /tmp/ImSwitch/main.py $params
        '''
        headless = config.is_headless
        http_port = str(config.http_port)
        socket_port = str(config.socket_port)
        config_folder = str(config.config_folder) if config.config_folder else "None"
        config_file = str(config.default_config) if config.default_config else "None"
        is_ssl = config.ssl
        scan_ext_drive_mount = config.scan_ext_data_folder
        ext_drive_mount = config.ext_drive_mount
        
        # Erstellen der Argumentliste
        args = [
            sys.executable,
            os.path.abspath(sys.argv[0]),
            '--http-port', http_port,
            '--socket-port', socket_port,
        ]
        
        if config_folder != "None":
            args.extend(['--config-folder', config_folder])
            
        if forceConfigFile and config_file != "None":
            args.extend(['--config-file', config_file])
            
        if headless:
            args.append('--headless')
            
        if not is_ssl:
            args.append('--no-ssl')
            
        if scan_ext_drive_mount:
            args.append('--scan-ext-drive-mount')
            
        if ext_drive_mount:
            args.extend(['--ext-drive-mount', ext_drive_mount])
            
        if config.data_folder:
            args.extend(['--ext-data-folder', config.data_folder])
        
        # execute script with new arguments
        os.execv(sys.executable, args)
    else:
        os.execv(sys.executable, ['"' + sys.executable + '"', '-m', module])


class OSToolsError(Exception):
    pass


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
