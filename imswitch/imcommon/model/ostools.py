import os
import subprocess
import sys
from imswitch import IS_HEADLESS
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
    if IS_HEADLESS:
        # we read the args from __argparse__ and restart the software using the same arguments
        # we need to add the module name to the arguments
        from imswitch import __argparse__

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
        headless = str(imswitch.IS_HEADLESS)
        http_port = str(imswitch.__httpport__)
        socket_port = str(imswitch.__socketport__)
        config_folder = str(imswitch.DEFAULT_CONFIG_PATH)
        config_file = str(imswitch.DEFAULT_SETUP_FILE)
        is_ssl = str(imswitch.__ssl__)
        scan_ext_drive_mount = bool(imswitch.SCAN_EXT_DATA_FOLDER)
        ext_drive_mount = imswitch.EXT_DRIVE_MOUNT
        # Erstellen der Argumentliste
        args = [
            sys.executable,
            os.path.abspath(sys.argv[0]),
            '--http-port', http_port,
            '--socket-port', socket_port,
            '--config-folder', config_folder,
        ]
        if forceConfigFile:
            args.append('--config-file')
            args.append(config_file)
        if headless == 'True':
            args.append('--headless')
        if is_ssl == 'False':
            args.append('--no-ssl')
        if scan_ext_drive_mount:
            args.append('--scan-ext-drive-mount')
        if ext_drive_mount:
            args.append('--ext-drive-mount')
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
