import logging
import os
import sys
import traceback
import time
import threading


# somewhere after all your variables & functions exist
try:
    import IPython
    #IPython.embed_kernel(local_ns={**locals(), **globals()})
    #from ipykernel.embed import embed_kernel   # pip install ipykernel
except ImportError:
    embed_kernel = None

from .model import dirtools, pythontools, initLogger
from imswitch.imcommon.framework import Signal, Thread
from imswitch import IS_HEADLESS
from imswitch.config import get_config
if not IS_HEADLESS:
    from qtpy import QtCore, QtGui, QtWidgets
    from .view.guitools import getBaseStyleSheet
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
else:
    from psygnal import emit_queued


def prepareApp():
    """ This function must be called before any views are created. """
    if IS_HEADLESS:
        """We won't have any GUI, so we don't need to prepare the app."""
        return None
    # Initialize exception handling
    pythontools.installExceptHook()

    # Set logging levels
    logging.getLogger('pyvisa').setLevel(logging.WARNING)
    logging.getLogger('lantz').setLevel(logging.WARNING)

    # Create app
    os.environ['IMSWITCH_FULL_APP'] = '1'  # Indicator that non-plugin version of ImSwitch is used
    os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'  # Force Qt to use PyQt5
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # Force HDF5 to not lock files
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # TODO: Some weird combination of the below settings may help us to scale Napari?
    # Set environment variables for high DPI scaling
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    #os.environ["QT_SCALE_FACTOR"] = ".8"  # Adjust this value as needed
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    # Set application attributes
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)  # Fixes Napari issues
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True) # proper scaling on Mac?
    #QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # https://stackoverflow.com/questions/72131093/pyqt5-qwebengineview-doesnt-load-url
    # The following element (sandbox) is to keep the app from crashing when using QWebEngineView
    app = QtWidgets.QApplication(['', '--no-sandbox'])
    app.setStyleSheet("QWidget { font-size: 9pt; }")  # Smaller default font
    app.setWindowIcon(QtGui.QIcon(os.path.join(dirtools.DataFileDirs.Root, 'icon.png')))
    app.setStyleSheet(getBaseStyleSheet())
    return app


def start_jupyter_kernel_in_main_thread(moduleMainControllers):
    """Start Jupyter kernel with access to all managers in main thread."""
    try:
        logger = initLogger('JupyterKernel')
        logger.info('Starting Jupyter kernel in main thread...')
        
        # Create namespace with access to all managers
        kernel_ns = {
            'moduleMainControllers': moduleMainControllers,
        }
        
        # Add direct access to imcontrol master controller if available
        if hasattr(moduleMainControllers, 'mapping') and 'imcontrol' in moduleMainControllers.mapping:
            master_controller = moduleMainControllers.mapping['imcontrol']._ImConMainController__masterController
            kernel_ns['master'] = master_controller
            
            # Add direct access to managers
            if hasattr(master_controller, 'lasersManager'):
                kernel_ns['lasersManager'] = master_controller.lasersManager
            if hasattr(master_controller, 'detectorsManager'):
                kernel_ns['detectorsManager'] = master_controller.detectorsManager
            if hasattr(master_controller, 'positionersManager'):
                kernel_ns['positionersManager'] = master_controller.positionersManager
            if hasattr(master_controller, 'recordingManager'):
                kernel_ns['recordingManager'] = master_controller.recordingManager
                
            logger.info('Managers added to kernel namespace')
        
        # Import commonly used packages in kernel namespace
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        
        kernel_ns.update({
            'np': np,
            'plt': plt,
            'time': time
        })
        
        # Start the kernel in main thread (will block)
        IPython.embed_kernel(local_ns=kernel_ns)
        
    except Exception as e:
        logger.error(f'Failed to start Jupyter kernel: {e}')


def start_jupyter_kernel_timer(moduleMainControllers):
    """Start Jupyter kernel with a small delay to let Qt initialize."""
    def start_delayed():
        time.sleep(2)  # Give Qt time to initialize
        try:
            logger = initLogger('JupyterKernel')
            logger.info('Starting Jupyter kernel after Qt initialization...')
            
            # Create namespace with access to all managers
            kernel_ns = {
                'moduleMainControllers': moduleMainControllers,
            }
            
            # Add direct access to imcontrol master controller if available
            if hasattr(moduleMainControllers, 'mapping') and 'imcontrol' in moduleMainControllers.mapping:
                master_controller = moduleMainControllers.mapping['imcontrol']._ImConMainController__masterController
                kernel_ns['master'] = master_controller
                
                # Add direct access to managers
                if hasattr(master_controller, 'lasersManager'):
                    kernel_ns['lasersManager'] = master_controller.lasersManager
                if hasattr(master_controller, 'detectorsManager'):
                    kernel_ns['detectorsManager'] = master_controller.detectorsManager
                if hasattr(master_controller, 'positionersManager'):
                    kernel_ns['positionersManager'] = master_controller.positionersManager
                if hasattr(master_controller, 'recordingManager'):
                    kernel_ns['recordingManager'] = master_controller.recordingManager
                    
                logger.info('Managers added to kernel namespace')
            
            # Import commonly used packages in kernel namespace
            import numpy as np
            import matplotlib.pyplot as plt
            import time
            
            kernel_ns.update({
                'np': np,
                'plt': plt,
                'time': time
            })
            
            # Start the kernel - this will block but in a separate thread
            IPython.embed_kernel(local_ns=kernel_ns)
            
        except Exception as e:
            logger.error(f'Failed to start Jupyter kernel: {e}')
    
    # Start in daemon thread for GUI mode
    threading.Thread(target=start_delayed, daemon=True).start()


def launchApp(app, mainView, moduleMainControllers):
    """ Launches the app. The program will exit when the app is exited. """
    logger = initLogger('launchApp')
    
    # Get kernel setting from configuration
    config = get_config()
    kernel_enabled = config.with_kernel

    if IS_HEADLESS:
        """We won't have any GUI, so we don't need to prepare the app."""
        # Keep python running
        if kernel_enabled:
            # Start kernel in main thread (this will block, but that's what we want in headless mode)
            start_jupyter_kernel_in_main_thread(moduleMainControllers)
        else:
            tDiskCheck = time.time()
            while True: # TODO: have webserver signal somehow?
                try:
                    emit_queued()
                    time.sleep(1)
                    if time.time() - tDiskCheck > 60 and dirtools.getDiskusage() > 0.9:
                        # if the storage is full or the user presses Ctrl+C, we want to stop the experiment
                        moduleMainControllers.mapping["imcontrol"]._ImConMainController__commChannel.sigExperimentStop.emit()
                        tDiskCheck = time.time()

                except KeyboardInterrupt:
                    exitCode = 0
                    break

    else:
        # GUI mode
        if kernel_enabled:
            # Start kernel after Qt initializes properly
            start_jupyter_kernel_timer(moduleMainControllers)
            logger.info('Jupyter kernel will start after Qt initialization')
        
        # Show app
        if mainView is not None:
            mainView.showMaximized()
            mainView.show()
        exitCode = app.exec_()

    # Clean up
    for controller in moduleMainControllers:
        try:
            controller.closeEvent()
        except Exception:
            logger.error(f'Error closing {type(controller).__name__}')
            logger.error(traceback.format_exc())

    # Exit
    sys.exit(exitCode)


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
