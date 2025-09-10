import logging
import os
import sys
import traceback
import time
import threading
import IPython

try:
    from ipykernel.embed import embed_kernel
except ImportError:
    embed_kernel = None

from .model import dirtools, pythontools, initLogger
from imswitch.imcommon.framework import Signal, Thread
from imswitch import IS_HEADLESS
from imswitch.config import get_config
from .kernel_state import set_embedded_kernel_connection_file, find_latest_kernel_connection_file
if not IS_HEADLESS:
    from qtpy import QtCore, QtGui, QtWidgets
    from .view.guitools import getBaseStyleSheet
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
else:
    from psygnal import emit_queued


def start_embedded_kernel(namespace):
    """
    Start an embedded Jupyter kernel in the current thread with thread-safe signal handling.
    
    This function works around the ValueError: signal only works in main thread 
    by patching signal handling during kernel initialization.
    """
    if embed_kernel is not None:
        logger = initLogger('embedded_kernel')
        logger.info("Starting embedded Jupyter kernel...")
        logger.info("Connect with: jupyter console --existing or select kernel in JupyterLab")
        
        # Import signal module for patching
        import signal
        
        # Store original signal function
        original_signal = signal.signal
        
        def thread_safe_signal(signum, handler):
            """Thread-safe signal handler that only works in main thread"""
            if threading.current_thread() is threading.main_thread():
                return original_signal(signum, handler)
            else:
                # In non-main threads, just return the handler without registering
                logger.debug(f"Skipping signal {signum} registration in thread {threading.current_thread().name}")
                return handler
        
        try:
            # Patch signal.signal to be thread-safe
            signal.signal = thread_safe_signal
            
            # Now embed_kernel should work without signal errors
            embed_kernel(local_ns=namespace)
            
            # After kernel starts, try to find and store the connection file
            connection_file = find_latest_kernel_connection_file()
            if connection_file:
                set_embedded_kernel_connection_file(connection_file)
                logger.info(f"Embedded kernel connection file: {connection_file}")
            
        except Exception as e:
            logger.error(f"Failed to start embedded kernel: {e}")
            logger.error(traceback.format_exc())
            
            # Try fallback approach
            logger.info("Attempting fallback kernel startup method...")
            try:
                # Alternative: Start kernel without signal handling
                import os
                os.environ['JUPYTER_KERNEL_DISABLE_SIGNALS'] = '1'
                embed_kernel(local_ns=namespace)
                logger.info("Fallback kernel startup successful")
                
                # Try to find connection file for fallback too
                connection_file = find_latest_kernel_connection_file()
                if connection_file:
                    set_embedded_kernel_connection_file(connection_file)
                    logger.info(f"Embedded kernel connection file: {connection_file}")
                
            except Exception as fallback_error:
                logger.error(f"Fallback kernel startup also failed: {fallback_error}")
                
        finally:
            # Always restore original signal function
            signal.signal = original_signal
            
    else:
        logger = initLogger('embedded_kernel')
        logger.warning("ipykernel not available. Install with: pip install ipykernel")


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


def launchApp(app, mainView, moduleMainControllers):
    """ Launches the app. The program will exit when the app is exited. """
    logger = initLogger('launchApp')
    config = get_config()
    IPython.embed_kernel(local_ns={**locals(), **globals()})

    # Start embedded Jupyter kernel if requested
    if config.enable_kernel:
        # Prepare namespace for kernel
        kernel_namespace = globals().copy()
        kernel_namespace.update({
            "moduleMainControllers": moduleMainControllers,
            "app": app,
            "mainView": mainView,
            "config": config,
        })
        
        # Add individual managers for easy access
        if isinstance(moduleMainControllers, dict):
            for module_id, controller in moduleMainControllers.items():
                kernel_namespace[f"{module_id}_controller"] = controller
                # Try to extract managers from imcontrol module if available
                if module_id == 'imcontrol' and hasattr(controller, '_ImConMainController__masterController'):
                    master_controller = controller._ImConMainController__masterController
                    kernel_namespace["master_controller"] = master_controller
                    
                    # Add individual managers for convenience
                    for attr_name in dir(master_controller):
                        if attr_name.endswith('Manager') and not attr_name.startswith('_'):
                            manager = getattr(master_controller, attr_name, None)
                            if manager is not None:
                                kernel_namespace[attr_name] = manager
        elif hasattr(moduleMainControllers, '__iter__'):
            # Handle case where it's a list/values()
            for i, controller in enumerate(moduleMainControllers):
                kernel_namespace[f"controller_{i}"] = controller
        
        # Start kernel in daemon thread
        kernel_thread = threading.Thread(
            target=start_embedded_kernel, 
            args=(kernel_namespace,), 
            daemon=True,
            name="EmbeddedJupyterKernel"
        )
        kernel_thread.start()
        logger.info("Embedded Jupyter kernel thread started")
        logger.info("To connect from Jupyter Notebook:")
        logger.info("1. Open the ImSwitch_Embedded_Kernel_Connection.ipynb notebook")
        logger.info("2. Follow the manual connection instructions in the notebook")
        logger.info("3. Or use: jupyter console --existing")
        logger.info("1. Wait 3 seconds for kernel spec registration")
        logger.info("2. Refresh browser and select 'ImSwitch Embedded' kernel")
        logger.info("3. Or use manual connection: jupyter console --existing")

    if IS_HEADLESS:
        """We won't have any GUI, so we don't need to prepare the app."""
        # Keep python running
        tDiskCheck = time.time()
        while True: # TODO: have webserver signal somehow?
            try:
                emit_queued()
                time.sleep(1)
                if time.time() - tDiskCheck > 60 and dirtools.getDiskusage() > 0.9:
                    # if the storage is full or the user presses Ctrl+C, we want to stop the experiment
                    if isinstance(moduleMainControllers, dict) and "imcontrol" in moduleMainControllers:
                        controller = moduleMainControllers["imcontrol"]
                        if hasattr(controller, '_ImConMainController__commChannel'):
                            controller._ImConMainController__commChannel.sigExperimentStop.emit()
                    tDiskCheck = time.time()

            except KeyboardInterrupt:
                exitCode = 0
                break

    else:
        # Show app
        if mainView is not None:
            mainView.showMaximized()
            mainView.show()
        exitCode = app.exec_()

    # Clean up
    controllers_to_close = []
    if isinstance(moduleMainControllers, dict):
        controllers_to_close = moduleMainControllers.values()
    elif hasattr(moduleMainControllers, '__iter__'):
        controllers_to_close = moduleMainControllers
    
    for controller in controllers_to_close:
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
