
from imswitch.config import get_config
import os
from imswitch.imcommon.model import dirtools
import sys
from .notebook_process import testnotebook, startnotebook, stopnotebook

class LaunchNotebookServer:

    def __init__(self):
        pass

    def startServer(self):
        config = get_config()
        
        python_exec_path = os.path.dirname(sys.executable)
        execname = os.path.join(python_exec_path, 'jupyter-lab')

        # check if jupyter notebook is installed
        if not testnotebook(execname):
            print("No jupyter notebook found")
            return False

        directory = None
        directory =  os.path.join(dirtools.UserFileDirs.Root, "imnotebook")
        if not os.path.exists(directory):
            os.makedirs(directory)

        # start the notebook process with configured port
        webaddr = startnotebook(execname, port=config.jupyter_port, directory=directory)
        return webaddr

    def stopServer(self):
        try:stopnotebook()
        except Exception as e:
            print("Could not stop Jupyter Notebook server: %s" % str(e))
        return True
