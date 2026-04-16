import webbrowser

import psutil

from imswitch import  __version__
from imswitch.imcommon.framework import Timer
from imswitch.imcommon.model import dirtools, modulesconfigtools, ostools, APIExport
from .basecontrollers import WidgetController


class MultiModuleWindowController(WidgetController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._moduleIdNameMap = {}

        self._updateMemBarTimer = Timer()
        self._updateMemBarTimer.timeout.connect(self.updateRAMUsage)
        self._updateMemBarTimer.start(1000)

        self.updateRAMUsage()


    def openUserDir(self):
        """ Shows the user files directory in system file explorer. """
        ostools.openFolderInOS(dirtools.UserFileDirs.Root)

    def showDocs(self):
        """ Opens the ImSwitch documentation in a web browser. """
        webbrowser.open(f'https://imswitch.readthedocs.io/en/v{__version__}/')

    def checkUpdates(self):
        """ Checks if there are any updates to ImSwitch available and notifies
        the user. """
        self.checkUpdatesController.checkForUpdates()
        self._widget.showCheckUpdatesDialogBlocking()

    def showAbout(self):
        """ Shows an "about" dialog. """
        self._widget.showAboutDialogBlocking()

    def moduleAdded(self, moduleId, moduleName):
        self._moduleIdNameMap[moduleId] = moduleName

    def updateRAMUsage(self):
        self._widget.updateRAMUsage(psutil.virtual_memory()[2] / 100)

    @APIExport(runOnUIThread=True)
    def setCurrentModule(self, moduleId: str) -> None:
        """ Sets the currently displayed module to the module with the
        specified ID (e.g. "imcontrol"). """
        moduleName = self._moduleIdNameMap[moduleId]
        self._widget.setSelectedModule(moduleName)
