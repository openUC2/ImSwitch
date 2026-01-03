
import os

class LightsheetReactWidget(object):
    """ Widget containing lightsheet interface. """

    def __init__(self):
        self._path = os.path.dirname(__file__)

    def getUIPath(self):
        return os.path.join(self._path, "lightsheetreactwidget")
