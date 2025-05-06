
import os
from imswitch.imcommon.model import UIExport

class LightsheetReactWidget(object):
    """ Widget containing lightsheet interface. """
    
    def __init__(self):
        self._path = os.path.dirname(__file__)
        
    @UIExport(
        path="/Users/bene/Downloads/imswitch-lightsheet/src/imswitch_lightsheet/ui/dist",                 # relative to **this** Python package
        name="Lightsheet",
        icon="ThreeDRotationIcon",
    )
    def getUIPath(self):
        return os.path.join(self._path, "lightsheetreactwidget")