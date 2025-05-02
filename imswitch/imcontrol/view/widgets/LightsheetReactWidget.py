
import os
from imswitch.imcommon.model import UIExport

class LightsheetReactWidget:
    """ Widget containing lightsheet interface. """
    
    def __init__(self):
        self._path = os.path.dirname(__file__)
        
    @UIExport(
        path="ui/dist",                 # relative to **this** Python package
        name="Lightsheet",
        icon="ThreeDRotationIcon",
        scope="lightsheet_plugin",      # ↔ webpack ModuleFederationPlugin.name
    )
    def getUIPath(self):
        return os.path.join(self._path, "lightsheetreactwidget")