"""SiLA 2 server of ImSwitch (UniteLabs CDK), for lab-automation workcells.

Enabled by "SiLa2" in availableWidgets plus a "sila2" block in the setup JSON. The features
(stage, imaging, well plates and experiments, plate hand-over) are in ``sila2/features.py``;
``sila2/check.py`` is a client that exercises every command against a running server.
"""

from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import initLogger, APIExport

try:
    from .sila2.features import create_features
    HAS_CDK = True
except ImportError:
    HAS_CDK = False


class SiLa2Controller(ImConWidgetController):
    """Registers the SiLA 2 features and starts the server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._features = []
        manager = getattr(self._master, "sila2Manager", None)
        if not HAS_CDK or manager is None or not manager.is_enabled():
            self._logger.info("SiLA2 disabled (unitelabs-cdk missing or sila2 not enabled).")
            return
        self._features = create_features(self._master, manager.get_config()["handoverPositionName"])
        for feature in self._features:
            manager.register_feature(feature)
        manager.start_server()

    def closeEvent(self):
        manager = getattr(self._master, "sila2Manager", None)
        if manager is not None:
            manager.shutdown()

    @APIExport(requestType="GET")
    def getSiLA2Status(self) -> dict:
        """SiLA 2 server state: enabled, running, last error, configuration (name, UUID, host,
        port, tls, discovery, cloud endpoint) and the plate on the stage as the hand-over
        feature knows it (type and id, empty when none)."""
        manager = getattr(self._master, "sila2Manager", None)
        if manager is None:
            return {"enabled": False, "running": False}
        site = next((f for f in self._features if hasattr(f, "labware")), None)
        return {
            "enabled": manager.is_enabled(),
            "running": manager.is_running(),
            "error": manager.last_error(),
            "config": manager.get_config(),
            "labwareOnStage": list(site.labware) if site else ["", ""],
        }
