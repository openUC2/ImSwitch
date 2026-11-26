"""SiLA 2 server of ImSwitch (UniteLabs CDK): the microscope as a device in a lab-automation
workcell.

Enabled by "SiLa2" in availableWidgets plus a "sila2" block in the setup JSON. The features
(stage, imaging, well plates and experiments, jobs, plate hand-over, lock) are in
``sila2/features.py``; ``imswitch/sila_check.py`` (``imswitch-sila-check``) is a client that
exercises every command against a running server. While a workcell holds the SiLA lock,
``model/workcell_lock.py`` keeps the HTTP API (the UI) from moving, lighting or starting
anything.
"""

from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import initLogger, APIExport
from imswitch.imcontrol.model import configfiletools, workcell_lock

try:
    from .sila2.features import create_features, lock_active
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
        info = manager.info
        self._features = create_features(self._master, info.handoverPositionName,
                                         plate_origins=info.plateOrigins,
                                         save_setup=self._save_setup)
        for feature in self._features:
            manager.register_feature(feature)
        workcell_lock.set_lock_state(self._lock_holder)
        manager.start_server()

    def _feature(self, attribute: str):
        return next((f for f in self._features if hasattr(f, attribute)), None)

    def _lock_holder(self) -> str:
        # Read without the property: it would signal the server's event loop from this thread.
        feature = self._feature("lock")
        return "SiLA client lock" if lock_active(getattr(feature, "_lock", None)) else ""

    def _save_setup(self) -> None:
        """Write the setup JSON (the sila2 block changed, e.g. a taught plate origin)."""
        options, _ = configfiletools.loadOptions()
        configfiletools.saveSetupInfo(options, self._setupInfo)

    def closeEvent(self):
        workcell_lock.set_lock_state(None)
        manager = getattr(self._master, "sila2Manager", None)
        if manager is not None:
            manager.shutdown()

    def isWorkcellLocked(self) -> bool:
        """True while a SiLA client holds the lock (False for calls made on its behalf)."""
        return bool(workcell_lock.holder())

    @APIExport(requestType="GET")
    def getSiLA2Status(self) -> dict:
        """SiLA 2 server state: enabled, running, last error, configuration (name, UUID, host,
        port, tls, discovery, cloud endpoint), whether a workcell holds the lock, the plate on
        the stage as the hand-over feature knows it (type and id, empty when none) and the
        taught plate origins."""
        manager = getattr(self._master, "sila2Manager", None)
        if manager is None:
            return {"enabled": False, "running": False}
        site = self._feature("labware")
        return {
            "enabled": manager.is_enabled(),
            "running": manager.is_running(),
            "error": manager.last_error(),
            "config": manager.get_config(),
            "locked": bool(self._lock_holder()),
            "labwareOnStage": list(site.labware) if site else ["", ""],
            "plateOrigins": dict(manager.info.plateOrigins) if manager.info else {},
        }

    @APIExport(requestType="POST")
    def releaseWorkcellLock(self) -> dict:
        """Free the microscope from a workcell's SiLA lock, e.g. when the workcell crashed
        while holding it. The workcell's next locked call then fails."""
        lock = self._feature("lock")
        manager = getattr(self._master, "sila2Manager", None)
        if lock is None or manager is None or not self._lock_holder():
            return {"released": False}
        manager.call_soon(setattr, lock, "lock", None)
        self._logger.warning("SiLA workcell lock released by hand.")
        return {"released": True}
