"""
SiLa2Manager for OpenUC2 ImSwitch.

Runs the SiLA 2 server (UniteLabs CDK ``Connector``) in a daemon thread with its own
asyncio event loop. Configuration comes from the setup JSON block ``sila2`` (SiLA2Info).
"""

import asyncio
import logging
import socket
import threading
import uuid
from typing import Any, Dict, List, Optional

from imswitch.imcommon.model import initLogger

try:
    import structlog
    from unitelabs.cdk import Connector
    from unitelabs.cdk.config import ConnectorBaseConfig, SiLAServerConfig, CloudServerConfig
    from unitelabs.cdk.config.connector_base_config import CloudServersConfig
    HAS_SILA2 = True
except ImportError:
    Connector = None
    HAS_SILA2 = False

SERVER_TYPE = "Microscope"
UUID_NAMESPACE = uuid.UUID("6f1b4c59-3d0e-4d3f-9a59-0c2a1b6e5e11")  # openUC2 ImSwitch SiLA servers


def stable_server_uuid(port: int) -> str:
    """One UUID per instrument: the same host and port always give the same UUID, so a
    workcell that registered the microscope finds it again after a restart."""
    return str(uuid.uuid5(UUID_NAMESPACE, f"{socket.gethostname()}:{port}"))


class SiLa2Manager:
    """Manager for the SiLA2 server integration in ImSwitch."""

    def __init__(self, setupInfo):
        self.__logger = initLogger(self)
        self._info = setupInfo
        self._connector: Optional[Any] = None
        self._features: List[Any] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._error = ""

        if not HAS_SILA2:
            self.__logger.warning("unitelabs-cdk not installed – SiLA2 disabled "
                                  "(pip install unitelabs-cdk)")
        elif setupInfo is None:
            self.__logger.info("No 'sila2' block in the setup – SiLA2 disabled")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return HAS_SILA2 and self._info is not None and bool(self._info.enabled)

    @property
    def server_uuid(self) -> str:
        return self._info.serverUUID or stable_server_uuid(self._info.serverPort)

    def get_config(self) -> Dict[str, Any]:
        """Configuration as reported by getSiLA2Status (no key material)."""
        if self._info is None:
            return {"enabled": False}
        i = self._info
        return {
            "enabled": bool(i.enabled), "name": i.serverName, "uuid": self.server_uuid,
            "host": i.serverHost, "port": i.serverPort, "tls": i.tls,
            "discovery": i.discovery, "cloudEndpoint": i.cloudEndpoint,
            "handoverPositionName": i.handoverPositionName,
        }

    def _connector_config(self):
        i = self._info
        sila_server = SiLAServerConfig(
            hostname=i.serverHost, port=i.serverPort, uuid=self.server_uuid,
            name=i.serverName, type=SERVER_TYPE, description=i.serverDescription,
            version=i.serverVersion, vendor_url=i.vendorUrl, tls=i.tls,
            certificate_chain=i.certificateChain, private_key=i.privateKey,
        )
        endpoints = []
        if i.cloudEndpoint:
            endpoints = [CloudServerConfig(hostname=i.cloudEndpoint, port=i.cloudPort, tls=True)]
        config = ConnectorBaseConfig(
            sila_server=sila_server,
            cloud_servers=CloudServersConfig(enabled=bool(endpoints), endpoints=endpoints),
        )
        if not i.discovery:
            config.discovery = None
        return config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_feature(self, feature) -> None:
        """Register a ``sila.Feature`` instance. Call before ``start_server()``."""
        self._features.append(feature)

    def start_server(self) -> None:
        """Start the Connector in a daemon thread with its own event loop.

        CDK's ``run()`` installs signal handlers, which only works on the main thread, so
        the thread calls ``Connector.start()`` directly.
        """
        if not self.is_enabled():
            self.__logger.info("SiLA2 server not started (disabled)")
            return
        if self._server_thread is not None and self._server_thread.is_alive():
            self.__logger.warning("SiLA2 server is already running")
            return
        self._server_thread = threading.Thread(
            target=self._run_server_loop, daemon=True, name="SiLA2Server")
        self._server_thread.start()

    def _run_server_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._start_connector())
            loop.run_forever()
        except Exception as e:
            self._error = str(e)
            self.__logger.error(f"SiLA2 server error: {e}")
        finally:
            self._running = False
            self._connector = None
            loop.close()
            self.__logger.info("SiLA2 server stopped")

    async def _start_connector(self) -> None:
        if not structlog.is_configured():  # the CDK logs every feature definition at debug
            structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
        app = Connector(self._connector_config())
        for feature in self._features:
            app.register(feature)
        self._connector = app
        # start() keeps running while the cloud connection retries; the local server is
        # up as soon as wait_for_ready() returns.
        task = asyncio.get_running_loop().create_task(app.start())
        task.add_done_callback(self._on_start_done)
        await app.wait_for_ready()
        self._running = True
        i = self._info
        self.__logger.info(f"SiLA2 server '{i.serverName}' {self.server_uuid} on "
                           f"{i.serverHost}:{i.serverPort} (tls={i.tls}, "
                           f"discovery={i.discovery}, cloud={i.cloudEndpoint or 'off'})")

    def _on_start_done(self, task) -> None:
        if not task.cancelled() and task.exception() is not None:
            self._error = str(task.exception())
            self.__logger.error(f"SiLA2 connector: {task.exception()}")

    def is_running(self) -> bool:
        return self._running

    def last_error(self) -> str:
        return self._error

    def stop_server(self) -> None:
        """Stop the Connector (graceful, 5 s) and the event loop."""
        loop, app = self._loop, self._connector
        if loop is None or not loop.is_running():
            return
        if app is not None:
            try:
                asyncio.run_coroutine_threadsafe(app.stop(), loop).result(timeout=10)
            except Exception as e:
                self.__logger.warning(f"SiLA2 stop: {e}")
        loop.call_soon_threadsafe(loop.stop)
        self._running = False

    def shutdown(self) -> None:
        self.stop_server()
