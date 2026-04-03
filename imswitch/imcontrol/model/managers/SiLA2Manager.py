"""
SiLA2Manager for OpenUC2 ImSwitch.

Manages the lifecycle of the SiLA2 server (UniteLabs CDK Connector).
Configuration is loaded from setupInfo, analogous to ArkitektManager.

The SiLA2 server runs in a dedicated daemon thread with its own asyncio
event loop. 
"""

import asyncio
import dataclasses
import threading
from typing import Any, Optional, Dict, List

from imswitch.imcommon.model import initLogger

try:
    from unitelabs.cdk import Connector
    from unitelabs.cdk.config import (
        ConnectorBaseConfig,
        SiLAServerConfig,
    )
    HAS_SILA2 = True

    @dataclasses.dataclass
    class OpenUC2MicroscopeConfig(ConnectorBaseConfig):
        """
        SiLA2 Connector configuration for the OpenUC2 ImSwitch microscope.

        All fields are populated from the ImSwitch setup JSON (SiLA2Info) at
        runtime; the defaults here serve as safe fallbacks.
        """

        sila_server: Any = dataclasses.field(
            default_factory=lambda: SiLAServerConfig(
                name="OpenUC2 ImSwitch",
                description="SiLA2 server for OpenUC2 ImSwitch microscope control",
                type="Microscope",
                version="0.1.0",
                vendor_url="https://openuc2.com/",
            )
        )

except ImportError:
    Connector = None
    ConnectorBaseConfig = None
    SiLAServerConfig = None
    OpenUC2MicroscopeConfig = None
    HAS_SILA2 = False


class SiLA2Manager:
    """Manager for the SiLA2 server integration in ImSwitch."""

    def __init__(self, setupInfo):
        """
        Initialize the SiLA2Manager.

        Args:
            setupInfo: SiLA2Info dataclass from the setup configuration.
        """
        self.__logger = initLogger(self)
        self._setupInfo = setupInfo
        self._config: Dict[str, Any] = {}
        self._connector: Optional[Any] = None
        self._features: List[Any] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        if not HAS_SILA2:
            self.__logger.warning(
                "unitelabs-cdk not installed – SiLA2 features disabled. "
                "Install with: pip install unitelabs-cdk"
            )
            return

        if self._setupInfo is None:
            self.__logger.info(
                "No SiLA2 configuration found in setupInfo – SiLA2 features disabled"
            )
            return

        self._load_config_from_setupinfo()

        if not self._config.get("enabled", True):
            self.__logger.info("SiLA2 integration disabled in configuration")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_config_from_setupinfo(self) -> None:
        """Load SiLA2 configuration from setupInfo."""
        try:
            info = self._setupInfo
            self._config = {
                "enabled": getattr(info, "enabled", True),
                "server_name": getattr(info, "serverName", "OpenUC2 ImSwitch"),
                "server_description": getattr(
                    info,
                    "serverDescription",
                    "SiLA2 server for OpenUC2 ImSwitch microscope control",
                ),
                "server_host": getattr(info, "serverHost", "0.0.0.0"),
                "server_port": getattr(info, "serverPort", 50052),
                "server_version": getattr(info, "serverVersion", "0.1.0"),
                "vendor_url": getattr(info, "vendorUrl", "https://openuc2.com"),
            }
            self.__logger.info(
                f"SiLA2 config loaded: name={self._config['server_name']}, "
                f"host={self._config['server_host']}:{self._config['server_port']}"
            )
        except Exception as e:
            self.__logger.error(f"Failed to load SiLA2 config from setupInfo: {e}")
            self._config = {"enabled": False}

        # self._config = 
    # ------------------------------------------------------------------
    # Feature registration
    # ------------------------------------------------------------------

    def register_feature(self, feature) -> None:
        """
        Register a SiLA2 feature to be served by the Connector.

        Call this *before* ``start_server()``.

        Args:
            feature: An instance of a ``sila.Feature`` subclass.
        """
        self._features.append(feature)
        self.__logger.debug(f"Registered SiLA2 feature: {type(feature).__name__}")

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start_server(self) -> None:
        """Start the SiLA2 server in a dedicated background daemon thread.

        The thread creates its own asyncio event loop and calls
        Connector.start() directly, avoiding CDK's run() which requires
        signal-handler access (set_wakeup_fd) restricted to the main thread.
        """
        if not HAS_SILA2:
            self.__logger.warning("Cannot start SiLA2 server – unitelabs-cdk not available")
            return
        if not self._config.get("enabled", True):
            self.__logger.info("SiLA2 server not started (disabled)")
            return
        if self._running:
            self.__logger.warning("SiLA2 server is already running")
            return
        '''
        TODO: Problematic since not main thread, but CDK's run() requires signal handling which is
        main-thread-only. We call Connector.start() directly in the thread to avoid this, 
        but it may still cause issues with some features that expect to be on the main thread. 
        We should test this thoroughly and consider alternatives if it causes problems.
        '''
        self._server_thread = threading.Thread( 
            target=self._run_server_loop,
            daemon=True,
            name="SiLA2Server",
        )
        self._server_thread.start()
        self.__logger.info("SiLA2 server thread started")

    def _run_server_loop(self) -> None:
        """Entry point for the SiLA2 daemon thread.

        Creates a dedicated event loop, starts the Connector, then calls
        loop.run_forever() so the Connector's background tasks keep running.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._setup_connector())
            self.__logger.info("SiLA2 Connector running – entering keep-alive loop")
            loop.run_forever()
        except Exception as e:
            self.__logger.error(f"SiLA2 server error: {e}")
        finally:
            self._running = False
            self._connector = None
            try:
                loop.close()
            except Exception:
                pass
            self.__logger.info("SiLA2 server stopped")

    async def _setup_connector(self) -> None:
        """Build and start the CDK Connector inside the background event loop."""
        cfg = self._config

        connector_config = OpenUC2MicroscopeConfig(
            sila_server=SiLAServerConfig(
                name=cfg.get("server_name", "OpenUC2 ImSwitch"),
                description=cfg.get(
                    "server_description",
                    "SiLA2 server for OpenUC2 ImSwitch microscope control",
                ),
                type="Microscope",
                version=cfg.get("server_version", "0.1.0"),
                vendor_url=cfg.get("vendor_url", "https://openuc2.com/"),
            )
        )
        # TODO: Load from config.json instead of hardcoding
        connector_config.cloud_server_endpoint.hostname = (
            "00000000-0000-0000-0000-000000000000.dev.unitelabs.io"
        )
        connector_config.cloud_server_endpoint.port = 443
        connector_config.cloud_server_endpoint.tls = True

        app = Connector(connector_config)
        self._connector = app

        for feature in self._features:
            app.register(feature)
            self.__logger.info(f"SiLA2: registered feature {type(feature).__name__}")

        # Non-blocking: starts the gRPC/SiLA2 listener; background tasks
        # remain active as long as loop.run_forever() is running.
        await app.start()
        self._running = True
        self.__logger.info(
            f"SiLA2 Connector started: {cfg.get('server_name', 'OpenUC2 ImSwitch')}"
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Check whether SiLA2 integration is enabled and the CDK is available."""
        return HAS_SILA2 and self._config.get("enabled", False)

    def is_running(self) -> bool:
        """Return True if the SiLA2 server is currently running."""
        return self._running

    def get_config(self) -> Dict[str, Any]:
        """Return a copy of the current configuration."""
        return self._config.copy()

    def get_connector(self) -> Optional[Any]:
        """Return the underlying Connector instance (or None)."""
        return self._connector

    def schedule_coroutine(self, coro):
        """
        Schedule a coroutine on the SiLA2 event loop from a sync context.

        Returns an ``asyncio.Future`` that can be used to retrieve the result.
        """
        if self._loop is None or not self._running:
            raise RuntimeError("SiLA2 event loop is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop_server(self) -> None:
        """Signal the background event loop to stop (best-effort graceful stop)."""
        self._running = False
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.__logger.info("SiLA2 stop requested")

    def shutdown(self) -> None:
        """Alias for stop_server, called on cleanup."""
        self.stop_server()
        self._connector = None

    def __del__(self):
        """Cleanup on garbage collection."""
        self.shutdown()
