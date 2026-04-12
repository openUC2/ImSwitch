"""
SiLA2Manager for OpenUC2 ImSwitch.

Manages the lifecycle of the SiLA2 server (UniteLabs CDK Connector).
Configuration is loaded from setupInfo, analogous to ArkitektManager.
The SiLA2 server runs in a background asyncio event loop and exposes
microscope capabilities as SiLA2 features.
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
        self._thread: Optional[threading.Thread] = None
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
        """Create the Connector, register features, and start serving in a background thread."""
        if not HAS_SILA2:
            self.__logger.warning("Cannot start SiLA2 server – unitelabs-cdk not available")
            return
        if not self._config.get("enabled", True):
            self.__logger.info("SiLA2 server not started (disabled)")
            return
        if self._running:
            self.__logger.warning("SiLA2 server is already running")
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_server_loop,
            daemon=True,
            name="SiLA2-Server",
        )
        self._thread.start()
        self.__logger.info("SiLA2 server thread started")

    def _run_server_loop(self) -> None:
        """Entry point for the background thread that runs the asyncio event loop."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            self.__logger.error(f"SiLA2 server loop exited with error: {e}")
        finally:
            self._running = False

    async def _serve(self) -> None:
        """Create the Connector, register all features, and run forever."""
        cfg = self._config

        # Build a typed config instance from the ImSwitch setup values
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

        self._connector = Connector(connector_config)

        for feature in self._features:
            self._connector.register(feature)
            self.__logger.info(f"SiLA2: registered feature {type(feature).__name__}")

        self._running = True
        self.__logger.info(
            f"SiLA2 Connector serving on "
            f"{cfg.get('server_host', '0.0.0.0')}:{cfg.get('server_port', 50052)}"
        )

        # The reference CDK pattern uses `app.start()` (UniteLabs CDK >= 0.2).
        # Older builds exposed a `serve(host, port)` coroutine.  We try
        # `start()` first, then `serve()`, and finally keep the loop alive
        # as a last resort so that the asyncio thread does not exit.
        if hasattr(self._connector, "start"):
            await self._connector.start()
        elif hasattr(self._connector, "serve"):
            await self._connector.serve(
                host=cfg.get("server_host", "0.0.0.0"),
                port=cfg.get("server_port", 50052),
            )
        else:
            # Last-resort fallback – keep the event loop alive
            self.__logger.warning(
                "SiLA2 Connector has neither start() nor serve() – "
                "running in no-op mode (features will not be reachable)"
            )
            while self._running:
                await asyncio.sleep(1)

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

    def shutdown(self) -> None:
        """Stop the SiLA2 server and clean up resources."""
        self._running = False
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._connector = None
        self.__logger.info("SiLA2 server shut down")

    def __del__(self):
        """Cleanup on garbage collection."""
        self.shutdown()
