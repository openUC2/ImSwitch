"""
SiLA2Manager for OpenUC2 ImSwitch.

Manages the lifecycle of the SiLA2 server (UniteLabs CDK Connector).
Configuration is loaded from setupInfo, analogous to ArkitektManager.
The SiLA2 server runs in a background asyncio event loop and exposes
microscope capabilities as SiLA2 features.
"""

import asyncio
import threading
from typing import Optional, Dict, Any, List

from imswitch.imcommon.model import initLogger

try:
    from unitelabs.cdk import Connector
    HAS_SILA2 = True
except ImportError:
    Connector = None
    HAS_SILA2 = False

import dataclasses
from unitelabs.cdk.config import (
    ConfigurationError,
    ConnectorBaseConfig,
    SiLAServerConfig,
    delayed_default,
    validate_config,
)
import typing_extensions as typing

        
@dataclasses.dataclass
class openUC2MicroscopeConfig(ConnectorBaseConfig):
    """Configuration for the openUC2 Microscope device connector."""

    sila_server: SiLAServerConfig = dataclasses.field(
        default_factory=delayed_default(
            lambda self: SiLAServerConfig(
                name=f"{self.device} Connector",
                description=f"A connector for the {self.device} openUC2 Microscope devices.",
                type="Microscope",
                version=str(0.1), # TODO: FIXME!
                vendor_url="https://openuc2.com/"
            )
        )
    )
    serial_port: str = "/dev/usb0"



    @validate_config()
    def validate_autodetect(self) -> typing.Self:
        """Validate that if autodetect is enabled, a serial number has been provided."""
        if self.autodetect and not self.serial_number:
            msg = "Setting autodetect=True requires a 'serial_number' configuration value to also be set."
            raise ConfigurationError(msg)
        return self


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
        
        # use the openUC2MicroscopeConfig dataclass to validate and structure the configuration
        self._connector = Connector(openUC2MicroscopeConfig)

        for feature in self._features:
            self._connector.register(feature)
            self.__logger.info(f"SiLA2: registered feature {type(feature).__name__}")

        self._running = True
        self.__logger.info(
            f"SiLA2 Connector serving on "
            f"{cfg.get('server_host', '0.0.0.0')}:{cfg.get('server_port', 50052)}"
        )

        # The Connector.serve() call blocks until cancelled.  If the CDK
        # version does not expose a `serve` coroutine directly we fall back
        # to keeping the loop alive indefinitely so that registered features
        # can be discovered via SiLA2 service discovery.
        if hasattr(self._connector, "serve"):
            await self._connector.serve(
                host=cfg.get("server_host", "0.0.0.0"),
                port=cfg.get("server_port", 50052),
            )
        else:
            # Fallback – keep the event loop alive
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
