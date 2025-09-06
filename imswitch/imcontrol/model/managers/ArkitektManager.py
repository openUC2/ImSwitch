"""
ArkitektManager for ImSwitch integration with Arkitekt services.

This manager handles the connection to Arkitekt services and provides
methods for image processing and deconvolution using remote services.
Configuration is loaded from setupInfo similar to FocusLockController.
"""

from typing import Optional, Dict, Any
from contextvars import copy_context, Context

from imswitch.imcommon.model import initLogger


def ensure_context_in_thread(context: Context):
    """Ensure context variables are available in the current thread."""
    for ctx, value in context.items():
        ctx.set(value)


class ArkitektManager:
    """Manager for Arkitekt integration in ImSwitch."""

    def __init__(self, setupInfo):
        """
        Initialize the ArkitektManager.
        
        Args:
            setupInfo: Setup information containing Arkitekt configuration
            masterController: Reference to the master controller
        """
        self.__logger = initLogger(self)
        
        self._setupInfo = setupInfo
        
        # Initialize Arkitekt-related attributes
        self.__arkitekt = None
        self.context = None
        
        # Check if Arkitekt configuration exists
        if self._setupInfo is None:
            self.__logger.info("No Arkitekt configuration found in setupInfo - Arkitekt features disabled")
            return
            
        # Load configuration from setupInfo (similar to FocusLockController pattern)
        self._load_config_from_setupinfo()
        
        # Initialize Arkitekt if enabled
        if self._config.get("enabled", True):
            self._initialize_arkitekt()
        else:
            self.__logger.info("Arkitekt integration disabled in configuration")

    def _load_config_from_setupinfo(self) -> None:
        """Load Arkitekt configuration from setupInfo (similar to FocusLockController pattern)."""
        try:
            arkitekt_info = self._setupInfo
            
            # Load parameters using getattr with defaults (like FocusLockController does)
            enabled = getattr(arkitekt_info, "enabled", True)
            app_name = getattr(arkitekt_info, "appName", "imswitch")
            redeem_token = getattr(arkitekt_info, "redeemToken", "")
            url = getattr(arkitekt_info, "url", "http://go.arkitekt.io")
            sync_in_async = getattr(arkitekt_info, "syncInAsync", True)
            deconvolve_action_hash = getattr(arkitekt_info, "deconvolveActionHash", 
                                           "c58c90edbf6e208e3deafdd6f885553d6e027573f0ddc3b59ced3911f016ef4f")
            # QUESTION: How can I retreive this hash? Is there a way to get a readable list of available services?
            self._config = {
                "enabled": enabled,
                "app_name": app_name,
                "redeem_token": redeem_token,
                "url": url,
                "sync_in_async": sync_in_async,
                "deconvolve_action_hash": deconvolve_action_hash
            }
            
            self.__logger.info(f"Loaded Arkitekt config from setupInfo: enabled={enabled}, app_name={app_name}")
            
        except Exception as e:
            self.__logger.error(f"Failed to load Arkitekt configuration from setupInfo: {e}")
            self._config = {"enabled": False}

    def _initialize_arkitekt(self) -> None:
        """Initialize Arkitekt connection if enabled."""
        if not self._config.get("enabled", True):
            self.__logger.info("Arkitekt integration is disabled in configuration")
            return

        try:
            # Import here to avoid issues if arkitekt_next is not installed
            from arkitekt_next import easy
            from koil import Koil
            
            # Create Arkitekt client with configuration
            self.__arkitekt = easy(
                identifier=self._config.get("app_name", "imswitch"),
                redeem_token=self._config.get("redeem_token"),
                url=self._config.get("url", "http://go.arkitekt.io")
            )
            
            # Set up Koil for async context handling
            self.__arkitekt.__koil = Koil(
                sync_in_async=self._config.get("sync_in_async", True)
            )
            
            # Enter the arkitekt client context (spawn background thread)
            self.__arkitekt.enter()
            
            # Copy context variables for thread handling
            self.context = copy_context()
            
            self.__logger.info("Arkitekt integration initialized successfully")
            
        except ImportError:
            self.__logger.warning("arkitekt_next not available - Arkitekt features disabled")
            self._config["enabled"] = False
        except Exception as e:
            self.__logger.error(f"Failed to initialize Arkitekt: {e}")
            self._config["enabled"] = False

    def is_enabled(self) -> bool:
        """Check if Arkitekt integration is enabled and available."""
        return self._config.get("enabled", False) and self.__arkitekt is not None

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy() if self._config else {}

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration and reinitialize if necessary.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        if self._config is None:
            self._config = {}
            
        # Update configuration
        self._config.update(new_config)
        
        # Save updated configuration
        config_dir = dirtools.UserFileDirs.Config
        arkitekt_config_path = os.path.join(config_dir, 'arkitekt_config.json')
        self._save_config(arkitekt_config_path)
        
        # Reinitialize if enabled status changed or important settings changed
        if new_config.get("enabled") or any(key in new_config for key in 
                                          ["app_name", "redeem_token", "url", "sync_in_async"]):
            self._initialize_arkitekt()
            
        self.__logger.info("Arkitekt configuration updated")

    def upload_and_deconvolve_image(self, image) -> Optional[Any]:
        """
        Upload an image to Arkitekt and perform deconvolution.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deconvolved image as numpy array, or None if operation fails
        """
        if not self.is_enabled():
            self.__logger.warning("Arkitekt not available for image deconvolution")
            return None

        try:
            # Import here to avoid issues if libraries are not available
            from arkitekt_next import find
            from mikro_next.api.schema import from_array_like
            
            # Ensure context is available in this thread
            ensure_context_in_thread(self.context)
            
            # Find the deconvolution action
            action = find(
                hash=self._config.get("deconvolve_action_hash",
                                    "c58c90edbf6e208e3deafdd6f885553d6e027573f0ddc3b59ced3911f016ef4f")
            )
            
            # Convert numpy array to mikro image
            mikro_image = from_array_like(
                image,
                name="ImSwitch_Image"
            )
            
            # Call the deconvolution action (blocks until result is available)
            result = action(image=mikro_image)
            
            # Download and return the processed image data
            deconvolved_image = result.data.compute()
            
            self.__logger.info("Image deconvolution completed successfully")
            return deconvolved_image
            
        except Exception as e:
            self.__logger.error(f"Failed to perform image deconvolution: {e}")
            return None

    def find_action(self, action_hash: str) -> Optional[Any]:
        """
        Find an Arkitekt action by its hash.
        
        Args:
            action_hash: Hash identifier of the action
            
        Returns:
            Action object or None if not found
        """
        if not self.is_enabled():
            self.__logger.warning("Arkitekt not available for action lookup")
            return None

        try:
            from arkitekt_next import find
            
            ensure_context_in_thread(self.context)
            action = find(hash=action_hash)
            
            self.__logger.debug(f"Found action with hash: {action_hash}")
            return action
            
        except Exception as e:
            self.__logger.error(f"Failed to find action {action_hash}: {e}")
            return None

    def execute_action(self, action_hash: str, **kwargs) -> Optional[Any]:
        """
        Execute an Arkitekt action with given parameters.
        
        Args:
            action_hash: Hash identifier of the action
            **kwargs: Parameters to pass to the action
            
        Returns:
            Action result or None if execution fails
        """
        action = self.find_action(action_hash)
        if action is None:
            return None

        try:
            result = action(**kwargs)
            self.__logger.info(f"Action {action_hash} executed successfully")
            return result
            
        except Exception as e:
            self.__logger.error(f"Failed to execute action {action_hash}: {e}")
            return None

    def shutdown(self) -> None:
        """Shutdown Arkitekt connection gracefully."""
        if self.__arkitekt is not None:
            try:
                self.__arkitekt.exit()
                self.__logger.info("Arkitekt connection closed")
            except Exception as e:
                self.__logger.error(f"Error during Arkitekt shutdown: {e}")
            finally:
                self.__arkitekt = None

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.shutdown()
