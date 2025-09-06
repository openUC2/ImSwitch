"""
Example integration of ArkitektManager into ImSwitch controllers.

This shows how to use the ArkitektManager in other parts of ImSwitch,
such as in detector managers or other controllers.
"""

from imswitch.imcontrol.model.managers.ArkitektManager import ArkitektManager


class ExampleController:
    """Example controller showing ArkitektManager integration."""
    
    def __init__(self, masterController):
        """Initialize controller with ArkitektManager."""
        self._master = masterController
        
        # Initialize ArkitektManager
        self.arkitekt_manager = ArkitektManager(masterController)
    
    def snap_and_deconvolve(self):
        """
        Example method: capture image and deconvolve using Arkitekt.
        
        This replaces the fragmented code from the original example.
        """
        try:
            # Get image from detector (assuming snapNumpy method exists)
            if hasattr(self._master, 'snapNumpy'):
                numpy_array = list(self._master.snapNumpy().values())[0]
            else:
                print("No snapNumpy method available")
                return None
            
            # Use ArkitektManager for deconvolution
            if self.arkitekt_manager.is_enabled():
                deconvolved_image = self.arkitekt_manager.upload_and_deconvolve_image(numpy_array)
                
                if deconvolved_image is not None:
                    print("Image deconvolution completed successfully")
                    return deconvolved_image
                else:
                    print("Deconvolution failed")
                    return None
            else:
                print("Arkitekt integration not available")
                return numpy_array  # Return original image
                
        except Exception as e:
            print(f"Error in snap_and_deconvolve: {e}")
            return None
    
    def configure_arkitekt(self, new_token: str = None, enabled: bool = None):
        """
        Example method: update Arkitekt configuration.
        
        Args:
            new_token: New redeem token for Arkitekt
            enabled: Enable/disable Arkitekt integration
        """
        config_updates = {}
        
        if new_token is not None:
            config_updates["redeem_token"] = new_token
            
        if enabled is not None:
            config_updates["enabled"] = enabled
            
        if config_updates:
            self.arkitekt_manager.update_config(config_updates)
            print(f"Arkitekt configuration updated: {config_updates}")
    
    def get_arkitekt_status(self):
        """Get current Arkitekt status and configuration."""
        return {
            "enabled": self.arkitekt_manager.is_enabled(),
            "config": self.arkitekt_manager.get_config()
        }


# Usage example in a detector manager or similar class
class ExampleDetectorManager:
    """Example showing integration into a detector manager."""
    
    def __init__(self, masterController, name):
        self._master = masterController
        self._name = name
        
        # Initialize ArkitektManager
        self.arkitekt_manager = ArkitektManager(masterController)
    
    def capture_and_process(self):
        """Capture image and optionally process with Arkitekt."""
        # Simulate image capture
        import numpy as np
        fake_image = np.random.rand(512, 512).astype(np.float32)
        
        print(f"Captured image with shape: {fake_image.shape}")
        
        # Process with Arkitekt if available
        if self.arkitekt_manager.is_enabled():
            processed_image = self.arkitekt_manager.upload_and_deconvolve_image(fake_image)
            
            if processed_image is not None:
                print("Image processed with Arkitekt deconvolution")
                return processed_image
            else:
                print("Arkitekt processing failed, returning original image")
                return fake_image
        else:
            print("Arkitekt not available, returning original image")
            return fake_image


# Configuration management example
def setup_arkitekt_from_user_input():
    """Example function to set up Arkitekt from user input."""
    # This could be part of a GUI setup dialog
    
    print("Setting up Arkitekt integration...")
    
    # In a real implementation, these would come from GUI inputs
    user_token = "your-actual-token-here"
    server_url = "http://go.arkitekt.io"
    enable_integration = True
    
    # Create configuration
    config = {
        "redeem_token": user_token,
        "url": server_url,
        "enabled": enable_integration,
        "app_name": "imswitch"
    }
    
    print(f"Configuration: {config}")
    
    # In practice, this would be done through the manager instance
    # arkitekt_manager.update_config(config)
    
    return config
