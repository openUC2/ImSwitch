# model.py
class ExperimentModel:
    """
    Example class that stores parameters for T, C, Z, X, Y, and positions.
    In a real system, you'd do more sophisticated config validation.
    """

    def __init__(self, number_z_steps: int = 5, timepoints: int = 1, 
                 x_pixels: int = 64, y_pixels: int = 48,
                 microscope_name: str = "FRAME", is_multiposition: bool = False,
                 stack_cycling_mode: str = "per_stack",
                 channels: dict = None, multi_positions: dict = None):
        """
        Initialize the experiment model with default values.
        """
        # Example config structure
        self.configuration = {
            "experiment": {
                "MicroscopeState": {
                    "microscope_name": microscope_name,
                    "number_z_steps": number_z_steps,
                    "is_multiposition": is_multiposition,
                    "timepoints": timepoints,
                    "stack_cycling_mode": stack_cycling_mode, # means Z changes fastest
                    "channels": {
                        "Ch0": {"is_selected": True, "camera_exposure_time": 50},
                        "Ch1": {"is_selected": True, "camera_exposure_time": 100},
                    },
                },
                "CameraParameters": {
                    "FRAME": {
                        "x_pixels": x_pixels,
                        "y_pixels": y_pixels,
                    }
                },
            }
        }

        # For convenience, store the final shape in attributes
        self.img_width = self.configuration["experiment"]["CameraParameters"]["FRAME"]["x_pixels"]
        self.img_height = self.configuration["experiment"]["CameraParameters"]["FRAME"]["y_pixels"]

        # You could do additional validation, or read from a real file
