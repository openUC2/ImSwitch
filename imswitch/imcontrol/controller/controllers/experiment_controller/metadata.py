# metadata.py
class MinimalMetadata:
    """
    Minimal stand-in for your custom metadata logic.
    - per_stack => whether Z changes faster or C changes faster in your scanning.
    - multiscales_dict(...) => builds the top-level dictionary that OME-Zarr demands.
    """

    def __init__(self, per_stack: bool = True):
        self.per_stack = per_stack

    def multiscales_dict(self, name: str, paths, resolutions, view: str = "") -> dict:
        """
        Returns a valid OME-Zarr 0.4 'multiscales' entry with:
         - version = '0.4'
         - axes = ['t','c','z','y','x']
         - datasets[] each has 'path' plus 'coordinateTransformations' with a 'scale'
        """
        datasets = []
        for p in paths:
            datasets.append(
                {
                    "path": p,
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [1, 1, 1, 1, 1],  # identity (t,c,z,y,x)
                        }
                    ],
                }
            )
        return {
            "name": name,
            "view": view,
            "version": "0.4",
            "axes": ["t", "c", "z", "y", "x"],
            "resolutions": str(resolutions),  # optional – just storing them
            "datasets": datasets,
        }
