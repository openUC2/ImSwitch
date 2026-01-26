"""
Minimal OME-Zarr metadata utilities.

This module provides helper classes for building OME-NGFF compliant metadata
for Zarr stores.

Migrated from: imswitch/imcontrol/controller/controllers/experiment_controller/metadata.py
"""

from typing import List, Optional, Dict, Any


class MinimalMetadata:
    """
    Minimal metadata builder for OME-Zarr 0.4 format.
    
    Builds proper OME-NGFF multiscales metadata with axes, 
    coordinate transformations, and resolution information.
    
    Attributes:
        per_stack: Whether Z changes faster than C in scanning order
        
    Example:
        >>> metadata = MinimalMetadata(per_stack=True)
        >>> multiscales = metadata.multiscales_dict(
        ...     name="experiment",
        ...     paths=["0", "1"],
        ...     resolutions=[(1,1,1), (1,2,2)]
        ... )
    """

    def __init__(self, per_stack: bool = True):
        """
        Initialize metadata builder.
        
        Args:
            per_stack: If True, Z changes faster than C during scanning.
                      If False, C changes faster than Z.
        """
        self.per_stack = per_stack

    def multiscales_dict(
        self, 
        name: str, 
        paths: List[str], 
        resolutions: List[tuple], 
        view: str = "",
        axes: Optional[List[Dict[str, str]]] = None,
        pixel_sizes: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Build a valid OME-Zarr 0.4 'multiscales' entry.
        
        Args:
            name: Name for this multiscales entry
            paths: List of array paths (e.g., ["0", "1", "2"] for pyramid levels)
            resolutions: List of (dz, dy, dx) downsampling factors per level
            view: Optional view name
            axes: Optional custom axes definition. Defaults to TCZYX.
            pixel_sizes: Optional dict with 't', 'c', 'z', 'y', 'x' pixel sizes in physical units
            
        Returns:
            Dictionary conforming to OME-NGFF 0.4 multiscales specification
        """
        # Default axes for 5D TCZYX data
        if axes is None:
            axes = [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ]
        
        # Default pixel sizes (identity)
        if pixel_sizes is None:
            pixel_sizes = {"t": 1, "c": 1, "z": 1, "y": 1, "x": 1}
        
        # Build datasets with coordinate transformations
        datasets = []
        for i, p in enumerate(paths):
            # Get resolution factors for this level
            if i < len(resolutions):
                dz, dy, dx = resolutions[i]
            else:
                dz, dy, dx = 1, 1, 1
            
            # Scale factors include pixel sizes and downsampling
            scale = [
                pixel_sizes.get("t", 1),
                pixel_sizes.get("c", 1),
                pixel_sizes.get("z", 1) * dz,
                pixel_sizes.get("y", 1) * dy,
                pixel_sizes.get("x", 1) * dx,
            ]
            
            datasets.append({
                "path": p,
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale}
                ],
            })
        
        result = {
            "name": name,
            "version": "0.4",
            "axes": [{"name": a["name"], "type": a["type"]} 
                     if "unit" not in a 
                     else {"name": a["name"], "type": a["type"], "unit": a["unit"]}
                     for a in axes],
            "datasets": datasets,
        }
        
        # Add optional fields
        if view:
            result["view"] = view
        if resolutions:
            result["resolutions"] = str(resolutions)
            
        return result

    def omero_channels_dict(
        self,
        channel_names: List[str],
        channel_colors: Optional[List[str]] = None,
        dtype_max: int = 65535
    ) -> Dict[str, Any]:
        """
        Build OME-Zarr omero metadata for channel visualization.
        
        Args:
            channel_names: List of channel names
            channel_colors: List of hex colors (without #). 
                           Defaults to green, red, blue, cyan, magenta, yellow
            dtype_max: Maximum value for contrast window (65535 for uint16)
            
        Returns:
            Dictionary conforming to OME-NGFF omero specification
        """
        if channel_colors is None:
            default_colors = ["00FF00", "FF0000", "0000FF", "00FFFF", "FF00FF", "FFFF00"]
            channel_colors = [default_colors[i % len(default_colors)] 
                             for i in range(len(channel_names))]
        
        channels = []
        for i, name in enumerate(channel_names):
            color = channel_colors[i] if i < len(channel_colors) else "FFFFFF"
            channels.append({
                "label": name,
                "color": color,
                "active": True,
                "coefficient": 1.0,
                "family": "linear",
                "inverted": False,
                "window": {
                    "start": 0,
                    "end": dtype_max,
                    "min": 0,
                    "max": dtype_max
                }
            })
        
        return {
            "id": 1,
            "version": "0.4",
            "channels": channels,
            "rdefs": {
                "defaultT": 0,
                "defaultZ": 0,
                "model": "color"
            }
        }
