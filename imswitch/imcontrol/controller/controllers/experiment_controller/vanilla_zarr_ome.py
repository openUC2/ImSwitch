"""
Vanilla Zarr implementation of OME-Zarr functionality.

This module provides vanilla Zarr implementations of the key ome-zarr functions
to eliminate the ome-zarr dependency while maintaining compatibility with
OME-Zarr format standards and viewers like vivZAR and vtkviewer.
"""

import zarr
import os


class VanillaZarrStore:
    """Simple store wrapper for vanilla Zarr functionality."""
    
    def __init__(self, path):
        self.path = path
        # Zarr 3.0 compatibility - DirectoryStore was replaced with LocalStore or direct path usage
        if hasattr(zarr.storage, 'DirectoryStore'):
            # Zarr 2.x compatibility
            self.store = zarr.storage.DirectoryStore(path)
        elif hasattr(zarr.storage, 'LocalStore'):
            # Zarr 3.x with LocalStore
            self.store = zarr.storage.LocalStore(path)
        else:
            # Zarr 3.x with direct path usage
            self.store = path
    
    def close(self):
        """Close the store."""
        pass  # No explicit close needed for directory stores


class VanillaZarrURL:
    """Simple URL wrapper for vanilla Zarr functionality."""
    
    def __init__(self, path, mode="r"):
        self.path = path
        self.mode = mode
        self.store = VanillaZarrStore(path)


def parse_url(path, mode="r"):
    """
    Vanilla Zarr implementation of ome_zarr.io.parse_url.
    
    Args:
        path: Path to Zarr store
        mode: Access mode (r, w, a)
        
    Returns:
        Simple URL object with store attribute
    """
    return VanillaZarrURL(path, mode)


def write_image(image, group, axes="zyx"):
    """
    Vanilla Zarr implementation of ome_zarr.writer.write_image.
    
    Args:
        image: Image data to write
        group: Zarr group to write to
        axes: Axis specification string
    """
    # Create the main image dataset with Zarr v3 compatible API
    # Choose reasonable chunk sizes based on image shape
    chunk_shape = []
    for dim_size in image.shape:
        chunk_size = min(dim_size, 512)  # Use 512 or dimension size, whichever is smaller
        chunk_shape.append(chunk_size)
    
    array = group.create_dataset(
        "0",
        shape=tuple(int(dim) for dim in image.shape),
        chunks=tuple(int(chunk_size) for chunk_size in chunk_shape),
        dtype=image.dtype
    )
    array[:] = image
    
    # Set OME-Zarr metadata for the image
    axes_metadata = []
    for axis in axes:
        if axis == 'z':
            axes_metadata.append({"name": "z", "type": "space"})
        elif axis == 'y':
            axes_metadata.append({"name": "y", "type": "space"})
        elif axis == 'x':
            axes_metadata.append({"name": "x", "type": "space"})
        elif axis == 't':
            axes_metadata.append({"name": "t", "type": "time"})
        elif axis == 'c':
            axes_metadata.append({"name": "c", "type": "channel"})
    
    multiscales = [{
        "version": "0.4",
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0] * len(axes)}
                ]
            }
        ],
        "axes": axes_metadata
    }]
    
    group.attrs["multiscales"] = multiscales


def write_multiscales_metadata(group, datasets, format_version, shape, **attrs):
    """
    Vanilla Zarr implementation of ome_zarr.writer.write_multiscales_metadata.
    
    Args:
        group: Zarr group to write metadata to
        datasets: List of dataset specifications or single dataset dict  
        format_version: OME-Zarr format version string
        shape: Shape of the image data
        **attrs: Additional attributes to add to the group
    """
    # Convert datasets to proper format
    if isinstance(datasets, list):
        dataset_list = []
        for d in datasets:
            if isinstance(d, dict) and 'path' in d:
                dataset_list.append({
                    'path': d['path'], 
                    'coordinateTransformations': [{'type': 'identity'}]
                })
            else:
                # Handle legacy format
                dataset_list.append({
                    'path': str(d), 
                    'coordinateTransformations': [{'type': 'identity'}]
                })
    else:
        # Single dataset case
        path = datasets['path'] if isinstance(datasets, dict) else str(datasets)
        dataset_list = [{
            'path': path, 
            'coordinateTransformations': [{'type': 'identity'}]
        }]
    
    # Create OME-Zarr multiscales metadata
    multiscales = [{
        'version': format_version,
        'datasets': dataset_list,
        'axes': [
            {'name': 'y', 'type': 'space'},
            {'name': 'x', 'type': 'space'}
        ]
    }]
    
    group.attrs['multiscales'] = multiscales
    
    # Add any additional attributes
    for key, value in attrs.items():
        group.attrs[key] = value


def format_from_version(version):
    """
    Vanilla Zarr implementation of ome_zarr.format.format_from_version.
    
    Args:
        version: Version string (e.g., "0.2", "0.4")
        
    Returns:
        Version string (passthrough for vanilla implementation)
    """
    return version