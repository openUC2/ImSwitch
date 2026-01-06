"""
Zarr Export Utility for OME-Zarr files.

This module provides utilities for exploring and exporting data from OME-Zarr files.
Users can list available channels, coordinates, timepoints, and extract specific slices.

Usage:
    from zarr_export import ZarrExporter
    
    # Open a zarr file
    exporter = ZarrExporter("/path/to/experiment.ome.zarr")
    
    # Get info about the dataset
    info = exporter.get_info()
    print(info)
    
    # Extract a specific slice
    slice_data = exporter.extract_slice(t=0, c=0, z=5)
    
    # Export a time series for one channel
    exporter.export_tiff_series(output_dir="/path/to/output", channel=0)
"""

import os
import json
import zarr
import numpy as np
import tifffile as tif
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple


@dataclass
class DatasetInfo:
    """Information about an OME-Zarr dataset."""
    path: str
    shape: Tuple[int, ...]  # t, c, z, y, x
    n_timepoints: int
    n_channels: int
    n_z_planes: int
    height: int
    width: int
    dtype: str
    pixel_size_x: float
    pixel_size_y: float
    pixel_size_z: float
    time_interval: float
    x_start: float
    y_start: float
    z_start: float
    channel_names: List[str]
    channel_colors: List[str]
    pyramid_levels: int

    def __str__(self):
        return f"""OME-Zarr Dataset Info:
  Path: {self.path}
  Shape (t,c,z,y,x): {self.shape}
  Timepoints: {self.n_timepoints}
  Channels: {self.n_channels} ({', '.join(self.channel_names)})
  Z-planes: {self.n_z_planes}
  Image size: {self.width} x {self.height}
  Pixel size XY: {self.pixel_size_x} µm
  Pixel size Z: {self.pixel_size_z} µm
  Time interval: {self.time_interval} s
  Physical origin: X={self.x_start}, Y={self.y_start}, Z={self.z_start} µm
  Pyramid levels: {self.pyramid_levels}
  Data type: {self.dtype}"""


class ZarrExporter:
    """
    Export utility for OME-Zarr files.
    
    Provides methods to explore and extract data from OME-Zarr datasets.
    """

    def __init__(self, zarr_path: str):
        """
        Initialize the exporter with a path to an OME-Zarr file.
        
        Args:
            zarr_path: Path to the .ome.zarr directory
        """
        self.zarr_path = zarr_path
        self.root = zarr.open_group(zarr_path, mode="r")
        self._parse_metadata()

    def _parse_metadata(self):
        """Parse OME-NGFF metadata from the zarr store."""
        # Get multiscales metadata
        self.multiscales = self.root.attrs.get("multiscales", [{}])[0]
        self.axes = self.multiscales.get("axes", [])
        self.datasets = self.multiscales.get("datasets", [])

        # Get omero metadata for channels
        self.omero = self.root.attrs.get("omero", {})
        self.channels = self.omero.get("channels", [])

        # Get the base resolution array (level 0)
        self.base_array = self.root["0"]
        self.shape = self.base_array.shape

        # Parse coordinate transformations
        self._parse_transforms()

    def _parse_transforms(self):
        """Parse coordinate transformations from metadata."""
        # Default values
        self.pixel_size_x = 1.0
        self.pixel_size_y = 1.0
        self.pixel_size_z = 1.0
        self.time_interval = 1.0
        self.x_start = 0.0
        self.y_start = 0.0
        self.z_start = 0.0

        # Try to get from dataset-level transforms
        if self.datasets:
            transforms = self.datasets[0].get("coordinateTransformations", [])
            for transform in transforms:
                if transform.get("type") == "scale":
                    scale = transform.get("scale", [1, 1, 1, 1, 1])
                    if len(scale) >= 5:
                        self.time_interval = scale[0]
                        self.pixel_size_z = scale[2]
                        self.pixel_size_y = scale[3]
                        self.pixel_size_x = scale[4]
                elif transform.get("type") == "translation":
                    translation = transform.get("translation", [0, 0, 0, 0, 0])
                    if len(translation) >= 5:
                        self.z_start = translation[2]
                        self.y_start = translation[3]
                        self.x_start = translation[4]

        # Also check global transforms
        global_transforms = self.multiscales.get("coordinateTransformations", [])
        for transform in global_transforms:
            if transform.get("type") == "scale":
                scale = transform.get("scale", [1, 1, 1, 1, 1])
                if len(scale) >= 5:
                    self.time_interval = scale[0]
                    self.pixel_size_z = scale[2]
                    self.pixel_size_y = scale[3]
                    self.pixel_size_x = scale[4]

    def get_info(self) -> DatasetInfo:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            DatasetInfo dataclass with all dataset metadata
        """
        # Count pyramid levels
        pyramid_levels = len([k for k in self.root.keys() if k.isdigit()])

        # Get channel names and colors
        channel_names = [ch.get("label", f"Channel_{i}") for i, ch in enumerate(self.channels)]
        if not channel_names:
            channel_names = [f"Channel_{i}" for i in range(self.shape[1])]

        channel_colors = [ch.get("color", "FFFFFF") for ch in self.channels]
        if not channel_colors:
            channel_colors = ["FFFFFF"] * self.shape[1]

        return DatasetInfo(
            path=self.zarr_path,
            shape=self.shape,
            n_timepoints=self.shape[0],
            n_channels=self.shape[1],
            n_z_planes=self.shape[2],
            height=self.shape[3],
            width=self.shape[4],
            dtype=str(self.base_array.dtype),
            pixel_size_x=self.pixel_size_x,
            pixel_size_y=self.pixel_size_y,
            pixel_size_z=self.pixel_size_z,
            time_interval=self.time_interval,
            x_start=self.x_start,
            y_start=self.y_start,
            z_start=self.z_start,
            channel_names=channel_names,
            channel_colors=channel_colors,
            pyramid_levels=pyramid_levels
        )

    def list_channels(self) -> List[Dict[str, Any]]:
        """
        List all channels with their metadata.
        
        Returns:
            List of channel dictionaries with name, color, etc.
        """
        if self.channels:
            return self.channels
        else:
            # Return default channel info
            return [{"label": f"Channel_{i}", "color": "FFFFFF"} for i in range(self.shape[1])]

    def get_physical_coordinates(self, t: int = 0, z: int = 0, y_pixel: int = 0, x_pixel: int = 0) -> Dict[str, float]:
        """
        Convert pixel coordinates to physical coordinates.
        
        Args:
            t: Timepoint index
            z: Z-plane index
            y_pixel: Y pixel coordinate
            x_pixel: X pixel coordinate
            
        Returns:
            Dictionary with physical coordinates in micrometers/seconds
        """
        return {
            "time_s": t * self.time_interval,
            "x_um": self.x_start + x_pixel * self.pixel_size_x,
            "y_um": self.y_start + y_pixel * self.pixel_size_y,
            "z_um": self.z_start + z * self.pixel_size_z
        }

    def extract_slice(self, t: int = 0, c: int = 0, z: int = 0, level: int = 0) -> np.ndarray:
        """
        Extract a single 2D slice from the dataset.
        
        Args:
            t: Timepoint index
            c: Channel index
            z: Z-plane index
            level: Pyramid level (0 = full resolution)
            
        Returns:
            2D numpy array
        """
        array = self.root[str(level)]
        return np.array(array[t, c, z, :, :])

    def extract_volume(self, t: int = 0, c: int = 0, level: int = 0) -> np.ndarray:
        """
        Extract a 3D volume (all z-planes) for a specific timepoint and channel.
        
        Args:
            t: Timepoint index
            c: Channel index
            level: Pyramid level (0 = full resolution)
            
        Returns:
            3D numpy array (z, y, x)
        """
        array = self.root[str(level)]
        return np.array(array[t, c, :, :, :])

    def extract_time_series(self, c: int = 0, z: int = 0, level: int = 0) -> np.ndarray:
        """
        Extract a time series for a specific channel and z-plane.
        
        Args:
            c: Channel index
            z: Z-plane index
            level: Pyramid level (0 = full resolution)
            
        Returns:
            3D numpy array (t, y, x)
        """
        array = self.root[str(level)]
        return np.array(array[:, c, z, :, :])

    def extract_region(self, t: int = 0, c: int = 0, z: int = 0,
                       y_start: int = 0, y_end: int = None,
                       x_start: int = 0, x_end: int = None,
                       level: int = 0) -> np.ndarray:
        """
        Extract a region of interest from the dataset.
        
        Args:
            t, c, z: Dimension indices
            y_start, y_end: Y range (pixels)
            x_start, x_end: X range (pixels)
            level: Pyramid level
            
        Returns:
            2D numpy array
        """
        array = self.root[str(level)]
        if y_end is None:
            y_end = array.shape[3]
        if x_end is None:
            x_end = array.shape[4]
        return np.array(array[t, c, z, y_start:y_end, x_start:x_end])

    def export_tiff(self, output_path: str, t: int = 0, c: int = 0, z: int = 0,
                    level: int = 0, compression: str = "zlib"):
        """
        Export a single slice to a TIFF file.
        
        Args:
            output_path: Path for the output TIFF
            t, c, z: Dimension indices
            level: Pyramid level
            compression: TIFF compression method
        """
        data = self.extract_slice(t, c, z, level)

        # Create OME metadata
        info = self.get_info()
        metadata = {
            'Pixels': {
                'PhysicalSizeX': info.pixel_size_x,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': info.pixel_size_y,
                'PhysicalSizeYUnit': 'µm',
            }
        }

        tif.imwrite(output_path, data, compression=compression, metadata=metadata)
        print(f"Exported slice to {output_path}")

    def export_z_stack(self, output_path: str, t: int = 0, c: int = 0,
                       level: int = 0, compression: str = "zlib"):
        """
        Export a z-stack to a TIFF file.
        
        Args:
            output_path: Path for the output TIFF
            t: Timepoint index
            c: Channel index
            level: Pyramid level
            compression: TIFF compression method
        """
        data = self.extract_volume(t, c, level)

        info = self.get_info()
        metadata = {
            'Pixels': {
                'PhysicalSizeX': info.pixel_size_x,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': info.pixel_size_y,
                'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': info.pixel_size_z,
                'PhysicalSizeZUnit': 'µm',
            }
        }

        tif.imwrite(output_path, data, compression=compression, metadata=metadata)
        print(f"Exported z-stack ({data.shape[0]} planes) to {output_path}")

    def export_tiff_series(self, output_dir: str, channel: int = None,
                           timepoint: int = None, z_plane: int = None,
                           level: int = 0, compression: str = "zlib"):
        """
        Export multiple slices to a series of TIFF files.
        
        Args:
            output_dir: Directory for output files
            channel: Specific channel (None = all channels)
            timepoint: Specific timepoint (None = all timepoints)
            z_plane: Specific z-plane (None = all z-planes)
            level: Pyramid level
            compression: TIFF compression
        """
        os.makedirs(output_dir, exist_ok=True)
        info = self.get_info()

        # Determine ranges
        t_range = [timepoint] if timepoint is not None else range(info.n_timepoints)
        c_range = [channel] if channel is not None else range(info.n_channels)
        z_range = [z_plane] if z_plane is not None else range(info.n_z_planes)

        count = 0
        for t in t_range:
            for c in c_range:
                for z in z_range:
                    channel_name = info.channel_names[c] if c < len(info.channel_names) else f"ch{c}"
                    filename = f"t{t:04d}_c{c}_{channel_name}_z{z:04d}.tif"
                    filepath = os.path.join(output_dir, filename)

                    data = self.extract_slice(t, c, z, level)
                    tif.imwrite(filepath, data, compression=compression)
                    count += 1

        print(f"Exported {count} TIFF files to {output_dir}")

    def export_ome_tiff(self, output_path: str, level: int = 0,
                        channels: List[int] = None, timepoints: List[int] = None,
                        z_planes: List[int] = None):
        """
        Export data to a single OME-TIFF file with full metadata.
        
        Args:
            output_path: Path for output OME-TIFF
            level: Pyramid level
            channels: List of channel indices (None = all)
            timepoints: List of timepoint indices (None = all)
            z_planes: List of z-plane indices (None = all)
        """
        info = self.get_info()
        array = self.root[str(level)]

        # Determine ranges
        t_list = timepoints if timepoints is not None else list(range(info.n_timepoints))
        c_list = channels if channels is not None else list(range(info.n_channels))
        z_list = z_planes if z_planes is not None else list(range(info.n_z_planes))

        # Extract data
        data = np.array(array[t_list, :, :, :, :][:, c_list, :, :, :][:, :, z_list, :, :])

        # Build OME-XML metadata
        channel_names = [info.channel_names[c] for c in c_list]

        # Write OME-TIFF
        with tif.TiffWriter(output_path, bigtiff=True) as tiff:
            tiff.write(
                data,
                photometric='minisblack',
                metadata={
                    'axes': 'TCZYX',
                    'Channel': {'Name': channel_names},
                    'Pixels': {
                        'PhysicalSizeX': info.pixel_size_x,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': info.pixel_size_y,
                        'PhysicalSizeYUnit': 'µm',
                        'PhysicalSizeZ': info.pixel_size_z,
                        'PhysicalSizeZUnit': 'µm',
                    }
                }
            )

        print(f"Exported OME-TIFF ({data.shape}) to {output_path}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Export dataset information to a dictionary.
        
        Returns:
            Dictionary with all metadata
        """
        info = self.get_info()
        return {
            "path": info.path,
            "shape": list(info.shape),
            "dimensions": {
                "timepoints": info.n_timepoints,
                "channels": info.n_channels,
                "z_planes": info.n_z_planes,
                "height": info.height,
                "width": info.width
            },
            "physical_size": {
                "pixel_x_um": info.pixel_size_x,
                "pixel_y_um": info.pixel_size_y,
                "pixel_z_um": info.pixel_size_z,
                "time_interval_s": info.time_interval
            },
            "origin": {
                "x_um": info.x_start,
                "y_um": info.y_start,
                "z_um": info.z_start
            },
            "channels": [
                {"index": i, "name": name, "color": color}
                for i, (name, color) in enumerate(zip(info.channel_names, info.channel_colors))
            ],
            "pyramid_levels": info.pyramid_levels,
            "dtype": info.dtype
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Export dataset information to JSON string.
        
        Args:
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_info(self, output_path: str):
        """
        Save dataset information to a JSON file.
        
        Args:
            output_path: Path for output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved dataset info to {output_path}")


def main():
    """Command-line interface for the Zarr exporter."""
    import argparse

    parser = argparse.ArgumentParser(description="OME-Zarr Export Utility")
    parser.add_argument("zarr_path", help="Path to the .ome.zarr directory")
    parser.add_argument("--info", action="store_true", help="Print dataset info")
    parser.add_argument("--json", action="store_true", help="Output info as JSON")
    parser.add_argument("--export-dir", help="Export all slices to this directory")
    parser.add_argument("--export-tiff", help="Export single slice to TIFF")
    parser.add_argument("--export-ome-tiff", help="Export to OME-TIFF")
    parser.add_argument("-t", type=int, default=0, help="Timepoint index")
    parser.add_argument("-c", type=int, default=0, help="Channel index")
    parser.add_argument("-z", type=int, default=0, help="Z-plane index")
    parser.add_argument("--level", type=int, default=0, help="Pyramid level")

    args = parser.parse_args()

    exporter = ZarrExporter(args.zarr_path)

    if args.info:
        print(exporter.get_info())

    if args.json:
        print(exporter.to_json())

    if args.export_tiff:
        exporter.export_tiff(args.export_tiff, t=args.t, c=args.c, z=args.z, level=args.level)

    if args.export_dir:
        exporter.export_tiff_series(args.export_dir, level=args.level)

    if args.export_ome_tiff:
        exporter.export_ome_tiff(args.export_ome_tiff, level=args.level)


if __name__ == "__main__":
    main()
