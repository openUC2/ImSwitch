# single_multiscale_zarr_data_source.py
import numpy as np
import zarr
import zarr.storage

class SingleMultiscaleZarrWriter:
    """
    A single "multiscale" image approach. We keep a single top-level group: "0",
    inside it a dataset "0" with shape = (t, c, z, bigY, bigX).
    We fill that bounding box on-the-fly, placing each tile at its region.

    For example usage:
       1) set_metadata(...): define total T, C, Z, plus your bounding box (bigY,bigX).
       2) open_store(): create '0/0' dataset with shape=(T, C, Z, bigY, bigX).
       3) write(tile, t, c, z, yStart, yEnd, xStart, xEnd).
       4) close().
    """

    def __init__(self, file_name: str, mode: str = "w"):
        self.file_name = file_name
        self.mode = mode

        self._store = None
        self.root_group = None
        self.dataset = None  # reference to the single dataset "/0/0"

        # placeholders for final shape
        self.shape_t = 1
        self.shape_c = 1
        self.shape_z = 1
        self.bigY = 1
        self.bigX = 1
        self.dtype = np.uint16

        # If your scanning logic has partial or no knowledge of T/C/Z,
        # you can track them dynamically. But for simplicity, we assume
        # you know them up front.
        self.axes = ["t", "c", "z", "y", "x"]  # standard for OME-Zarr 0.4

    def set_metadata(
        self,
        t: int,
        c: int,
        z: int,
        bigY: int,
        bigX: int,
        dtype = np.uint16
    ):
        """
        Let the user define final bounding-box shape: (T, C, Z, bigY, bigX).
        We store them and create the dataset in open_store().
        """
        self.shape_t = t
        self.shape_c = c
        self.shape_z = z
        self.bigY = bigY
        self.bigX = bigX
        self.dtype = dtype

    def open_store(self):
        """
        Create the FSStore with dimension_separator='/', create group '0'
        with a single dataset '0' shaped (t, c, z, bigY, bigX).
        Updated for Zarr 3.0 compatibility.

        We'll define a minimal 'multiscales' attribute so napari recognizes
        it as one image. There's only one scale, so there's just one item
        in 'datasets'.
        """
        # Zarr 3.0 compatibility for FSStore
        if hasattr(zarr.storage, 'FSStore'):
            # Try to use FSStore with Zarr 3.0 API first
            try:
                store = zarr.storage.FSStore(self.file_name, mode=self.mode, dimension_separator="/")
            except TypeError:
                # Fallback for Zarr 3.0 where FSStore API might have changed
                store = zarr.storage.FSStore(self.file_name, mode=self.mode)
        else:
            # Direct path usage for newer Zarr versions
            store = self.file_name
        
        self.root_group = zarr.group(store=store, overwrite=True)
        self._store = store

        # Create a sub-group named '0' (the top-level multiscale image)
        group_0 = self.root_group.create_group("0", overwrite=True)

        shape_5d = (int(self.shape_t), int(self.shape_c), int(self.shape_z), int(self.bigY), int(self.bigX))

        # dataset name is '0' under group '0'
        ds = group_0.create_dataset(
            "0",
            shape=shape_5d,
            chunks=(1,1,1,512,512),  # or however you want chunking
            dtype=self.dtype,
        )
        ds.attrs["_ARRAY_DIMENSIONS"] = self.axes

        # create minimal 'multiscales' in the group_0 .zattrs
        multiscales = [
            {
                "version": "0.4",
                "axes": self.axes,
                "datasets": [
                    {
                        "path": "0",
                        # if you want a transform:
                        # "coordinateTransformations": [{"type": "translation", "translation": [0,0,0,0,0]}]
                    }
                ],
            }
        ]
        group_0.attrs["multiscales"] = multiscales

        self.dataset = ds

    def write_tile(
        self,
        tile: np.ndarray,
        t: int,
        c: int,
        z: int,
        y_start: int,
        x_start: int
    ):
        """
        Writes the tile (2D) at [t, c, z, y_start:y_end, x_start:x_end].
        'tile.shape' => (tileY, tileX).
        We do no checks if it fits in the bounding box, so make sure you computed them.

        If you do a Z-stack or time-lapse, you can index them here with t,c,z
        or just fix them to 0 if you want.
        """
        if self.dataset is None:
            self.open_store()

        tileY, tileX = tile.shape
        y_end = int(y_start + tileY)
        x_end = int(x_start + tileX)

        self.dataset[t, c, z, y_start:y_end, x_start:x_end] = tile.astype(self.dtype)

    def close(self):
        """Close the store if needed."""
        if self._store:
            # Only close if it's a store object with close method
            if hasattr(self._store, 'close'):
                self._store.close()
            self._store = None
