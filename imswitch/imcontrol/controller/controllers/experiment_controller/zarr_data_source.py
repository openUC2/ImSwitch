# zarr_data_source.py

import numpy as np
import zarr
import zarr.storage
from imswitch.imcontrol.controller.controllers.experiment_controller.metadata import MinimalMetadata




GROUP_PREFIX = "Pos_"

class MinimalZarrDataSource:
    """
    Writes frames to an OME-Zarr store:
     - set_metadata_from_configuration_experiment(config)
       picks out T, C, Z, X, Y, positions, etc.
     - new_position() for each position, creates arrays for multi-resolution
     - write() appends a new 2D frame into the right (t,c,z) slice
    """

    def __init__(self, file_name: str, mode: str = "w"):
        self.file_name = file_name
        self.mode = mode
        self._store = None
        self.image = None

        # placeholders for shape
        self.shape_t = 1
        self.shape_c = 1
        self.shape_z = 1
        self.shape_y = 64
        self.shape_x = 64
        self.per_stack = True
        self.dtype = np.uint16

        # multi-resolution arrays
        self.shapes = []
        self.resolutions = []

        self.metadata = None

        # Bookkeeping for scanning
        self._current_position = -1
        self._current_frame = 0

    def set_metadata_from_configuration_experiment(self, config: dict):
        """
        Read T, C, Z, X, Y, positions from 'config'.
        Then define shapes for multi-res, plus create 'MinimalMetadata'.
        """
        ms = config["experiment"]["MicroscopeState"]
        self.shape_z = ms["number_z_steps"]
        self.shape_t = ms["timepoints"]

        channels = [ck for ck, cinfo in ms["channels"].items() if cinfo["is_selected"]]
        self.shape_c = len(channels)

        cam_key = ms["microscope_name"]
        cp = config["experiment"]["CameraParameters"][cam_key]
        self.shape_x = cp["x_pixels"]
        self.shape_y = cp["y_pixels"]

        # do we let z vary faster or c vary faster
        self.per_stack = ms.get("stack_cycling_mode", "per_stack") == "per_stack"

        # Example: 2-level pyramid
        shape0 = (self.shape_z, self.shape_y, self.shape_x)
        shape1 = (self.shape_z, max(1, self.shape_y // 2), max(1, self.shape_x // 2))
        self.shapes = [shape0, shape1]

        # each resolution => (dz,dy,dx)
        self.resolutions = [(1,1,1), (1,2,2)]

        self.metadata = MinimalMetadata(per_stack=self.per_stack)

    def _setup_store(self):
        """
        Create a FSStore with dimension_separator='/' for OME-Zarr 0.4,
        then top-level group.
        """
        store = zarr.storage.FSStore(
            self.file_name, mode=self.mode, dimension_separator="/"
        )
        self.image = zarr.group(store=store, overwrite=True)
        self._store = store
        self.image.attrs["description"] = "OME-Zarr from MinimalZarrDataSource"

    def new_position(self, pos_index: int, **kw):
        """
        For each multi-res level, create a (t,c,z,y,x) array,
        then append an entry to 'multiscales'.
        """
        name = f"{GROUP_PREFIX}{pos_index}"
        paths = []
        for i, zyx in enumerate(self.shapes):
            shape_5d = (self.shape_t, self.shape_c, *zyx)
            arr_name = f"{name}_{i}"

            arr = self.image.create(
                name=arr_name,
                shape=shape_5d,
                chunks=(1,1) + zyx,  # chunk T,C => store entire Z,Y,X in one chunk
                dtype=self.dtype,
            )
            paths.append(arr.path)
            arr.attrs["_ARRAY_DIMENSIONS"] = ["t","c","z","y","x"]

        ms_list = self.image.attrs.get("multiscales", [])
        ms_entry = self.metadata.multiscales_dict(
            name, paths, self.resolutions, view=kw.get("view","")
        )
        ms_list.append(ms_entry)
        self.image.attrs["multiscales"] = ms_list

    def _cztp_indices(self, frame_id: int):
        """Map overall frame index -> c,z,t,position, depending on per_stack."""
        if self.per_stack:
            c = (frame_id // self.shape_z) % self.shape_c
            z = frame_id % self.shape_z
        else:
            c = frame_id % self.shape_c
            z = (frame_id // self.shape_c) % self.shape_z
        t = (frame_id // (self.shape_c*self.shape_z)) % self.shape_t
        p = frame_id // (self.shape_c*self.shape_z*self.shape_t)
        return c,z,t,p

    def write(self, data: np.ndarray, x=None, y=None, z=None, theta=None, f=None):
        """
        Writes a new 2D image to the next (t,c,z,p). If we detect a new position p,
        call new_position(...). Then store a downsample for each resolution.
        """
        if self._store is None:
            self._setup_store()

        c,zslice,t,p = self._cztp_indices(self._current_frame)

        if p != self._current_position:
            self._current_position = p
            self.new_position(p, x=x, y=y, z=z, theta=theta, f=f)

        # store in each resolution
        for i, (dz, dy, dx) in enumerate(self.resolutions):
            arr_name = f"{GROUP_PREFIX}{p}_{i}"
            # compute which z-plane to store
            zs = zslice // dz
            ds_data = data[::dy, ::dx]
            self.image[arr_name][t,c,zs,:,:] = ds_data.astype(self.dtype)

        self._current_frame += 1

    def close(self):
        """Close the store if needed."""
        if self._store is not None:
            self._store.close()
            self._store = None
