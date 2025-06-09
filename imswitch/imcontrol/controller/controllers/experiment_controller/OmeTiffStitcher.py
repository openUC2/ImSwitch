import time
import threading
import numpy as np
import tifffile
from collections import deque
import os
class OmeTiffStitcher:
    def __init__(self, file_path, bigtiff=True):
        """
        file_path: Where to write the OME-TIFF
        bigtiff:   Whether to use bigtiff=True (recommended if large or many images)
        """
        self.file_path = file_path
        self.bigtiff = bigtiff
        self.queue = deque()       # Holds (image_array, metadata_dict)
        self.lock = threading.Lock()
        self.is_running = False
        self._thread = None

    def start(self):
        """Begin the background thread that writes images to disk as they arrive."""
        self.is_running = True
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the thread to stop, then join it."""
        self.is_running = False
        if self._thread is not None:
            self._thread.join()

    def add_image(self, image, position_x, position_y, index_x, index_y, pixel_size):
        """
        Enqueue an image for writing.
        :param image: 2D or 3D NumPy array (e.g. grayscale or color).
        :param position_x: stage X coordinate in microns
        :param position_y: stage Y coordinate in microns
        :param index_x:   tile index X (used for some readers)
        :param index_y:   tile index Y
        :param pixel_size: pixel size in microns
        """
        # A minimal OME-like metadata block that Fiji can often interpret.
        # The "Plane" section stores stage position; "Pixels" sets physical pixel size.
        metadata = {
            "Pixels": {
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            },
            "Plane": {
                "PositionX": position_x,
                "PositionY": position_y,
                "IndexX": index_x,
                "IndexY": index_y
            },
        }
        with self.lock:
            self.queue.append((image, metadata))

    def _process_queue(self):
        """
        Background loop: open the OME-TIFF in append mode, pop images from queue,
        and write them with embedded metadata.
        """
        # ensure the folder exists if it does not create it
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff, append=True) as tif:
            # Keep running until stop() is called AND the queue is empty
            while self.is_running or len(self.queue) > 0:
                with self.lock:
                    if self.queue:
                        image, metadata = self.queue.popleft()
                    else:
                        image = None

                if image is not None:
                    # Each call writes a new series/plane in append mode.
                    try:
                        tif.write(data=image, metadata=metadata)
                    except Exception as e:
                        print(f"Error writing image: {e}")
                else:
                    # Sleep briefly to avoid busy loop when queue is empty
                    time.sleep(0.01)

    def close(self):
        """Close the OME-TIFF file. Not strictly necessary if using stop()."""
        self.stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.is_running = False
        self.queue.clear()
        self.lock = None
        self._thread = None
        self.queue = None
        self.file_path = None