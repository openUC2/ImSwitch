import os
import cv2
import math
import time
from imswitch import __file__
import threading
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue, Empty

from skimage.draw import line
from scipy.signal import convolve2d
from imswitch.imcommon.model import initLogger

try:
    import NanoImagingPack as nip

    IS_NIP = True
except:
    IS_NIP = False


def _extract_image(image, shape):
    if IS_NIP:
        return nip.extract(image, shape)
    height, width = shape
    img_h, img_w = image.shape[:2]
    start_y = max((img_h - height) // 2, 0)
    start_x = max((img_w - width) // 2, 0)
    end_y = start_y + height
    end_x = start_x + width
    cropped = image[start_y:end_y, start_x:end_x]
    if cropped.shape[0] == height and cropped.shape[1] == width:
        return cropped
    out = np.zeros((height, width), dtype=image.dtype)
    out[: cropped.shape[0], : cropped.shape[1]] = cropped
    return out

# Makes sure code still executes without numba, albeit extremely slow
try:
    from numba import njit, prange
except ModuleNotFoundError:
    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper

"""
End-to-end astigmatism autofocus simulation:
- Simulated microscope with Z-scan, rotated astigmatism, and XY drift
- Rotation-invariant second-moment focus metric (fast, no fits)
- Plots + saves stack (NPZ) and metrics (CSV)
"""

from math import cos, sin

# ----------------------- Simulation -----------------------

class AstigmaticMicroscopeSimulator:
    def __init__(
        self,
        H=128,
        W=128,
        roi_half=28,
        phi_deg=33.0,      # astig axis rotation w.r.t. camera x
        s0=1.7,            # base sigma at nominal focus (px)
        astig_slope=0.33,  # sigma_x = s0 + a*z, sigma_y = s0 - a*z
        amp=2400.0,
        bg=35.0,
        read_noise=0*2.2,
        poisson=0*True,
        seed=7,
    ):
        self.H, self.W = H, W
        self.roi_half = roi_half
        self.phi = np.deg2rad(phi_deg)
        self.s0 = float(s0)
        self.a = float(astig_slope)
        self.amp = float(amp)
        self.bg = float(bg)
        self.read_noise = float(read_noise)
        self.poisson = bool(poisson)
        self.rng = np.random.default_rng(seed)

        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xgrid = x
        self.ygrid = y

        # ROI indices (fixed crop around center; centroiding handles drift)
        cx = W // 2
        cy = H // 2
        r = roi_half
        self.roi_slice = np.s_[cy - r : cy + r, cx - r : cx + r]

    def xy_drift(self, z):
        # linear + sinusoidal drift to stress-test the metric
        dx = 0.25 * z + 1.2 * np.sin(0.8 * z)
        dy = -0.18 * z + 1.0 * np.cos(0.6 * z)
        return dx, dy

    def render_frame(self, z):
        x, y = self.xgrid, self.ygrid
        dx, dy = self.xy_drift(z)
        cx = self.W / 2 + dx
        cy = self.H / 2 + dy

        sx = max(self.s0 + self.a * z, 0.5)
        sy = max(self.s0 - self.a * z, 0.5)

        # rotate coords by phi (principal axes of astigmatism)
        xp = (x - cx) * cos(self.phi) + (y - cy) * sin(self.phi)
        yp = -(x - cx) * sin(self.phi) + (y - cy) * cos(self.phi)

        g = np.exp(-0.5 * ((xp / sx) ** 2 + (yp / sy) ** 2))
        I = self.amp * g + self.bg

        if self.poisson:
            I = self.rng.poisson(I).astype(np.float32)
        else:
            I = I.astype(np.float32)

        I += self.rng.normal(0, self.read_noise, I.shape).astype(np.float32)
        return np.clip(I, 0, None)



class VirtualMicroscopeManager:
    """A low-level wrapper for TCP-IP communication (ESP32 REST API)
       with added objective control that toggles the objective lens.
       Toggling the objective will double the image magnification by
       binning the pixels (2x2 binning).
    """

    def __init__(self, rs232Info, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self._settings = rs232Info.managerProperties
        self._name = name

        try:
            self._imagePath = rs232Info.managerProperties["imagePath"]
            if self._imagePath not in ["simplant", "smlm", "astigmatism", "wellplatecalib", "april_tag"]:
                raise NameError
        except:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            self._imagePath = os.path.join(
                package_dir, "_data/images/histoASHLARStitch.jpg"
            )
            self.__logger.info(
                "If you want to use the plant, use 'imagePath': 'simplant', 'astigmatism' in your setup.json"
            )

        self._virtualMicroscope = VirtualMicroscopy(self._imagePath)
        self._positioner = self._virtualMicroscope.positioner
        self._camera = self._virtualMicroscope.camera
        self._illuminator = self._virtualMicroscope.illuminator
        self._objective = self._virtualMicroscope.objective

        # Focus surface tilt simulation – makes effective focus depend on XY
        # Configure via managerProperties in the setup JSON, e.g.:
        #   "focusSurface": { "tiltX": 0.01, "tiltY": 0.005, "curvature": 0.0001 }
        focus_surface = self._settings.get("focusSurface", {})
        self._positioner.set_focus_surface(
            tilt_x=focus_surface.get("tiltX", 0.0),
            tilt_y=focus_surface.get("tiltY", 0.0),
            curvature=focus_surface.get("curvature", 0.0),
            offset=focus_surface.get("offset", 0.0),
        )
        if any(v != 0 for v in [focus_surface.get("tiltX", 0), focus_surface.get("tiltY", 0), focus_surface.get("curvature", 0)]):
            self.__logger.info(
                f"Focus surface tilt enabled: tiltX={focus_surface.get('tiltX', 0)}, "
                f"tiltY={focus_surface.get('tiltY', 0)}, "
                f"curvature={focus_surface.get('curvature', 0)}"
            )

        # Initialize objective state: 1 (default) => no binning, 2 => binned image (2x magnification)
        self.currentObjective = 1
        self._camera.binning = False

    def toggleObjective(self):
        """
        Toggle the objective lens.
        When toggled, the virtual objective move is simulated,
        and the image magnification is changed by binning the pixels.
        """
        if self.currentObjective == 1:
            # Move to objective 2: simulate move and apply 2x binning
            self.__logger.info("Switching to Objective 2: Applying 2x binning")
            # Here one could call a REST API endpoint like:
            # /ObjectiveController/moveToObjective?slot=2
            self.currentObjective = 2
            self._camera.binning = True
        else:
            # Move back to objective 1: remove binning
            self.__logger.info("Switching to Objective 1: Removing binning")
            # Here one could call a REST API endpoint like:
            # /ObjectiveController/moveToObjective?slot=1
            self.currentObjective = 1
            self._camera.binning = False

    def finalize(self):
        self._virtualMicroscope.stop()



class Positioner:
    def __init__(self, parent):
        self._parent = parent
        self.position = {"X": 0, "Y": 0, "Z": 0, "A": 0}
        self.mDimensions = (self._parent.camera.SensorHeight, self._parent.camera.SensorWidth)
        self.lock = threading.Lock()

        # Focus surface tilt parameters (set via VirtualMicroscopeManager)
        self._focus_tilt_x = 0.0   # dZ/dX slope (µm focus per µm X)
        self._focus_tilt_y = 0.0   # dZ/dY slope
        self._focus_curvature = 0.0  # Quadratic bowl term
        self._focus_offset = 0.0   # Constant offset

        if IS_NIP:
            self.psf = self.compute_psf(dz=0)
        else:
            self.psf = None

    def set_focus_surface(self, tilt_x: float = 0.0, tilt_y: float = 0.0,
                          curvature: float = 0.0, offset: float = 0.0):
        """
        Configure XY-dependent focus surface for realistic simulation.

        The effective defocus at position (x, y) is:
            dz_surface = tilt_x * x + tilt_y * y + curvature * (x² + y²) + offset

        The total defocus seen by the camera is:
            dz_effective = Z_stage - dz_surface(X, Y)

        When Z_stage == dz_surface → in focus (dz_effective == 0).

        Args:
            tilt_x: Linear tilt in X (units: Z per X, e.g. 0.01 means 10µm defocus per 1000µm X)
            tilt_y: Linear tilt in Y
            curvature: Quadratic bowl curvature (positive = concave up)
            offset: Constant Z offset
        """
        self._focus_tilt_x = tilt_x
        self._focus_tilt_y = tilt_y
        self._focus_curvature = curvature
        self._focus_offset = offset

    def get_focus_surface_z(self, x: float, y: float) -> float:
        """
        Get the ideal focus Z for a given XY position.

        Returns the Z value at which the sample is in focus at (x, y).
        """
        return (
            self._focus_tilt_x * x +
            self._focus_tilt_y * y +
            self._focus_curvature * (x * x + y * y) +
            self._focus_offset
        )

    def get_effective_defocus(self) -> float:
        """
        Compute effective defocus accounting for the focus surface.

        Returns defocus = Z_stage - ideal_focus_Z(X, Y).
        When this is 0, the image is in perfect focus.
        """
        pos = self.position
        ideal_z = self.get_focus_surface_z(pos["X"], pos["Y"])
        return pos["Z"] - ideal_z

    def move(self, x=None, y=None, z=None, a=None, is_absolute=False):
        with self.lock:
            if is_absolute:
                if x is not None:
                    self.position["X"] = x
                if y is not None:
                    self.position["Y"] = y
                if z is not None:
                    self.position["Z"] = z
                if a is not None:
                    self.position["A"] = a
            else:
                if x is not None:
                    self.position["X"] += x
                if y is not None:
                    self.position["Y"] += y
                if z is not None:
                    self.position["Z"] += z
                if a is not None:
                    self.position["A"] += a

            # Recompute PSF using effective defocus (accounts for focus surface)
            effective_dz = self.get_effective_defocus()
            self.compute_psf(effective_dz)

    def get_position(self):
        with self.lock:
            return self.position.copy()

    def compute_psf(self, dz):
        dz = np.float32(dz)
        # print("Defocus:" + str(dz))
        if IS_NIP and dz != 0:
            obj = nip.image(np.zeros(self.mDimensions))
            obj.pixelsize = (100.0, 100.0) # TODO: adjust based on objective
            paraAbber = nip.PSF_PARAMS()
            paraAbber.aberration_types = [paraAbber.aberration_zernikes.spheric]
            paraAbber.aberration_strength = [np.float32(dz / 10.0)]  # scale factor for defocus
            psf = nip.psf(obj, paraAbber)
            self.psf = psf.copy()
            del psf
            del obj
        else:
            self.psf = None

    def get_psf(self):
        return self.psf


class Illuminator:
    def __init__(self, parent):
        self._parent = parent
        self.intensity = 0
        self.lock = threading.Lock()

    def set_intensity(self, channel=1, intensity=0):
        with self.lock:
            self.intensity = intensity

    def get_intensity(self, channel):
        with self.lock:
            return self.intensity


class VirtualMicroscopy:
    def __init__(self, filePath="path_to_image.jpeg"):
        self.camera = Camera(self, filePath)
        self.positioner = Positioner(self)
        self.illuminator = Illuminator(self)
        self.objective = Objective(self)
        self.galvo = VirtualGalvoScanner(self)  # Virtual galvo scanner for testing

    def startAcquisition(self):
        """Start continuous frame acquisition"""
        self.camera.startAcquisition()

    def stopAcquisition(self):
        """Stop continuous frame acquisition"""
        self.camera.stopAcquisition()

    def stop(self):
        """Stop all operations and clean up"""
        self.camera.stopAcquisition()


@njit(parallel=True)
def FromLoc2Image_MultiThreaded(
    xc_array: np.ndarray, yc_array: np.ndarray, photon_array: np.ndarray,
    sigma_array: np.ndarray, image_height: int, image_width: int, pixel_size: float
):
    Image = np.zeros((image_height, image_width))
    for ij in prange(image_height * image_width):
        j = int(ij / image_width)
        i = ij - j * image_width
        for xc, yc, photon, sigma in zip(xc_array, yc_array, photon_array, sigma_array):
            if (photon > 0) and (sigma > 0):
                S = sigma * math.sqrt(2)
                x = i * pixel_size - xc
                y = j * pixel_size - yc
                if (x + pixel_size / 2) ** 2 + (y + pixel_size / 2) ** 2 < 16 * sigma**2:
                    ErfX = math.erf((x + pixel_size) / S) - math.erf(x / S)
                    ErfY = math.erf((y + pixel_size) / S) - math.erf(y / S)
                    Image[j][i] += 0.25 * photon * ErfX * ErfY
    return Image


def binary2locs(img: np.ndarray, density: float):
    all_locs = np.nonzero(img == 1)
    n_points = int(len(all_locs[0]) * density)
    selected_idx = np.random.choice(len(all_locs[0]), n_points, replace=False)
    filtered_locs = all_locs[0][selected_idx], all_locs[1][selected_idx]
    return filtered_locs


def createBranchingTree(width=5000, height=5000, lineWidth=3):
    np.random.seed(0)
    image = np.ones((height, width), dtype=np.uint8) * 255

    def draw_vessel(start, end, image):
        rr, cc = line(start[0], start[1], end[0], end[1])
        try:
            image[rr, cc] = 0
        except:
            return

    def draw_tree(start, angle, length, depth, image, reducer, max_angle=40):
        if depth == 0:
            return
        end = (int(start[0] + length * np.sin(np.radians(angle))),
               int(start[1] + length * np.cos(np.radians(angle))))
        draw_vessel(start, end, image)
        angle += np.random.uniform(-10, 10)
        new_length = length * reducer
        new_depth = depth - 1
        draw_tree(end, angle - max_angle * np.random.uniform(-1, 1), new_length, new_depth, image, reducer)
        draw_tree(end, angle + max_angle * np.random.uniform(-1, 1), new_length, new_depth, image, reducer)

    start_point = (height - 1, width // 2)
    initial_angle = -90
    initial_length = np.max((width, height)) * 0.15
    depth = 7
    reducer = 0.9
    draw_tree(start_point, initial_angle, initial_length, depth, image, reducer)
    rectangle = np.ones((lineWidth, lineWidth))
    from scipy.signal import convolve2d
    image = convolve2d(image, rectangle, mode="same", boundary="fill", fillvalue=0)
    return image


if __name__ == "__main__":
    imagePath = "smlm"
    microscope = VirtualMicroscopy(filePath=imagePath)
    vmManager = VirtualMicroscopeManager(rs232Info=type("RS232", (), {"managerProperties": {"imagePath": "smlm"}})(), name="VirtualScope")
    microscope.illuminator.set_intensity(intensity=1000)

    # Toggle objective to simulate switching and doubling magnification via binning
    vmManager.toggleObjective()
    for i in range(5):
        microscope.positioner.move(
            x=1400 + i * (-200), y=-800 + i * (-10), z=0, is_absolute=True
        )
        frame = microscope.camera.getLast()
        plt.imsave(f"frame_{i}.png", frame)
    cv2.destroyAllWindows()

class Objective:
    def __init__(self, parent):
        self._parent = parent


class VirtualGalvoScanner:
    """
    Virtual Galvo Scanner for testing and simulation.
    Simulates a galvo mirror scanner with DAC range 0-4095.
    """
    
    def __init__(self, parent):
        self._parent = parent
        self._logger = None
        self.lock = threading.Lock()
        
        # Scan configuration (matching GalvoScanConfig parameters)
        self._config = {
            'nx': 256,
            'ny': 256,
            'x_min': 500,
            'x_max': 3500,
            'y_min': 500,
            'y_max': 3500,
            'sample_period_us': 1,
            'frame_count': 0,  # 0 = infinite
            'bidirectional': False
        }
        
        # Scan state
        self._running = False
        self._current_frame = 0
        self._current_line = 0
        self._scan_thread = None
        self._stop_event = threading.Event()
        
        # DAC position (current mirror position)
        self._x_position = 2048  # Center position
        self._y_position = 2048  # Center position
        
    def set_galvo_scan(self, nx=None, ny=None, x_min=None, x_max=None, 
                       y_min=None, y_max=None, sample_period_us=None,
                       frame_count=None, bidirectional=None):
        """Set scan configuration and start scanning."""
        with self.lock:
            # Update configuration
            if nx is not None:
                self._config['nx'] = int(nx)
            if ny is not None:
                self._config['ny'] = int(ny)
            if x_min is not None:
                self._config['x_min'] = max(0, min(4095, int(x_min)))
            if x_max is not None:
                self._config['x_max'] = max(0, min(4095, int(x_max)))
            if y_min is not None:
                self._config['y_min'] = max(0, min(4095, int(y_min)))
            if y_max is not None:
                self._config['y_max'] = max(0, min(4095, int(y_max)))
            if sample_period_us is not None:
                self._config['sample_period_us'] = max(0, int(sample_period_us))
            if frame_count is not None:
                self._config['frame_count'] = max(0, int(frame_count))
            if bidirectional is not None:
                self._config['bidirectional'] = bool(bidirectional)
                
        # Start the scan
        self._start_scan()
        
        return {
            'success': True,
            'config': self.get_config()
        }
    
    def _start_scan(self):
        """Start the virtual scan in a background thread."""
        # Stop any existing scan
        self.stop_galvo_scan()
        
        self._stop_event.clear()
        self._running = True
        self._current_frame = 0
        self._current_line = 0
        
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        
    def _scan_loop(self):
        """Simulated scan loop running in background thread."""
        config = self._config.copy()
        nx = config['nx']
        ny = config['ny']
        x_min = config['x_min']
        x_max = config['x_max']
        y_min = config['y_min']
        y_max = config['y_max']
        sample_period_us = config['sample_period_us']
        frame_count = config['frame_count']
        bidirectional = config['bidirectional']
        
        # Calculate step sizes
        x_step = (x_max - x_min) / max(nx - 1, 1)
        y_step = (y_max - y_min) / max(ny - 1, 1)
        
        # Scan delay (convert µs to seconds, minimum 1µs)
        delay = max(sample_period_us / 1_000_000, 0.000001)
        
        frame = 0
        while not self._stop_event.is_set():
            # Check frame limit
            if frame_count > 0 and frame >= frame_count:
                break
                
            self._current_frame = frame
            
            # Scan each line
            for y_idx in range(ny):
                if self._stop_event.is_set():
                    break
                    
                self._current_line = y_idx
                y_pos = y_min + y_idx * y_step
                self._y_position = int(y_pos)
                
                # Determine scan direction (bidirectional: alternate directions)
                reverse = bidirectional and (y_idx % 2 == 1)
                
                # Scan each pixel in the line
                for x_idx in range(nx):
                    if self._stop_event.is_set():
                        break
                        
                    # Calculate X position
                    if reverse:
                        x_pos = x_max - x_idx * x_step
                    else:
                        x_pos = x_min + x_idx * x_step
                    
                    self._x_position = int(x_pos)
                    
                    # Simulate sample period delay
                    time.sleep(delay)
                    
            frame += 1
            
        self._running = False
        
    def stop_galvo_scan(self):
        """Stop the current scan."""
        self._stop_event.set()
        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=1.0)
        self._running = False
        return {'success': True}
    
    def get_galvo_status(self):
        """Get current scan status."""
        with self.lock:
            return {
                'running': self._running,
                'current_frame': self._current_frame,
                'current_line': self._current_line,
                'x_position': self._x_position,
                'y_position': self._y_position
            }
    
    def get_config(self):
        """Get current configuration."""
        with self.lock:
            return self._config.copy()
            
    def set_position(self, x=None, y=None):
        """Set galvo position directly (without scanning)."""
        with self.lock:
            if x is not None:
                self._x_position = max(0, min(4095, int(x)))
            if y is not None:
                self._y_position = max(0, min(4095, int(y)))
        return {'success': True, 'x': self._x_position, 'y': self._y_position}
    
    def get_position(self):
        """Get current galvo position."""
        with self.lock:
            return {'x': self._x_position, 'y': self._y_position}





class Camera:
    def __init__(self, parent, filePath="path_to_image.jpeg"):
        self._parent = parent
        self.filePath = filePath

        if self.filePath == "simplant":
            self._image = createBranchingTree(width=5000, height=5000)
            self._image /= np.max(self._image)
            self.SensorHeight = 300  # self._image.shape[1]
            self.SensorWidth = 400  # self._image.shape[0]
        elif self.filePath == "april_tag":
            ''' load april_tag_36h11_01.svg '''
            package_dir = os.path.dirname(os.path.abspath(__file__))
            self._imagePath = os.path.join(
                package_dir, "_data/images/april_tag_fromscreen.jpg" # apriltag_grid.png"
            )
            self._image = cv2.imread(self._imagePath, cv2.IMREAD_GRAYSCALE)
            # downscale for performance
            subsamplingfactor = 3
            self._image = cv2.resize(self._image, (self._image.shape[1]//subsamplingfactor, self._image.shape[0]//subsamplingfactor), interpolation=cv2.INTER_AREA)
            # flip top/bottom
            #self._image = cv2.flip(self._image, 0)
            # invert image
            self._image = 255 - self._image
            self._image = self._image / np.max(self._image)
            self.SensorHeight = 300  # self._image.shape[1]
            self.SensorWidth = 400  # self._image.shape[0]
        elif self.filePath == "clock":
            pass # TODO: IMPLEMENT
        elif self.filePath == "wellplatecalib":
            ''' load calibration_front.svg '''
            package_dir = os.path.dirname(os.path.abspath(__file__))
            self._imagePath = os.path.join(
                package_dir, "_data/images/calibration_front.png"
            )
            self._image = cv2.imread(self._imagePath, cv2.IMREAD_GRAYSCALE)
            # invert image
            self._image = 255 - self._image
            self._image = self._image / np.max(self._image)
            self.SensorHeight = 300  # self._image.shape[1]
            self.SensorWidth = 400  # self._image.shape[0]


        elif self.filePath == "astigmatism":
            self.SensorHeight = 512  # self._image.shape[1]
            self.SensorWidth = 512  # self._image.shape[0]

            self.astimulator = AstigmaticMicroscopeSimulator(W=self.SensorHeight, H=self.SensorWidth, roi_half=256)

        elif self.filePath == "smlm":
            self.SensorHeight = 300  # self._image.shape[1]
            self.SensorWidth = 400  # self._image.shape[0]

            tmp = createBranchingTree(width=5000, height=5000)
            tmp_min = np.min(tmp)
            tmp_max = np.max(tmp)
            self._image = (
                1 - ((tmp - tmp_min) / (tmp_max - tmp_min)) > 0
            )  # generating binary image
        else:
            self._image = np.mean(cv2.imread(filePath), axis=2)
            self._image /= np.max(self._image)
            self.SensorHeight = 300  # self._image.shape[1]
            self.SensorWidth = 400  # self._image.shape[0]

        self.lock = threading.Lock()
        self.model = "VirtualCamera"
        self.PixelSize = 1.0
        self.isRGB = False
        self.flipX = False
        self.flipY = False
        self.flipImage = (self.flipY, self.flipX)
        self.frameNumber = 0
        # precompute noise so that we will save energy and trees
        self.noiseStack = np.abs(
            np.random.randn(self.SensorHeight, self.SensorWidth, 100) * 2
        )

        # Thread-safe frame queue and acquisition thread
        self.frame_queue = Queue(maxsize=5)  # Limit queue size to prevent memory overflow
        self.acquisition_active = False
        self.acquisition_thread = None
        self.acquisition_lock = threading.Lock()

        # Cached parameters to avoid locking parent constantly
        self._cached_position = {"X": 0, "Y": 0, "Z": 0, "A": 0}
        self._cached_intensity = 1.0
        self._cached_psf = None
        self.binning = False  # For objective binning support

    def startAcquisition(self):
        """Start the continuous frame production thread"""
        with self.acquisition_lock:
            if not self.acquisition_active:
                self.acquisition_active = True
                self.acquisition_thread = threading.Thread(target=self._frame_producer_loop, daemon=True)
                self.acquisition_thread.start()

    def stopAcquisition(self):
        """Stop the continuous frame production thread"""
        with self.acquisition_lock:
            if self.acquisition_active:
                self.acquisition_active = False
                if self.acquisition_thread:
                    self.acquisition_thread.join(timeout=1.0)
                    self.acquisition_thread = None
                # Clear remaining frames from queue
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        break

    def _update_cached_parameters(self):
        """Update cached parameters from parent to minimize locking overhead"""
        try:
            self._cached_position = self._parent.positioner.get_position()
            self._cached_intensity = self._parent.illuminator.get_intensity(1)
            self._cached_psf = np.squeeze(self._parent.positioner.get_psf())
        except Exception:
            # In case parent doesn't have these methods yet
            pass

    def _frame_producer_loop(self):
        """Continuous frame production loop running in separate thread"""
        while self.acquisition_active:
            try:
                # Update cached parameters from parent (reduces lock contention)
                self._update_cached_parameters()
                # Produce frame with cached parameters
                frame = self.produce_frame(
                    x_offset=self._cached_position["X"],
                    y_offset=self._cached_position["Y"],
                    z_offset=self._cached_position["Z"],
                    light_intensity=self._cached_intensity,
                    defocusPSF=self._cached_psf,
                )

                self.frameNumber += 1
                # Try to put frame in queue (non-blocking to avoid backlog)
                try:
                    self.frame_queue.put((frame, self.frameNumber), block=False)
                except:
                    # Queue is full, drop oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((frame, self.frameNumber), block=False)
                    except:
                        pass
                # Small sleep to reduce CPU load (adjust based on desired frame rate)
                time.sleep(0.01)  # ~100 fps max, adjust as needed
            except Exception as e:
                print(f"Error in frame producer loop: {e}")
                time.sleep(0.1)  # Back off on error

    def produce_frame(
        self, x_offset=0, y_offset=0, z_offset=0, light_intensity=1.0, defocusPSF=None
    ):
        """Generate a frame based on the current settings."""
        if self.filePath == "smlm": # There is likely a better way of handling this
            frame = self.produce_smlm_frame(x_offset, y_offset, light_intensity).astype(np.uint16)
        elif self.filePath == "astigmatism":
            frame = self.produce_astigmatism_frame(z_offset).astype(np.uint16)
        else:
            # Removed lock here since we're using cached parameters
            # add noise
            image = self._image#.copy()
            # Adjust image based on offsets
            image = np.roll(
                np.roll(image, int(x_offset), axis=1), int(y_offset), axis=0
            )
            image = _extract_image(
                image, (self.SensorHeight, self.SensorWidth)
            )  # extract the image to the sensor size
            # do all post-processing on cropped image
            if IS_NIP and defocusPSF is not None and not defocusPSF.shape == ():
                image = np.array(np.real(nip.convolve(image, defocusPSF)))
            image = np.float32(image) * np.float32(light_intensity)
            image += self.noiseStack[:, :, np.random.randint(0, 100)]
            # Adjust illumination
            frame = image.astype(np.uint16)
            # Removed sleep here - controlled in producer loop

        # Apply flip if needed (zero-CPU operation using numpy)
        if self.flipImage[0]:  # flipY
            frame = np.flip(frame, axis=0)
        if self.flipImage[1]:  # flipX
            frame = np.flip(frame, axis=1)

        return frame

    def produce_astigmatism_frame(self, z_offset=0):
        #!/usr/bin/env python3
        return self.astimulator.render_frame(z=z_offset)

    def produce_smlm_frame(self, x_offset=0, y_offset=0, light_intensity=5000):
        """Generate a SMLM frame based on the current settings."""
        # Removed lock here since we're using cached parameters
        # add noise
        image = self._image.copy()
        # Adjust image based on offsets
        image = np.roll(
            np.roll(image, int(x_offset), axis=1), int(y_offset), axis=0
        )
        image = np.array(
            _extract_image(image, (self.SensorHeight, self.SensorWidth))
        )

        yc_array, xc_array = binary2locs(image, density=0.05)
        photon_array = np.random.normal(
            light_intensity * 5, light_intensity * 0.05, size=len(xc_array)
        )

        wavelenght = 6  # change to get it from microscope settings
        wavelenght_std = 0.5  # change to get it from microscope settings
        NA = 1.2  # change to get it from microscope settings
        sigma = 0.21 * wavelenght / NA  # change to get it from microscope settings
        sigma_std = (
            0.21 * wavelenght_std / NA
        )  # change to get it from microscope settings
        sigma_array = np.random.normal(sigma, sigma_std, size=len(xc_array))

        ADC_per_photon_conversion = 1.0  # change to get it from microscope settings
        readout_noise = 50  # change to get it from microscope settings
        ADC_offset = 100  # change to get it from microscope settings

        out = FromLoc2Image_MultiThreaded(
            xc_array,
            yc_array,
            photon_array,
            sigma_array,
            self.SensorHeight,
            self.SensorWidth,
            self.PixelSize,
        )
        out = (
            ADC_per_photon_conversion * np.random.poisson(out)
            + readout_noise
            * np.random.normal(size=(self.SensorHeight, self.SensorWidth))
            + ADC_offset
        )
        # Removed sleep here - controlled in producer loop
        return np.array(out)

    def getLast(self, returnFrameNumber=False):
        """Get the latest frame from the queue or generate one if acquisition not active"""
        # Try to get frame from queue if acquisition is active
        now = time.time()
        if self.acquisition_active:
            try:
                frame, frame_number = self.frame_queue.get(timeout=0.02)
                if frame is None:
                    return None
                self.frame = frame
                if self.binning:
                    # Apply 2x2 binning for objective magnification
                    self.frame = self._apply_binning(self.frame)
                if returnFrameNumber:
                    return self.frame, frame_number
                else:
                    return self.frame
            except Empty:
                # Queue is empty, fall back to direct generation
                pass

        # Fallback: generate frame directly (legacy behavior for non-acquisition mode)
        position = self._parent.positioner.get_position()
        defocusPSF = np.squeeze(self._parent.positioner.get_psf())
        intensity = self._parent.illuminator.get_intensity(1)
        self.frameNumber += 1

        frame = self.produce_frame(
            x_offset=position["X"],
            y_offset=position["Y"],
            z_offset=position["Z"],
            light_intensity=intensity,
            defocusPSF=defocusPSF,
        )

        if self.binning:
            frame = self._apply_binning(frame)

        if returnFrameNumber:
            return frame, self.frameNumber
        else:
            return frame

    def _apply_binning(self, frame):
        """Apply 2x2 binning to simulate objective magnification"""
        h, w = frame.shape
        binned_h, binned_w = h // 2, w // 2
        binned = frame.reshape(binned_h, 2, binned_w, 2).mean(axis=(1, 3))
        return binned.astype(frame.dtype)

    def getLastChunk(self):
        """Get the latest frame chunk"""
        mFrame = self.getLast()
        return np.expand_dims(mFrame, axis=0), [self.frameNumber]

    def setPropertyValue(self, propertyName, propertyValue):
        pass


@njit(parallel=True)
def FromLoc2Image_MultiThreaded(
    xc_array: np.ndarray, yc_array: np.ndarray, photon_array: np.ndarray, sigma_array: np.ndarray, image_height: int, image_width: int, pixel_size: float
):
    """
    Generate an image from localized emitters using multi-threading.

    Parameters
    ----------
    xc_array : array_like
        Array of x-coordinates of the emitters.
    yc_array : array_like
        Array of y-coordinates of the emitters.
    photon_array : array_like
        Array of photon counts for each emitter.
    sigma_array : array_like
        Array of standard deviations (sigmas) for each emitter.
    image_height : int
        Height of the output image in pixels.
    image_width : int
        Width of the output image in pixels.
    pixel_size : float
        Size of each pixel in the image.

    Returns
    -------
    Image : ndarray
        2D array representing the generated image.

    Notes
    -----
    The function utilizes multi-threading for parallel processing using Numba's
    `njit` decorator with `parallel=True`. Emitters with non-positive photon
    counts or non-positive sigma values are ignored. Only emitters within a
    distance of 4 sigma from the center of the pixel are considered to save
    computation time.

    The calculation involves error functions (`erf`) to determine the contribution
    of each emitter to the pixel intensity.

    Originally from: https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/Deep-STORM_2D_ZeroCostDL4Mic.ipynb
    """
    Image = np.zeros((image_height, image_width))
    for ij in prange(image_height * image_width):
        j = int(ij / image_width)
        i = ij - j * image_width
        for xc, yc, photon, sigma in zip(xc_array, yc_array, photon_array, sigma_array):
            # Don't bother if the emitter has photons <= 0 or if Sigma <= 0
            if (photon > 0) and (sigma > 0):
                S = sigma * math.sqrt(2)
                x = i * pixel_size - xc
                y = j * pixel_size - yc
                # Don't bother if the emitter is further than 4 sigma from the centre of the pixel
                if (x + pixel_size / 2) ** 2 + (
                    y + pixel_size / 2
                ) ** 2 < 16 * sigma**2:
                    ErfX = math.erf((x + pixel_size) / S) - math.erf(x / S)
                    ErfY = math.erf((y + pixel_size) / S) - math.erf(y / S)
                    Image[j][i] += 0.25 * photon * ErfX * ErfY
    return Image


def binary2locs(img: np.ndarray, density: float):
    """
    Selects a subset of locations from a binary image based on a specified density.

    Parameters
    ----------
    img : np.ndarray
        2D binary image array where 1s indicate points of interest.
    density : float
        Proportion of points to randomly select from the points of interest.
        Should be a value between 0 and 1.

    Returns
    -------
    filtered_locs : tuple of np.ndarray
        Tuple containing two arrays. The first array contains the row indices
        and the second array contains the column indices of the selected points.

    Notes
    -----
    The function identifies all locations in the binary image where the value is 1.
    It then randomly selects a subset of these locations based on the specified
    density and returns their coordinates.
    """
    all_locs = np.nonzero(img == 1)
    n_points = int(len(all_locs[0]) * density)
    selected_idx = np.random.choice(len(all_locs[0]), n_points, replace=False)
    filtered_locs = all_locs[0][selected_idx], all_locs[1][selected_idx]
    return filtered_locs


def createBranchingTree(width=5000, height=5000, lineWidth=3):
    np.random.seed(0)  # Set a random seed for reproducibility
    # Define the dimensions of the image
    width, height = 5000, 5000

    # Create a blank white image
    image = np.ones((height, width), dtype=np.uint8) * 255

    # Function to draw a line (blood vessel) on the image
    def draw_vessel(start, end, image):
        rr, cc = line(start[0], start[1], end[0], end[1])
        try:
            image[rr, cc] = 0  # Draw a black line
        except:
            end = 0
            return

    # Recursive function to draw a tree-like structure
    def draw_tree(start, angle, length, depth, image, reducer, max_angle=40):
        if depth == 0:
            return

        # Calculate the end point of the branch
        end = (
            int(start[0] + length * np.sin(np.radians(angle))),
            int(start[1] + length * np.cos(np.radians(angle))),
        )

        # Draw the branch
        draw_vessel(start, end, image)

        # change the angle slightly to add some randomness
        angle += np.random.uniform(-10, 10)

        # Recursively draw the next level of branches
        new_length = length * reducer  # Reduce the length for the next level
        new_depth = depth - 1
        draw_tree(
            end,
            angle - max_angle * np.random.uniform(-1, 1),
            new_length,
            new_depth,
            image,
            reducer,
        )
        draw_tree(
            end,
            angle + max_angle * np.random.uniform(-1, 1),
            new_length,
            new_depth,
            image,
            reducer,
        )

    # Starting point and parameters
    start_point = (height - 1, width // 2)
    initial_angle = -90  # Start by pointing upwards
    initial_length = np.max((width, height)) * 0.15  # Length of the first branch
    depth = 7  # Number of branching levels
    reducer = 0.9
    # Draw the tree structure
    draw_tree(start_point, initial_angle, initial_length, depth, image, reducer)

    # convolve image with rectangle
    rectangle = np.ones((lineWidth, lineWidth))
    image = convolve2d(image, rectangle, mode="same", boundary="fill", fillvalue=0)

    return image


if __name__ == "__main__":

    # Read the image locally
    # mFWD = os.path.dirname(os.path.realpath(__file__)).split("imswitch")[0]
    # imagePath = mFWD + "imswitch/_data/images/histoASHLARStitch.jpg"
    imagePath = "smlm"
    microscope = VirtualMicroscopy(filePath=imagePath)
    microscope.illuminator.set_intensity(intensity=1000)

    for i in range(5):
        microscope.positioner.move(
            x=1400 + i * (-200), y=-800 + i * (-10), z=0, is_absolute=True
        )
        frame = microscope.camera.getLast()
        plt.imsave(f"frame_{i}.png", frame)
    cv2.destroyAllWindows()

# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
