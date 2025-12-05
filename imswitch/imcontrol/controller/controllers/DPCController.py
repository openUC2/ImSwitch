import json
import os
import io
import queue

import numpy as np
import time
import threading
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import tifffile as tif
import cv2

from imswitch.imcommon.model import dirtools, initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex, Timer

from ..basecontrollers import LiveUpdatedController

import os
import time
import numpy as np
from pathlib import Path
import tifffile
from imswitch import IS_HEADLESS

pi    = np.pi
naxis = np.newaxis
F     = lambda x: np.fft.fft2(x)
IF    = lambda x: np.fft.ifft2(x)

# =========================
# Dataclasses (API-stable)
# =========================
@dataclass
class DPCParams:
    """DPC processing parameters"""
    pixelsize: float = 0.2  # micrometers
    wavelength: float = 0.53  # micrometers
    na: float = 0.3
    nai: float = 0.3
    n: float = 1.0
    led_intensity_r: int = 0  # Red channel intensity
    led_intensity_g: int = 255  # Green channel intensity (default)
    led_intensity_b: int = 0  # Blue channel intensity
    wait_time: float = 0.2  # seconds between LED changes
    frame_sync: int = 2  # number of frames to wait for fresh frame
    save_images: bool = False
    save_directory: str = ""
    reg_u: float = 1e-1  # Tikhonov regularization for absorption
    reg_p: float = 5e-3  # Tikhonov regularization for phase
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DPCState:
    """DPC processing state"""
    is_processing: bool = False
    is_paused: bool = False
    frame_count: int = 0
    processed_count: int = 0
    last_process_time: float = 0.0
    processing_fps: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DPCController(ImConWidgetController):
    """
    DPC Controller with backend processing and API control.
    
    Features:
    - Uses LEDMatrixController setHalves() for pattern generation
    - MJPEG streaming of reconstructed images
    - API control via RESTful endpoints
    - Optional data saving
    """

    sigImageReceived = Signal()
    sigDPCProcessorImageComputed = Signal(np.ndarray, str)
    sigDPCStateChanged = Signal(object)  # state_dict
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # switch to detect if a recording is in progress
        self.isRecording = False

        # load config file
        if self._setupInfo.dpc is None: # TODO: We should have a default config example here 
            self._logger.error('DPC is not configured in your setup file.')

        # Pattern names for DPC (using LEDMatrix setHalves)
        self.allDPCPatternNames = ("top", "bottom", "right", "left")

        # dpc parameters from setup

        wavelength = self._master.dpcManager.wavelength if hasattr(self._master.dpcManager, 'wavelength') else 0.53
        pixelsize = self._master.dpcManager.pixelsize if hasattr(self._master.dpcManager, 'pixelsize') else 0.2
        na = self._master.dpcManager.NA if hasattr(self._master.dpcManager, 'NA') else 0.3
        nai = self._master.dpcManager.NAi if hasattr(self._master.dpcManager, 'NAi') else 0.3
        n = self._master.dpcManager.n if hasattr(self._master.dpcManager, 'n') else 1.0

        # Initialize parameters
        self._params = DPCParams(
            pixelsize=pixelsize,
            wavelength=wavelength,
            na=na,
            nai=nai,
            n=n,
            led_intensity_r=0,
            led_intensity_g=255,
            led_intensity_b=0,
            wait_time=0.2,
            frame_sync=2,
            save_images=False,
            save_directory=str(os.path.join(dirtools.UserFileDirs.Data, "DPC"))
        )
        self._state = DPCState()
        self._processing_lock = threading.Lock()

        # Get LEDMatrix manager
        allLEDMatrixNames = self._master.LEDMatrixsManager.getAllDeviceNames()
        if len(allLEDMatrixNames) == 0:
            self._logger.error("No LEDMatrix found in setup")
            return
        self.ledMatrix = self._master.LEDMatrixsManager[allLEDMatrixNames[0]]

        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detector = self._master.detectorsManager[allDetectorNames[0]]
        self.detector.startAcquisition()
        self.frameShape = self.detector.shape
        
        # initialize DPC processor
        self.DPCProcessor = DPCProcessor(self, self.frameShape, self._params)

        # MJPEG streaming
        self._mjpeg_queue = queue.Queue(maxsize=2)
        self._jpeg_quality = 85
        
        # Processing queue - decouples capture from reconstruction
        self._processing_queue = queue.Queue(maxsize=2)
        
        # Processing threads
        self._capture_thread = None
        self._reconstruction_thread = None
        self._stop_processing_event = threading.Event()
        
        # Performance monitoring
        self._perf_process_times = []
        self._perf_window_size = 30

        # connect the reconstructed image to the displayer
        self._logger.info("DPCController initialized successfully")

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_dpc_processing()
        
        if self._capture_thread is not None:
            self._stop_processing_event.set()
            self._capture_thread.join(timeout=2.0)
        
        if self._reconstruction_thread is not None:
            self._stop_processing_event.set()
            self._reconstruction_thread.join(timeout=2.0)
    
    # =========================
    # Core Processing Methods
    # =========================
    def _set_led_pattern(self, pattern_name: str):
        """Set LED pattern using LEDMatrix manager setHalves method"""
        intensity = (self._params.led_intensity_r, 
                    self._params.led_intensity_g, 
                    self._params.led_intensity_b)
        self.ledMatrix.setHalves(
            intensity=intensity,
            region=pattern_name, 
            getReturn=True,
            timeout=.2
        )
    
    def _get_fresh_frame(self, timeout: float = 1.0, nFrameSync: int = 2) -> Optional[np.ndarray]:
        """Get a fresh frame with frame sync (like ExperimentController)"""
        cTime = time.time()
        lastFrameNumber = -1
        
        while True:
            # Get frame and frame number to ensure fresh frame
            mFrame, currentFrameNumber = self.detector.getLatestFrame(returnFrameNumber=True)
            
            if lastFrameNumber == -1:
                # First round
                lastFrameNumber = currentFrameNumber
            
            if time.time() - cTime > timeout:
                # Timeout - use whatever we have
                if mFrame is None:
                    mFrame = self.detector.getLatestFrame(returnFrameNumber=False)
                break
            
            if currentFrameNumber <= lastFrameNumber + nFrameSync:
                time.sleep(0.01)  # Off-load CPU
            else:
                break
        
        return mFrame
    
    def _capture_dpc_stack(self):
        """Capture 4 images with different LED patterns"""
        stack = []
        
        # Turn off all LEDs first
        #self.ledMatrix.setAll(state=(0,0,0), getReturn=False)
        
        for pattern_name in self.allDPCPatternNames:
            if self._stop_processing_event.is_set():
                return None

            # Set pattern
            self._set_led_pattern(pattern_name)
            self._logger.debug(f"Showing pattern: {pattern_name}")
            time.sleep(self._params.wait_time / 2)
                        
            # Capture fresh frame with sync
            frame = self._get_fresh_frame(timeout=1.0, nFrameSync=self._params.frame_sync)
            if frame is None:
                self._logger.warning(f"Failed to get frame for pattern {pattern_name}")
                continue
            
            if len(frame.shape) > 2:
                frame = np.mean(frame, axis=2)
            stack.append(frame)
            
            self._state.frame_count += 1
        
        
        return np.array(stack) if len(stack) == 4 else None
    
    def _capture_loop(self):
        """Capture loop - captures DPC stacks and adds to processing queue"""
        self._logger.info("DPC capture loop started")
        
        while not self._stop_processing_event.is_set():
            try:
                # Capture 4-image stack
                stack = self._capture_dpc_stack()
                
                if stack is None:
                    if self._stop_processing_event.is_set():
                        break
                    continue
                
                # Add to processing queue (non-blocking, drop old if full)
                try:
                    self._processing_queue.put_nowait(stack)
                except queue.Full:
                    # Drop oldest stack and add new one
                    try:
                        self._processing_queue.get_nowait()
                        self._processing_queue.put_nowait(stack)
                    except:
                        pass
                
            except Exception as e:
                self._logger.error(f"Error in DPC capture loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        # Turn off LEDs when done
        self.ledMatrix.setAll(state=(0,0,0), getReturn=False)
        self._logger.info("DPC capture loop stopped")
    
    def _reconstruction_loop(self):
        """Reconstruction loop - processes stacks from queue"""
        self._logger.info("DPC reconstruction loop started")
        
        while not self._stop_processing_event.is_set():
            try:
                # Get stack from queue with timeout
                try:
                    stack = self._processing_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                t_start = time.time()
                
                # Process stack
                result = self.DPCProcessor.process_stack(stack, save=self._params.save_images)
                
                if result is not None:
                    dpc_lr, dpc_tb, qdpc = result
                    
                    # Emit signals for display
                    self.sigDPCProcessorImageComputed.emit(dpc_lr, "DPC left/right")
                    self.sigDPCProcessorImageComputed.emit(dpc_tb, "DPC top/bottom")
                    
                    # Add to MJPEG stream (use phase or combined image)
                    if qdpc is not None and len(qdpc) > 0:
                        phase_image = np.angle(qdpc[0])
                        self._add_to_mjpeg_stream(phase_image)
                    else:
                        # Fallback to gradient image
                        self._add_to_mjpeg_stream(dpc_lr)
                    
                    self._state.processed_count += 1
                
                # Update performance metrics
                t_end = time.time()
                process_time = t_end - t_start
                self._state.last_process_time = process_time
                
                self._perf_process_times.append(process_time)
                if len(self._perf_process_times) > self._perf_window_size:
                    self._perf_process_times.pop(0)
                
                if len(self._perf_process_times) > 0:
                    avg_time = np.mean(self._perf_process_times)
                    self._state.processing_fps = 1.0 / avg_time if avg_time > 0 else 0.0
                
                self._emit_state_changed()
                
            except Exception as e:
                self._logger.error(f"Error in DPC reconstruction loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        self._logger.info("DPC reconstruction loop stopped")
    
    def _add_to_mjpeg_stream(self, image):
        """Add processed image to MJPEG stream"""
        try:
            # Normalize to 0-255 with NaN handling
            img_normalized = image.copy()
            
            # Replace NaN and inf with 0
            img_normalized = np.nan_to_num(img_normalized, nan=0.0, posinf=0.0, neginf=0.0)
            
            img_min, img_max = img_normalized.min(), img_normalized.max()
            if img_max > img_min and np.isfinite(img_max) and np.isfinite(img_min):
                img_normalized = (img_normalized - img_min) / (img_max - img_min) * 255.0
            else:
                # If min == max or invalid, just clip to 0-255
                img_normalized = np.clip(img_normalized, 0, 255)
            
            # Safely convert to uint8
            img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
            
            # Convert to BGR for JPEG encoding
            if len(img_normalized.shape) == 2:
                img_bgr = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2BGR)
            
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            _, buffer = cv2.imencode('.jpg', img_bgr, encode_param)
            
            # Add to queue (non-blocking, drop old frames)
            try:
                self._mjpeg_queue.put_nowait(buffer.tobytes())
            except queue.Full:
                try:
                    self._mjpeg_queue.get_nowait()
                    self._mjpeg_queue.put_nowait(buffer.tobytes())
                except:
                    pass
        except Exception as e:
            self._logger.error(f"Error adding image to MJPEG stream: {e}")
    
    def _emit_state_changed(self):
        """Emit state changed signal"""
        self.sigDPCStateChanged.emit(self._state.to_dict())
    
    # =========================
    # API: Parameter Control
    # =========================
    @APIExport(runOnUIThread=True)
    def get_dpc_params(self) -> Dict[str, Any]:
        """Get current DPC parameters"""
        return self._params.to_dict()
    
    @APIExport(runOnUIThread=True, requestType="POST")
    def set_dpc_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set DPC parameters
        
        Args:
            params: Dictionary of parameters to update
            
        Returns:
            Updated parameter dictionary
        """
        with self._processing_lock:
            for key, value in params.items():
                if hasattr(self._params, key):
                    setattr(self._params, key, value)
                    self._logger.debug(f"Updated DPC parameter: {key} = {value}")
            
            # Update processor if it exists
            if hasattr(self, 'DPCProcessor'):
                self.DPCProcessor.update_parameters(self._params)
        
        return self.get_dpc_params()
    
    @APIExport(runOnUIThread=True)
    def get_dpc_state(self) -> Dict[str, Any]:
        """Get current DPC processing state"""
        return self._state.to_dict()
    
    # =========================
    # API: Processing Control
    # =========================
    @APIExport(runOnUIThread=True)
    def start_dpc_processing(self) -> Dict[str, Any]:
        """Start DPC processing"""
        with self._processing_lock:
            if self._state.is_processing:
                return {"status": "already_running", "state": self._state.to_dict()}
            
            # Reset state
            self._state.is_processing = True
            self._state.is_paused = False
            self._state.frame_count = 0
            self._state.processed_count = 0
            self._perf_process_times = []
            
            # Ensure camera is running
            self._ensure_camera_running()
            
            # Clear processing queue
            while not self._processing_queue.empty():
                try:
                    self._processing_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Start capture and reconstruction threads
            self._stop_processing_event.clear()
            
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="DPC-Capture"
            )
            self._capture_thread.start()
            
            self._reconstruction_thread = threading.Thread(
                target=self._reconstruction_loop,
                daemon=True,
                name="DPC-Reconstruction"
            )
            self._reconstruction_thread.start()
            
            self._emit_state_changed()
            self._logger.info("DPC processing started")
            
            return {"status": "started", "state": self._state.to_dict()}
    
    @APIExport(runOnUIThread=True)
    def stop_dpc_processing(self) -> Dict[str, Any]:
        """Stop DPC processing"""
        with self._processing_lock:
            if not self._state.is_processing:
                return {"status": "not_running", "state": self._state.to_dict()}
            
            # Stop processing
            self._stop_processing_event.set()
            
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=2.0)
                self._capture_thread = None
            
            if self._reconstruction_thread is not None:
                self._reconstruction_thread.join(timeout=2.0)
                self._reconstruction_thread = None
            
            self._state.is_processing = False
            self._state.is_paused = False
            
            self._emit_state_changed()
            self._logger.info("DPC processing stopped")
            
            return {"status": "stopped", "state": self._state.to_dict()}
    
    @APIExport(runOnUIThread=False)
    def mjpeg_stream_dpc(self, startStream: bool = True, jpeg_quality: int = 85) -> Any:
        """
        MJPEG stream of processed DPC images
        
        Args:
            startStream: Whether to start streaming
            jpeg_quality: JPEG quality (1-100)
            
        Yields:
            MJPEG frames
        """
        if not startStream:
            return
        
        self._jpeg_quality = jpeg_quality
        
        def generate():
            self._logger.info("Starting MJPEG stream for DPC")
            try:
                while True:
                    try:
                        frame = self._mjpeg_queue.get(timeout=1.0)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self._logger.error(f"Error in MJPEG stream: {e}")
                        break
            except GeneratorExit:
                self._logger.info("MJPEG stream closed")
        
        return generate()
    
    def _ensure_camera_running(self):
        """Ensure camera is running"""
        try:
            if not self.detector.isAcquiring():
                self.detector.startAcquisition()
        except:
            pass
    
    # =========================
    # Legacy GUI compatibility
    # =========================

    def toggleRecording(self):
        """Toggle recording mode"""
        if IS_HEADLESS:
            return
        self._params.save_images = not self._params.save_images

'''#####################################
# DPC PROCESSOR
#####################################'''

class DPCProcessor(object):

    def __init__(self, parent, shape, params: DPCParams):
        '''
        setup parameters
        '''
        # initialize logger
        self._logger = initLogger(self, tryInheritParent=False)
        self.parent = parent

        # Default values from params
        self.shape = shape
        self.pixelsize = params.pixelsize
        self.NA = params.na
        self.NAi = params.nai
        self.n = params.n
        self.wavelength = params.wavelength
        self.rotation = [0, 180, 90, 270]  # top, bottom, right, left
        self.save_directory = params.save_directory

        # Create DPC solver
        self.dpc_solver_obj = DPCSolver(
            shape=self.shape,
            wavelength=self.wavelength,
            na=self.NA,
            NAi=self.NAi,
            pixelsize=self.pixelsize,
            rotation=self.rotation
        )
        
        # Set Tikhonov regularization
        self.dpc_solver_obj.setTikhonovRegularization(
            reg_u=params.reg_u,
            reg_p=params.reg_p
        )

        # Ensure save directory exists
        if self.save_directory:
            os.makedirs(self.save_directory, exist_ok=True)

    def update_parameters(self, params: DPCParams):
        """Update processor parameters from DPCParams"""
        self.pixelsize = params.pixelsize
        self.NA = params.na
        self.NAi = params.nai
        self.n = params.n
        self.wavelength = params.wavelength
        self.save_directory = params.save_directory
        
        # Recreate solver with new parameters
        self.dpc_solver_obj = DPCSolver(
            shape=self.shape,
            wavelength=self.wavelength,
            na=self.NA,
            NAi=self.NAi,
            pixelsize=self.pixelsize,
            rotation=self.rotation
        )
        
        self.dpc_solver_obj.setTikhonovRegularization(
            reg_u=params.reg_u,
            reg_p=params.reg_p
        )

    def process_stack(self, stack: np.ndarray, save: bool = False) -> Optional[Tuple]:
        """
        Process 4-image DPC stack
        
        Args:
            stack: (4, H, W) array of DPC images
            save: Whether to save results
            
        Returns:
            Tuple of (dpc_lr, dpc_tb, qdpc_result) or None on error
        """
        try:
            if stack.shape[0] != 4:
                self._logger.error(f"Expected 4 images, got {stack.shape[0]}")
                return None
            
            # Compute qDPC reconstruction
            qdpc_result = self.dpc_solver_obj.solve(dpc_imgs=stack.astype('float64'))
            
            # Compute gradient images (normalized difference)
            dpc_result_1 = (stack[0] - stack[1]) / (stack[0] + stack[1] + 1e-10)  # top - bottom
            dpc_result_2 = (stack[2] - stack[3]) / (stack[2] + stack[3] + 1e-10)  # right - left
            
            # Save images if requested
            if save and self.save_directory:
                date = datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
                
                # Save qDPC reconstruction
                filename_recon = os.path.join(self.save_directory, f"{date}_DPC_Reconstruction.tif")
                tif.imwrite(filename_recon, qdpc_result)
                
                # Save gradient images
                filename_lr = os.path.join(self.save_directory, f"{date}_DPC_LeftRight.tif")
                filename_tb = os.path.join(self.save_directory, f"{date}_DPC_TopBottom.tif")
                tif.imwrite(filename_lr, dpc_result_2.astype(np.float32))
                tif.imwrite(filename_tb, dpc_result_1.astype(np.float32))
                
                self._logger.info(f"Saved DPC results to {self.save_directory}")
            
            return dpc_result_2, dpc_result_1, qdpc_result
            
        except Exception as e:
            self._logger.error(f"Error during DPC reconstruction: {e}", exc_info=True)
            return None

# (C) Wallerlab 2019
# https://github.com/Waller-Lab/DPC/blob/master/python_code/dpc_algorithm.py
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from skimage import io
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage import uniform_filter

class DPCSolver:
    def __init__(self, shape, wavelength, na, NAi, pixelsize, rotation):
        self.shape = (shape[1], shape[0]) # the image shape comes in X,Y but here we assume height, width => so transpose 
        if self.shape[0] == 0:
            self.shape = (512, 512)

        self.wavelength = wavelength
        self.na         = na
        self.NAi      = NAi
        self.pixel_size = pixelsize
        self.dpc_num    = len(rotation)
        self.rotation   = rotation
        self.fxlin      = np.fft.ifftshift(self.genGrid(self.shape[-1], 1.0/self.shape[-1]/self.pixel_size))
        self.fylin      = np.fft.ifftshift(self.genGrid(self.shape[-2], 1.0/self.shape[-2]/self.pixel_size))
        self.pupil      = self.pupilGen(self.fxlin, self.fylin, self.wavelength, self.na)
        self.sourceGen()
        self.WOTFGen()

    def setTikhonovRegularization(self, reg_u = 1e-6, reg_p = 1e-6):
        self.reg_u      = reg_u
        self.reg_p      = reg_p

    def normalization(self):
        for img in self.dpc_imgs:
            img          /= uniform_filter(img, size=img.shape[0]//2)
            meanIntensity = img.mean()
            img          /= meanIntensity        # normalize intensity with DC term
            img          -= 1.0                  # subtract the DC term

    def sourceGen(self):
        self.source = []
        pupil = self.pupilGen(self.fxlin, self.fylin, self.wavelength, self.na, NAi=self.NAi)
        for rotIdx in range(self.dpc_num):
            self.source.append(np.zeros((self.shape)))
            rotdegree = self.rotation[rotIdx]
            if rotdegree < 180:
                self.source[-1][self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15>=
                                self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = 1.0
                self.source[-1] *= pupil
            else:
                self.source[-1][self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15<
                                self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = -1.0
                self.source[-1] *= pupil
                self.source[-1] += pupil
        self.source = np.asarray(self.source)

    def WOTFGen(self):
        self.Hu = []
        self.Hp = []
        for rotIdx in range(self.source.shape[0]):
            FSP_cFP  = F(self.source[rotIdx]*self.pupil)*F(self.pupil).conj()
            I0       = (self.source[rotIdx]*self.pupil*self.pupil.conj()).sum()
            self.Hu.append(2.0*IF(FSP_cFP.real)/I0)
            self.Hp.append(2.0j*IF(1j*FSP_cFP.imag)/I0)
        self.Hu = np.asarray(self.Hu)
        self.Hp = np.asarray(self.Hp)

    def solve(self, dpc_imgs, **kwargs):
        self.dpc_imgs = dpc_imgs.astype('float64')
        self.normalization()

        dpc_result  = []
        AHA         = [(self.Hu.conj()*self.Hu).sum(axis=0)+self.reg_u,            (self.Hu.conj()*self.Hp).sum(axis=0),\
                       (self.Hp.conj()*self.Hu).sum(axis=0)           , (self.Hp.conj()*self.Hp).sum(axis=0)+self.reg_p]
        determinant = AHA[0]*AHA[3]-AHA[1]*AHA[2]
        
        # Avoid division by zero - add small epsilon where determinant is zero
        determinant = np.where(np.abs(determinant) < 1e-10, 1e-10, determinant)
        
        for frame_index in range(self.dpc_imgs.shape[0]//self.dpc_num):
            fIntensity = np.asarray([F(self.dpc_imgs[frame_index*self.dpc_num+image_index]) for image_index in range(self.dpc_num)])
            AHy        = np.asarray([(self.Hu.conj()*fIntensity).sum(axis=0), (self.Hp.conj()*fIntensity).sum(axis=0)])
            
            # Compute with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                absorption = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
                phase      = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
            
            # Replace NaN/inf with 0
            absorption = np.nan_to_num(absorption, nan=0.0, posinf=0.0, neginf=0.0)
            phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
            
            dpc_result.append(absorption+1.0j*phase)

        return np.asarray(dpc_result)


    def pupilGen(self, fxlin, fylin, wavelength, na, NAi=0.0):
        pupil = np.array(fxlin[naxis, :]**2+fylin[:, naxis]**2 <= (na/wavelength)**2)
        if NAi != 0.0:
            pupil[fxlin[naxis, :]**2+fylin[:, naxis]**2 < (NAi/wavelength)**2] = 0.0
        return pupil

    def genGrid(self, size, dx):
        xlin = np.arange(size, dtype='complex128')
        return (xlin-size//2)*dx




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
