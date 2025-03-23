import os
import time
import datetime
import subprocess
import platform
import numpy as np
import cv2
from threading import Thread, Event
from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcommon.framework import Signal
from ..basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS
from typing import Dict

# We map FastAPI GET -> @APIExport()
# and FastAPI POST -> @APIExport(requestType="POST")

# Minimal default config matching the React needs:
#  - exposureTime, gain, timelapsePeriod, storagePath, isRunning
#  - isFocusMode, freeSpace, currentImageCount, etc.
DEFAULT_CONFIG = {
    "exposureTime": 100.0,
    "gain": 0.0,
    "timelapsePeriod": 60,  # in seconds
    "storagePath": "/mnt/usb_drive",  # example
    "isRunning": False,
    "wasRunning": False,
    "numberOfFrames": 10,
    "experimentName": "LepMonTest",
    "axislepmon": "Z",
    "axisFocus": "X",
    "isRecordVideo": True,
    "fileFormat": "JPG",
    "frameRate": 1,
    "delayTimeAfterRestart": 1.0,
    "time": "00:00:00",
    "date": "2021-01-01",
}

class LepmonController(LiveUpdatedController):
    """
    Example Lepmon Controller which provides:
    - GET endpoints to retrieve status (isRunning, imageCount, freeSpace, etc.)
    - POST endpoints to start/stop an experiment, set camera exposure/gain, etc.
    - A background thread capturing frames (lepmonExperimentThread).
    - Minimal code to illustrate a FastAPI-like structure using the @APIExport decorator.
    """

    # Signals -> used to broadcast to WebSocket in the background
    sigImagesTaken = Signal(int)      # e.g. "imageCounter" WS message
    sigIsRunning = Signal(bool)       # e.g. "isRunning" WS message
    sigFocusSharpness = Signal(float) # e.g. "focusSharpness" WS message
    temperatureUpdate = Signal(dict)  # e.g. "temperatureUpdate" WS message
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # Mock reading config
        self._master.LepmonManager.defaultConfig = DEFAULT_CONFIG
        self.mExperimentParameters = self._master.LepmonManager.defaultConfig
        
        self.is_measure = False
        self.imagesTaken = 0
        
        # Detector (camera)
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.detectorlepmonCam = self._master.detectorsManager[allDetectorNames[0]]

        # Possibly set default exposure/gain
        self.changeAutoExposureTime("auto")

        # If was running, start automatically
        if self.mExperimentParameters["wasRunning"]:
            self._logger.debug("Resuming experiment because 'wasRunning' was True.")
            
        # start thread that pulls sensor data
        self.sensorThread = Thread(target=self._pullSensorData, args=(10,))
        self.sensorThread.start()
        
        # initialize temperature and humidity
        self.innerTemp = np.round(np.random.uniform(20, 25), 2)
        self.outerTemp = np.round(np.random.uniform(15, 20), 2)
        self.humidity = np.round(np.random.uniform(40, 50), 2)
        
        
    # ---------------------- GET-Like Endpoints ---------------------- #

    @APIExport(requestType="POST")
    def setSensorData(self, sensorData: dict) -> dict: 
        """
        A GET-like endpoint that sets the inner and outer temperature and humidity.
        {"innerTemp": 25.0, "outerTemp": 20.0, "humidity": 45.0}   
        """
        try:
            innerTemp = sensorData["innerTemp"]
            outerTemp = sensorData["outerTemp"]
            humidity = sensorData["humidity"]
            self.innerTemp = innerTemp
            self.outerTemp = outerTemp
            self.humidity = humidity
            sensor_data = {"innerTemp": self.innerTemp, "outerTemp": self.outerTemp, "humidity": self.humidity}
            self.temperatureUpdate.emit(sensor_data)
            return {"success": True, "message": "Sensor data updated."}
        except Exception as e:
            self._logger.error(f"Could not update sensor data: {e}")
            return {"success": False, "message": "Could not update sensor data:"}
        

    @APIExport()
    def getStatus(self) -> dict:
        """
        A GET-like endpoint that returns a dict with isRunning, currentImageCount, freeSpace, serverTime, etc.
        """
        free_space_str = self._computeFreeSpace()
        status = {
            "isRunning": self.is_measure,
            "currentImageCount": self.imagesTaken,
            "serverTime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "freeSpace": free_space_str
        }
        return status

    @APIExport()
    def getInitialParams(self) -> dict:
        """
        Another GET-like endpoint that returns camera/timelapse settings, storage path, etc.
        """
        result = {
            "exposureTime": self.mExperimentParameters["exposureTime"],
            "gain": self.mExperimentParameters["gain"],
            "timelapsePeriod": self.mExperimentParameters["timelapsePeriod"],
            "storagePath": self.mExperimentParameters["storagePath"],
        }
        return result

    # ---------------------- POST-Like Endpoints --------------------- #

    @APIExport(requestType="POST")
    def startExperiment(self,
                        deviceTime: str,
                        deviceLat: float = None,
                        deviceLng: float = None,
                        exposureTime: float = 100.0,
                        gain: float = 0.0,
                        timelapsePeriod: int = 60, 
                        time: str = None, 
                        date: str = None) -> dict:
        """
        Called by the frontend to start an experiment.
        Here we store the user deviceTime, lat/lng, exposure, etc.
        Then we call self.startLepmonExperiment(...) in a thread.
        """
        self._logger.debug(f"startExperiment from deviceTime={deviceTime}, lat={deviceLat}, lng={deviceLng}")
        
        # We can set camera exposure/gain
        self.changeAutoExposureTime("manual")
        self.changeExposureTime(exposureTime)
        self.changeGain(gain)
        
        # Also set timelapse period
        self.mExperimentParameters["timelapsePeriod"] = timelapsePeriod
        self.mExperimentParameters["time"] = time
        self.mExperimentParameters["date"] = date
       
        
        self.mExperimentParameters["timeStamp"] = (time + "_" + date)
        self.mExperimentParameters["storagePath"] = "/mnt/usb_drive"
        self.mExperimentParameters["numImages"] = -1
        self.mExperimentParameters["fileFormat"] = "TIF"
        self.mExperimentParameters["frameRate"] = timelapsePeriod
        self.mExperimentParameters["experimentName"] = "LepMonTest" 
        self.mExperimentParameters["uniqueId"] = np.random.randint(0, 1000),
        
        # Start thread
        self.mExperimentThread = Thread(target=self.lepmonExperimentThread, args=(
            self.mExperimentParameters["timeStamp"],
            self.mExperimentParameters["experimentName"],
            self.mExperimentParameters["uniqueId"],
            self.mExperimentParameters["numImages"],
            self.mExperimentParameters["frameRate"],
            self.mExperimentParameters["storagePath"],
            self.mExperimentParameters["fileFormat"],
        ), daemon=True)
        self.mExperimentThread.start()
    
    
        # Actually start the experiment logic
        self.is_measure = True
        self.imagesTaken = 0
        self.sigIsRunning.emit(True)  # Websocket signal
        return {"success": True, "message": "Experiment started", "lat": deviceLat, "lng": deviceLng}

    @APIExport(requestType="POST")
    def stopLepmonExperiment(self) -> dict:
        """
        Called by the frontend to stop any running experiment.
        """
        self._logger.debug("Experiment stopped by user.")
        self.is_measure = False
        self.sigIsRunning.emit(False)
        return {"success": True, "message": "Experiment stopped"}

    @APIExport(requestType="POST")
    def focusMode(self) -> dict:
        """
        The user triggers a 15s focus mode in the backend.
        We simulate sending sharpness values via self.sigFocusSharpness.emit(...) in a small thread.
        """
        self._logger.debug("Focus Mode requested.")
        def focus_thread():
            start_time = time.time()
            while time.time() - start_time < 15.0:
                # Example: generate a random "sharpness" or measure it from camera
                # For now, let's just do random
                import random
                sharp_val = random.uniform(10, 300)
                self.sigFocusSharpness.emit(sharp_val)
                time.sleep(0.5)
        t = Thread(target=focus_thread, daemon=True)
        t.start()
        return {"success": True, "message": "Focus mode started for 15s"}

    @APIExport(requestType="POST")
    def reboot(self) -> dict:
        """
        The user triggers a device reboot.
        """
        self._logger.debug("Reboot requested.")
        # e.g. os.system("sudo reboot")
        return {"success": True, "message": "System is rebooting (mock)"}

    # ---------------------- Main experiment thread ------------------- #
    def lepmonExperimentThread(self,
                             timeStamp: str,
                             experimentName: str,
                             uniqueId: str,
                             numImages: int,
                             frameRate: float,
                             filePath: str,
                             fileFormat: str):

        """
        Example background thread that acquires images up to numImages or until is_measure==False.
        """


        self.is_measure = True
        self.imagesTaken = 0
        self.sigIsRunning.emit(True)

        # Possibly create a folder
        dirPath = os.path.join(filePath, timeStamp)
        if not os.path.exists(dirPath):
            try:
                os.makedirs(dirPath)
            except Exception as e:
                self._logger.error(e)


        while self.is_measure and self.imagesTaken < numImages or numImages == -1:
            currentTime = time.time()
            self.imagesTaken += 1

            # Notify WebSocket about new image count
            self.sigImagesTaken.emit(self.imagesTaken)

            # Snap image
            filename = os.path.join(dirPath, f"{timeStamp}_{experimentName}_{uniqueId}_{self.imagesTaken}")
            try:
                self.snapImagelepmonCam(filename, fileFormat=fileFormat)
            except Exception as e:
                self._logger.error(f"Could not snap image: {e}")

            # Sleep to maintain framerate
            while (time.time() - currentTime) < (frameRate):
                if not self.is_measure:
                    break
                time.sleep(0.05)
                # TAKE IMAGE HERE
            if not self.is_measure:
                break
        self.is_measure = False
        self.sigIsRunning.emit(False)
        self._logger.debug("lepmonExperimentThread done.")

    # ----------------------- Snap single image ----------------------- #
    @APIExport(runOnUIThread=True)
    def snapImagelepmonCam(self, fileName=None, fileFormat="JPG"):
        """Just captures the latest frame from the camera and saves it."""
        if not fileName:
            fileName = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        frame = self.detectorlepmonCam.getLatestFrame()
        if frame is None:
            self._logger.warning("No frame received from the camera.")
            return

        if fileFormat.upper() == "TIF":
            import tifffile as tif
            tif.imsave(fileName + ".tif", frame, append=False)
        elif fileFormat.upper() == "JPG":
            cv2.imwrite(fileName + ".jpg", frame)
        elif fileFormat.upper() == "PNG":
            cv2.imwrite(fileName + ".png", frame)
        else:
            self._logger.warning(f"No valid fileFormat selected: {fileFormat}")

    # ----------------------- Camera controls ------------------------- #
    @APIExport(runOnUIThread=True)
    def changeExposureTime(self, value):
        """Change exposure time (manual mode)."""
        try:
            self.mExperimentParameters["exposureTime"] = value
            self.detectorlepmonCam.setParameter(name="exposure", value=value)
        except Exception as e:
            self._logger.error(f"Could not set exposure: {e}")

    @APIExport(runOnUIThread=True)
    def changeAutoExposureTime(self, value):
        """Enable/disable auto-exposure, e.g. 'auto' or 'manual'."""
        try:
            self.detectorlepmonCam.setParameter(name="exposure_mode", value=value)
        except Exception as e:
            self._logger.error(f"Could not set auto exposure mode: {e}")

    @APIExport(runOnUIThread=True)
    def changeGain(self, value):
        """Change camera gain."""
        try:
            self.mExperimentParameters["gain"] = value
            self.detectorlepmonCam.setGain(value)
        except Exception as e:
            self._logger.error(f"Could not set gain: {e}")

    # ---------------------- Helper functions -------------------------- #
    
    def _pullSensorData(self, interval):
        while True:
            # Get sensor data
            # e.g. temperature, humidity, pressure, etc.
            # sensor_data = getSensorData()
            # self.sigSensorData.emit(sensor_data)
            time.sleep(interval)
            # simulate inner/outer temperature and humidity
            # join them in dictionary 
            sensor_data = {"innerTemp": self.innerTemp, "outerTemp": self.outerTemp, "humidity": self.humidity}
            self.temperatureUpdate.emit(sensor_data)
            
    def _computeFreeSpace(self) -> str:
        # Simplistic approach or call your existing function
        usage = dirtools.getDiskusage()  # returns fraction used, e.g. 0.8 => 80%
        used_prc = usage * 100
        free_prc = 100.0 - used_prc
        return f"{free_prc:.1f}% free"

    def detect_external_drives(self):
        """If you want to keep an external drive detection method, do so here."""
        system = platform.system()
        external_drives = []
        if system in ["Linux", "Darwin"]:
            df_result = subprocess.run(['df', '-h'], stdout=subprocess.PIPE)
            output = df_result.stdout.decode('utf-8')
            lines = output.splitlines()
            for line in lines:
                if '/media/' in line or '/Volumes/' in line:
                    drive_info = line.split()
                    mount_point = " ".join(drive_info[5:])
                    external_drives.append(mount_point)
        elif system == "Windows":
            wmic_result = subprocess.run(['wmic', 'logicaldisk', 'get', 'caption,description'],
                                         stdout=subprocess.PIPE)
            output = wmic_result.stdout.decode('utf-8')
            lines = output.splitlines()
            for line in lines:
                if 'Removable Disk' in line:
                    drive_info = line.split()
                    drive_letter = drive_info[0]
                    external_drives.append(drive_letter)
        return external_drives


