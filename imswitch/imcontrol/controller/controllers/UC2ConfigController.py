import json
import os
import threading
from imswitch import IS_HEADLESS
import numpy as np
import datetime
from imswitch.imcommon.model import APIExport, initLogger, dirtools, ostools
from imswitch.imcommon.framework import Signal
from ..basecontrollers import ImConWidgetController
from imswitch.imcontrol.model import configfiletools
import tifffile as tif
from imswitch.imcontrol.model import Options
from imswitch.imcontrol.view.guitools import ViewSetupInfo
import json
import os
import tempfile
import threading
import requests
import shutil
from pathlib import Path
from serial.tools import list_ports
import serial

try:
    import esptool
    HAS_ESPTOOL = True
except ImportError:
    HAS_ESPTOOL = False


CAN_ADDRESS_MAP = {
    "master": 1,
    "a": 10,
    "x": 11,
    "y": 12,
    "z": 13,
    "laser": 20,
    "led": 30,
}

class UC2ConfigController(ImConWidgetController):
    """Linked to UC2ConfigWidget."""

    sigUC2SerialReadMessage = Signal(str)
    sigUC2SerialWriteMessage = Signal(str)
    sigUC2SerialIsConnected = Signal(bool)
    sigOTAStatusUpdate = Signal(object)  # Emits OTA status updates


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        # OTA update tracking
        self._ota_status = {}  # Dictionary to track OTA status by CAN ID
        self._ota_lock = threading.Lock()
        self._firmware_server_url = "http://localhost:9000"  # Default firmware server URL
        self._firmware_cache_dir = Path(tempfile.gettempdir()) / "uc2_ota_firmware_cache"
        self._firmware_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # WiFi credentials for OTA (can be overridden via API)
        self._ota_wifi_ssid = None
        self._ota_wifi_password = None

        try:
            self.stages = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]
        except Exception as e:
            self.__logger.error("No Stages found in the config file? " +e )
            self.stages = None

        #
        # register the callback to take a snapshot triggered by the ESP32
        self.registerCaptureCallback()
        
        # register OTA callback
        self.registerOTACallback()

        # register the callbacks for emitting serial-related signals
        if hasattr(self._master.UC2ConfigManager, "ESP32"):
            try:
                self._master.UC2ConfigManager.ESP32.serial.setWriteCallback(self.processSerialWriteMessage)
                self._master.UC2ConfigManager.ESP32.serial.setReadCallback(self.processSerialReadMessage)
            except Exception as e:
                self._logger.error(f"Could not register serial callbacks: {e}")


        # Connect buttons to the logic handlers
        if IS_HEADLESS:
            return
        # Connect buttons to the logic handlers
        self._widget.setPositionXBtn.clicked.connect(self.set_positionX)
        self._widget.setPositionYBtn.clicked.connect(self.set_positionY)
        self._widget.setPositionZBtn.clicked.connect(self.set_positionZ)
        self._widget.setPositionABtn.clicked.connect(self.set_positionA)

        self._widget.autoEnableBtn.clicked.connect(self.set_auto_enable)
        self._widget.unsetAutoEnableBtn.clicked.connect(self.unset_auto_enable)
        self._widget.reconnectButton.clicked.connect(self.reconnect)
        self._widget.closeConnectionButton.clicked.connect(self.closeConnection)
        self._widget.btpairingButton.clicked.connect(self.btpairing)
        self._widget.stopCommunicationButton.clicked.connect(self.interruptSerialCommunication)

    def processSerialWriteMessage(self, message):
        self.sigUC2SerialWriteMessage.emit(message)

    def processSerialReadMessage(self, message):
        self.sigUC2SerialReadMessage.emit(message)

    def registerCaptureCallback(self):
        # This will capture an image based on a signal coming from the ESP32
        def snapImage(value):
            self.detector_names = self._master.detectorsManager.getAllDeviceNames()
            self.detector = self._master.detectorsManager[self.detector_names[0]]
            mImage = self.detector.getLatestFrame()
            # save image
            drivePath = dirtools.UserFileDirs.Data
            timeStamp = datetime.datetime.now().strftime("%Y_%m_%d")
            dirPath = os.path.join(drivePath, 'recordings', timeStamp)
            fileName  = "Snapshot_"+datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            filePath = os.path.join(dirPath, fileName)
            self.__logger.debug(f"Saving image to {filePath}.tif")
            if mImage is not None:
                if mImage.ndim == 2:
                    tif.imwrite(filePath + ".tif", mImage)
                elif mImage.ndim == 3:
                    tif.imwrite(filePath + ".tif", mImage[0])
                else:
                    self.__logger.error("Image is not 2D or 3D")
            else:
                self.__logger.error("Image is None")

            # (detectorName, image, init, scale, isCurrentDetector)
            self._commChannel.sigUpdateImage.emit('Image', mImage, True, 1, False)

        def printCallback(value):
            self.__logger.debug(f"Callback called with value: {value}")
        try:
            self.__logger.debug("Registering callback for snapshot")
            # register default callback
            for i in range(1, self._master.UC2ConfigManager.ESP32.message.nCallbacks):
                self._master.UC2ConfigManager.ESP32.message.register_callback(i, printCallback)
            self._master.UC2ConfigManager.ESP32.message.register_callback(1, snapImage) # FIXME: Too hacky?

        except Exception as e:
            self.__logger.error(f"Could not register callback: {e}")

    def registerOTACallback(self):
        """Register callback for OTA status updates from CAN devices."""
        def ota_callback(ota_response):
            """
            Handle OTA status updates from CAN devices.
            
            :param ota_response: Dictionary containing:
                - canId: CAN ID of the device
                - status: 0=success, 1=wifi_failed, 2=ota_failed
                - statusMsg: Human readable message
                - ip: Device IP address (if successful)
                - hostname: Device hostname (e.g., "UC2-CAN-14.local")
                - success: True if status == 0
            """
            can_id = ota_response.get("canId")
            
            # Update internal status tracking
            with self._ota_lock:
                if can_id not in self._ota_status:
                    self._ota_status[can_id] = {}
                
                self._ota_status[can_id].update({
                    "status": ota_response.get("status"),
                    "statusMsg": ota_response.get("statusMsg"),
                    "ip": ota_response.get("ip"),
                    "hostname": ota_response.get("hostname"),
                    "success": ota_response.get("success"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "upload_status": "pending" if ota_response.get("success") else "failed"
                })
            
            # Log the status
            if ota_response.get("success"):
                self.__logger.info(f"Device {can_id} ready for OTA at {ota_response.get('ip')}")
                
                # Trigger firmware upload if we have a firmware server configured
                if self._firmware_server_url:
                    threading.Thread(
                        target=self._upload_firmware_to_device,
                        args=(can_id, ota_response.get("ip")),
                        daemon=True
                    ).start()
            else:
                self.__logger.error(f"❌ OTA setup failed for device {can_id}: {ota_response.get('statusMsg')}")
            
            # Emit signal for external listeners
            self.sigOTAStatusUpdate.emit(ota_response)
        
        try:
            # Register callback with UC2 client's canota module
            if hasattr(self._master.UC2ConfigManager, "ESP32") and hasattr(self._master.UC2ConfigManager.ESP32, "canota"):
                self._master.UC2ConfigManager.ESP32.canota.register_callback(0, ota_callback)
                self.__logger.debug("OTA callback registered successfully")
            else:
                self.__logger.warning("UC2 ESP32 client does not have canota module")
        except Exception as e:
            self.__logger.error(f"Could not register OTA callback: {e}")
    
    def _upload_firmware_to_device(self, can_id, ip_address):
        """
        Upload firmware to a device via Arduino OTA using espota.py.
        
        Uses the official espota.py script from Arduino/PlatformIO framework.
        This is the most reliable method as it uses the exact same tool
        that PlatformIO and Arduino IDE use internally.
        
        :param can_id: CAN ID of the device
        :param ip_address: IP address of the device
        """
        try:
            import subprocess
            import sys
            
            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["upload_status"] = "downloading"
            
            # Download firmware from server
            firmware_path = self._download_firmware_for_device(can_id)
            
            if not firmware_path:
                self.__logger.error(f"No firmware found for device {can_id}")
                with self._ota_lock:
                    if can_id in self._ota_status:
                        self._ota_status[can_id]["upload_status"] = "failed"
                        self._ota_status[can_id]["upload_error"] = "No firmware file found on server"
                return
            
            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["upload_status"] = "uploading"
            
            self.__logger.info(f"Uploading firmware to device {can_id} at {ip_address}: {firmware_path.name}")
            
            # Try to find espota.py in common locations
            espota_locations = [
                # PlatformIO locations
                os.path.expanduser("~/.platformio/packages/framework-arduinoespressif32/tools/espota.py"),
                os.path.expanduser("~/.platformio/packages/tool-espotapy/espota.py"),
                # Arduino IDE locations (macOS)
                "/Applications/Arduino.app/Contents/Java/hardware/espressif/esp32/tools/espota.py",
                # System-wide
                "/usr/local/bin/espota.py",
                "/usr/bin/espota.py",
            ]
            
            espota_path = None
            for location in espota_locations:
                if os.path.exists(location):
                    espota_path = location
                    self.__logger.debug(f"Found espota.py at: {espota_path}")
                    break
            
            if not espota_path:
                # Try to find espota.py using which/where
                try:
                    result = subprocess.run(['which', 'espota.py'], 
                                          capture_output=True, text=True, check=True)
                    espota_path = result.stdout.strip()
                except:
                    pass
            
            if not espota_path:
                self.__logger.error("espota.py not found in common locations")
                self.__logger.info("Searched locations: " + ", ".join(espota_locations))
                raise Exception("espota.py not found. Please install PlatformIO or Arduino IDE with ESP32 support")
            
            # Get firmware size for logging
            firmware_size = os.path.getsize(firmware_path)
            self.__logger.info(f"Firmware size: {firmware_size:,} bytes")
            
            # Prepare espota.py command
            # Use sys.executable to ensure we use the same Python interpreter
            cmd = [
                sys.executable,
                espota_path,
                "--debug",
                "--progress",
                "-i", ip_address,
                "-p", "3232",  # Default ESP32 OTA port
                "-f", str(firmware_path)
            ]
            
            self.__logger.debug(f"Running: {' '.join(cmd)}")
            
            # Run espota.py as subprocess
            try:
                # Run with real-time output capture
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                # Track progress
                last_progress = 0
                
                # Read output in real-time
                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.strip()
                        
                        # Log important messages
                        if "Uploading:" in line or "%" in line:
                            # Extract progress percentage
                            try:
                                if "%" in line:
                                    percent_str = line.split("%")[0].split()[-1]
                                    progress = int(percent_str)
                                    if progress >= last_progress + 10:
                                        self.__logger.info(f"Upload progress: {progress}%")
                                        last_progress = progress
                            except:
                                pass
                        elif "Success" in line:
                            self.__logger.info(f"✅ {line}")
                        elif "Error" in line or "FAIL" in line:
                            self.__logger.error(f"espota.py: {line}")
                        else:
                            self.__logger.debug(f"espota.py: {line}")
                
                # Get return code
                return_code = process.poll()
                
                if return_code == 0:
                    self.__logger.info(f"✅ Firmware uploaded successfully to device {can_id}")
                    with self._ota_lock:
                        if can_id in self._ota_status:
                            self._ota_status[can_id]["upload_status"] = "success"
                            self._ota_status[can_id]["upload_timestamp"] = datetime.datetime.now().isoformat()
                else:
                    # Read any remaining error output
                    _, stderr = process.communicate()
                    error_msg = stderr.strip() if stderr else f"Exit code: {return_code}"
                    
                    self.__logger.error(f"espota.py failed: {error_msg}")
                    with self._ota_lock:
                        if can_id in self._ota_status:
                            self._ota_status[can_id]["upload_status"] = "failed"
                            self._ota_status[can_id]["upload_error"] = error_msg
                            
            except subprocess.CalledProcessError as e:
                error_msg = f"espota.py failed with exit code {e.returncode}"
                self.__logger.error(error_msg)
                if e.stderr:
                    self.__logger.error(f"Error output: {e.stderr}")
                    
                with self._ota_lock:
                    if can_id in self._ota_status:
                        self._ota_status[can_id]["upload_status"] = "failed"
                        self._ota_status[can_id]["upload_error"] = error_msg
                        
        except FileNotFoundError as e:
            self.__logger.error(f"File not found: {e}")
            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["upload_status"] = "failed"
                    self._ota_status[can_id]["upload_error"] = str(e)
                    
        except Exception as e:
            self.__logger.error(f"Error uploading firmware to device {can_id}: {e}")
            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["upload_status"] = "failed"
                    self._ota_status[can_id]["upload_error"] = str(e)
    
    def _download_firmware_for_device(self, can_id):
        """
        Download firmware for a specific CAN ID from the firmware server.
        
        Queries the server for available firmware matching the pattern id_<CANID>_*.bin
        and downloads it to a local cache.
        
        :param can_id: CAN ID of the device
        :return: Path to downloaded firmware file or None
        """
        try:
            # First, get list of available firmware files from server
            firmware_list_url = f"{self._firmware_server_url}/latest/"
            
            self.__logger.debug(f"Fetching firmware list from {firmware_list_url}")
            response = requests.get(firmware_list_url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML directory listing to find matching firmware
            from html.parser import HTMLParser
            
            class FirmwareLinkParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.firmware_files = []
                
                def handle_starttag(self, tag, attrs):
                    if tag == 'a':
                        for attr, value in attrs:
                            if attr == 'href' and value.endswith('.bin'):
                                self.firmware_files.append(value)
            
            parser = FirmwareLinkParser()
            parser.feed(response.text)
            
            # Find firmware file matching the CAN ID pattern: id_<CANID>_*.bin
            target_pattern = f"id_{can_id}_"
            matching_files = [f for f in parser.firmware_files if f.startswith(target_pattern)]
            
            if not matching_files:
                # Try to find generic firmware based on device type
                device_type_map = {
                    range(10, 14): "motor",  # 10-13: motors (A, X, Y, Z)
                    range(20, 30): "laser",  # 20-29: lasers
                    range(30, 40): "led",    # 30-39: LEDs
                }
                
                device_type = None
                for id_range, dtype in device_type_map.items():
                    if can_id in id_range:
                        device_type = dtype
                        break
                
                if device_type:
                    # Look for any firmware containing the device type
                    matching_files = [f for f in parser.firmware_files if device_type in f]
                    if matching_files:
                        self.__logger.warning(f"Using generic {device_type} firmware for device {can_id}: {matching_files[0]}")
            
            if not matching_files:
                self.__logger.error(f"No firmware found for device {can_id} on server")
                return None
            
            # Use the first matching file
            firmware_filename = matching_files[0]
            
            # Download the firmware file
            firmware_url = f"{self._firmware_server_url}/latest/{firmware_filename}" # TODO: We should make this adaptable 
            local_path = self._firmware_cache_dir / firmware_filename
            
            self.__logger.info(f"Downloading firmware from {firmware_url}")
            
            # Check if already cached
            if False and local_path.exists():#TODO: Always redownload for now
                self.__logger.debug(f"Using cached firmware: {local_path}")
                return local_path
            
            # Download firmware
            with requests.get(firmware_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            
            self.__logger.info(f"Downloaded firmware to {local_path}")
            return local_path
            
        except Exception as e:
            self.__logger.error(f"Error downloading firmware for device {can_id}: {e}")
            return None

    def set_motor_positions(self, a, x, y, z):
        # Add your logic to set motor positions here.
        self.__logger.debug(f"Setting motor positions: A={a}, X={x}, Y={y}, Z={z}")
        # push the positions to the motor controller
        if a is not None: self.stages.setPositionOnDevice(value=float(a), axis="A")
        if x is not None:  self.stages.setPositionOnDevice(value=float(x), axis="X")
        if y is not None: self.stages.setPositionOnDevice(value=float(y), axis="Y")
        if z is not None: self.stages.setPositionOnDevice(value=float(z), axis="Z")

        # retrieve the positions from the motor controller
        positions = self.stages.getPosition()
        if not IS_HEADLESS: self._widget.reconnectDeviceLabel.setText("Motor positions: A="+str(positions["A"])+", X="+str(positions["X"])+", \n Y="+str(positions["Y"])+", Z="+str(positions["Z"]))
        # update the GUI
        self._commChannel.sigUpdateMotorPosition.emit()

    def interruptSerialCommunication(self):
        self._master.UC2ConfigManager.interruptSerialCommunication()
        if not IS_HEADLESS: self._widget.reconnectDeviceLabel.setText("We are intrrupting the last command")

    def set_auto_enable(self):
        # Add your logic to auto-enable the motors here.
        # get motor controller
        self.stages.enalbeMotors(enableauto=True)

    def unset_auto_enable(self):
        # Add your logic to unset auto-enable for the motors here.
        self.stages.enalbeMotors(enable=True, enableauto=False)

    def set_positionX(self):
        if not IS_HEADLESS: x = self._widget.motorXEdit.text() # TODO: Should be a signal for all motors
        self.set_motor_positions(None, x, None, None)

    def set_positionY(self):
        if not IS_HEADLESS: y = self._widget.motorYEdit.text()
        self.set_motor_positions(None, None, y, None)

    def set_positionZ(self):
        if not IS_HEADLESS: z = self._widget.motorZEdit.text()
        self.set_motor_positions(None, None, None, z)

    def set_positionA(self):
        if not IS_HEADLESS: a = self._widget.motorAEdit.text()
        self.set_motor_positions(a, None, None, None)

    def reconnectThread(self, baudrate=None):
        self._master.UC2ConfigManager.initSerial(baudrate=baudrate)
        if not IS_HEADLESS:
            self._widget.reconnectDeviceLabel.setText("We are connected: "+str(self._master.UC2ConfigManager.isConnected()))
        else:
            self.__logger.debug("We are connected: "+str(self._master.UC2ConfigManager.isConnected()))
            self.sigUC2SerialIsConnected.emit(self._master.UC2ConfigManager.isConnected())

    def closeConnection(self):
        self._master.UC2ConfigManager.closeSerial()
        if not IS_HEADLESS: self._widget.reconnectDeviceLabel.setText("Connection to ESP32 closed.")

    @APIExport(runOnUIThread=True)
    def moveToSampleMountingPosition(self):
        self._logger.debug('Moving to sample loading position.')
        self.stages.moveToSampleMountingPosition()

    @APIExport(runOnUIThread=False)
    def stopImSwitch(self):
        self._commChannel.sigExperimentStop.emit()
        return {"message": "ImSwitch is shutting down"}

    @APIExport(runOnUIThread=False)
    def restartImSwitch(self):
        ostools.restartSoftware()
        return {"message": "ImSwitch is restarting"}

    @APIExport(runOnUIThread=False)
    def isImSwitchRunning(self):
        return True

    @APIExport(runOnUIThread=False)
    def getDiskUsage(self):
        return dirtools.getDiskusage()

    @APIExport(runOnUIThread=False)
    def getDataPath(self):
        return dirtools.UserFileDirs.Data

    @APIExport(runOnUIThread=False)
    def setDataPathFolder(self, path):
        dirtools.UserFileDirs.Data = path
        self._logger.debug(f"Data path set to {path}")
        return {"message": f"Data path set to {path}"}

    @APIExport(runOnUIThread=True)
    def reconnect(self):
        self._logger.debug('Reconnecting to ESP32 device.')
        baudrate = None
        if not IS_HEADLESS:
            self._widget.reconnectDeviceLabel.setText("Reconnecting to ESP32 device.")
            if self._widget.getBaudRateGui() in (115200, 500000):
                baudrate = self._widget.getBaudRateGui()
        mThread = threading.Thread(target=self.reconnectThread, args=(baudrate,))
        mThread.start()

    @APIExport(runOnUIThread=True)
    def writeSerial(self, payload):
        return self._master.UC2ConfigManager.ESP32.serial.writeSerial(payload)

    @APIExport(runOnUIThread=True)
    def is_connected(self):
        return self._master.UC2ConfigManager.isConnected()

    @APIExport(runOnUIThread=True)
    def btpairing(self):
        self._logger.debug('Pairing BT device.')
        mThread = threading.Thread(target=self._master.UC2ConfigManager.pairBT)
        mThread.start()
        mThread.join()
        if not IS_HEADLESS: self._widget.reconnectDeviceLabel.setText("Bring the PS controller into pairing mode")

    @APIExport(runOnUIThread=True)
    def restartCANDevice(self, device_id=0):
        self._logger.debug('Restarting CAN device.')
        self._master.UC2ConfigManager.restartCANDevice(device_id)

    @APIExport(runOnUIThread=False)
    def espRestart(self):
        try:
            self._master.UC2ConfigManager.restartESP()
            return {"status": "ESP32 restarted successfully"}
        except Exception as e:
            return {"error": str(e)}
         
    
    
    ''' CAN OTA Update Methods '''
    
    @APIExport(runOnUIThread=False)
    def setOTAWiFiCredentials(self, ssid, password):
        """
        Set WiFi credentials for OTA updates.
        
        :param ssid: WiFi network name
        :param password: WiFi password
        :return: Status message
        """
        self._ota_wifi_ssid = ssid
        self._ota_wifi_password = password
        self.__logger.info(f"OTA WiFi credentials set: SSID={ssid}")
        return {"status": "success", "message": f"WiFi credentials set for SSID: {ssid}"}
    
    @APIExport(runOnUIThread=False)
    def setOTAFirmwareServer(self, server_url="http://localhost:9000"):
        """
        Set the firmware server URL for OTA updates.
        
        The server should serve firmware files at <server_url>/latest/ with the naming convention:
        - id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
        - id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
        - id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin
        - id_21_esp32_seeed_xiao_esp32s3_can_slave_led_debug.bin
        
        :param server_url: URL of the firmware server (default: http://localhost:9000)
        :return: Status message with list of available firmware files
        """
        # Remove trailing slash if present
        server_url = server_url.rstrip('/')
        
        try:
            # Test server connectivity
            test_url = f"{server_url}/latest/"
            self.__logger.debug(f"Testing firmware server: {test_url}")
            
            response = requests.get(test_url, timeout=5)
            response.raise_for_status()
            
            # Parse available firmware files
            from html.parser import HTMLParser
            
            class FirmwareLinkParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.firmware_files = []
                
                def handle_starttag(self, tag, attrs):
                    if tag == 'a':
                        for attr, value in attrs:
                            if attr == 'href' and value.endswith('.bin'):
                                self.firmware_files.append(value)
            
            parser = FirmwareLinkParser()
            parser.feed(response.text)
            
            self._firmware_server_url = server_url
            
            self.__logger.info(f"OTA firmware server set: {server_url}")
            self.__logger.info(f"Found {len(parser.firmware_files)} firmware files")
            
            return {
                "status": "success",
                "message": f"Firmware server set: {server_url}",
                "server_url": server_url,
                "firmware_files": parser.firmware_files,
                "count": len(parser.firmware_files)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to connect to firmware server: {str(e)}",
                "server_url": server_url
            }
    
    @APIExport(runOnUIThread=False)
    def setOTAFirmwareDirectory(self, directory):
        """
        DEPRECATED: Use setOTAFirmwareServer instead.
        
        This method is kept for backward compatibility but now configures
        the firmware server URL based on a local directory assumption.
        """
        self.__logger.warning("setOTAFirmwareDirectory is deprecated. Use setOTAFirmwareServer instead.")
        return self.setOTAFirmwareServer()
    
    @APIExport(runOnUIThread=False)
    def listAvailableFirmware(self):
        """
        List all available firmware files from the configured server.
        
        Fetches the directory listing from <server_url>/latest/ and parses
        available firmware files.
        
        :return: Dictionary with firmware files organized by device type
        """
        if not self._firmware_server_url:
            return {
                "status": "error",
                "message": "Firmware server not set. Use setOTAFirmwareServer first."
            }
        
        try:
            # Fetch firmware list from server
            list_url = f"{self._firmware_server_url}/latest/"
            self.__logger.debug(f"Fetching firmware list from {list_url}")
            
            response = requests.get(list_url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML directory listing
            from html.parser import HTMLParser
            
            class FirmwareLinkParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.firmware_files = []
                
                def handle_starttag(self, tag, attrs):
                    if tag == 'a':
                        for attr, value in attrs:
                            if attr == 'href' and value.endswith('.bin'):
                                self.firmware_files.append(value)
            
            parser = FirmwareLinkParser()
            parser.feed(response.text)
            
            # Organize by device ID
            firmware_by_id = {}
            for fw_file in parser.firmware_files:
                # Extract CAN ID from filename (e.g., "id_20_..." -> 20)
                try:
                    parts = fw_file.split('_')
                    if len(parts) >= 2 and parts[0] == 'id':
                        can_id = int(parts[1])
                        firmware_url = f"{self._firmware_server_url}/latest/{fw_file}"
                        
                        firmware_by_id[can_id] = {
                            "filename": fw_file,
                            "url": firmware_url,
                            "can_id": can_id
                        }
                except (ValueError, IndexError):
                    continue
            
            return {
                "status": "success",
                "firmware_server": self._firmware_server_url,
                "firmware_count": len(firmware_by_id),
                "firmware": firmware_by_id
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to fetch firmware list from server: {str(e)}",
                "server_url": self._firmware_server_url
            }
    
    @APIExport(runOnUIThread=False)
    def startSingleDeviceOTA(self, can_id:int, ssid:str=None, password:str=None, timeout:int=300000):
        """
        Start OTA update for a single CAN device.
        
        This will:
        1. Send OTA command to the device via CAN
        2. Device connects to WiFi and starts ArduinoOTA server
        3. Device sends back IP address via callback
        4. Firmware is automatically uploaded (if firmware directory is set)
        
        :param can_id: CAN ID of the device (e.g., 11=Motor X, 20=Laser, 30=LED)
        :param ssid: WiFi SSID (optional, uses configured credentials if not provided)
        :param password: WiFi password (optional, uses configured credentials if not provided)
        :param timeout: OTA timeout in milliseconds (default: 5 minutes)
        :return: Status message
        """
        # Use provided credentials or fall back to configured ones
        wifi_ssid = ssid or self._ota_wifi_ssid
        wifi_password = password or self._ota_wifi_password
        
        if not wifi_ssid or not wifi_password:
            return {
                "status": "error",
                "message": "WiFi credentials not provided. Use setOTAWiFiCredentials or provide ssid/password parameters."
            }
        
        # Check if ESP32 client has canota module
        if not hasattr(self._master.UC2ConfigManager, "ESP32"):
            return {"status": "error", "message": "ESP32 client not available"}
        
        if not hasattr(self._master.UC2ConfigManager.ESP32, "canota"):
            return {"status": "error", "message": "CAN OTA module not available in UC2 client"}
        
        # Initialize status tracking
        with self._ota_lock:
            self._ota_status[can_id] = {
                "status": "initiated",
                "timestamp": datetime.datetime.now().isoformat(),
                "can_id": can_id,
                "upload_status": "waiting"
            }
        
        try:
            # Send OTA command to device
            self.__logger.info(f"Starting OTA update for device {can_id}")
            response = self._master.UC2ConfigManager.ESP32.canota.start_ota_update(
                can_id=can_id,
                ssid=wifi_ssid,
                password=wifi_password,
                timeout=timeout,
                is_blocking=False
            )
            
            return {
                "status": "success",
                "message": f"OTA update initiated for device {can_id}",
                "can_id": can_id,
                "command_response": response
            }
            
        except Exception as e:
            self.__logger.error(f"Error starting OTA for device {can_id}: {e}")
            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["status"] = "error"
                    self._ota_status[can_id]["error"] = str(e)
            
            return {
                "status": "error",
                "message": f"Failed to start OTA for device {can_id}: {str(e)}"
            }
    
    @APIExport(runOnUIThread=False)
    def startMultipleDeviceOTA(self, can_ids, ssid=None, password=None, timeout=300000, delay_between=2):
        """
        Start OTA update for multiple CAN devices sequentially.
        
        :param can_ids: List of CAN IDs (e.g., [11, 12, 13, 20, 30])
        :param ssid: WiFi SSID (optional, uses configured credentials if not provided)
        :param password: WiFi password (optional, uses configured credentials if not provided)
        :param timeout: OTA timeout in milliseconds per device
        :param delay_between: Delay in seconds between starting each device
        :return: Status message with results for each device
        """
        if not isinstance(can_ids, list):
            return {"status": "error", "message": "can_ids must be a list"}
        
        results = []
        
        for can_id in can_ids:
            result = self.startSingleDeviceOTA(
                can_id=can_id,
                ssid=ssid,
                password=password,
                timeout=timeout
            )
            results.append({
                "can_id": can_id,
                "result": result
            })
            
            # Small delay between commands to avoid overwhelming the CAN bus
            if delay_between > 0:
                import time
                time.sleep(delay_between)
        
        return {
            "status": "success",
            "message": f"OTA update initiated for {len(can_ids)} devices",
            "results": results
        }
    
    @APIExport(runOnUIThread=False)
    def getOTAStatus(self, can_id=None):
        """
        Get OTA status for one or all devices.
        
        :param can_id: CAN ID of specific device (optional, returns all if not provided)
        :return: Dictionary with OTA status information
        """
        with self._ota_lock:
            if can_id is not None:
                # Return status for specific device
                if can_id in self._ota_status:
                    return {
                        "status": "success",
                        "can_id": can_id,
                        "ota_status": self._ota_status[can_id]
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"No OTA status available for device {can_id}"
                    }
            else:
                # Return status for all devices
                return {
                    "status": "success",
                    "device_count": len(self._ota_status),
                    "devices": self._ota_status
                }
    
    @APIExport(runOnUIThread=False)
    def clearOTAStatus(self, can_id=None):
        """
        Clear OTA status tracking.
        
        :param can_id: CAN ID of specific device (optional, clears all if not provided)
        :return: Status message
        """
        with self._ota_lock:
            if can_id is not None:
                if can_id in self._ota_status:
                    del self._ota_status[can_id]
                    return {"status": "success", "message": f"Cleared OTA status for device {can_id}"}
                else:
                    return {"status": "error", "message": f"No OTA status found for device {can_id}"}
            else:
                count = len(self._ota_status)
                self._ota_status.clear()
                return {"status": "success", "message": f"Cleared OTA status for {count} devices"}
    
    @APIExport(runOnUIThread=False)
    def getOTADeviceMapping(self):
        """
        Get mapping of CAN IDs to device types.
        
        :return: Dictionary with device type mappings
        """
        return {
            "status": "success",
            "mapping": {
                "motors": {
                    "A": 10,
                    "X": 11,
                    "Y": 12,
                    "Z": 13
                },
                "laser": {
                    "laser_0": 20,
                    "laser_1": 21,
                    "laser_2": 22
                },
                "led": {
                    "led_0": 30,
                    "led_1": 31
                },
                "master": 1
            },
            "description": "CAN ID mapping for UC2 devices"
        }
    
    @APIExport(runOnUIThread=False)
    def clearOTAFirmwareCache(self):
        """
        Clear the local firmware cache directory.
        
        This removes all downloaded firmware files from the cache.
        Useful for forcing fresh downloads on next OTA update.
        
        :return: Status message
        """
        try:
            if self._firmware_cache_dir.exists():
                cache_files = list(self._firmware_cache_dir.glob("*.bin"))
                count = len(cache_files)
                
                for cache_file in cache_files:
                    cache_file.unlink()
                
                self.__logger.info(f"Cleared {count} files from firmware cache")
                return {
                    "status": "success",
                    "message": f"Cleared {count} cached firmware files",
                    "cache_directory": str(self._firmware_cache_dir)
                }
            else:
                return {
                    "status": "success",
                    "message": "Cache directory does not exist",
                    "cache_directory": str(self._firmware_cache_dir)
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to clear cache: {str(e)}"
            }
    
    @APIExport(runOnUIThread=False)
    def getOTAFirmwareCacheStatus(self):
        """
        Get status of the firmware cache directory.
        
        :return: Dictionary with cache information
        """
        try:
            if not self._firmware_cache_dir.exists():
                return {
                    "status": "success",
                    "cache_directory": str(self._firmware_cache_dir),
                    "exists": False,
                    "cached_files": [],
                    "total_size": 0
                }
            
            cache_files = list(self._firmware_cache_dir.glob("*.bin"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            file_info = []
            for cache_file in cache_files:
                stat = cache_file.stat()
                file_info.append({
                    "filename": cache_file.name,
                    "size": stat.st_size,
                    "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            return {
                "status": "success",
                "cache_directory": str(self._firmware_cache_dir),
                "exists": True,
                "cached_files": file_info,
                "file_count": len(cache_files),
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get cache status: {str(e)}"
            }
    



# Copyright (C) Benedict Diederich
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
