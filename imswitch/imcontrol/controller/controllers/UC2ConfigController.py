import os
import threading
from imswitch import IS_HEADLESS
import datetime
from imswitch.imcommon.model import APIExport, initLogger, dirtools, ostools
from imswitch.imcommon.framework import Signal
from ..basecontrollers import ImConWidgetController
import tifffile as tif
import tempfile
import requests
import shutil
from pathlib import Path
from serial.tools import list_ports
import socket
import sys
import subprocess
import time

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
    sigUSBFlashStatusUpdate = Signal(object)  # Emits USB flash status updates
    sigCameraTrigger = Signal(object)  # Emits camera trigger events from hardware


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        # OTA update tracking
        self._ota_status = {}  # Dictionary to track OTA status by CAN ID
        self._ota_lock = threading.Lock()
        self._firmware_server_url = "http://localhost/firmware"  # Firmware server URL (must end with /)
        self._firmware_cache_dir = Path(tempfile.gettempdir()) / "uc2_ota_firmware_cache"
        self._firmware_cache_dir.mkdir(parents=True, exist_ok=True)

        # Prevent concurrent flashing attempts
        self._usb_flash_lock = threading.Lock()


        # WiFi credentials for OTA (can be overridden via API)
        self._ota_wifi_ssid = socket.gethostname().split(".local")[0]
        self._ota_wifi_password = "youseetoo" # this is the default password for forklifted UC2 firmwares

        try:
            self.stages = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]
        except Exception as e:
            self.__logger.error("No Stages found in the config file? ", e )
            self.stages = None

        #
        # register the callback to take a snapshot triggered by the ESP32
        self.registerCaptureCallback()

        # register OTA callback
        self.registerOTACallback()

        # register CAN callback
        self.registerCANCallback()

        # register camera trigger callback for performance mode
        self.registerCameraTriggerCallback()

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

    def _get_can_id_firmware_mapping(self):
        """
        Centralized CAN ID to firmware filename mapping.
        
        This mapping is used by both firmware download and listing operations
        to ensure consistent firmware assignment across the system.
        
        :return: Dictionary mapping CAN IDs to firmware filenames
        """
        return {
            1: "esp32_UC2_3_CAN_HAT_Master.bin",  # Master firmware (USB connected, use esptool)
            10: "esp32_seeed_xiao_esp32s3_can_slave_motor.bin",  # Motor A
            11: "esp32_seeed_xiao_esp32s3_can_slave_motor.bin",  # Motor X
            12: "esp32_seeed_xiao_esp32s3_can_slave_motor.bin",  # Motor Y
            13: "esp32_seeed_xiao_esp32s3_can_slave_motor.bin",  # Motor Z
            14: "esp32_seeed_xiao_esp32s3_can_slave_motor.bin",  # Additional motor
            15: "esp32_seeed_xiao_esp32s3_can_slave_motor.bin",  # Additional motor
            20: "esp32_seeed_xiao_esp32s3_can_slave_illumination.bin",  # Laser 0
            21: "esp32_seeed_xiao_esp32s3_can_slave_illumination.bin",  # Laser 1
            22: "esp32_seeed_xiao_esp32s3_can_slave_illumination.bin",  # Laser 2
            30: "esp32_seeed_xiao_esp32s3_can_slave_illumination.bin",  # LED 0
            31: "esp32_seeed_xiao_esp32s3_can_slave_illumination.bin",  # LED 1
            40: "esp32_seeed_xiao_esp32s3_can_slave_galvo.bin",  # Galvo mirror
        }

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
            drivePath = dirtools.UserFileDirs.getValidatedDataPath()
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
                    "can_id": can_id,  # Ensure can_id is always in the status
                    "status": ota_response.get("status"),
                    "statusMsg": ota_response.get("statusMsg"),
                    "ip": ota_response.get("ip"),
                    "hostname": ota_response.get("hostname"),
                    "success": ota_response.get("success"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "upload_status": "wifi_connected" if ota_response.get("success") else "wifi_failed",
                    "message": f"Device {can_id} connected to WiFi at {ota_response.get('ip')}" if ota_response.get("success") else f"WiFi connection failed: {ota_response.get('statusMsg')}"
                })

                # Get the current status dict for emission
                current_status = self._ota_status[can_id].copy()

            # Log the status
            if ota_response.get("success"):
                self.__logger.info(f"✅ Device {can_id} ready for OTA at {ota_response.get('ip')}")

                # Emit WiFi connection success to frontend BEFORE starting upload
                self.sigOTAStatusUpdate.emit(current_status)

                # Trigger firmware upload if we have a firmware server configured
                if self._firmware_server_url:
                    threading.Thread(
                        target=self._upload_firmware_to_device,
                        args=(can_id, ota_response.get("ip")),
                        daemon=True
                    ).start()
                else:
                    # No firmware server configured, mark as ready for manual upload
                    with self._ota_lock:
                        self._ota_status[can_id]["upload_status"] = "ready_for_upload"
                        self._ota_status[can_id]["message"] = f"Device {can_id} ready for firmware upload (no server configured)"
                    self.sigOTAStatusUpdate.emit(self._ota_status[can_id])
            else:
                self.__logger.error(f"❌ OTA setup failed for device {can_id}: {ota_response.get('statusMsg')}")
                # Emit failure status to frontend
                self.sigOTAStatusUpdate.emit(current_status)

        try:
            # Register callback with UC2 client's canota module
            if hasattr(self._master.UC2ConfigManager, "ESP32") and hasattr(self._master.UC2ConfigManager.ESP32, "canota"):
                self._master.UC2ConfigManager.ESP32.canota.register_callback(0, ota_callback)
                self.__logger.debug("OTA callback registered successfully")
            else:
                self.__logger.warning("UC2 ESP32 client does not have canota module")
        except Exception as e:
            self.__logger.error(f"Could not register OTA callback: {e}")

    def registerCANCallback(self):
        """Register callback for CAN device scan updates."""
        def can_scan_callback(scan_results):
            """
            Handle CAN scan results from ESP32.
            
            :param scan_results: List of CAN devices with their information
                                Example: [
                                    {"canId": 20, "deviceType": 1, "status": 0, "deviceTypeStr": "laser", "statusStr": "idle"},
                                    {"canId": 10, "deviceType": 0, "status": 0, "deviceTypeStr": "motor", "statusStr": "idle"}
                                ]
            """
            try:
                self.__logger.info(f"CAN scan callback received {len(scan_results)} devices")
                for device in scan_results:
                    self.__logger.debug(f"CAN Device: ID={device.get('canId')}, "
                                      f"Type={device.get('deviceTypeStr')}, "
                                      f"Status={device.get('statusStr')}")

                # Emit signal to update GUI or notify other components
                self._commChannel.sigUpdateCANDevices.emit(scan_results)

            except Exception as e:
                self.__logger.error(f"Error in CAN scan callback: {e}")

        try:
            if hasattr(self._master.UC2ConfigManager, "ESP32") and hasattr(self._master.UC2ConfigManager.ESP32, "can"):
                self._master.UC2ConfigManager.ESP32.can.register_callback(0, can_scan_callback)
                self.__logger.debug("CAN scan callback registered successfully")
            else:
                self.__logger.warning("ESP32 CAN not available - CAN scan callbacks won't work")
        except Exception as e:
            self.__logger.error(f"Could not register CAN callback: {e}")

    def registerCameraTriggerCallback(self):
        """
        Register callback for camera trigger events from firmware.
        
        When the firmware sends {"cam":1}, this callback emits a signal
        that can be used by the ExperimentController for software-triggered
        acquisition in performance mode.
        """
        def camera_trigger_callback(trigger_info):
            """
            Handle camera trigger events from ESP32.
            
            :param trigger_info: Dictionary containing:
                - trigger: Trigger signal (1 = trigger)
                - frame_id: Frame number
                - timestamp: Unix timestamp
                - illumination: Optional illumination channel index
            """
            try:
                self.__logger.debug(f"Camera trigger received: frame {trigger_info.get('frame_id')}")
                
                # Emit signal for ExperimentController or other listeners
                self.sigCameraTrigger.emit(trigger_info)
                
            except Exception as e:
                self.__logger.error(f"Error in camera trigger callback: {e}")

        try:
            if hasattr(self._master.UC2ConfigManager, "ESP32") and hasattr(self._master.UC2ConfigManager.ESP32, "camera_trigger"):
                self._master.UC2ConfigManager.ESP32.camera_trigger.register_callback(0, camera_trigger_callback)
                self.__logger.debug("Camera trigger callback registered successfully")
            else:
                self.__logger.debug("ESP32 camera_trigger module not available - hardware camera triggers won't work")
        except Exception as e:
            self.__logger.error(f"Could not register camera trigger callback: {e}")


    @APIExport(runOnUIThread=False)
    def scan_canbus(self, timeout:int=5) -> dict:
        """
        Scan the CAN bus for connected devices.

        :param timeout: Timeout for the scan in seconds (default: 5)
        :return: List of detected CAN IDs
        
        returns:
        {
            "scan": [
                {
                "canId": 20,
                "deviceType": 1,
                "status": 0,
                "deviceTypeStr": "laser",
                "statusStr": "idle"
                }
            ],
            "detected_ids": [20],
            "qid": 23,
            "count": 1
            }
        """
        try:
            if not hasattr(self._master.UC2ConfigManager, "ESP32") or not hasattr(self._master.UC2ConfigManager.ESP32, "can"):
                self.__logger.error("CAN bus module not available in UC2 client")
                return []

            self.__logger.debug("Starting CAN bus scan...")
            return_message = self._master.UC2ConfigManager.ESP32.can.scan(timeout=timeout)
            scan_results = return_message[-1]
            detected_ids = [device["canId"] for device in scan_results.get("scan", [])]
            scan_results["detected_ids"] = detected_ids
            self.__logger.info(f"Detected CAN devices: {detected_ids}")
            return scan_results

        except Exception as e:
            self.__logger.error(f"Error scanning CAN bus: {e}")
            return {}

    @APIExport(runOnUIThread=False)
    def get_canbus_devices(self, timeout:int=2):
        """
        Get list of available CAN devices.

        :param timeout: Timeout for the command in seconds (default: 2)
        :return: List of available CAN IDs
        """
        try:
            if not hasattr(self._master.UC2ConfigManager, "ESP32") or not hasattr(self._master.UC2ConfigManager.ESP32, "can"):
                self.__logger.error("CAN bus module not available in UC2 client")
                return []

            self.__logger.debug("Fetching available CAN devices...")
            response = self._master.UC2ConfigManager.ESP32.can.get_available_devices(timeout=timeout)
            # the second response should hold the available IDs
            scan_response = response[-1]
            ''' example response:
            {"scan": [
                {"canId": 10, "deviceType": 0, "status": 1, "deviceTypeStr": "motor", "statusStr": "busy"}, 
                {"canId": 11, "deviceType": 0, "status": 0, "deviceTypeStr": "motor", "statusStr": "idle"}, 
                {"canId": 12, "deviceType": 0, "status": 0, "deviceTypeStr": "motor", "statusStr": "idle"}, 
                {"canId": 13, "deviceType": 0, "status": 1, "deviceTypeStr": "motor", "statusStr": "busy"}, 
                {"canId": 20, "deviceType": 1, "status": 0, "deviceTypeStr": "laser", "statusStr": "idle"}, 
                {"canId": 30, "deviceType": 2, "status": 0, "deviceTypeStr": "led", "statusStr": "idle"}], 
                "qid": 125, "count": 6}
            '''
            available_ids = [device["canId"] for device in scan_response.get("scan", [])]

            self.__logger.info(f"Available CAN devices: {available_ids}")
            return available_ids

        except Exception as e:
            self.__logger.error(f"Error fetching CAN devices: {e}")
            return []

    def _upload_firmware_to_device(self, can_id, ip_address):
        """
        Upload firmware to a device via Arduino OTA.
        
        Uses the integrated espota module for OTA updates.
        This is a Python implementation of the standard ESP32 OTA protocol
        used by Arduino IDE and PlatformIO.
        
        :param can_id: CAN ID of the device
        :param ip_address: IP address of the device
        """
        try:
            from . import espota

            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["upload_status"] = "downloading"

            # Download firmware from server
            firmware_path = self._download_firmware_for_device(can_id)

            if not firmware_path:
                error_msg = f"❌ No firmware found for device {can_id}"
                self.__logger.error(error_msg)
                with self._ota_lock:
                    if can_id in self._ota_status:
                        self._ota_status[can_id]["upload_status"] = "failed"
                        self._ota_status[can_id]["upload_error"] = "No firmware file found on server"
                        self._ota_status[can_id]["message"] = error_msg
                        # Emit error update to frontend
                        self.sigOTAStatusUpdate.emit(self._ota_status[can_id])
                return

            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["upload_status"] = "uploading"
                    self._ota_status[can_id]["upload_progress"] = 0
                    self._ota_status[can_id]["message"] = f"Starting upload to device {can_id}..."
                    # Emit upload start to frontend
                    self.sigOTAStatusUpdate.emit(self._ota_status[can_id])

            self.__logger.info(f"Uploading firmware to device {can_id} at {ip_address}: {firmware_path.name}")

            # Get firmware size for logging
            firmware_size = os.path.getsize(firmware_path)
            self.__logger.info(f"Firmware size: {firmware_size:,} bytes")

            # Define progress callback that updates status and emits signal
            def progress_callback(percent):
                self.__logger.info(f"Upload progress: {percent}%")
                with self._ota_lock:
                    if can_id in self._ota_status:
                        self._ota_status[can_id]["upload_progress"] = percent
                        # Emit progress update to frontend
                        self.sigOTAStatusUpdate.emit(self._ota_status[can_id])

            # Upload firmware using integrated espota module
            result = espota.upload_ota(
                esp_ip=ip_address,
                firmware_path=str(firmware_path),
                esp_port=3232,  # Default ESP32 OTA port
                host_ip="0.0.0.0",
                password="",  # No password by default
                timeout=20,
                show_progress=True,
                logger=self.__logger,
                progress_callback=progress_callback
            )

            if result == 0:
                success_msg = f"✅ Firmware uploaded successfully to device {can_id}"
                self.__logger.info(success_msg)
                with self._ota_lock:
                    if can_id in self._ota_status:
                        self._ota_status[can_id]["upload_status"] = "success"
                        self._ota_status[can_id]["upload_progress"] = 100
                        self._ota_status[can_id]["upload_timestamp"] = datetime.datetime.now().isoformat()
                        self._ota_status[can_id]["message"] = success_msg
                        # Emit final success update to frontend
                        self.sigOTAStatusUpdate.emit(self._ota_status[can_id])
            else:
                error_msg = f"❌ OTA upload failed with code {result}"
                self.__logger.error(error_msg)
                with self._ota_lock:
                    if can_id in self._ota_status:
                        self._ota_status[can_id]["upload_status"] = "failed"
                        self._ota_status[can_id]["upload_error"] = error_msg
                        self._ota_status[can_id]["message"] = error_msg
                        # Emit failure update to frontend
                        self.sigOTAStatusUpdate.emit(self._ota_status[can_id])

        except FileNotFoundError as e:
            error_msg = f"❌ File not found: {e}"
            self.__logger.error(error_msg)
            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["upload_status"] = "failed"
                    self._ota_status[can_id]["upload_error"] = str(e)
                    self._ota_status[can_id]["message"] = error_msg
                    # Emit error update to frontend
                    self.sigOTAStatusUpdate.emit(self._ota_status[can_id])

        except Exception as e:
            error_msg = f"❌ Error uploading firmware to device {can_id}: {e}"
            self.__logger.error(error_msg)
            with self._ota_lock:
                if can_id in self._ota_status:
                    self._ota_status[can_id]["upload_status"] = "failed"
                    self._ota_status[can_id]["upload_error"] = str(e)
                    self._ota_status[can_id]["message"] = error_msg
                    # Emit error update to frontend
                    self.sigOTAStatusUpdate.emit(self._ota_status[can_id])

    def _download_firmware_for_device(self, can_id):
        """
        Download firmware for a specific CAN ID from the firmware server.
        
        Queries the server for available firmware using JSON API.
        Matches firmware based on CAN device type (motor, laser, led, etc.).
        
        :param can_id: CAN ID of the device
        :return: Path to downloaded firmware file or None
        """
        try:
            # Get list of available firmware files from server via JSON API
            self.__logger.debug(f"Fetching firmware list from {self._firmware_server_url}")
            response = requests.get(
                self._firmware_server_url,
                headers={"Accept": "application/json"},
                timeout=10
            )
            response.raise_for_status()

            # Parse JSON response
            firmware_data = response.json()

            # Extract firmware file names
            firmware_files = [item['name'] for item in firmware_data if item['name'].endswith('.bin')]

            self.__logger.debug(f"Available firmware files: {firmware_files}")

            # Use centralized firmware mapping
            can_id_to_firmware = self._get_can_id_firmware_mapping()

            # Find firmware file matching the CAN ID pattern: id_<CANID>_*.bin (custom firmware)
            target_pattern = f"id_{can_id}_"
            matching_files = [f for f in firmware_files if f.startswith(target_pattern)]

            if matching_files:
                # Custom firmware for specific device found
                firmware_filename = matching_files[0]
                self.__logger.info(f"Using custom firmware for device {can_id}: {firmware_filename}")
            elif can_id in can_id_to_firmware:
                # Use standard firmware from mapping
                firmware_filename = can_id_to_firmware[can_id]
                if firmware_filename in firmware_files:
                    self.__logger.info(f"Using standard firmware for device {can_id}: {firmware_filename}")
                else:
                    self.__logger.error(f"Standard firmware not found on server: {firmware_filename}")
                    return None
            else:
                self.__logger.error(f"No firmware mapping found for CAN ID {can_id}")
                return None

            # Download the firmware file - construct full URL
            # The server provides relative URLs (./filename.bin), so we build the full URL
            firmware_url = f"{self._firmware_server_url}/{firmware_filename}"
            local_path = self._firmware_cache_dir / firmware_filename

            self.__logger.info(f"Downloading firmware from {firmware_url}")

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
        if not IS_HEADLESS: x = self._widget.motorXEdit.text()
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
        return dirtools.UserFileDirs.getValidatedDataPath()

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
    def getOTAWiFiCredentials(self):
        """
        Get current WiFi credentials for OTA updates.
        
        :return: Dictionary with SSID and password
        """
        return {
            "ssid": self._ota_wifi_ssid,
            "password": self._ota_wifi_password
        }

    @APIExport(runOnUIThread=False)
    def setOTAFirmwareServer(self, server_url="http://localhost/firmware"):
        """
        Set the firmware server URL for OTA updates.
        
        The server should serve firmware files at <server_url>/latest/ with the naming convention:
        - id_10_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
        - id_11_esp32_seeed_xiao_esp32s3_can_slave_motor.bin
        - id_20_esp32_seeed_xiao_esp32s3_can_slave_laser_debug.bin
        - id_21_esp32_seeed_xiao_esp32s3_can_slave_led_debug.bin
        
        :param server_url: URL of the firmware server (default: http://localhost/firmware)
        :return: Status message with list of available firmware files
        """
        # Remove trailing slash if present
        server_url = server_url.rstrip('/')

        try:
            # Test server connectivity
            self.__logger.debug(f"Testing firmware server: {server_url}")
            response = requests.get(
                server_url,
                headers={"Accept": "application/json"},
                timeout=1
            )
            response.raise_for_status()

            # Parse JSON response
            firmware_data = response.json()

            if not isinstance(firmware_data, list):
                raise ValueError("Invalid response format from firmware server")
            if not all('name' in item for item in firmware_data):
                raise ValueError("Firmware server response missing 'name' fields")
            self._firmware_server_url = server_url

            self.__logger.info(f"OTA firmware server set: {server_url}")
            self.__logger.info(f"Found {len(firmware_data)} firmware files")

            return {
                "status": "success",
                "message": f"Firmware server set: {server_url}",
                "server_url": server_url,
                "firmware_files": firmware_data,
                "count": len(firmware_data)
            }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to connect to firmware server: {str(e)}",
                "server_url": server_url
            }

    @APIExport(runOnUIThread=False)
    def getOTAFirmwareServer(self):
        """
        Get the current firmware server URL for OTA updates.
        
        :return: Current firmware server URL
        """
        return {
            "firmware_server_url": self._firmware_server_url
        }

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
            list_url = f"{self._firmware_server_url}/"
            self.__logger.debug(f"Fetching firmware list from {list_url}")

            response = requests.get(list_url, timeout=10, headers={"Accept": "application/json"})
            response.raise_for_status()


            # Parse JSON response (same format as setOTAFirmwareServer)
            firmware_data = response.json()

            if not isinstance(firmware_data, list):
                raise ValueError("Invalid response format from firmware server")

            # Extract firmware file names from JSON response
            firmware_files = [item['name'] for item in firmware_data if item['name'].endswith('.bin')]

            self.__logger.info(f"Available firmware files: {firmware_files}")

            # Use centralized firmware mapping
            can_id_to_firmware = self._get_can_id_firmware_mapping()

            # Organize by device ID using lookup table
            firmware_by_id = {}
            for can_id, expected_firmware in can_id_to_firmware.items():
                # Check if expected firmware exists in available files
                # print(expected_firmware)
                if expected_firmware in firmware_files:
                    firmware_url = f"{self._firmware_server_url}/{expected_firmware}"
                    # print(firmware_url)
                    # Find matching item in firmware_data to get size and mod_time
                    firmware_info = next((item for item in firmware_data if item['name'] == expected_firmware), None)

                    firmware_by_id[can_id] = {
                        "filename": expected_firmware,
                        "url": firmware_url,
                        "can_id": can_id,
                        "size": firmware_info.get('size', 0) if firmware_info else 0,
                        "mod_time": firmware_info.get('mod_time', '') if firmware_info else ''
                    }

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
                "upload_status": "command_sent",
                "message": f"OTA command sent to device {can_id}, waiting for WiFi connection..."
            }

        # Emit initial status to frontend
        self.sigOTAStatusUpdate.emit(self._ota_status[can_id])

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

    @APIExport(runOnUIThread=False, requestType="POST")
    def startMultipleDeviceOTA(self, can_ids:list[int]=[10,20], ssid=None, password=None, timeout=300000, delay_between:int=2):
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

    '''ESPTOOL related uploads'''

    # USB flash status tracking
    _usb_flash_status = {
        "status": "idle",
        "progress": 0,
        "message": "",
        "details": None
    }

    def _emit_usb_flash_status(self, status, progress, message, details=None):
        """Emit USB flash status update to frontend via signal."""
        self._usb_flash_status = {
            "status": status,
            "progress": progress,
            "message": message,
            "details": details,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.sigUSBFlashStatusUpdate.emit(self._usb_flash_status)
        self.__logger.info(f"USB Flash: {message} ({progress}%)")

    @APIExport(runOnUIThread=False)
    def listSerialPorts(self):
        """
        List all available serial ports on the system.
        
        :return: List of serial port information dictionaries
        """
        return self._list_serial_ports()

    @APIExport(runOnUIThread=False)
    def getUSBFlashStatus(self):
        """
        Get current USB flash status.
        
        :return: Dictionary with current flash status
        """
        return self._usb_flash_status

    # -----------------------------
    # USB flashing for CAN HAT (master)
    # -----------------------------
    def _list_serial_ports(self):
        ports = []
        try:
            for p in list_ports.comports():
                ports.append({
                    "device": p.device,
                    "description": getattr(p, "description", "") or "",
                    "manufacturer": getattr(p, "manufacturer", "") or "",
                    "product": getattr(p, "product", "") or "",
                    "hwid": getattr(p, "hwid", "") or "",
                    "vid": getattr(p, "vid", None),
                    "pid": getattr(p, "pid", None),
                    "serial_number": getattr(p, "serial_number", None),
                })
        except Exception as e:
            self.__logger.error(f"Failed to list serial ports: {e}")
        return ports

    def _find_hat_serial_port(self, match: str = "HAT", preferred_can_id: int = 1) -> str:
        """
        Find the USB serial port for the master CAN HAT by matching a substring
        in the port metadata (description/manufacturer/product/hwid).
        """
        ports = self._list_serial_ports()
        if not ports:
            raise RuntimeError("No serial ports found.")

        match_l = (match or "").strip().lower()
        candidates = []
        for p in ports:
            hay = " ".join([
                p.get("device", ""),
                p.get("description", ""),
                p.get("manufacturer", ""),
                p.get("product", ""),
                p.get("hwid", ""),
            ]).lower()
            if match_l and match_l in hay:
                candidates.append(p)

        # Fallback heuristics if no direct match
        if not candidates:
            # Prefer common USB-UART/CDC device names
            usb_like = []
            for p in ports:
                dev = (p.get("device") or "").lower()
                hwid = (p.get("hwid") or "").lower()
                desc = (p.get("description") or "").lower()
                if any(x in dev for x in ["/dev/ttyusb", "/dev/ttyacm", "com"]) or "usb" in hwid or "usb" in desc:
                    usb_like.append(p)
            candidates = usb_like

        if not candidates:
            raise RuntimeError(f"No candidate serial ports found for match='{match}'. Ports={ports}")

        # If ImSwitch currently uses one of these ports, prefer it
        try:
            current_port = getattr(self._master.UC2ConfigManager, "serialport", None)
            if current_port:
                for c in candidates:
                    if c["device"] == current_port:
                        return current_port
        except Exception:
            pass

        # Prefer stable ordering: ttyACM before ttyUSB (often native USB), then the first
        def rank(p):
            dev = (p.get("device") or "").lower()
            if "/dev/ttyacm" in dev:
                return 0
            if "/dev/ttyusb" in dev:
                return 1
            return 2

        candidates = sorted(candidates, key=rank)
        return candidates[0]["device"]

    def _download_firmware_by_name(self, firmware_filename: str) -> Path | None:
        server_url = (self._firmware_server_url or "").rstrip("/")
        if not server_url:
            self.__logger.error("Firmware server URL is not set.")
            return None

        firmware_filename = firmware_filename.lstrip("./")
        local_path = self._firmware_cache_dir / firmware_filename
        if local_path.exists():
            return local_path

        url = f"{server_url}/{firmware_filename}"
        try:
            self.__logger.info(f"Downloading firmware: {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            return local_path
        except Exception as e:
            self.__logger.error(f"Failed to download {url}: {e}")
            try:
                if local_path.exists():
                    local_path.unlink()
            except Exception:
                pass
            return None

    def _download_master_firmware(self) -> Path | None:
        """
        Resolve + download the master (CAN HAT) firmware from the firmware server.
        Tries (in this order):
          1) exact known names (HAT/master)
          2) id_1_*.bin
          3) anything containing 'hat' and 'master'
        """
        server_url = (self._firmware_server_url or "").rstrip("/")
        if not server_url:
            self.__logger.error("Firmware server URL is not set.")
            return None

        try:
            resp = requests.get(server_url, headers={"Accept": "application/json"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            names = [i.get("name", "") for i in data if isinstance(i, dict) and i.get("name", "").endswith(".bin")]
        except Exception as e:
            self.__logger.error(f"Failed to query firmware server '{server_url}': {e}")
            return None

        # 1) preferred exact names
        preferred = [
            "esp32_UC2_3_CAN_HAT_Master.bin",
            "esp32_UC2_CAN_HAT_Master.bin",
            "UC2_CAN_HAT_Master.bin",
            "CAN_HAT_Master.bin",
        ]
        for fn in preferred:
            if fn in names:
                return self._download_firmware_by_name(fn)

        # 2) id_1_*.bin
        id1 = [n for n in names if n.startswith("id_1_")]
        if id1:
            return self._download_firmware_by_name(id1[0])

        # 3) fuzzy match
        fuzzy = [n for n in names if ("hat" in n.lower() and "master" in n.lower())]
        if fuzzy:
            return self._download_firmware_by_name(fuzzy[0])

        self.__logger.error(f"No master firmware found on server. Available={names}")
        return None

    def _run_esptool(self, args: list[str]) -> tuple[bool, str]:
        """
        Run esptool either via python module (preferred) or subprocess fallback.
        Returns (success, message).
        """
        # Preferred: imported esptool (no external process)
        if HAS_ESPTOOL:
            try:
                # esptool.main() calls sys.exit(); catch it.
                esptool.main(args)
                return True, "OK"
            except SystemExit as e:
                code = int(getattr(e, "code", 1) or 0)
                return (code == 0), f"esptool exited with code {code}"
            except Exception as e:
                return False, f"esptool failed: {e}"

        # Fallback: subprocess (requires esptool to be installed as module)
        try:
            cmd = [sys.executable, "-m", "esptool"] + args
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            ok = (proc.returncode == 0)
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            return ok, out.strip()
        except Exception as e:
            return False, f"Failed to run esptool subprocess: {e}"

    @APIExport(runOnUIThread=False, requestType="POST")
    def flashMasterFirmwareUSB(
        self,
        port: str | None = None,
        match: str = "HAT",
        baud: int = 921600,
        flash_offset: int = 0x10000,
        erase_flash: bool = False,
        reconnect_after: bool = True,
    ):
        """
        Flash the master CAN HAT firmware via USB serial using esptool.

        - Disconnects ImSwitch from the ESP32 first (master carries comms).
        - Downloads firmware from the configured firmware server.
        - Flashes firmware to the detected (or provided) USB serial port.

        Parameters:
          port: explicit serial port (e.g. "/dev/ttyACM0"). If None, auto-detect via 'match'.
          match: substring to identify the HAT in serial port metadata (default "HAT").
          baud: flashing baudrate (default 921600).
          flash_offset: address to flash the BIN to. Use 0x0 for merged images; 0x10000 for app-only images.
          erase_flash: if True, erase flash before writing.
          reconnect_after: if True, reconnect ImSwitch to the master after flashing.
        """
        with self._usb_flash_lock:
            if not (HAS_ESPTOOL or True):
                self._emit_usb_flash_status("failed", 0, "esptool not available", "Install with: pip install esptool")
                return {"status": "error", "message": "esptool not available. Install with: pip install esptool"}

            # 1) disconnect master from ImSwitch
            self._emit_usb_flash_status("disconnecting", 5, "Disconnecting from ESP32...")
            try:
                self.__logger.info("Disconnecting ImSwitch from master before USB flashing…")
                try:
                    self._master.UC2ConfigManager.interruptSerialCommunication()
                except Exception:
                    pass
                self._master.UC2ConfigManager.closeSerial()
            except Exception as e:
                self.__logger.warning(f"Could not fully close serial before flashing: {e}")

            time.sleep(0.5)  # give OS time to release port
            self._emit_usb_flash_status("downloading", 10, "Downloading firmware from server...")

            # 2) resolve firmware
            fw_path = self._download_master_firmware()
            if not fw_path:
                self._emit_usb_flash_status("failed", 10, "No master firmware found on server")
                return {"status": "error", "message": "No master firmware found/downloaded."}

            self._emit_usb_flash_status("downloading", 20, f"Firmware downloaded: {fw_path.name}")

            # 3) resolve port
            try:
                flash_port = port or self._find_hat_serial_port(match=match)
                self._emit_usb_flash_status("flashing", 25, f"Using port: {flash_port}")
            except Exception as e:
                self._emit_usb_flash_status("failed", 20, f"Failed to find serial port: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to resolve HAT serial port: {e}",
                    "available_ports": self._list_serial_ports(),
                }

            self.__logger.info(f"Flashing master firmware via {flash_port} (baud={baud}, offset=0x{flash_offset:x})")

            # 4) optionally erase
            if erase_flash:
                self._emit_usb_flash_status("erasing", 30, "Erasing flash memory...")
                ok, msg = self._run_esptool(["--port", flash_port, "--baud", str(baud), "erase_flash"])
                if not ok:
                    self._emit_usb_flash_status("failed", 30, "Flash erase failed", msg)
                    return {
                        "status": "error",
                        "message": "erase_flash failed",
                        "details": msg,
                        "port": flash_port,
                        "firmware": str(fw_path),
                    }
                self._emit_usb_flash_status("flashing", 40, "Flash erased successfully")

            # 5) write flash
            self._emit_usb_flash_status("flashing", 45, "Writing firmware to device...")
            write_args = [
                "--port", flash_port,
                "--baud", str(baud),
                "--chip", "esp32",
                "write_flash",
                "--flash_mode", "dio",
                "--flash_freq", "80m",
                "--flash_size", "4MB",
                "0x%X" % int(flash_offset),
                str(fw_path),
            ]
            '''should be:
            esptool.py \
                --chip esp32 \
                --port /dev/cu.SLAB_USBtoUART \
                --baud 921600 \
                write_flash \
                --flash_mode dio \
                --flash_freq 80m \
                --flash_size 4MB \
                0x10000 firmware.bin'''
            ok, msg = self._run_esptool(write_args)
            if not ok:
                self._emit_usb_flash_status("failed", 50, "Firmware write failed", msg)
                return {
                    "status": "error",
                    "message": "write_flash failed",
                    "details": msg,
                    "port": flash_port,
                    "firmware": str(fw_path),
                    "flash_offset": int(flash_offset),
                }

            self._emit_usb_flash_status("flashing", 85, "Firmware written successfully!")

            # 6) reconnect
            if reconnect_after:
                self._emit_usb_flash_status("reconnecting", 90, "Reconnecting to device...")
                try:
                    time.sleep(1.0)
                    self.__logger.info("Reconnecting ImSwitch to master after flashing…")
                    self._master.UC2ConfigManager.initSerial(baudrate=None)
                    self._emit_usb_flash_status("success", 100, "✅ Firmware flashed and reconnected!")
                except Exception as e:
                    self._emit_usb_flash_status("warning", 95, "Flashed OK, but reconnect failed", str(e))
                    return {
                        "status": "warning",
                        "message": "Flashed OK, but reconnect failed",
                        "port": flash_port,
                        "firmware": str(fw_path),
                        "details": str(e),
                    }
            else:
                self._emit_usb_flash_status("success", 100, "✅ Firmware flashed successfully!")

            return {
                "status": "success",
                "message": "Master firmware flashed via USB",
                "port": flash_port,
                "firmware": str(fw_path),
                "baud": int(baud),
                "flash_offset": int(flash_offset),
                "erase_flash": bool(erase_flash),
                "reconnect_after": bool(reconnect_after),
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
