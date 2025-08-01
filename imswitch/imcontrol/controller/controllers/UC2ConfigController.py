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

GITHUB_API_LATEST_RELEASE = "https://api.github.com/repos/youseetoo/uc2-esp32/releases/latest"
try:
    FIRMWARE_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "uc2_esp32_fw"
    FIRMWARE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
except:
    # Fallback for systems where tempfile.gettempdir() does not return a valid path
    FIRMWARE_DOWNLOAD_DIR = Path("uc2_esp32_fw")
    FIRMWARE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _fetch_latest_firmware_assets():
    """Return list of (name, download_url) tuples for the latest release."""
    response = requests.get(GITHUB_API_LATEST_RELEASE, timeout=10)
    response.raise_for_status()
    data = response.json()
    assets = data.get("assets", [])
    return [(asset["name"], asset["browser_download_url"]) for asset in assets]


def _download_firmware(filename: str, url: str) -> Path:
    """Download *filename* from *url* to temporary folder and return Path."""
    target = FIRMWARE_DOWNLOAD_DIR / filename
    if target.exists():
        return target
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return target


def _run_esptool(args):
    """Wrapper around esptool.main() that converts SystemExit to Exception."""
    try:
        esptool.main(args)
    except SystemExit as exc:
        if exc.code != 0:
            raise RuntimeError(f"esptool failed with code {exc.code}, args: {args}")

class UC2ConfigController(ImConWidgetController):
    """Linked to UC2ConfigWidget."""

    sigUC2SerialReadMessage = Signal(str)
    sigUC2SerialWriteMessage = Signal(str)
    sigUC2SerialIsConnected = Signal(bool)


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        try:
            self.stages = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]
        except Exception as e:
            self.__logger.error("No Stages found in the config file? " +e )
            self.stages = None

        #
        # register the callback to take a snapshot triggered by the ESP32
        self.registerCaptureCallback()

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


        
    ''' ESPTOOL related methods '''
    @APIExport(runOnUIThread=False)
    def list_firmware_and_ports(self):
        if not HAS_ESPTOOL:
            return {"error": "esptool not installed"}
        try:
            assets = _fetch_latest_firmware_assets()
        except Exception as e:
            return {"error": str(e)}
        firmware_names = [name for name, _ in assets]
        ports = [p.device for p in list_ports.comports()]
        return {"firmware": firmware_names, "ports": ports}

    @APIExport(runOnUIThread=False)
    def espRestart(self):
        if not HAS_ESPTOOL:
            return {"error": "esptool not installed"}
        try:
            self._master.UC2ConfigManager.restartESP()
            return {"status": "ESP32 restarted successfully"}
        except Exception as e:
            return {"error": str(e)}
          
    @APIExport(runOnUIThread=False)
    def flash_firmware(
        self,
        filename: str,
        dev_type: str,
        axis_or_id: str,
        port: str,
        erase_flash: bool = True,
        baud_write: int = 460800,
    ):
        if not HAS_ESPTOOL:
            return {"error": "esptool not installed"}
        # locate asset
        assets = dict(_fetch_latest_firmware_assets())
        if filename not in assets:
            return {"error": "unknown firmware filename"}
        file_path = _download_firmware(filename, assets[filename])

        # close existing serial if open
        try:
            self._master.UC2ConfigManager.closeSerial()
        except Exception:
            pass

        # erase flash
        esptool_args_base = [
            "--chip",
            "esp32",
            "--port",
            port,
            "--baud",
            str(baud_write),
        ]
        if erase_flash:
            _run_esptool(esptool_args_base + ["erase_flash"])

        # write flash (offset 0x0)
        _run_esptool(esptool_args_base + ["write_flash", "0x0", str(file_path)])

        # set CAN address after flashing
        address_key = dev_type.lower()
        if dev_type.lower() == "motor":
            address_key = axis_or_id.lower()
        address = CAN_ADDRESS_MAP.get(address_key)
        if address is None:
            return {"error": f"No CAN address mapping for {dev_type}/{axis_or_id}"}

        ser = serial.Serial(port, 115200, timeout=2)
        msg = json.dumps({"task": "/can_act", "address": int(address)}) + "\n"
        ser.write(msg.encode())
        ser.flush()
        ser.close()

        # reconnect ImSwitch serial if necessary (async)
        threading.Thread(target=self.reconnectThread, daemon=True).start()
        return {"status": "flashed", "file": filename, "address": address, "port": port}
    

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
