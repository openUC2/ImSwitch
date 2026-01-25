import enum
import os
import time
from io import BytesIO
from typing import Dict, Optional, Type, List
import h5py
try:
    import zarr
except:
    pass
import numpy as np
import tifffile as tiff
import cv2


from imswitch.imcommon.framework import Signal, SignalInterface, Thread, Worker
from imswitch.imcommon.model import initLogger
import abc
import logging

from imswitch.imcontrol.model.managers.DetectorsManager import DetectorsManager

logger = logging.getLogger(__name__)
# Fallback to ome-zarr if vanilla implementation is not available
try:
    from ome_zarr.writer import write_multiscales_metadata # TODO: This fails with newer numpy versions!
    from ome_zarr.format import format_from_version
    IS_OME_ZARR = True
except ImportError:
    IS_OME_ZARR = False


def _create_zarr_store(path):
    """
    Create a Zarr store compatible with both Zarr 2.x and 3.x
    
    Args:
        path: Path to the store
        
    Returns:
        Store object compatible with current Zarr version
    """
    if hasattr(zarr.storage, 'DirectoryStore'):
        # Zarr 2.x compatibility
        return zarr.storage.DirectoryStore(path)
    elif hasattr(zarr.storage, 'LocalStore'):
        # Zarr 3.x with LocalStore
        return zarr.storage.LocalStore(path)
    else:
        # Zarr 3.x with direct path usage
        return path
class AsTemporayFile(object):
    """ A temporary file that when exiting the context manager is renamed to its original name. """
    def __init__(self, filepath, tmp_extension='.tmp'):
        if os.path.exists(filepath):
            raise FileExistsError(f'File {filepath} already exists.')
        self.path = filepath
        self.tmp_path = filepath + tmp_extension

    def __enter__(self):
        return self.tmp_path

    def __exit__(self, *args, **kwargs):
        os.rename(self.tmp_path, self.path)


class Storer(abc.ABC):
    """ Base class for storing data"""
    def __init__(self, filepath, detectorManager):
        self.filepath = filepath
        self.detectorManager: DetectorsManager = detectorManager

    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        """ Stores images and attributes according to the spec of the storer """
        raise NotImplementedError

    def stream(self, data = None, **kwargs):
        """ Stores data in a streaming fashion. """
        raise NotImplementedError


class ZarrStorer(Storer):
    """ A storer that stores the images in a zarr file store """
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        if not IS_OME_ZARR:
            logger.error("OME Zarr is not installed. Please install ome-zarr.")
            return
        with AsTemporayFile(f'{self.filepath}.zarr') as path:
            datasets: List[dict] = []
            store = _create_zarr_store(path)
            root = zarr.group(store=store)

            for channel, image in images.items():
                shape = self.detectorManager[channel].shape
                root.create_dataset(channel, data=image, shape=tuple(reversed(shape)),
                                        chunks=(512, 512), dtype='i2') #TODO: why not dynamic chunking?

                datasets.append({"path": channel, "transformation": None})
            write_multiscales_metadata(root, datasets, format_from_version("0.2"), shape, **attrs)
            logger.info(f"Saved image to zarr file {path}")




class TiffStorer(Storer):
    """ A storer that stores the images in a series of tiff files with OME metadata """
    
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        for channel, image in images.items():
            path = f'{self.filepath}_{channel}.ome.tiff'
            if not hasattr(image, "shape"):
                logger.error(f"Could not save image to tiff file {path}")
                continue
            
            try:
                # Build OME-TIFF metadata from attrs
                ome_metadata = self._build_ome_metadata(channel, image, attrs)
                
                if ome_metadata:
                    # Save as OME-TIFF with metadata
                    tiff.imwrite(
                        path, 
                        image,
                        metadata=ome_metadata,
                        imagej=False,  # Use OME metadata, not ImageJ
                    )
                else:
                    # Fallback to basic TIFF
                    tiff.imwrite(path, image)
                
                logger.info(f"Saved image to tiff file {path}")
            except Exception as e:
                logger.error(f"Error saving tiff file {path}: {e}")
                # Fallback to basic save
                tiff.imwrite(path, image)
    
    def _build_ome_metadata(self, channel: str, image: np.ndarray, attrs: Dict[str, str]) -> Optional[Dict]:
        """
        Build OME-TIFF metadata dictionary from attributes.
        
        Args:
            channel: Channel/detector name
            image: Image array
            attrs: Attributes dictionary (may contain SharedAttrValue objects)
            
        Returns:
            Dictionary with OME metadata or None if attrs is empty
        """
        if not attrs:
            return None
        
        metadata = {}
        
        try:
            # Get channel-specific attrs
            channel_attrs = attrs.get(channel, attrs)
            
            # Extract pixel size
            pixel_size = None
            for key in ['PixelSizeUm', 'Detector:PixelSizeUm', f'Detector:{channel}:PixelSizeUm']:
                if key in channel_attrs:
                    val = channel_attrs[key]
                    pixel_size = val.value if hasattr(val, 'value') else val
                    break
            
            # Build resolution info
            if pixel_size:
                # Physical size in micrometers
                metadata['PhysicalSizeX'] = float(pixel_size)
                metadata['PhysicalSizeY'] = float(pixel_size)
                metadata['PhysicalSizeXUnit'] = 'µm'
                metadata['PhysicalSizeYUnit'] = 'µm'
            
            # Extract exposure
            for key in ['ExposureMs', 'Detector:ExposureMs', f'Detector:{channel}:ExposureMs']:
                if key in channel_attrs:
                    val = channel_attrs[key]
                    exposure = val.value if hasattr(val, 'value') else val
                    metadata['ExposureTime'] = float(exposure) / 1000.0  # Convert to seconds
                    metadata['ExposureTimeUnit'] = 's'
                    break
            
            # Extract stage positions
            for axis in ['X', 'Y', 'Z']:
                for key_pattern in [f'Positioner:Stage:{axis}:Position', f'Stage:{axis}:Position']:
                    matching_keys = [k for k in channel_attrs.keys() if key_pattern in str(k)]
                    for key in matching_keys:
                        val = channel_attrs[key]
                        pos = val.value if hasattr(val, 'value') else val
                        metadata[f'Position{axis}'] = float(pos)
                        metadata[f'Position{axis}Unit'] = 'µm'
                        break
            
            # Add objective info if available
            for key in ['Objective:Name', 'ObjectiveName']:
                if key in channel_attrs:
                    val = channel_attrs[key]
                    metadata['Objective'] = val.value if hasattr(val, 'value') else str(val)
                    break
            
            for key in ['Objective:Magnification', 'ObjectiveMagnification']:
                if key in channel_attrs:
                    val = channel_attrs[key]
                    metadata['Magnification'] = float(val.value if hasattr(val, 'value') else val)
                    break
            
            ''' lasers metadata
            e.g.:
            'Laser:LED:WavelengthNm' = 635
            'Laser:LED:Value' =  25538
            'Laser:LASER:WavelengthNm' = 488
            'Laser:LASER:Value' =  0
            '''
            laser_infos = []
            for key in channel_attrs.keys():
                if 'Laser:' in key and 'WavelengthNm' in key:
                    parts = key.split(':')
                    if len(parts) >= 3:
                        laser_name = parts[1]
                        wavelength_key = key
                        value_key = f'Laser:{laser_name}:Value'
                        
                        wavelength_val = channel_attrs[wavelength_key]
                        wavelength = wavelength_val.value if hasattr(wavelength_val, 'value') else wavelength_val
                        
                        power = None
                        if value_key in channel_attrs:
                            power_val = channel_attrs[value_key]
                            power = power_val.value if hasattr(power_val, 'value') else power_val
                        
                        laser_info = {
                            'Name': laser_name,
                            'WavelengthNm': float(wavelength),
                        }
                        if power is not None:
                            laser_info['Power'] = float(power)
                        laser_infos.append(laser_info)

            if laser_infos:
                metadata['Lasers'] = laser_infos

            # Add channel name
            metadata['Channel'] = channel
            
            # Add timestamp
            import datetime
            metadata['DateTime'] = datetime.datetime.now().isoformat()
            
            return metadata if metadata else None
            
        except Exception as e:
            logger.warning(f"Error building OME metadata: {e}")
            return None

class PNGStorer(Storer):
    """ A storer that stores the images in a series of png files """
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        for channel, image in images.items():
            #with AsTemporayFile(f'{self.filepath}_{channel}.png') as path:
            path = f'{self.filepath}_{channel}.png'
            # if image is BW only, we have to convert it to RGB
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = cv2.convertScaleAbs(image)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(path, image)
            del image
            logger.info(f"Saved image to png file {path}")


class JPGStorer(Storer):
    """ A storer that stores the images in a series of jpg files """
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        for channel, image in images.items():
            #with AsTemporayFile(f'{self.filepath}_{channel}.jpg') as path:
            path = f'{self.filepath}_{channel}.jpg'
            # if image is BW only, we have to convert it to RGB
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(path, image)
            logger.info(f"Saved image to jpg file {path}")
class MP4Storer(Storer):
    """ A storer that writes the frames to an MP4 file """

    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        # not yet implemented
        pass


class SaveMode(enum.Enum):
    Disk = 1
    RAM = 2
    DiskAndRAM = 3
    Numpy = 4


class SaveFormat(enum.Enum):
    TIFF = 1
    ZARR = 3
    MP4 = 4
    PNG = 5
    JPG = 6


DEFAULT_STORER_MAP: Dict[str, Type[Storer]] = {
    SaveFormat.ZARR: ZarrStorer,
    SaveFormat.TIFF: TiffStorer,
    SaveFormat.MP4: MP4Storer,
    SaveFormat.PNG: PNGStorer,
    SaveFormat.JPG: JPGStorer
}


class RecordingManager(SignalInterface):
    """ RecordingManager handles single frame captures as well as continuous
    recordings of detector data. """

    sigRecordingStarted = Signal()
    sigRecordingEnded = Signal()
    sigRecordingFrameNumUpdated = Signal(int)  # (frameNumber)
    sigRecordingTimeUpdated = Signal(int)  # (recTime)
    sigMemorySnapAvailable = Signal(
        str, np.ndarray, object, bool
    )  # (name, image, filePath, savedToDisk)
    sigMemoryRecordingAvailable = Signal(
        str, object, object, bool
    )  # (name, file, filePath, savedToDisk)

    def __init__(self, detectorsManager, storerMap: Optional[Dict[str, Type[Storer]]] = None):
        super().__init__()
        self.__logger = initLogger(self)
        self.__storerMap = storerMap or DEFAULT_STORER_MAP
        self._memRecordings = {}  # { filePath: bytesIO }
        self.__detectorsManager = detectorsManager
        self.__record = False

        if 1: #not IS_HEADLESS: # TODO: Merge the two RecordingWorkers
            self._thread = Thread()
            self.__recordingWorker = RecordingWorker(self)
            self.__recordingWorker.moveToThread(self._thread)
            self._thread.started.connect(self.__recordingWorker.run)
        else:
            self.__recordingWorker = RecordingWorkerNoQt(self)
            self._thread = Thread(target=self.__recordingWorker.run)

    def __del__(self):
        self.endRecording(emitSignal=False, wait=True)
        if hasattr(super(), '__del__'):
            super().__del__()

    @property
    def record(self):
        """ Whether a recording is currently being recorded. """
        return self.__record

    @property
    def detectorsManager(self):
        return self.__detectorsManager

    def startRecording(self, detectorNames, recMode, savename, saveMode, attrs,
                       saveFormat=SaveFormat.TIFF, singleMultiDetectorFile=False, singleLapseFile=False,
                       recFrames=None, recTime=None):
        """ Starts a recording with the specified detectors, recording mode,
        file name prefix and attributes to save to the recording per detector.
        In SpecFrames mode, recFrames (the number of frames) must be specified,
        and in SpecTime mode, recTime (the recording time in seconds) must be
        specified. """

        self.__logger.info('Starting recording')
        self.__record = True
        self.__recordingWorker.detectorNames = detectorNames
        self.__recordingWorker.recMode = recMode
        self.__recordingWorker.savename = savename
        self.__recordingWorker.saveMode = saveMode
        self.__recordingWorker.saveFormat = saveFormat
        self.__recordingWorker.attrs = attrs
        self.__recordingWorker.recFrames = recFrames
        self.__recordingWorker.recTime = recTime
        self.__recordingWorker.singleMultiDetectorFile = singleMultiDetectorFile
        self.__recordingWorker.singleLapseFile = singleLapseFile
        self.__detectorsManager.execOnAll(lambda c: c.flushBuffers(),
                                          condition=lambda c: c.forAcquisition)
        if 0: #IS_HEADLESS:
            self._thread = Thread(target=self.__recordingWorker.run) # TODO: Merge the two RecordingWorkers
        self._thread.start()

    def endRecording(self, emitSignal=True, wait=True):
        """ Ends the current recording. Unless emitSignal is false, the
        sigRecordingEnded signal will be emitted. Unless wait is False, this
        method will wait until the recording is complete before returning. """

        self.__detectorsManager.execOnAll(lambda c: c.flushBuffers(),
                                          condition=lambda c: c.forAcquisition)

        if self.__record:
            self.__logger.info('Stopping recording')
        self.__record = False
        self._thread.quit()
        if emitSignal:
            self.sigRecordingEnded.emit()
        if wait:
            self._thread.wait()

    def snap(self, detectorNames=None, savename="", saveMode=SaveMode.Disk, saveFormat=SaveFormat.TIFF, attrs=None):
        """ Saves an image with the specified detectors to a file
        with the specified name prefix, save mode, file format and attributes
        to save to the capture per detector. """
        acqHandle = self.__detectorsManager.startAcquisition()

        if detectorNames is None:
            detectorNames = self.__detectorsManager.detectorNames

        try:
            images = {}

            # Acquire data
            for detectorName in detectorNames:
                images[detectorName] = self.__detectorsManager[detectorName].getLatestFrame()
                image = images[detectorName]

            if saveFormat:
                storer = self.__storerMap[saveFormat]

                if saveMode == SaveMode.Disk or saveMode == SaveMode.DiskAndRAM:
                    # Save images to disk
                    store = storer(savename, self.__detectorsManager)
                    store.snap(images, attrs)

                if saveMode == SaveMode.RAM or saveMode == SaveMode.DiskAndRAM:
                    for channel, image in images.items():
                        name = os.path.basename(f'{savename}_{channel}')
                        self.sigMemorySnapAvailable.emit(name, image, savename, saveMode == SaveMode.DiskAndRAM)

        except Exception as e:
            self.__logger.error(f'Failed to snap image: {e}')

        finally:
            # self.__detectorsManager.stopAcquisition(acqHandle)
            if saveMode == SaveMode.Numpy:
                return images


    def snapImagePrev(self, detectorName, savename, saveFormat, image, attrs):
        """ Saves a previously taken image to a file with the specified name prefix,
        file format and attributes to save to the capture per detector. """
        fileExtension = str(saveFormat.name).lower()
        filePath = self.getSaveFilePath(f'{savename}_{detectorName}.{fileExtension}')

        if saveFormat == SaveFormat.TIFF:
            tiff.imwrite(filePath, image)
        elif saveFormat == SaveFormat.PNG:
            cv2.imwrite(filePath, image)
        elif saveFormat == SaveFormat.JPG:
            cv2.imwrite(filePath, image)
        elif saveFormat == SaveFormat.ZARR:
            if not IS_OME_ZARR:
                logger.error("OME Zarr is not installed. Please install ome-zarr.")
                return
            path = self.getSaveFilePath(f'{savename}.{fileExtension}')
            store = _create_zarr_store(path)
            root = zarr.group(store=store)
            shape = self.__detectorsManager[detectorName].shape
            d = root.create_dataset(detectorName, data=image, shape=tuple(reversed(shape)), chunks=(512, 512),
                                    dtype='i2')
            datasets = {"path": detectorName, "transformation": None}
            write_multiscales_metadata(root, datasets, format_from_version("0.2"), shape, **attrs)
            store.close()
        else:
            raise ValueError(f'Unsupported save format "{saveFormat}"')

    def getSaveFilePath(self, path, allowOverwriteDisk=False, allowOverwriteMem=False):
        newPath = path
        numExisting = 0

        def existsFunc(pathToCheck):
            if not allowOverwriteDisk and os.path.exists(pathToCheck):
                return True
            if not allowOverwriteMem and pathToCheck in self._memRecordings:
                return True
            return False

        while existsFunc(newPath):
            numExisting += 1
            pathWithoutExt, pathExt = os.path.splitext(path)
            newPath = f'{pathWithoutExt}_{numExisting}{pathExt}'
        return newPath




class RecordingWorker(Worker):
    def __init__(self, recordingManager):
        super().__init__()
        self.__logger = initLogger(self)
        self.__recordingManager = recordingManager
        self.__logger = initLogger(self)

    def run(self):
        acqHandle = self.__recordingManager.detectorsManager.startAcquisition()
        try:
            self._record()

        finally:
            self.__recordingManager.detectorsManager.stopAcquisition(acqHandle)

    def _record(self):
        if self.saveFormat == SaveFormat.ZARR:
            files, fileDests, filePaths = self._getFiles()

        shapes = {detectorName: self.__recordingManager.detectorsManager[detectorName].shape
                  for detectorName in self.detectorNames}

        currentFrame = {}
        datasets = {}
        filenames = {}

        for detectorName in self.detectorNames:
            currentFrame[detectorName] = 0

            datasetName = detectorName
            if self.recMode == RecMode.ScanLapse and self.singleLapseFile:
                # Add scan number to dataset name
                scanNum = 0
                datasetNameWithScan = f'{datasetName}_scan{scanNum}'
                while datasetNameWithScan in files[detectorName]:
                    scanNum += 1
                    datasetNameWithScan = f'{datasetName}_scan{scanNum}'
                datasetName = datasetNameWithScan

            # Initial number of frames must not be 0; otherwise, too much disk space may get
            # allocated. We remove this default frame later on if no frames are captured.
            shape = shapes[detectorName]
            if len(shape) > 2:
                shape = shape[-2:]

            if self.saveFormat == SaveFormat.TIFF:
                fileExtension = str(self.saveFormat.name).lower()
                filenames[detectorName] = self.__recordingManager.getSaveFilePath(
                    f'{self.savename}_{detectorName}.{fileExtension}', False, False)

            elif self.saveFormat == SaveFormat.ZARR:
                if not IS_OME_ZARR:
                    logger.error("OME Zarr is not installed. Please install ome-zarr.")
                    return
                datasets[detectorName] = files[detectorName].create_dataset(datasetName, shape=(1, *reversed(shape)),
                                                                            dtype='i2', chunks=(1, 512, 512)
                                                                            )
                datasets[detectorName].attrs['detector_name'] = detectorName
                # For ImageJ compatibility
                datasets[detectorName].attrs['element_size_um'] \
                    = self.__recordingManager.detectorsManager[detectorName].pixelSizeUm
                datasets[detectorName].attrs['writing'] = True
                info: List[dict] = [{"path": datasetName, "transformation": None}]
                write_multiscales_metadata(files[detectorName], info, format_from_version("0.2"), shape, **self.attrs[detectorName])

        self.__recordingManager.sigRecordingStarted.emit()
        try:
            if len(self.detectorNames) < 1:
                raise ValueError('No detectors to record specified')

            if self.recMode in [RecMode.SpecFrames, RecMode.ScanOnce, RecMode.ScanLapse]:
                recFrames = self.recFrames
                if recFrames is None:
                    raise ValueError('recFrames must be specified in SpecFrames, ScanOnce or'
                                     ' ScanLapse mode')

                while (self.__recordingManager.record and
                       any([currentFrame[detectorName] < recFrames
                            for detectorName in self.detectorNames])):
                    for detectorName in self.detectorNames:
                        if currentFrame[detectorName] >= recFrames:
                            continue  # Reached requested number of frames with this detector, skip

                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)

                        if n > 0:
                            it = currentFrame[detectorName]
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue
                            elif self.saveFormat == SaveFormat.ZARR:
                                dataset = datasets[detectorName]
                                if it == 0:
                                    dataset[0, :, :] = newFrames[0, :, :]
                                    if n > 0:
                                        dataset.append(newFrames[1:n, :, :])
                                else:
                                    dataset.append(newFrames)
                                currentFrame[detectorName] += n

                            # Things get a bit weird if we have multiple detectors when we report
                            # the current frame number, since the detectors may not be synchronized.
                            # For now, we will report the lowest number.
                            self.__recordingManager.sigRecordingFrameNumUpdated.emit(
                                min(list(currentFrame.values()))
                            )
                    time.sleep(0.0001)  # Prevents freezing for some reason

                self.__recordingManager.sigRecordingFrameNumUpdated.emit(0)
            elif self.recMode == RecMode.SpecTime:
                recTime = self.recTime
                if recTime is None:
                    raise ValueError('recTime must be specified in SpecTime mode')

                start = time.time()
                currentRecTime = 0
                shouldStop = False
                while True:
                    for detectorName in self.detectorNames:
                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)
                        if n > 0:
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue
                            elif  self.saveFormat == SaveFormat.ZARR:
                                it = currentFrame[detectorName]
                                dataset = datasets[detectorName]
                                dataset.resize(n + it, axis=0)
                                dataset[it:it + n, :, :] = newFrames
                            currentFrame[detectorName] += n
                            self.__recordingManager.sigRecordingTimeUpdated.emit(
                                np.around(currentRecTime, decimals=2)
                            )
                            currentRecTime = time.time() - start

                    if shouldStop:
                        break  # Enter loop one final time, then stop

                    if not self.__recordingManager.record or currentRecTime >= recTime:
                        shouldStop = True

                    time.sleep(0.0001)  # Prevents freezing for some reason

                self.__recordingManager.sigRecordingTimeUpdated.emit(0)
            elif self.recMode == RecMode.UntilStop:
                shouldStop = False
                while True:
                    for detectorName in self.detectorNames:
                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)
                        if n > 0:
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue

                            elif self.saveFormat == SaveFormat.ZARR:
                                it = currentFrame[detectorName]
                                dataset = datasets[detectorName]
                                if it == 0:
                                    dataset[0, :, :] = newFrames[0, :, :]
                                    if n > 0:
                                        dataset.append(newFrames[1:n, :, :])
                                else:
                                    dataset.append(newFrames)

                            currentFrame[detectorName] += n

                    if shouldStop:
                        break

                    if not self.__recordingManager.record:
                        shouldStop = True  # Enter loop one final time, then stop

                    time.sleep(0.0001)  # Prevents freezing for some reason
            else:
                raise ValueError('Unsupported recording mode specified')
        finally:

            if self.saveFormat == SaveFormat.ZARR:
                for detectorName, file in files.items():
                    # Remove default frame if no frames have been captured
                    if self.saveMode == SaveMode.RAM or self.saveMode == SaveMode.DiskAndRAM:
                        filePath = filePaths[detectorName]
                        name = os.path.basename(filePath)
                        if self.saveMode == SaveMode.RAM:
                            file.close()
                            self.__recordingManager.sigMemoryRecordingAvailable.emit(
                                name, fileDests[detectorName], filePath, False
                            )
                        else:
                            file.flush()
                            self.__recordingManager.sigMemoryRecordingAvailable.emit(
                                name, file, filePath, True
                            )
                    else:
                        datasets[detectorName].attrs['writing'] = False
                        self.store.close()
            emitSignal = True
            if self.recMode in [RecMode.SpecFrames, RecMode.ScanOnce, RecMode.ScanLapse]:
                emitSignal = False
            self.__recordingManager.endRecording(emitSignal=emitSignal, wait=False)

    def _getFiles(self):
        singleMultiDetectorFile = self.singleMultiDetectorFile
        singleLapseFile = self.recMode == RecMode.ScanLapse and self.singleLapseFile

        files = {}
        fileDests = {}
        filePaths = {}
        extension = 'zarr'

        for detectorName in self.detectorNames:
            if singleMultiDetectorFile:
                baseFilePath = f'{self.savename}.{extension}'
            else:
                baseFilePath = f'{self.savename}_{detectorName}.{extension}'

            filePaths[detectorName] = self.__recordingManager.getSaveFilePath(
                baseFilePath,
                allowOverwriteDisk=singleLapseFile and self.saveMode != SaveMode.RAM,
                allowOverwriteMem=singleLapseFile and self.saveMode == SaveMode.RAM
            )

        for detectorName in self.detectorNames:
            if self.saveMode == SaveMode.RAM:
                memRecordings = self.__recordingManager._memRecordings
                if (filePaths[detectorName] not in memRecordings or
                        memRecordings[filePaths[detectorName]].closed):
                    memRecordings[filePaths[detectorName]] = BytesIO()
                fileDests[detectorName] = memRecordings[filePaths[detectorName]]
            else:
                fileDests[detectorName] = filePaths[detectorName]

            if singleMultiDetectorFile and len(files) > 0:
                files[detectorName] = list(files.values())[0]
            else:
                if  self.saveFormat == SaveFormat.ZARR:
                    self.store = _create_zarr_store(fileDests[detectorName])
                    files[detectorName] = zarr.group(store=self.store, overwrite=True)

        return files, fileDests, filePaths

    def _getNewFrames(self, detectorName):
        newFrames, frameIndices = self.__recordingManager.detectorsManager[detectorName].getChunk()
        newFrames = np.array(newFrames)
        return newFrames

class RecordingWorkerNoQt(Worker):
    def __init__(self, recordingManager):
        super().__init__()
        self.__logger = initLogger(self)
        self.__recordingManager = recordingManager
        self.__logger = initLogger(self)

    def run(self):
        self.__logger.info('Recording worker NoQT started')
        acqHandle = self.__recordingManager.detectorsManager.startAcquisition()
        try:
            self._record()

        finally:
            self.__recordingManager.detectorsManager.stopAcquisition(acqHandle)

    def moveToThread(self, thread) -> None:
        return super().moveToThread(thread)

    def _record(self):
        self.__logger.info('Recording started in mode: ' + str(self.recMode))
        if self.saveFormat == SaveFormat.ZARR:
            files, fileDests, filePaths = self._getFiles()

        shapes = {detectorName: self.__recordingManager.detectorsManager[detectorName].shape
                  for detectorName in self.detectorNames}

        currentFrame = {}
        datasets = {}
        filenames = {}

        for detectorName in self.detectorNames:
            currentFrame[detectorName] = 0

            datasetName = detectorName
            if self.recMode == RecMode.ScanLapse and self.singleLapseFile:
                # Add scan number to dataset name
                scanNum = 0
                datasetNameWithScan = f'{datasetName}_scan{scanNum}'
                while datasetNameWithScan in files[detectorName]:
                    scanNum += 1
                    datasetNameWithScan = f'{datasetName}_scan{scanNum}'
                datasetName = datasetNameWithScan

            # Initial number of frames must not be 0; otherwise, too much disk space may get
            # allocated. We remove this default frame later on if no frames are captured.
            shape = shapes[detectorName]
            if len(shape) > 2:
                shape = shape[-2:]

            if self.saveFormat == SaveFormat.MP4:
                # Need to initiliaze videowriter for each detector
                self.__logger.debug("Initialize MP4 recorder")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fileExtension = str(self.saveFormat.name).lower()
                filePath = self.__recordingManager.getSaveFilePath(f'{self.savename}_{detectorName}.{fileExtension}')
                self.__logger.debug("Saving Video to file: " + filePath)
                filenames[detectorName] = filePath
                datasets[detectorName] = cv2.VideoWriter(filePath, fourcc, 20.0, shapes[detectorName])
                #datasets[detectorName] = cv2.VideoWriter(filePath, cv2.VideoWriter_fourcc(*'MJPG'), 10, shapes[detectorName])


            elif self.saveFormat == SaveFormat.TIFF:
                fileExtension = str(self.saveFormat.name).lower()
                filenames[detectorName] = self.__recordingManager.getSaveFilePath(
                    f'{self.savename}_{detectorName}.{fileExtension}', False, False)

            elif self.saveFormat == SaveFormat.PNG:
                fileExtension = str(self.saveFormat.name).lower()
                filenames[detectorName] = self.__recordingManager.getSaveFilePath(
                    f'{self.savename}_{detectorName}.{fileExtension}', False, False)

            elif self.saveFormat == SaveFormat.JPG:
                fileExtension = str(self.saveFormat.name).lower()
                filenames[detectorName] = self.__recordingManager.getSaveFilePath(
                    f'{self.savename}_{detectorName}.{fileExtension}', False, False)

            elif self.saveFormat == SaveFormat.ZARR:
                if not IS_OME_ZARR:
                    logger.error("OME Zarr is not installed. Please install ome-zarr.")
                    return
                datasets[detectorName] = files[detectorName].create_dataset(datasetName, shape=(1, *reversed(shape)),
                                                                            dtype='i2', chunks=(1, 512, 512)
                                                                            )
                datasets[detectorName].attrs['detector_name'] = detectorName
                # For ImageJ compatibility
                datasets[detectorName].attrs['element_size_um'] \
                    = self.__recordingManager.detectorsManager[detectorName].pixelSizeUm
                datasets[detectorName].attrs['writing'] = True
                info: List[dict] = [{"path": datasetName, "transformation": None}]
                write_multiscales_metadata(files[detectorName], info, format_from_version("0.2"), shape, **self.attrs[detectorName])


        self.__recordingManager.sigRecordingStarted.emit()
        try:
            if len(self.detectorNames) < 1:
                raise ValueError('No detectors to record specified')

            if self.recMode in [RecMode.SpecFrames, RecMode.ScanOnce, RecMode.ScanLapse]:
                recFrames = self.recFrames
                if recFrames is None:
                    raise ValueError('recFrames must be specified in SpecFrames, ScanOnce or'
                                     ' ScanLapse mode')

                while (self.__recordingManager.record and
                       any([currentFrame[detectorName] < recFrames
                            for detectorName in self.detectorNames])):
                    for detectorName in self.detectorNames:
                        if currentFrame[detectorName] >= recFrames:
                            continue  # Reached requested number of frames with this detector, skip

                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)

                        if n > 0:
                            it = currentFrame[detectorName]
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue

                            elif self.saveFormat == SaveFormat.ZARR:
                                dataset = datasets[detectorName]
                                if it == 0:
                                    dataset[0, :, :] = newFrames[0, :, :]
                                    if n > 0:
                                        dataset.append(newFrames[1:n, :, :])
                                else:
                                    dataset.append(newFrames)
                                currentFrame[detectorName] += n
                            elif self.saveFormat == SaveFormat.MP4:
                                for iframe in range(n):
                                    frame = newFrames[iframe,:,:]
                                    #https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
                                    frame = cv2.cvtColor(cv2.convertScaleAbs(frame), cv2.COLOR_GRAY2BGR)
                                    self.__logger.debug(type(frame))

                                    datasets[detectorName].write(frame)


                            # Things get a bit weird if we have multiple detectors when we report
                            # the current frame number, since the detectors may not be synchronized.
                            # For now, we will report the lowest number.
                            self.__recordingManager.sigRecordingFrameNumUpdated.emit(
                                min(list(currentFrame.values()))
                            )
                    time.sleep(0.0001)  # Prevents freezing for some reason

                self.__recordingManager.sigRecordingFrameNumUpdated.emit(0)
            elif self.recMode == RecMode.SpecTime:
                recTime = self.recTime
                if recTime is None:
                    raise ValueError('recTime must be specified in SpecTime mode')

                start = time.time()
                currentRecTime = 0
                shouldStop = False
                while True:
                    for detectorName in self.detectorNames:
                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)
                        if n > 0:
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue
                            elif self.saveFormat == SaveFormat.ZARR:
                                it = currentFrame[detectorName]
                                dataset = datasets[detectorName]
                                dataset.resize(n + it, axis=0)
                                dataset[it:it + n, :, :] = newFrames
                            elif self.saveFormat == SaveFormat.MP4:
                                for iframe in range(n):
                                    frame = newFrames[iframe,:,:]
                                    #https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
                                    frame = cv2.cvtColor(cv2.convertScaleAbs(frame), cv2.COLOR_GRAY2BGR)
                                    self.__logger.debug(type(frame))

                                    datasets[detectorName].write(frame)

                            currentFrame[detectorName] += n
                            self.__recordingManager.sigRecordingTimeUpdated.emit(
                                np.around(currentRecTime, decimals=2)
                            )
                            currentRecTime = time.time() - start

                    if shouldStop:
                        break  # Enter loop one final time, then stop

                    if not self.__recordingManager.record or currentRecTime >= recTime:
                        shouldStop = True

                    time.sleep(0.0001)  # Prevents freezing for some reason

                self.__recordingManager.sigRecordingTimeUpdated.emit(0)
            elif self.recMode == RecMode.UntilStop:
                shouldStop = False
                while True:
                    for detectorName in self.detectorNames:
                        newFrames = self._getNewFrames(detectorName)
                        n = len(newFrames)
                        if n > 0:
                            if self.saveFormat == SaveFormat.TIFF:
                                try:
                                    filePath = filenames[detectorName]
                                    tiff.imwrite(filePath, newFrames, append=True)
                                except ValueError:
                                    self.__logger.error("TIFF File exceeded 4GB.")
                                    if self.saveFormat == SaveFormat.TIFF:
                                        filePath = self.__recordingManager.getSaveFilePath(
                                            f'{self.savename}_{detectorName}.{fileExtension}', False, False)
                                        continue

                            elif self.saveFormat == SaveFormat.ZARR:
                                it = currentFrame[detectorName]
                                dataset = datasets[detectorName]
                                if it == 0:
                                    dataset[0, :, :] = newFrames[0, :, :]
                                    if n > 0:
                                        dataset.append(newFrames[1:n, :, :])
                                else:
                                    dataset.append(newFrames)
                            elif self.saveFormat == SaveFormat.MP4:
                                for iframe in range(n):
                                    frame = newFrames[iframe,:,:]
                                    #https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
                                    frame = cv2.cvtColor(cv2.convertScaleAbs(frame), cv2.COLOR_GRAY2BGR)

                                    datasets[detectorName].write(frame)


                            currentFrame[detectorName] += n

                    if shouldStop:
                        break

                    if not self.__recordingManager.record:
                        shouldStop = True  # Enter loop one final time, then stop

                    time.sleep(0.0001)  # Prevents freezing for some reason
            else:
                raise ValueError('Unsupported recording mode specified')
        finally:

            if self.saveFormat == SaveFormat.ZARR:
                for detectorName, file in files.items():
                    # Remove default frame if no frames have been captured

                    # Handle memory recordings
                    if self.saveMode == SaveMode.RAM or self.saveMode == SaveMode.DiskAndRAM:
                        filePath = filePaths[detectorName]
                        name = os.path.basename(filePath)
                        if self.saveMode == SaveMode.RAM:
                            file.close()
                            self.__recordingManager.sigMemoryRecordingAvailable.emit(
                                name, fileDests[detectorName], filePath, False
                            )
                        else:
                            file.flush()
                            self.__recordingManager.sigMemoryRecordingAvailable.emit(
                                name, file, filePath, True
                            )
                    else:
                        datasets[detectorName].attrs['writing'] = False
                        if self.saveFormat == SaveFormat.MP4:
                            for detectorName, file in files.items():
                                datasets[detectorName].release()
                        else:
                            self.store.close()
            emitSignal = True
            if self.recMode in [RecMode.SpecFrames, RecMode.ScanOnce, RecMode.ScanLapse]:
                emitSignal = False
            self.__recordingManager.endRecording(emitSignal=emitSignal, wait=False)

    def _getFiles(self):
        singleMultiDetectorFile = self.singleMultiDetectorFile
        singleLapseFile = self.recMode == RecMode.ScanLapse and self.singleLapseFile

        files = {}
        fileDests = {}
        filePaths = {}
        extension = 'zarr'

        for detectorName in self.detectorNames:
            if singleMultiDetectorFile:
                baseFilePath = f'{self.savename}.{extension}'
            else:
                baseFilePath = f'{self.savename}_{detectorName}.{extension}'

            filePaths[detectorName] = self.__recordingManager.getSaveFilePath(
                baseFilePath,
                allowOverwriteDisk=singleLapseFile and self.saveMode != SaveMode.RAM,
                allowOverwriteMem=singleLapseFile and self.saveMode == SaveMode.RAM
            )

        for detectorName in self.detectorNames:
            if self.saveMode == SaveMode.RAM:
                memRecordings = self.__recordingManager._memRecordings
                if (filePaths[detectorName] not in memRecordings or
                        memRecordings[filePaths[detectorName]].closed):
                    memRecordings[filePaths[detectorName]] = BytesIO()
                fileDests[detectorName] = memRecordings[filePaths[detectorName]]
            else:
                fileDests[detectorName] = filePaths[detectorName]

            if singleMultiDetectorFile and len(files) > 0:
                files[detectorName] = list(files.values())[0]
            else:
                if self.saveFormat == SaveFormat.ZARR:
                    self.store = _create_zarr_store(fileDests[detectorName])
                    files[detectorName] = zarr.group(store=self.store, overwrite=True)

        return files, fileDests, filePaths

    def _getNewFrames(self, detectorName):
        newFrames = self.__recordingManager.detectorsManager[detectorName].getChunk()
        newFrames = np.array(newFrames)
        return newFrames


class RecMode(enum.Enum):
    SpecFrames = 1
    SpecTime = 2
    ScanOnce = 3
    ScanLapse = 4
    UntilStop = 5


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
