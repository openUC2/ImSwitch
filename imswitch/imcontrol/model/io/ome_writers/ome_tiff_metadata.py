"""
OME-TIFF Metadata Builder using ome_types library.

This module provides functions for building properly formatted OME-XML metadata
strings for TIFF files, following the OME specification.

The metadata is compatible with tifffile's description parameter and can be
read by standard OME tools like ImageJ/Fiji, OMERO, and Python ome-types.

Usage:
    >>> from imswitch.imcontrol.model.io.ome_writers.ome_tiff_metadata import (
    ...     build_ome_metadata, OMEMetadataParams
    ... )
    >>> 
    >>> params = OMEMetadataParams(
    ...     width_px=512, height_px=512,
    ...     pixel_size_um=0.325,
    ...     channel_name="DAPI",
    ...     exposure_time_ms=100.0
    ... )
    >>> ome_xml = build_ome_metadata(params)
    >>> tifffile.imwrite("image.tiff", data, description=ome_xml)
"""

import datetime as dt
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from ome_types import OME
    from ome_types.model import (
        Channel,
        Detector,
        Detector_Type,
        Image,
        Instrument,
        Laser,
        LightEmittingDiode,
        LightSourceSettings,
        Microscope,
        Microscope_Type,
        Objective,
        ObjectiveSettings,
        Objective_Correction,
        Objective_Immersion,
        Pixels,
        Pixels_DimensionOrder,
        PixelType,
        Plane,
        StageLabel,
        TiffData,
    )
    OME_TYPES_AVAILABLE = True
except ImportError:
    OME_TYPES_AVAILABLE = False


# =============================================================================
# Helper Functions
# =============================================================================

def extract_wavelength_nm(channel_name: str) -> Optional[int]:
    """
    Extract wavelength in nm from channel name using regex.

    Examples:
        "Fluorescence 405 nm Ex" -> 405
        "LED_638" -> 638
        "BF LED matrix full" -> None
    """
    if not channel_name:
        return None
    match = re.search(r'(\d+)\s*nm?', channel_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def numpy_dtype_to_ome_pixel_type(dtype_str: str) -> "PixelType":
    """
    Convert numpy dtype string to OME PixelType.
    
    Args:
        dtype_str: numpy dtype as string (e.g., 'uint8', 'uint16', 'float32')
        
    Returns:
        OME PixelType enum value
    """
    if not OME_TYPES_AVAILABLE:
        return None
        
    dtype_map = {
        "uint8": PixelType.UINT8,
        "uint16": PixelType.UINT16,
        "uint32": PixelType.UINT32,
        "int8": PixelType.INT8,
        "int16": PixelType.INT16,
        "int32": PixelType.INT32,
        "float32": PixelType.FLOAT,
        "float64": PixelType.DOUBLE,
        # Mono formats from cameras
        "mono8": PixelType.UINT8,
        "mono10": PixelType.UINT16,
        "mono12": PixelType.UINT16,
        "mono14": PixelType.UINT16,
        "mono16": PixelType.UINT16,
    }
    return dtype_map.get(dtype_str.lower(), PixelType.UINT16)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OMEMetadataParams:
    """Parameters for building OME-XML metadata."""
    # Image dimensions
    width_px: int = 512
    height_px: int = 512
    size_z: int = 1
    size_c: int = 1
    size_t: int = 1
    
    # Physical sizes (in micrometers)
    pixel_size_um: float = 1.0
    z_step_um: Optional[float] = None
    
    # Pixel type
    dtype: str = "uint16"
    
    # Channel info
    channel_name: str = "Unknown"
    channel_index: int = 0
    
    # Acquisition info
    exposure_time_ms: float = 0.0
    timestamp: Optional[float] = None
    
    # Position info (in mm for OME compatibility)
    position_x_mm: float = 0.0
    position_y_mm: float = 0.0
    position_z_mm: float = 0.0
    
    # Z-plane index within stack
    z_index: int = 0
    
    # Additional info
    image_name: Optional[str] = None
    description: Optional[str] = None
    
    # Illumination
    illumination_channel: Optional[str] = None
    illumination_value: float = 0.0
    
    # Equipment info
    microscope_name: str = "openUC2"
    camera_vendor: Optional[str] = None
    camera_model: Optional[str] = None
    camera_serial: Optional[str] = None
    objective_magnification: Optional[float] = None
    objective_na: Optional[float] = None
    
    # Extra metadata to include in description
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OMEInstrumentTemplate:
    """
    Reusable OME instrument template for a microscope setup.
    
    Build once per microscope configuration and reuse for all images.
    """
    ome: Optional["OME"] = None
    microscope_name: str = "openUC2"
    channel_names: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if template is properly initialized."""
        return self.ome is not None and OME_TYPES_AVAILABLE


# =============================================================================
# OME Metadata Building Functions
# =============================================================================

def build_ome_instrument(
    microscope_name: str = "openUC2",
    camera_vendor: Optional[str] = None,
    camera_model: Optional[str] = None,
    camera_serial: Optional[str] = None,
    objective_magnification: Optional[float] = None,
    objective_na: Optional[float] = None,
    channel_names: Optional[List[str]] = None,
    microscope_type: str = "inverted",
) -> Optional[OMEInstrumentTemplate]:
    """
    Build reusable OME instrument structure with microscope-wide metadata.
    
    This creates the portions of OME metadata that remain constant for all
    images from the same microscope: Microscope, Detector, Objective, and 
    light sources.
    
    Args:
        microscope_name: Name/model of the microscope
        camera_vendor: Camera manufacturer
        camera_model: Camera model name
        camera_serial: Camera serial number
        objective_magnification: Objective magnification (e.g., 4, 10, 20)
        objective_na: Objective numerical aperture
        channel_names: List of available channel/illumination names
        microscope_type: "inverted", "upright", or "other"
        
    Returns:
        OMEInstrumentTemplate with pre-built OME object, or None if ome_types unavailable
    """
    if not OME_TYPES_AVAILABLE:
        return None
    
    # Create OME root
    ome_uuid = f"urn:uuid:{uuid.uuid4()}"
    ome = OME(creator="ImSwitch/openUC2", uuid=ome_uuid)
    
    # Create instrument
    instrument = Instrument(id="Instrument:0")
    
    # Add microscope info
    micro_type_map = {
        "inverted": Microscope_Type.INVERTED,
        "upright": Microscope_Type.UPRIGHT,
    }
    micro_type = micro_type_map.get(microscope_type.lower(), Microscope_Type.OTHER)
    
    microscope = Microscope(
        manufacturer="openUC2",
        model=microscope_name,
        type=micro_type,
    )
    instrument.microscope = microscope
    
    # Add objective if magnification provided
    if objective_magnification is not None:
        objective = Objective(
            id="Objective:0",
            nominal_magnification=float(objective_magnification),
            lens_na=objective_na,
            correction=Objective_Correction.PLAN_APO,
            immersion=Objective_Immersion.AIR,
        )
        instrument.objectives = [objective]
    
    # Add detector (camera)
    detector = Detector(
        id="Detector:0",
        manufacturer=camera_vendor,
        model=camera_model,
        serial_number=camera_serial,
        type=Detector_Type.CCD,
    )
    instrument.detectors = [detector]
    
    # Add light sources based on channel names
    if channel_names:
        lasers = []
        leds = []
        
        for i, ch_name in enumerate(channel_names):
            wavelength_nm = extract_wavelength_nm(ch_name)
            
            # Heuristic: if wavelength found and >= 400, likely a laser
            # otherwise treat as LED
            ch_lower = ch_name.lower()
            if wavelength_nm and wavelength_nm >= 400 and ("laser" in ch_lower or wavelength_nm <= 650):
                light_source = Laser(
                    id=f"LightSource:{i}",
                    wavelength=wavelength_nm,
                )
                lasers.append(light_source)
            else:
                light_source = LightEmittingDiode(
                    id=f"LightSource:{i}",
                )
                leds.append(light_source)
        
        if lasers:
            instrument.lasers = lasers
        if leds:
            instrument.light_emitting_diodes = leds
    
    ome.instruments = [instrument]
    
    return OMEInstrumentTemplate(
        ome=ome,
        microscope_name=microscope_name,
        channel_names=channel_names or []
    )


def build_ome_metadata(
    params: OMEMetadataParams,
    instrument_template: Optional[OMEInstrumentTemplate] = None,
) -> Optional[str]:
    """
    Build OME-XML metadata string for an image.
    
    Args:
        params: OMEMetadataParams with image-specific metadata
        instrument_template: Optional pre-built instrument template.
                           If None, creates basic instrument from params.
    
    Returns:
        OME-XML string for use with tifffile's description parameter,
        or None if ome_types is not available.
    """
    if not OME_TYPES_AVAILABLE:
        return None
    
    # Create new OME object for this image
    ome_uuid = f"urn:uuid:{uuid.uuid4()}"
    ome = OME(creator="ImSwitch/openUC2", uuid=ome_uuid)
    
    # Copy instrument from template or build minimal one
    if instrument_template and instrument_template.ome:
        ome.instruments = instrument_template.ome.instruments
    else:
        # Build minimal instrument from params
        instrument = Instrument(id="Instrument:0")
        microscope = Microscope(
            manufacturer="openUC2",
            model=params.microscope_name,
            type=Microscope_Type.OTHER,
        )
        instrument.microscope = microscope
        
        if params.camera_model:
            detector = Detector(
                id="Detector:0",
                manufacturer=params.camera_vendor,
                model=params.camera_model,
                serial_number=params.camera_serial,
                type=Detector_Type.CCD,
            )
            instrument.detectors = [detector]
        
        ome.instruments = [instrument]
    
    # Determine pixel type
    pixel_type = numpy_dtype_to_ome_pixel_type(params.dtype)
    
    # Create Pixels element
    pixels = Pixels(
        id="Pixels:0",
        dimension_order=Pixels_DimensionOrder.XYCZT,
        size_x=params.width_px,
        size_y=params.height_px,
        size_c=params.size_c,
        size_z=params.size_z,
        size_t=params.size_t,
        type=pixel_type,
        physical_size_x=params.pixel_size_um,
        physical_size_y=params.pixel_size_um,
        physical_size_z=params.z_step_um if params.z_step_um and params.z_step_um > 0 else None,
    )
    
    # Create channel
    wavelength_nm = extract_wavelength_nm(params.illumination_channel or params.channel_name)
    
    channel = Channel(
        id=f"Channel:0:{params.channel_index}",
        name=params.channel_name,
        samples_per_pixel=1,
    )
    
    # Add light source settings if we have illumination info
    if params.illumination_channel:
        # Try to find matching light source in instrument
        light_source_id = f"LightSource:{params.channel_index}"
        
        # Normalize attenuation to 0-1 range
        # Values may come as 0-100 (percentage) or 0-255 (PWM) or already 0-1
        attenuation = None
        if params.illumination_value > 0:
            if params.illumination_value > 100:
                # Assume 0-255 PWM range, convert to 0-1
                attenuation = min(params.illumination_value / 255.0, 1.0)
            elif params.illumination_value > 1:
                # Assume 0-100 percentage, convert to 0-1
                attenuation = min(params.illumination_value / 100.0, 1.0)
            else:
                # Already in 0-1 range
                attenuation = params.illumination_value
        
        light_settings = LightSourceSettings(
            id=light_source_id,
            wavelength=wavelength_nm,
            attenuation=attenuation,
        )
        channel.light_source_settings = light_settings
    
    pixels.channels = [channel]
    
    # Create Plane element with acquisition info
    plane = Plane(
        the_z=params.z_index,
        the_c=params.channel_index,
        the_t=0,
        exposure_time=params.exposure_time_ms / 1000.0 if params.exposure_time_ms > 0 else None,
        position_x=params.position_x_mm,
        position_y=params.position_y_mm,
        position_z=params.position_z_mm,
    )
    pixels.planes = [plane]
    
    # Create TiffData to link Plane to TIFF IFD
    tiffdata = TiffData(
        ifd=0,
        first_z=params.z_index,
        first_c=params.channel_index,
        first_t=0,
        plane_count=1,
    )
    pixels.tiff_data_blocks = [tiffdata]
    
    # Create Image element
    timestamp = params.timestamp or dt.datetime.now().timestamp()
    image_name = params.image_name or f"{params.channel_name}_z{params.z_index:03d}"
    
    image = Image(
        id="Image:0",
        name=image_name,
        acquisition_date=dt.datetime.fromtimestamp(timestamp, dt.timezone.utc),
        pixels=pixels,
    )
    
    # Add stage label with position info
    stage_label = StageLabel(
        name=image_name,
        x=params.position_x_mm,
        y=params.position_y_mm,
        z=params.position_z_mm,
    )
    image.stage_label = stage_label
    
    # Link objective if available
    if instrument_template and instrument_template.ome:
        instr = instrument_template.ome.instruments[0] if instrument_template.ome.instruments else None
        if instr and instr.objectives:
            image.objective_settings = ObjectiveSettings(id="Objective:0")
    
    # Build description with extra metadata (use ASCII-only characters for TIFF compatibility)
    desc_lines = [f"ImSwitch Acquisition"]
    desc_lines.append(f"Microscope: {params.microscope_name}")
    desc_lines.append(f"Channel: {params.channel_name}")
    desc_lines.append(f"Position: X={params.position_x_mm:.4f}mm, Y={params.position_y_mm:.4f}mm, Z={params.position_z_mm:.4f}mm")
    desc_lines.append(f"Pixel Size: {params.pixel_size_um:.4f} um")  # Use 'um' instead of 'µm' for ASCII compatibility
    
    if params.exposure_time_ms > 0:
        desc_lines.append(f"Exposure: {params.exposure_time_ms:.2f} ms")
    if params.illumination_channel:
        desc_lines.append(f"Illumination: {params.illumination_channel} @ {params.illumination_value:.1f}%")
    if params.description:
        desc_lines.append(f"Notes: {params.description}")
    
    # Add extra metadata
    for key, value in params.extra_metadata.items():
        desc_lines.append(f"{key}: {value}")
    
    image.description = "\n".join(desc_lines)
    
    ome.images = [image]
    
    # Convert to XML string
    return ome.to_xml()


def build_ome_metadata_from_dict(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Build OME-XML from a metadata dictionary.
    
    This is a convenience function that maps ImSwitch metadata dictionaries
    to OMEMetadataParams and builds the OME-XML.
    
    Args:
        metadata: Dictionary with metadata keys. Supported keys:
            - width, height, dtype
            - pixel_size, pixelSizeUm
            - channel, channel_name, illuminationChannel
            - exposure_time_ms, exposure
            - x, y, z (position in mm or microns)
            - time_index, channel_index, z_index
            - detector_name, camera_model
            - microscope_name
            - Any other keys are added to extra_metadata
            
    Returns:
        OME-XML string or None if ome_types unavailable
    """
    if not OME_TYPES_AVAILABLE:
        return None
    
    # Extract known fields
    width = metadata.get("width", metadata.get("Width", 512))
    height = metadata.get("height", metadata.get("Height", 512))
    dtype = str(metadata.get("dtype", "uint16"))
    
    # Pixel size (check multiple possible keys)
    pixel_size = metadata.get("pixel_size", 
                              metadata.get("pixelSizeUm",
                              metadata.get("pixel_size_um", 1.0)))
    
    # Channel name
    channel_name = metadata.get("channel_name",
                                metadata.get("illuminationChannel",
                                metadata.get("channel",
                                metadata.get("detector_name", "Unknown"))))
    
    # Exposure
    exposure = metadata.get("exposure_time_ms",
                           metadata.get("exposure",
                           metadata.get("exposureTime", 0.0)))
    
    # Position (convert from microns to mm if needed based on magnitude)
    x = metadata.get("x", metadata.get("position_x", 0.0))
    y = metadata.get("y", metadata.get("position_y", 0.0))
    z = metadata.get("z", metadata.get("position_z", 0.0))
    
    # Heuristic: if position values > 100, assume microns and convert to mm
    if abs(x) > 100 or abs(y) > 100:
        x = x / 1000.0
        y = y / 1000.0
        z = z / 1000.0
    
    # Indices
    z_index = metadata.get("z_index", 0)
    channel_index = metadata.get("channel_index", 0)
    time_index = metadata.get("time_index", 0)
    
    # Illumination
    illum_channel = metadata.get("illuminationChannel")
    illum_value = metadata.get("illuminationValue", 0.0)
    
    # Equipment
    microscope = metadata.get("microscope_name", "openUC2")
    camera_model = metadata.get("camera_model", metadata.get("detector_name"))
    
    # Collect extra metadata (anything not in known fields)
    known_keys = {
        "width", "height", "dtype", "pixel_size", "pixelSizeUm", "pixel_size_um",
        "channel_name", "illuminationChannel", "channel", "detector_name",
        "exposure_time_ms", "exposure", "exposureTime",
        "x", "y", "z", "position_x", "position_y", "position_z",
        "z_index", "channel_index", "time_index",
        "microscope_name", "camera_model",
        "Width", "Height"
    }
    extra = {k: v for k, v in metadata.items() if k not in known_keys}
    
    params = OMEMetadataParams(
        width_px=width,
        height_px=height,
        dtype=dtype,
        pixel_size_um=pixel_size,
        channel_name=channel_name,
        channel_index=channel_index,
        exposure_time_ms=exposure,
        position_x_mm=x,
        position_y_mm=y,
        position_z_mm=z,
        z_index=z_index,
        illumination_channel=illum_channel,
        illumination_value=illum_value,
        microscope_name=microscope,
        camera_model=camera_model,
        extra_metadata=extra,
    )
    
    return build_ome_metadata(params)
