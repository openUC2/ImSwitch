"""
Pydantic models for STORM controller parameters and data structures.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class FittingMethodType(str, Enum):
    """Enum for available fitting methods in microEye."""
    PHASOR_2D_CPU = "2D_Phasor_CPU"
    GAUSS_MLE_FIXED_SIGMA = "2D_Gauss_MLE_fixed_sigma"
    GAUSS_MLE_FREE_SIGMA = "2D_Gauss_MLE_free_sigma"
    GAUSS_MLE_ELLIPTICAL_SIGMA = "2D_Gauss_MLE_elliptical_sigma"
    GAUSS_MLE_CSPLINE = "3D_Gauss_MLE_cspline_sigma"


class FilterType(str, Enum):
    """Enum for available filter types."""
    BANDPASS = "bandpass"
    DOG = "difference_of_gaussians"
    TEMPORAL_MEDIAN = "temporal_median"


class BandpassFilterParameters(BaseModel):
    """Parameters for bandpass filter."""
    
    center: float = Field(default=40.0, gt=0, 
                         description="Center frequency for bandpass filter")
    
    width: float = Field(default=90.0, gt=0,
                        description="Width of bandpass filter")
    
    filter_type: str = Field(default="gauss", pattern="^(gauss|butterworth|ideal)$",
                           description="Type of bandpass filter")
    
    show_filter: bool = Field(default=False,
                            description="Show filter visualization")


class BlobDetectorParameters(BaseModel):
    """Parameters for OpenCV blob detector."""
    
    min_threshold: float = Field(default=0.0, ge=0, le=255,
                               description="Minimum threshold for blob detection")
    
    max_threshold: float = Field(default=255.0, ge=0, le=255,
                               description="Maximum threshold for blob detection")
    
    min_area: float = Field(default=1.5, gt=0,
                          description="Minimum area for blob detection")
    
    max_area: float = Field(default=80.0, gt=0,
                          description="Maximum area for blob detection")
    
    min_circularity: Optional[float] = Field(default=None, ge=0, le=1,
                                           description="Minimum circularity (None to disable)")
    
    min_convexity: Optional[float] = Field(default=None, ge=0, le=1,
                                         description="Minimum convexity (None to disable)")
    
    min_inertia_ratio: Optional[float] = Field(default=None, ge=0, le=1,
                                             description="Minimum inertia ratio (None to disable)")
    
    blob_color: int = Field(default=255, ge=0, le=255,
                          description="Color of blobs to detect")
    
    min_dist_between_blobs: float = Field(default=0.0, ge=0,
                                        description="Minimum distance between blobs")


class STORMProcessingParameters(BaseModel):
    """Parameters for STORM frame processing via microeye."""
    
    # Detection and fitting parameters
    threshold: float = Field(default=0.2, ge=0.0, le=1.0, 
                           description="Relative detection threshold (0-1)")
    
    fit_roi_size: int = Field(default=13, ge=7, le=99,
                            description="ROI size for fitting in pixels (must be odd)")
    
    fitting_method: FittingMethodType = Field(default=FittingMethodType.PHASOR_2D_CPU,
                                            description="Fitting method for localization")
    
    # Filter parameters
    filter_type: FilterType = Field(default=FilterType.BANDPASS,
                                  description="Pre-processing filter type")
    
    temporal_median_enabled: bool = Field(default=False,
                                        description="Enable temporal median filtering")
    
    # Processing parameters
    update_rate: int = Field(default=10, ge=1, le=100,
                           description="Update rate for live processing")
    
    pixel_size_nm: float = Field(default=117.5, gt=0,
                                description="Pixel size in nanometers")
    
    super_resolution_pixel_size_nm: float = Field(default=10.0, gt=0,
                                                 description="Super-resolution pixel size in nanometers")
    
    # Bandpass filter parameters
    bandpass_filter: BandpassFilterParameters = Field(default_factory=BandpassFilterParameters,
                                                     description="Bandpass filter parameters")
    
    # Blob detector parameters
    blob_detector: BlobDetectorParameters = Field(default_factory=BlobDetectorParameters,
                                                 description="Blob detector parameters")
    
    class Config:
        """Pydantic config."""
        use_enum_values = True


class STORMAcquisitionParameters(BaseModel):
    """Parameters for STORM frame acquisition."""
    
    # Session information
    session_id: Optional[str] = Field(default=None,
                                    description="Unique session identifier")
    
    # Acquisition parameters
    exposure_time: Optional[float] = Field(default=None, gt=0,
                                         description="Exposure time in milliseconds")
    
    max_frames: int = Field(default=-1,
                          description="Maximum frames to acquire (-1 for unlimited)")
    
    # Cropping parameters
    crop_enabled: bool = Field(default=False,
                             description="Enable cropping of acquired frames")
    
    crop_x: Optional[int] = Field(default=None, ge=0,
                                description="Top-left X coordinate for cropping")
    
    crop_y: Optional[int] = Field(default=None, ge=0,
                                description="Top-left Y coordinate for cropping")
    
    crop_width: Optional[int] = Field(default=None, gt=0,
                                    description="Width of crop region")
    
    crop_height: Optional[int] = Field(default=None, gt=0,
                                     description="Height of crop region")
    
    # Saving parameters
    save_enabled: bool = Field(default=False,
                             description="Enable saving of acquired frames")
    
    save_directory: Optional[str] = Field(default=None,
                                        description="Directory to save frames")
    
    save_format: str = Field(default="tiff", pattern="^(tiff|omezarr)$",
                           description="Format for saving frames")
    
    # Processing options
    process_locally: bool = Field(default=True,
                                description="Process frames locally with microeye")
    
    process_arkitekt: bool = Field(default=False,
                                 description="Process frames via Arkitekt")
    
    # Processing parameters for local processing
    processing_parameters: Optional[Dict[str, Any]] = Field(default=None,
                                                          description="Processing parameters for local reconstruction")


class STORMReconstructionResult(BaseModel):
    """Result from STORM frame reconstruction."""
    
    # Frame information
    frame_number: int = Field(description="Frame number in sequence")
    timestamp: str = Field(description="Timestamp of frame acquisition")
    session_id: str = Field(description="Session identifier")
    
    # Processing results
    num_localizations: int = Field(description="Number of localizations found")
    localization_parameters: Optional[List[List[float]]] = Field(
        default=None, description="Localization parameters (x, y, intensity, etc.)")
    
    # File paths
    raw_frame_path: Optional[str] = Field(default=None,
                                        description="Path to saved raw frame")
    
    reconstructed_frame_path: Optional[str] = Field(default=None,
                                                  description="Path to saved reconstructed frame")
    
    # Metadata
    processing_parameters: STORMProcessingParameters = Field(
        description="Parameters used for processing")
    
    acquisition_parameters: STORMAcquisitionParameters = Field(
        description="Parameters used for acquisition")


class STORMStatusResponse(BaseModel):
    """Response model for STORM status queries."""
    
    # Acquisition status
    acquisition_active: bool = Field(description="Whether acquisition is currently active")
    session_id: Optional[str] = Field(default=None, description="Current session ID")
    frames_acquired: int = Field(default=0, description="Number of frames acquired")
    
    # Processing status
    processing_active: bool = Field(default=False, description="Whether processing is active")
    frames_processed: int = Field(default=0, description="Number of frames processed")
    total_localizations: int = Field(default=0, description="Total localizations found")
    
    # System status
    microeye_available: bool = Field(description="Whether microEye is available")
    arkitekt_available: bool = Field(description="Whether Arkitekt is available")
    
    # Current parameters
    current_processing_params: Optional[STORMProcessingParameters] = Field(
        default=None, description="Current processing parameters")
    
    current_acquisition_params: Optional[STORMAcquisitionParameters] = Field(
        default=None, description="Current acquisition parameters")
    
    # Last reconstruction info
    last_reconstruction_path: Optional[str] = Field(
        default=None, description="Path to last reconstructed image")


class STORMControlCommand(BaseModel):
    """Base command model for STORM control operations."""
    
    command: str = Field(description="Command to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict,
                                     description="Command parameters")


class STORMReconstructionRequest(BaseModel):
    """Request model for starting STORM reconstruction."""
    
    # Session information
    session_id: Optional[str] = Field(default=None,
                                    description="Unique session identifier")
    
    # Acquisition parameters
    acquisition_parameters: STORMAcquisitionParameters = Field(
        default_factory=STORMAcquisitionParameters,
        description="Parameters for acquisition")
    
    # Processing parameters
    processing_parameters: STORMProcessingParameters = Field(
        default_factory=STORMProcessingParameters,
        description="Parameters for processing")
    
    # Additional options
    save_enabled: bool = Field(default=True,
                             description="Enable saving of results")


class STORMProcessingRequest(BaseModel):
    """Request model for setting STORM processing parameters."""
    
    processing_parameters: STORMProcessingParameters = Field(
        description="Processing parameters to set")


class STORMErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = Field(default=False, description="Operation success status")
    error: bool = Field(default=True, description="Indicates this is an error response")
    message: str = Field(description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class STORMSuccessResponse(BaseModel):
    """Success response model."""
    
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")


if __name__ == "__main__":
    # Test the models
    print("Testing STORM pydantic models...")
    
    # Test default parameters
    proc_params = STORMProcessingParameters()
    print(f"✓ Default processing parameters: {proc_params.model_dump()}")
    
    acq_params = STORMAcquisitionParameters()  
    print(f"✓ Default acquisition parameters: {acq_params.model_dump()}")
    
    # Test the new models
    bandpass_params = BandpassFilterParameters()
    print(f"✓ Default bandpass parameters: {bandpass_params.model_dump()}")
    
    blob_params = BlobDetectorParameters()
    print(f"✓ Default blob detector parameters: {blob_params.model_dump()}")
    
    # Test validation
    try:
        invalid_params = STORMProcessingParameters(threshold=1.5)
        print("✗ Validation failed - should have caught invalid threshold")
    except Exception as e:
        print(f"✓ Validation working: {type(e).__name__}")
    
    print("✓ All STORM pydantic models working correctly!")