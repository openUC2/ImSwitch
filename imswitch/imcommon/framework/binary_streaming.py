"""
Binary streaming utilities for ImSwitch.

This module provides functionality for streaming binary image frames over Socket.IO
with lossless compression and configurable subsampling.
"""

import struct
import time
import numpy as np
from typing import Optional, Tuple, Union
import logging

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

logger = logging.getLogger(__name__)

# Binary frame constants
UC2_MAGIC = b'UC2F'
HDR_FMT = "<4sB3xIIIHBBQ"  # magic, ver, pad, w,h,stride,bitdepth,ch,pixfmt,ts
PIXFMT_GRAY16 = 0
PIXFMT_BAYER_RG16 = 1
PIXFMT_RGB48 = 2

# Compression algorithm constants
COMPRESSION_NONE = "none"
COMPRESSION_LZ4 = "lz4"
COMPRESSION_ZSTD = "zstd"

class CompressionError(Exception):
    """Exception raised when compression fails."""
    pass

class BinaryFrameEncoder:
    """Encoder for binary image frames with compression and subsampling."""
    
    def __init__(self, 
                 compression_algorithm: str = COMPRESSION_LZ4,
                 compression_level: int = 0,
                 subsampling_factor: int = 1,
                 bitdepth: int = 12,
                 pixfmt: str = "GRAY16"):
        """
        Initialize the binary frame encoder.
        
        Args:
            compression_algorithm: Compression algorithm ("none", "lz4", "zstd")
            compression_level: Compression level (algorithm-specific)
            subsampling_factor: Subsampling factor (1=no downscale, 2/3/4 reduces resolution)
            bitdepth: Bit depth of input data (e.g., 12)
            pixfmt: Pixel format ("GRAY16", "BAYER_RG16", "RGB48")
        """
        self.compression_algorithm = compression_algorithm
        self.compression_level = compression_level
        self.subsampling_factor = subsampling_factor
        self.bitdepth = bitdepth
        self.pixfmt = self._get_pixfmt_code(pixfmt)
        
        # Validate compression algorithm availability
        if compression_algorithm == COMPRESSION_LZ4 and not HAS_LZ4:
            logger.warning("LZ4 not available, falling back to 'none'")
            self.compression_algorithm = COMPRESSION_NONE
        elif compression_algorithm == COMPRESSION_ZSTD and not HAS_ZSTD:
            logger.warning("Zstandard not available, falling back to 'none'")
            self.compression_algorithm = COMPRESSION_NONE
    
    def _get_pixfmt_code(self, pixfmt: str) -> int:
        """Convert pixel format string to code."""
        pixfmt_map = {
            "GRAY16": PIXFMT_GRAY16,
            "BAYER_RG16": PIXFMT_BAYER_RG16,
            "RGB48": PIXFMT_RGB48
        }
        return pixfmt_map.get(pixfmt, PIXFMT_GRAY16)
        
    def update_config(self, 
                     compression_algorithm: Optional[str] = None,
                     compression_level: Optional[int] = None,
                     subsampling_factor: Optional[int] = None):
        """Update encoder configuration at runtime."""
        if compression_algorithm is not None:
            # Validate availability
            if compression_algorithm == COMPRESSION_LZ4 and not HAS_LZ4:
                logger.warning("LZ4 not available, keeping current algorithm")
                compression_algorithm = None
            elif compression_algorithm == COMPRESSION_ZSTD and not HAS_ZSTD:
                logger.warning("Zstandard not available, keeping current algorithm")
                compression_algorithm = None
            
            if compression_algorithm is not None:
                self.compression_algorithm = compression_algorithm
                
        if compression_level is not None:
            self.compression_level = compression_level
        if subsampling_factor is not None:
            self.subsampling_factor = max(1, subsampling_factor)

    
    def subsample(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply subsampling to the image.
        
        Args:
            img: Input image array
            
        Returns:
            Tuple of (subsampled_image, effective_factor)
        """
        if len(img.shape) < 2:
            return img, 1
            
        h, w = img.shape[:2]
        effective_factor = self.subsampling_factor 

        # Use nearest-neighbor stride slicing
        subsampled = img[::effective_factor, ::effective_factor]
        return np.ascontiguousarray(subsampled), effective_factor
    
    def compress_block(self, buf: memoryview) -> bytes:
        """
        Compress a memory buffer.
        
        Args:
            buf: Memory view of the data to compress
            
        Returns:
            Compressed bytes
            
        Raises:
            CompressionError: If compression fails
        """
        try:
            if self.compression_algorithm == COMPRESSION_NONE:
                return bytes(buf)
            elif self.compression_algorithm == COMPRESSION_LZ4:
                if not HAS_LZ4:
                    raise CompressionError("LZ4 not available")
                level = max(0, min(self.compression_level, 16))
                return lz4.compress(buf, compression_level=level)
            elif self.compression_algorithm == COMPRESSION_ZSTD:
                if not HAS_ZSTD:
                    raise CompressionError("Zstandard not available")
                level = max(1, min(self.compression_level, 22))
                cctx = zstd.ZstdCompressor(level=level)
                return cctx.compress(buf)
            else:
                raise CompressionError(f"Unsupported compression algorithm: {self.compression_algorithm}")
        except Exception as e:
            raise CompressionError(f"Compression failed: {e}")
    
    def encode_frame(self, img: np.ndarray) -> Tuple[bytes, dict]:
        """
        Encode a frame into binary format.
        
        Args:
            img: Input image array
            
        Returns:
            Tuple of (binary_packet, metadata_dict)
        """
        start_time = time.time()
        # if image is rgb we need to convert it to grayscale for now # TODO: support rgb
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(img.dtype)
            # TODO: logger.warning("RGB image converted to grayscale for binary streaming")
    
        # Convert to uint16 if needed
        if img.dtype != np.uint16:
            if img.dtype == np.uint8:
                u16_img = img.astype(np.uint16) << 8  # Scale 8-bit to 16-bit range
            elif img.dtype == np.uint32:
                u16_img = (img >> 4).astype(np.uint16)  # Assume 16-bit data in upper bits
            else:
                # For 12-bit data packed in uint16, shift to use full 16-bit range
                u16_img = (img << (16 - self.bitdepth)).astype(np.uint16)
        else:
            u16_img = img
            
        # Apply subsampling
        subsampled_img, effective_factor = self.subsample(u16_img)
        #print("Subsampled image shape took ", time.time()-start_time)
        # Ensure contiguous array
        if not subsampled_img.flags.c_contiguous:
            subsampled_img = np.ascontiguousarray(subsampled_img)
        
        h, w = subsampled_img.shape[:2]
        channels = 1 if len(subsampled_img.shape) == 2 else subsampled_img.shape[2]
        stride = w * 2 * channels  # 2 bytes per pixel for uint16
        
        # Build header
        timestamp_ns = int(time.time_ns())
        header = struct.pack(HDR_FMT, UC2_MAGIC, 1, w, h, stride, 
                           self.bitdepth, channels, self.pixfmt, timestamp_ns)
        
        # Compress data
        raw_size = subsampled_img.nbytes
        try:
            compressed = self.compress_block(memoryview(subsampled_img))
            compression_success = True
        except CompressionError as e:
            logger.warning(f"Compression failed, falling back to uncompressed: {e}")
            compressed = bytes(memoryview(subsampled_img))
            compression_success = False
        #print("Compressed image took ", time.time()-start_time  )
        # Build packet: [header][u32 compressed_size][compressed_bytes]
        packet = header + struct.pack("<I", len(compressed)) + compressed
        
        encode_time_ms = (time.time() - start_time) * 1000
        
        # Build metadata
        metadata = {
            "width": w,
            "height": h,
            "original_width": img.shape[1],
            "original_height": img.shape[0],
            "subsampling_factor": effective_factor,
            "compression_algorithm": self.compression_algorithm if compression_success else "none",
            "compression_level": self.compression_level if compression_success else 0,
            "raw_bytes": raw_size,
            "compressed_bytes": len(compressed),
            "packet_bytes": len(packet),
            "compression_ratio": raw_size / len(compressed) if len(compressed) > 0 else 1.0,
            "encode_time_ms": encode_time_ms,
            "timestamp_ns": timestamp_ns
        }
        #print("Total encoding took ", time.time()-start_time  )
        return packet, metadata


    def decode_frame_header(self, data: bytes) -> dict:
        """
        Decode a binary frame header.
        
        Args:
            data: Binary data starting with frame header
            
        Returns:
            Dictionary with header fields
            
        Raises:
            ValueError: If header is invalid
        """
        if len(data) < struct.calcsize(HDR_FMT):
            raise ValueError("Data too short for header")
        
        header_size = struct.calcsize(HDR_FMT)
        header_data = data[:header_size]
        
        fields = struct.unpack(HDR_FMT, header_data)
        magic, ver, w, h, stride, bitdepth, channels, pixfmt, timestamp_ns = fields
        
        if magic != UC2_MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")
        
        if ver != 1:
            raise ValueError(f"Unsupported version: {ver}")
        
        # Extract compressed size
        if len(data) < header_size + 4:
            raise ValueError("Data too short for compressed size")
        
        compressed_size = struct.unpack("<I", data[header_size:header_size + 4])[0]
        
        return {
            "magic": magic,
            "version": ver,
            "width": w,
            "height": h,
            "stride": stride,
            "bitdepth": bitdepth,
            "channels": channels,
            "pixfmt": pixfmt,
            "timestamp_ns": timestamp_ns,
            "header_size": header_size,
            "compressed_size": compressed_size,
            "total_expected_size": header_size + 4 + compressed_size
        }

    def decode_frame(self, data: bytes) -> Tuple[np.ndarray, dict]:
        """
        Decode a complete binary frame into image and metadata.
        
        Args:
            data: Complete binary packet from encode_frame()
            
        Returns:
            Tuple of (decoded_image_array, metadata_dict)
            
        Raises:
            ValueError: If frame is invalid
            CompressionError: If decompression fails
        """
        # First decode the header
        header = self.decode_frame_header(data)
        
        # Validate we have enough data
        if len(data) < header["total_expected_size"]:
            raise ValueError(f"Incomplete frame: got {len(data)} bytes, expected {header['total_expected_size']}")
        
        # Extract compressed data
        header_size = header["header_size"]
        compressed_size = header["compressed_size"]
        compressed_data = data[header_size + 4:header_size + 4 + compressed_size]
        
        # Decompress the data
        try:
            decompressed = self._decompress_block(compressed_data)
        except Exception as e:
            raise CompressionError(f"Decompression failed: {e}")
        
        # Reconstruct the numpy array
        width = header["width"]
        height = header["height"]
        channels = header["channels"]
        
        # Calculate expected array size
        expected_bytes = width * height * channels * 2  # 2 bytes per uint16 pixel
        if len(decompressed) != expected_bytes:
            raise ValueError(f"Decompressed size mismatch: got {len(decompressed)}, expected {expected_bytes}")
        
        # Reshape into numpy array
        if channels == 1:
            shape = (height, width)
        else:
            shape = (height, width, channels)
            
        image = np.frombuffer(decompressed, dtype=np.uint16).reshape(shape)
        
        # Build metadata
        metadata = {
            "width": width,
            "height": height,
            "channels": channels,
            "bitdepth": header["bitdepth"],
            "pixfmt": header["pixfmt"],
            "timestamp_ns": header["timestamp_ns"],
            "compressed_bytes": compressed_size,
            "raw_bytes": len(decompressed),
            "compression_ratio": len(decompressed) / compressed_size if compressed_size > 0 else 1.0
        }
        
        return image, metadata

    def _decompress_block(self, compressed_data: bytes) -> bytes:
        """
        Decompress a compressed data block.
        
        Args:
            compressed_data: Compressed bytes
            
        Returns:
            Decompressed bytes
            
        Raises:
            CompressionError: If decompression fails
        """
        try:
            if self.compression_algorithm == COMPRESSION_NONE:
                return compressed_data
            elif self.compression_algorithm == COMPRESSION_LZ4:
                if not HAS_LZ4:
                    raise CompressionError("LZ4 not available for decompression")
                return lz4.decompress(compressed_data)
            elif self.compression_algorithm == COMPRESSION_ZSTD:
                if not HAS_ZSTD:
                    raise CompressionError("Zstandard not available for decompression")
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(compressed_data)
            else:
                raise CompressionError(f"Unsupported compression algorithm: {self.compression_algorithm}")
        except Exception as e:
            raise CompressionError(f"Decompression failed: {e}")