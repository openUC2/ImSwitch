"""
Unit tests for binary streaming functionality.
"""

import unittest
import numpy as np
import struct
import time
from unittest.mock import patch, MagicMock

from binary_streaming import (
    BinaryFrameEncoder, decode_frame_header, 
    UC2_MAGIC, HDR_FMT, PIXFMT_GRAY16,
    COMPRESSION_NONE, COMPRESSION_LZ4, COMPRESSION_ZSTD,
    CompressionError
)


class TestBinaryFrameEncoder(unittest.TestCase):
    """Test the BinaryFrameEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = BinaryFrameEncoder(
            compression_algorithm=COMPRESSION_NONE,
            compression_level=0,
            subsampling_factor=1,
            bitdepth=12,
            pixfmt="GRAY16"
        )
        
    def test_encoder_initialization(self):
        """Test encoder initialization with default values."""
        self.assertEqual(self.encoder.compression_algorithm, COMPRESSION_NONE)
        self.assertEqual(self.encoder.compression_level, 0)
        self.assertEqual(self.encoder.subsampling_factor, 1)
        self.assertEqual(self.encoder.bitdepth, 12)
        self.assertEqual(self.encoder.pixfmt, PIXFMT_GRAY16)
        
    def test_subsample_no_downscale(self):
        """Test subsampling with factor 1 (no downscale)."""
        img = np.random.randint(0, 4096, (100, 100), dtype=np.uint16)
        result, factor = self.encoder.subsample(img)
        
        self.assertEqual(factor, 1)
        np.testing.assert_array_equal(result, img)
        
    def test_subsample_factor_2(self):
        """Test subsampling with factor 2."""
        self.encoder.subsampling_factor = 2
        img = np.random.randint(0, 4096, (100, 100), dtype=np.uint16)
        result, factor = self.encoder.subsample(img)
        
        self.assertEqual(factor, 2)
        self.assertEqual(result.shape, (50, 50))
        np.testing.assert_array_equal(result, img[::2, ::2])
        
    def test_subsample_auto_max_dim(self):
        """Test automatic subsampling based on max dimension."""
        self.encoder.subsampling_auto_max_dim = 50
        img = np.random.randint(0, 4096, (100, 100), dtype=np.uint16)
        result, factor = self.encoder.subsample(img)
        
        self.assertEqual(factor, 2)  # 100 -> 50
        self.assertEqual(result.shape, (50, 50))
        
    def test_compress_block_none(self):
        """Test compression with 'none' algorithm."""
        data = np.random.randint(0, 65535, 100, dtype=np.uint16)
        compressed = self.encoder.compress_block(memoryview(data))
        
        self.assertEqual(compressed, data.tobytes())
        
    @patch('binary_streaming.HAS_LZ4', True)
    @patch('binary_streaming.lz4')
    def test_compress_block_lz4(self, mock_lz4):
        """Test compression with LZ4 algorithm."""
        self.encoder.compression_algorithm = COMPRESSION_LZ4
        mock_lz4.compress.return_value = b'compressed_data'
        
        data = np.random.randint(0, 65535, 100, dtype=np.uint16)
        compressed = self.encoder.compress_block(memoryview(data))
        
        self.assertEqual(compressed, b'compressed_data')
        mock_lz4.compress.assert_called_once()
        
    def test_encode_frame_basic(self):
        """Test basic frame encoding."""
        # Create test image
        img = np.random.randint(0, 4096, (10, 10), dtype=np.uint16)
        
        packet, metadata = self.encoder.encode_frame(img)
        
        # Check packet structure
        self.assertIsInstance(packet, bytes)
        self.assertGreater(len(packet), struct.calcsize(HDR_FMT) + 4)
        
        # Check metadata
        self.assertEqual(metadata['width'], 10)
        self.assertEqual(metadata['height'], 10)
        self.assertEqual(metadata['subsampling_factor'], 1)
        self.assertEqual(metadata['compression_algorithm'], COMPRESSION_NONE)
        self.assertIn('encode_time_ms', metadata)
        self.assertIn('timestamp_ns', metadata)
        
    def test_encode_frame_with_subsampling(self):
        """Test frame encoding with subsampling."""
        self.encoder.subsampling_factor = 2
        img = np.random.randint(0, 4096, (20, 20), dtype=np.uint16)
        
        packet, metadata = self.encoder.encode_frame(img)
        
        self.assertEqual(metadata['width'], 10)
        self.assertEqual(metadata['height'], 10)
        self.assertEqual(metadata['original_width'], 20)
        self.assertEqual(metadata['original_height'], 20)
        self.assertEqual(metadata['subsampling_factor'], 2)
        
    def test_encode_frame_uint8_conversion(self):
        """Test frame encoding with uint8 to uint16 conversion."""
        img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        
        packet, metadata = self.encoder.encode_frame(img)
        
        # Should handle conversion properly
        self.assertEqual(metadata['width'], 10)
        self.assertEqual(metadata['height'], 10)
        
    def test_update_config(self):
        """Test runtime configuration updates."""
        self.encoder.update_config(
            compression_algorithm=COMPRESSION_LZ4,
            compression_level=5,
            subsampling_factor=3
        )
        
        # Note: LZ4 might not be available, so it might fall back
        self.assertEqual(self.encoder.compression_level, 5)
        self.assertEqual(self.encoder.subsampling_factor, 3)


class TestDecodeFrameHeader(unittest.TestCase):
    """Test the decode_frame_header function."""
    
    def test_decode_valid_header(self):
        """Test decoding a valid frame header."""
        # Create a mock packet
        w, h, stride = 100, 100, 200
        bitdepth, channels, pixfmt = 12, 1, PIXFMT_GRAY16
        timestamp_ns = int(time.time_ns())
        
        header = struct.pack(HDR_FMT, UC2_MAGIC, 1, w, h, stride, 
                           bitdepth, channels, pixfmt, timestamp_ns)
        compressed_size = 1000
        size_bytes = struct.pack("<I", compressed_size)
        
        packet = header + size_bytes + b'x' * compressed_size
        
        decoded = decode_frame_header(packet)
        
        self.assertEqual(decoded['magic'], UC2_MAGIC)
        self.assertEqual(decoded['version'], 1)
        self.assertEqual(decoded['width'], w)
        self.assertEqual(decoded['height'], h)
        self.assertEqual(decoded['stride'], stride)
        self.assertEqual(decoded['bitdepth'], bitdepth)
        self.assertEqual(decoded['channels'], channels)
        self.assertEqual(decoded['pixfmt'], pixfmt)
        self.assertEqual(decoded['timestamp_ns'], timestamp_ns)
        self.assertEqual(decoded['compressed_size'], compressed_size)
        
    def test_decode_invalid_magic(self):
        """Test decoding with invalid magic bytes."""
        header = struct.pack(HDR_FMT, b'XXXX', 1, 100, 100, 200, 12, 1, 0, 0)
        size_bytes = struct.pack("<I", 100)
        packet = header + size_bytes
        
        with self.assertRaises(ValueError) as cm:
            decode_frame_header(packet)
        self.assertIn("Invalid magic bytes", str(cm.exception))
        
    def test_decode_invalid_version(self):
        """Test decoding with invalid version."""
        header = struct.pack(HDR_FMT, UC2_MAGIC, 99, 100, 100, 200, 12, 1, 0, 0)
        size_bytes = struct.pack("<I", 100)
        packet = header + size_bytes
        
        with self.assertRaises(ValueError) as cm:
            decode_frame_header(packet)
        self.assertIn("Unsupported version", str(cm.exception))
        
    def test_decode_too_short(self):
        """Test decoding with insufficient data."""
        packet = b'short'
        
        with self.assertRaises(ValueError) as cm:
            decode_frame_header(packet)
        self.assertIn("Data too short", str(cm.exception))


class TestRoundTripEncoding(unittest.TestCase):
    """Test round-trip encoding and decoding."""
    
    def test_round_trip_uncompressed(self):
        """Test encoding and header decoding round trip."""
        encoder = BinaryFrameEncoder(
            compression_algorithm=COMPRESSION_NONE,
            subsampling_factor=1
        )
        
        # Create test image
        original_img = np.random.randint(0, 4096, (50, 50), dtype=np.uint16)
        
        # Encode
        packet, metadata = encoder.encode_frame(original_img)
        
        # Decode header
        decoded_header = decode_frame_header(packet)
        
        # Verify consistency
        self.assertEqual(decoded_header['width'], metadata['width'])
        self.assertEqual(decoded_header['height'], metadata['height'])
        self.assertEqual(decoded_header['compressed_size'], metadata['compressed_bytes'])
        
        # Verify we can extract the compressed data
        header_size = decoded_header['header_size']
        data_start = header_size + 4
        data_end = data_start + decoded_header['compressed_size']
        compressed_data = packet[data_start:data_end]
        
        self.assertEqual(len(compressed_data), decoded_header['compressed_size'])
        
        # For uncompressed data, should match original (after subsampling)
        expected_data = original_img.tobytes()
        self.assertEqual(compressed_data, expected_data)


if __name__ == '__main__':
    unittest.main()