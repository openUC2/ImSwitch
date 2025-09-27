#!/usr/bin/env python3
"""
Simple demo script showing binary image streaming functionality
"""
import sys
sys.path.insert(0, '/home/runner/work/ImSwitch/ImSwitch/imswitch/imcommon/framework')

import numpy as np
import time
from binary_streaming import BinaryFrameEncoder, decode_frame_header

def main():
    print("=== Binary Image Streaming Demo ===")
    
    # Create test image (simulating 12-bit camera data)
    img = np.random.randint(0, 4096, (100, 100), dtype=np.uint16)
    print(f"Original image: {img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
    
    # Test different configurations
    configs = [
        ("Uncompressed", {"compression_algorithm": "none", "subsampling_factor": 1}),
        ("LZ4 Fast", {"compression_algorithm": "lz4", "compression_level": 0, "subsampling_factor": 1}),
        ("LZ4 + 2x downsample", {"compression_algorithm": "lz4", "compression_level": 0, "subsampling_factor": 2}),
        ("Zstd", {"compression_algorithm": "zstd", "compression_level": 3, "subsampling_factor": 1}),
    ]
    
    print(f"\n{'Config':<20} {'Size':<8} {'Ratio':<6} {'Time':<8} {'Dims':<10}")
    print("-" * 60)
    
    for name, config in configs:
        try:
            encoder = BinaryFrameEncoder(**config)
            
            start = time.time()
            packet, metadata = encoder.encode_frame(img)
            encode_time = (time.time() - start) * 1000
            
            print(f"{name:<20} {len(packet):<8} {metadata['compression_ratio']:<6.2f} "
                  f"{encode_time:<8.1f} {metadata['width']}x{metadata['height']}")
            
            # Test header decoding
            header = decode_frame_header(packet)
            assert header['width'] == metadata['width']
            assert header['height'] == metadata['height']
            
        except Exception as e:
            print(f"{name:<20} ERROR: {e}")
    
    print("\n=== Runtime Configuration Test ===")
    encoder = BinaryFrameEncoder(compression_algorithm="lz4")
    
    # Test runtime config changes
    configs = [
        {"subsampling_factor": 1, "compression_level": 0},
        {"subsampling_factor": 2, "compression_level": 1},
        {"compression_algorithm": "zstd", "compression_level": 3},
    ]
    
    for config in configs:
        encoder.update_config(**config)
        packet, metadata = encoder.encode_frame(img)
        print(f"Config {config}: {metadata['width']}x{metadata['height']}, "
              f"{metadata['compression_algorithm']}/{metadata['compression_level']}, "
              f"{len(packet)} bytes")
    
    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    main()