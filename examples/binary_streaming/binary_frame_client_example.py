#!/usr/bin/env python3
"""
Example client for consuming binary image frames from ImSwitch.
This shows how to decode the binary frame format on the client side.
"""
import struct
import numpy as np
import time

# Constants from the binary streaming protocol
UC2_MAGIC = b'UC2F'
HDR_FMT = "<4sB3xIIIHBBQ"  # magic, ver, pad, w,h,stride,bitdepth,ch,pixfmt,ts
PIXFMT_GRAY16 = 0

def decode_binary_frame(data: bytes) -> tuple:
    """
    Decode a binary frame from ImSwitch.
    
    Args:
        data: Raw binary frame data from Socket.IO
        
    Returns:
        tuple: (image_array, metadata_dict)
    """
    # Parse header
    header_size = struct.calcsize(HDR_FMT)
    if len(data) < header_size + 4:
        raise ValueError("Frame data too short")
    
    header_data = data[:header_size]
    magic, ver, w, h, stride, bitdepth, channels, pixfmt, timestamp_ns = struct.unpack(HDR_FMT, header_data)
    
    if magic != UC2_MAGIC:
        raise ValueError(f"Invalid magic: {magic}")
    if ver != 1:
        raise ValueError(f"Unsupported version: {ver}")
    
    # Get compressed size
    compressed_size = struct.unpack("<I", data[header_size:header_size + 4])[0]
    
    # Extract compressed data
    data_start = header_size + 4
    data_end = data_start + compressed_size
    if len(data) < data_end:
        raise ValueError("Frame data truncated")
    
    compressed_data = data[data_start:data_end]
    
    # Decompress data (you would use lz4 or zstd here based on your needs)
    # For now, assume uncompressed data
    raw_data = compressed_data
    
    # Reshape to image
    expected_size = h * w * channels * 2  # 2 bytes per uint16 pixel
    if len(raw_data) != expected_size:
        # Data was compressed - you need to decompress it here
        # import lz4.frame
        # raw_data = lz4.frame.decompress(compressed_data)
        pass
    
    # Convert to numpy array
    img = np.frombuffer(raw_data, dtype=np.uint16).reshape((h, w))
    
    metadata = {
        'width': w,
        'height': h,
        'bitdepth': bitdepth,
        'channels': channels,
        'pixfmt': pixfmt,
        'timestamp_ns': timestamp_ns,
        'compressed_size': compressed_size,
        'stride': stride
    }
    
    return img, metadata

def socketio_client_example():
    """
    Example of how to use the binary frames with python-socketio client.
    """
    import socketio
    
    sio = socketio.Client()
    
    @sio.on('frame')
    def on_binary_frame(data):
        """Handle binary frame data."""
        try:
            img, metadata = decode_binary_frame(data)
            
            print(f"Received frame: {metadata['width']}x{metadata['height']}, "
                  f"bitdepth={metadata['bitdepth']}, "
                  f"timestamp={metadata['timestamp_ns']}")
            
            # Process the 16-bit image
            # - Apply window/level adjustments
            # - Convert to display format
            # - Update UI
            
            # Example: Simple windowing for display
            display_img = np.clip((img - 1000) / 2000 * 255, 0, 255).astype(np.uint8)
            
            # Here you would update your display/UI
            # update_display(display_img)
            
        except Exception as e:
            print(f"Error decoding frame: {e}")
    
    @sio.on('signal')
    def on_signal(data):
        """Handle JSON metadata signals."""
        if isinstance(data, str):
            import json
            data = json.loads(data)
        
        if data.get('name') == 'frame_meta':
            print(f"Frame metadata: {data}")
    
    # Connect to ImSwitch
    sio.connect('http://localhost:8002')  # Adjust URL as needed
    
    print("Connected to ImSwitch, listening for frames...")
    sio.wait()

def websocket_client_example():
    """
    Example using websockets directly.
    """
    import asyncio
    import websockets
    import json
    
    async def client():
        uri = "ws://localhost:8002/ws/frames"  # Adjust URL as needed
        
        async with websockets.connect(uri) as websocket:
            print("Connected to ImSwitch websocket")
            
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary frame
                    try:
                        img, metadata = decode_binary_frame(message)
                        print(f"Frame: {metadata['width']}x{metadata['height']}")
                    except Exception as e:
                        print(f"Error decoding frame: {e}")
                else:
                    # JSON message
                    try:
                        data = json.loads(message)
                        print(f"JSON: {data}")
                    except:
                        pass
    
    asyncio.run(client())

if __name__ == "__main__":
    print("Binary Frame Client Examples")
    print("1. Socket.IO client (requires python-socketio)")
    print("2. WebSocket client (requires websockets)")
    print("3. Demo decode function")
    
    choice = input("Choose example (1/2/3): ").strip()
    
    if choice == "1":
        try:
            socketio_client_example()
        except ImportError:
            print("python-socketio not available. Install with: pip install python-socketio")
    elif choice == "2":
        try:
            websocket_client_example()
        except ImportError:
            print("websockets not available. Install with: pip install websockets")
    elif choice == "3":
        # Demo with fake data
        print("Creating fake binary frame...")
        
        # Create fake frame data
        w, h = 100, 100
        img_data = np.random.randint(0, 4096, (h, w), dtype=np.uint16)
        timestamp_ns = int(time.time_ns())
        
        # Pack header
        header = struct.pack(HDR_FMT, UC2_MAGIC, 1, w, h, w*2, 12, 1, PIXFMT_GRAY16, timestamp_ns)
        raw_bytes = img_data.tobytes()
        size_bytes = struct.pack("<I", len(raw_bytes))
        
        fake_frame = header + size_bytes + raw_bytes
        
        print(f"Fake frame: {len(fake_frame)} bytes")
        
        # Decode it
        decoded_img, metadata = decode_binary_frame(fake_frame)
        print(f"Decoded: {decoded_img.shape}, metadata: {metadata}")
        print("Original and decoded match:", np.array_equal(img_data, decoded_img))
    else:
        print("Invalid choice")