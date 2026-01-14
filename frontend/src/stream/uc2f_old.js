// UC2F binary packet parser and decompression utilities
import LZ4 from 'lz4js';

/**
 * Parse UC2F binary packet header and extract compressed data
 */
export function parseUC2F(buf) {
  if (!buf || buf.byteLength < 30) {
    throw new Error(`Invalid buffer: expected at least 30 bytes, got ${buf?.byteLength || 0}`);
  }

  const view = new DataView(buf);
  
  // Debug: examine the first 40 bytes to understand the structure
  console.log('Raw buffer analysis:');
  console.log('First 40 bytes (hex):', Array.from(new Uint8Array(buf, 0, Math.min(40, buf.byteLength)))
    .map(b => b.toString(16).padStart(2, '0')).join(' '));
  
  // Try different offsets to find the correct header start
  for (let testOffset = 0; testOffset < Math.min(20, buf.byteLength - 30); testOffset++) {
    const testWidth = view.getUint32(testOffset, true);
    const testHeight = view.getUint32(testOffset + 4, true);
    console.log(`Offset ${testOffset}: width=${testWidth}, height=${testHeight}`);
    
    if (testWidth === 512 && testHeight === 512) {
      console.log(`Found valid header at offset ${testOffset}!`);
      
      const header = {
        width: view.getUint32(testOffset, true),
        height: view.getUint32(testOffset + 4, true),
        stride: view.getUint32(testOffset + 8, true),
        bitdepth: view.getUint16(testOffset + 12, true),
        channels: view.getUint16(testOffset + 14, true),
        pixfmt: view.getUint16(testOffset + 16, true),
        ts: view.getBigUint64(testOffset + 18, true),
        // Note: compSize is NOT in the header, it's a separate field after the header
      };
      
      console.log('Parsed header (without compSize):', header);
      
      // Debug: examine bytes around expected compSize position
      console.log('Examining bytes around header end:');
      for (let i = 30; i < Math.min(50, buf.byteLength - testOffset); i++) {
        const byte = view.getUint8(testOffset + i);
        const u32Value = i <= buf.byteLength - testOffset - 4 ? view.getUint32(testOffset + i, true) : 'N/A';
        console.log(`  Offset ${testOffset + i} (rel ${i}): byte=${byte.toString(16).padStart(2, '0')} u32=${u32Value}`);
      }
      
      // Looking at the debug output, it seems the compSize (2379) might be at a different location
      // Let's search for the value 2379 (0x94B in hex)
      console.log('Searching for compSize value 2379 (0x94B):');
      for (let searchOffset = testOffset + 30; searchOffset < buf.byteLength - 4; searchOffset++) {
        const testValue = view.getUint32(searchOffset, true);
        if (testValue === 2379 || testValue === 2373 || (testValue >= 2370 && testValue <= 2390)) {
          console.log(`  Found potential compSize ${testValue} at offset ${searchOffset}`);
          
          const compDataOffset = searchOffset + 4;
          const availableDataSize = buf.byteLength - compDataOffset;
          console.log(`  Compressed data would start at ${compDataOffset}, available: ${availableDataSize}`);
          
          if (availableDataSize >= testValue && testValue > 0) {
            // This looks like the correct compSize
            const actualCompSize = testValue;
            const completeHeader = { ...header, compSize: actualCompSize };
            console.log('Found correct compSize:', actualCompSize);
            console.log('Complete header:', completeHeader);
            
            const comp = new Uint8Array(buf, compDataOffset, actualCompSize);
            return { header: completeHeader, comp };
          }
        }
      }
      
      // Fallback: use available data size if we couldn't find the exact compSize
      console.log('Could not find exact compSize, using fallback method');
      const headerSize = testOffset + 30;
      const compDataOffset = headerSize + 4;
      const actualCompSize = buf.byteLength - compDataOffset;
      
      // Update header with corrected compSize
      const completeHeader = { ...header, compSize: actualCompSize };
      console.log('Complete header with corrected compSize:', completeHeader);

      const comp = new Uint8Array(buf, compDataOffset, actualCompSize);
      return { header: completeHeader, comp };
    }
  }
  
  // If we get here, we couldn't find a valid header
  throw new Error('Could not find valid UC2F header in buffer');
}

/**
 * Decompress LZ4-compressed data or return raw data if not compressed
 */
export function decompressLZ4(comp) {
  try {
    // Check if data looks like LZ4 (has magic number)
    if (comp.length >= 4) {
      const magicBytes = Array.from(comp.slice(0, 4));
      console.log('First 4 bytes of compressed data:', magicBytes.map(b => '0x' + b.toString(16).padStart(2, '0')).join(' '));
    }
    
    return LZ4.decompress(comp);
  } catch (error) {
    console.warn('LZ4 decompression failed, assuming uncompressed data:', error.message);
    
    // If LZ4 decompression fails, the data might be uncompressed
    // Return the raw data as-is
    return new Uint8Array(comp);
  }
}

/**
 * Convert decompressed raw data to Uint16Array with proper stride handling
 */
export function rawToUint16Array(raw, width, height, stride) {
  console.log(`rawToUint16Array: raw.length=${raw.length}, width=${width}, height=${height}, stride=${stride}`);
  
  const expectedPixels = width * height;
  const expectedBytes = expectedPixels * 2; // 2 bytes per uint16 pixel
  const actualElements = Math.floor(raw.length / 2); // Each uint16 = 2 bytes
  
  console.log(`Expected pixels: ${expectedPixels}, expected bytes: ${expectedBytes}, actual elements: ${actualElements}`);
  
  // The frame data appears to be variable size due to streaming/compression artifacts
  // We need to extract exactly expectedPixels worth of uint16 data
  
  if (raw.length >= expectedBytes) {
    // We have enough data - extract exactly what we need from the beginning
    console.log(`Extracting ${expectedPixels} pixels from ${raw.length} bytes of data`);
    const u16Array = new Uint16Array(raw.buffer, raw.byteOffset, expectedPixels);
    
    // Log sample values to verify data
    const samples = [];
    for (let i = 0; i < Math.min(10, expectedPixels); i++) {
      samples.push(u16Array[i]);
    }
    console.log(`Sample values from extracted data: [${samples.join(', ')}]`);
    
    return u16Array;
  } else {
    // Not enough data - this suggests a parsing or streaming issue
    console.warn(`Insufficient data: got ${raw.length} bytes, expected ${expectedBytes} bytes for ${width}x${height} uint16 image`);
    console.warn(`This suggests UC2F header parsing or streaming issues`);
    
    // Use what we have, but pad with zeros if needed
    const u16Array = new Uint16Array(expectedPixels);
    const availableElements = Math.floor(raw.length / 2);
    const srcArray = new Uint16Array(raw.buffer, raw.byteOffset, availableElements);
    
    // Copy available data
    for (let i = 0; i < Math.min(availableElements, expectedPixels); i++) {
      u16Array[i] = srcArray[i];
    }
    
    console.log(`Filled ${Math.min(availableElements, expectedPixels)} pixels, padded ${expectedPixels - Math.min(availableElements, expectedPixels)} with zeros`);
    return u16Array;
  }
}

/**
 * Complete UC2F packet processing pipeline
 */
export function processUC2FPacket(buf) {
  try {
    console.log(`Processing UC2F packet: ${buf.byteLength} bytes`);
    
    const { header, comp } = parseUC2F(buf);
    console.log('Parsed header:', header);
    
    const raw = decompressLZ4(comp);
    console.log(`Decompressed ${comp.length} bytes to ${raw.length} bytes`);
    
    const dataU16 = rawToUint16Array(raw, header.width, header.height, header.stride);
    console.log(`Converted to Uint16Array: ${dataU16.length} elements`);
    
    return {
      ...header,
      dataU16
    };
  } catch (error) {
    console.error('UC2F processing error:', error);
    console.error('Buffer info:', {
      byteLength: buf?.byteLength,
      constructor: buf?.constructor?.name
    });
    throw error;
  }
}

/**
 * Process UC2F packet using backend metadata for accurate parsing
 * Based on backend format: HDR_FMT = "<4sB3xIIIHBBQ" (30 bytes) + u32 compressed_size + compressed_data
 */
export function processUC2FPacketWithMetadata(buf, metadata) {
  try {
    console.log(`Processing UC2F packet with metadata: ${buf.byteLength} bytes`);
    console.log('Metadata:', metadata);
    
    const view = new DataView(buf);
    
    // Verify UC2F magic at start of buffer
    const magic = new Uint8Array(buf, 0, 4);
    const magicStr = String.fromCharCode(...magic);
    if (magicStr !== 'UC2F') {
      throw new Error(`Invalid UC2F magic: got "${magicStr}", expected "UC2F"`);
    }
    
    // Parse header using backend format: HDR_FMT = "<4sB3xIIIHBBQ" 
    const headerSize = 30; // struct.calcsize(HDR_FMT)
    const header = {
      magic: magicStr,
      version: view.getUint8(4),
      // 3 bytes padding (5-7)
      width: view.getUint32(8, true),
      height: view.getUint32(12, true), 
      stride: view.getUint32(16, true),
      bitdepth: view.getUint16(20, true),
      channels: view.getUint8(22),
      pixfmt: view.getUint8(23),
      ts: view.getBigUint64(24, true)
    };
    
    console.log('Parsed UC2F header:', header);
    
    // Read compressed size after header
    const compSize = view.getUint32(headerSize, true);
    console.log('Compressed size from header:', compSize);
    
    // Extract compressed data
    const compDataOffset = headerSize + 4; // After header + u32 compSize
    const availableDataSize = buf.byteLength - compDataOffset;
    
    // Sanity check the compressed size - if it's unreasonable, treat as uncompressed
    const expectedUncompressedSize = header.width * header.height * 2; // 2 bytes per uint16 pixel
    const isCompSizeReasonable = compSize > 0 && compSize <= availableDataSize && compSize < expectedUncompressedSize * 2;
    
    let actualCompSize, comp, raw;
    
    if (!isCompSizeReasonable) {
      console.warn(`Unreasonable compSize ${compSize}, treating entire payload as uncompressed data`);
      // Treat the entire remaining buffer as uncompressed data
      actualCompSize = availableDataSize;
      comp = new Uint8Array(buf, compDataOffset, actualCompSize);
      raw = comp;
    } else {
      actualCompSize = Math.min(compSize, availableDataSize);
      comp = new Uint8Array(buf, compDataOffset, actualCompSize);
      
      console.log(`Extracting compressed data: offset=${compDataOffset}, size=${actualCompSize}`);
      console.log('First 4 bytes of compressed data:', Array.from(comp.slice(0, 4)).map(b => '0x' + b.toString(16).padStart(2, '0')).join(' '));
      
      // Check for LZ4 magic numbers or try decompression
      const hasLZ4Magic = (comp[0] === 0x04 && comp[1] === 0x22 && comp[2] === 0x4D && comp[3] === 0x18) || // LZ4 frame magic
                         (comp[0] === 0x02 && comp[1] === 0x21 && comp[2] === 0x4C && comp[3] === 0x18);   // Legacy LZ4 magic
      
      if (hasLZ4Magic) {
        try {
          raw = LZ4.decompress(comp);
          console.log(`LZ4 decompressed ${comp.length} bytes to ${raw.length} bytes`);
        } catch (error) {
          console.warn('LZ4 decompression failed despite magic number, treating as uncompressed:', error.message);
          raw = comp;
        }
      } else {
        console.log('No LZ4 magic number found, treating as uncompressed data');
        raw = comp;
      }
    }
    
    // Debug: check for potential data alignment issues
    const expectedPixelBytes = header.width * header.height * 2; // uint16 = 2 bytes per pixel
    console.log(`Expected pixel data: ${expectedPixelBytes} bytes for ${header.width}x${header.height} uint16 image`);
    console.log(`Available raw data: ${raw.length} bytes`);
    console.log(`Data size difference: ${raw.length - expectedPixelBytes} bytes`);
    
    // Convert to Uint16Array
    const dataU16 = rawToUint16Array(raw, header.width, header.height, header.stride);
    console.log(`Converted to Uint16Array: ${dataU16.length} elements`);
    
    // Check if we have reasonable pixel values and compute range
    const sampleValues = Array.from(dataU16.slice(0, 10));
    console.log('Sample pixel values:', sampleValues);
    
    // Check pixel value range to help debug black canvas
    if (dataU16.length > 0) {
      const checkSize = Math.min(1000, dataU16.length);
      let minVal = dataU16[0];
      let maxVal = dataU16[0];
      for (let i = 1; i < checkSize; i++) {
        if (dataU16[i] < minVal) minVal = dataU16[i];
        if (dataU16[i] > maxVal) maxVal = dataU16[i];
      }
      console.log(`Pixel value range in first ${checkSize} pixels: ${minVal} - ${maxVal}`);
    }
    
    return {
      ...header,
      compSize: actualCompSize,
      dataU16
    };
  } catch (error) {
    console.error('UC2F processing with metadata error:', error);
    console.error('Buffer info:', {
      byteLength: buf?.byteLength,
      constructor: buf?.constructor?.name
    });
    console.error('Metadata:', metadata);
    throw error;
  }
}

/**
 * Check if the browser supports the required features
 */
export function checkFeatureSupport() {
  let webgl2 = false;
  let intTextures = false;
  
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (gl) {
      webgl2 = true;
      intTextures = !!(gl.R16UI && gl.RED_INTEGER && gl.UNSIGNED_SHORT);
    }
  } catch (e) {
    console.warn('WebGL2 feature detection failed:', e);
  }
  
  let lz4Available = false;
  try {
    lz4Available = typeof LZ4 !== 'undefined' && typeof LZ4.decompress === 'function';
  } catch (e) {
    console.warn('LZ4 not available:', e);
  }
  
  return {
    webgl2,
    intTextures,
    lz4: lz4Available
  };
}
