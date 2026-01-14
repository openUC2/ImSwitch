// UC2F binary packet parser and decompression utilities
import LZ4 from 'lz4js';

/**
 * Parse UC2F binary packet header and extract compressed data
 * Backend format: HDR_FMT = "<4sB3xIIIHBBQ" (30 bytes) + u32 compressed_size + compressed_data
 */
export function parseUC2F(buf) {
  if (!buf || buf.byteLength < 30) { // 30 byte header minimum
    throw new Error(`Invalid buffer: expected at least 30 bytes, got ${buf?.byteLength || 0}`);
  }

  const view = new DataView(buf);
  
  // Check UC2F magic number at start
  const magic = new Uint8Array(buf, 0, 4);
  const magicStr = String.fromCharCode(...magic);
  if (magicStr !== 'UC2F') {
    console.warn(`Invalid UC2F magic: got "${magicStr}", expected "UC2F"`);
    // console.log('First 20 bytes (hex):', Array.from(new Uint8Array(buf, 0, Math.min(20, buf.byteLength)))
    //.map(b => b.toString(16).padStart(2, '0')).join(' '));
    throw new Error(`Invalid UC2F magic: got "${magicStr}", expected "UC2F"`);
  }
  
  // Parse header according to backend format: HDR_FMT = "<4sB3xIIIHBBQ"
  const header = {
    magic: magicStr,
    version: view.getUint8(4),
    // 3 bytes padding (5-7)
    width: view.getUint32(8, true),    // little endian
    height: view.getUint32(12, true),
    stride: view.getUint32(16, true),
    bitdepth: view.getUint16(20, true),
    channels: view.getUint8(22),
    pixfmt: view.getUint8(23),
    ts: view.getBigUint64(24, true)    // 64-bit timestamp
  };
  
  // Try to read compressed size after header (30 bytes), but handle cases where it's not present
  const headerSize = 32;
  let compSize = 0;
  let compDataOffset = headerSize;
  
  // Check if there's a compressed size field (4 more bytes)
  if (buf.byteLength >= headerSize + 4) {
    const potentialCompSize = view.getUint32(headerSize, true);
    const remainingAfterCompSizeField = buf.byteLength - (headerSize + 4);
    
    // If the potential compressed size makes sense (not too large), use it
    if (potentialCompSize > 0 && potentialCompSize <= remainingAfterCompSizeField) {
      compSize = potentialCompSize;
      compDataOffset = headerSize + 4; // After header + u32 compSize
      // console.log('UC2F: Using compressed size field:', compSize);
    } else {
      // Compressed size field seems wrong, assume no compressed size field
      compSize = buf.byteLength - headerSize;
      compDataOffset = headerSize;
      // console.log('UC2F: Ignoring invalid compressed size field, using remaining data:', compSize);
    }
  } else {
    // No space for compressed size field, use all remaining data
    compSize = buf.byteLength - headerSize;
    compDataOffset = headerSize;
    // console.log('UC2F: No compressed size field, using remaining data:', compSize);
  }
  
  const comp = new Uint8Array(buf, compDataOffset, compSize);
  const completeHeader = { ...header, compSize };
  
  // console.log('UC2F parsed:', completeHeader);
  return { header: completeHeader, comp };
}

/**
 * Decompress LZ4-compressed data or return raw data if not compressed
 */
export function decompressLZ4(comp) {
  try {
    // Check if data has LZ4 frame magic number
    if (comp.length >= 4) {
      const hasLZ4Magic = (comp[0] === 0x04 && comp[1] === 0x22 && comp[2] === 0x4D && comp[3] === 0x18) || // LZ4 frame magic
                         (comp[0] === 0x02 && comp[1] === 0x21 && comp[2] === 0x4C && comp[3] === 0x18);   // Legacy LZ4 magic
      
      if (hasLZ4Magic) {
        const decompressed = LZ4.decompress(comp);
        // console.log(`LZ4 decompressed ${comp.length} bytes to ${decompressed.length} bytes`);
        return decompressed;
      }
    }
    
    // console.log('No LZ4 magic found, treating as uncompressed');
    return new Uint8Array(comp);
  } catch (error) {
    console.warn('LZ4 decompression failed, treating as uncompressed:', error.message);
    return new Uint8Array(comp);
  }
}

/**
 * Convert decompressed raw data to Uint16Array with proper stride handling
 */
export function rawToUint16Array(raw, width, height, stride) {
  const expectedPixels = width * height;
  const expectedBytes = expectedPixels * 2; // 2 bytes per uint16 pixel
  
  // console.log(`rawToUint16Array: raw.length=${raw.length}, width=${width}, height=${height}, stride=${stride}`);
  // console.log(`Expected: ${expectedPixels} pixels (${expectedBytes} bytes)`);
  
  if (raw.length >= expectedBytes) {
    // We have enough data - extract exactly what we need
    const u16Array = new Uint16Array(raw.buffer, raw.byteOffset, expectedPixels);
    
    // Log sample values to verify data
    const sampleCount = Math.min(10, expectedPixels);
    const samples = [];
    for (let i = 0; i < sampleCount; i++) {
      samples.push(u16Array[i]);
    }
    // console.log(`Sample pixel values: [${samples.join(', ')}]`);
    
    return u16Array;
  } else {
    console.warn(`Insufficient data: got ${raw.length} bytes, expected ${expectedBytes} bytes`);
    
    // Use what we have, pad with zeros if needed
    const u16Array = new Uint16Array(expectedPixels);
    const availableElements = Math.floor(raw.length / 2);
    const srcArray = new Uint16Array(raw.buffer, raw.byteOffset, availableElements);
    
    // Copy available data
    for (let i = 0; i < Math.min(availableElements, expectedPixels); i++) {
      u16Array[i] = srcArray[i];
    }
    
    return u16Array;
  }
}

/**
 * Complete UC2F packet processing pipeline
 */
export function processUC2FPacket(buf) {
  try {
    // console.log(`Processing UC2F packet: ${buf.byteLength} bytes`);
    
    const { header, comp } = parseUC2F(buf);
    const raw = decompressLZ4(comp);
    const dataU16 = rawToUint16Array(raw, header.width, header.height, header.stride);
    
    // Log pixel value range for debugging
    if (dataU16.length > 0) {
      const sampleSize = Math.min(1000, dataU16.length);
      let minVal = dataU16[0];
      let maxVal = dataU16[0];
      for (let i = 1; i < sampleSize; i++) {
        if (dataU16[i] < minVal) minVal = dataU16[i];
        if (dataU16[i] > maxVal) maxVal = dataU16[i];
      }
      // console.log(`Pixel value range (first ${sampleSize} pixels): ${minVal} - ${maxVal}`);
    }
    
    return {
      ...header,
      dataU16
    };
  } catch (error) {
    console.error('UC2F processing error:', error);
    throw error;
  }
}

/**
 * Process UC2F packet using backend metadata for accurate parsing
 */
export function processUC2FPacketWithMetadata(buf, metadata) {
  // For now, just use the standard parser - metadata could be used for validation
  return processUC2FPacket(buf);
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