/**
 * flimBinaryParser.js - Parser for the FLIM LABS flim-server binary WebSocket protocol.
 *
 * Targets flim-imager 2.x. The server (ws://<host>:5249/data) concatenates up
 * to 100 messages into one binary WebSocket frame (no framing headers). Each
 * message starts with a 1-byte type tag followed by a type-specific
 * little-endian payload. Layouts mirror `BinaryMessage::serialize()` in
 * flim-lib (src-tauri/flim-lib/src/process/message.rs of flim-imager-2.0).
 *
 * v2 protocol differences vs v1 (this parser is NOT v1-compatible):
 *  - most messages carry a leading `step: u8` (0=preview, 1=scouting,
 *    2=insta-flim, 3=tcspc, 4=calibration, 5=phasors);
 *  - CPS is a u64 count (was f64), IntensityLimits are u64;
 *  - IMAGING_END carries a trailing u32 last_complete_frame;
 *  - new tags 16-27 (insta-flim, traces, completed frames, diagnostics...).
 *
 * Every tag must be decodable: an unknown tag has unknown length, which
 * forces dropping the rest of the frame - so all v2 tags are handled even
 * where the panel only consumes a few of them.
 */

export const FLIM_MSG = {
  LINE: 0,
  CURVE: 1,
  CALIBRATION: 2,
  PHASOR: 3,
  IMAGING_END: 4,
  END_EXPERIMENT: 6,
  LASER_PERIOD: 8,
  CHANNELS_DETECTION: 10,
  SKIP_DATA: 11,
  CPS: 12,
  SBR: 13,
  INTENSITY_LIMITS: 14,
  INTENSITY_HISTOGRAM: 15,
  LINE_INSTA_FLIM: 16,
  INSTA_FLIM_LIMITS: 17,
  INSTA_FLIM_HISTOGRAM: 18,
  INTENSITY_TRACE: 19,
  PEAK_KCPS_MATRIX: 20,
  COMPLETED_FRAME_LINES: 21,
  COMPLETED_FRAME_INSTA_FLIM_LINES: 22,
  IMAGE_DIAGNOSTIC: 23,
  IMAGE_DIAGNOSTIC_END: 24,
  ROI_CURVE: 25,
  PIXEL_DWELL_TIME: 26,
  LASER_FREQUENCY: 27,
};

/**
 * Parse one binary chunk into an array of message objects ({ type, ...fields }).
 * Unknown/truncated trailing data stops parsing gracefully.
 */
export function parseFlimChunk(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const messages = [];
  let o = 0;
  const remaining = () => arrayBuffer.byteLength - o;

  const u8 = () => {
    const v = view.getUint8(o);
    o += 1;
    return v;
  };
  const u32 = () => {
    const v = view.getUint32(o, true);
    o += 4;
    return v;
  };
  const u64 = () => {
    const v = view.getBigUint64(o, true);
    o += 8;
    return Number(v);
  };
  const f32 = () => {
    const v = view.getFloat32(o, true);
    o += 4;
    return v;
  };
  const f64 = () => {
    const v = view.getFloat64(o, true);
    o += 8;
    return v;
  };
  const u32Array = (n) => {
    // Byte offset of a WS message buffer is not always aligned; slice to be safe
    const arr = new Uint32Array(arrayBuffer.slice(o, o + n * 4));
    o += n * 4;
    return arr;
  };
  const f32Array = (n) => {
    const arr = new Float32Array(arrayBuffer.slice(o, o + n * 4));
    o += n * 4;
    return arr;
  };
  const f64Array = (n) => {
    const arr = new Float64Array(arrayBuffer.slice(o, o + n * 8));
    o += n * 8;
    return arr;
  };
  // n interleaved (u32 intensity, f32 lifetime) pairs
  const instaFlimPairs = (n) => {
    const intensities = new Uint32Array(n);
    const lifetimes = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      intensities[i] = u32();
      lifetimes[i] = f32();
    }
    return { intensities, lifetimes };
  };

  try {
    while (remaining() >= 1) {
      const type = u8();
      switch (type) {
        case FLIM_MSG.LINE: {
          const step = u8();
          const frame = u32();
          const line = u32();
          const channel = u32();
          const pixels = u32Array(u32());
          messages.push({ type, step, frame, line, channel, pixels });
          break;
        }
        case FLIM_MSG.CURVE:
        case FLIM_MSG.ROI_CURVE: {
          const step = u8();
          const frame = u32();
          const channel = u32();
          const data = u32Array(u32());
          messages.push({ type, step, frame, channel, data });
          break;
        }
        case FLIM_MSG.CALIBRATION: {
          messages.push({
            type,
            frame: u32(),
            channel: u32(),
            harmonic: u32(),
            phase: f64(),
            modulation: f64(),
          });
          break;
        }
        case FLIM_MSG.PHASOR: {
          const step = u8();
          const frame = u32();
          const channel = u32();
          const harmonic = u32();
          const gRows = u32();
          const gCols = u32();
          const gData = f64Array(gRows * gCols);
          const sRows = u32();
          const sCols = u32();
          const sData = f64Array(sRows * sCols);
          messages.push({ type, step, frame, channel, harmonic, gRows, gCols, gData, sRows, sCols, sData });
          break;
        }
        case FLIM_MSG.IMAGING_END: {
          const len = u32();
          let dataFile = null;
          if (len > 0) {
            dataFile = new TextDecoder().decode(arrayBuffer.slice(o, o + len));
            o += len;
          }
          const lastCompleteFrame = u32();
          messages.push({ type, dataFile, lastCompleteFrame });
          break;
        }
        case FLIM_MSG.END_EXPERIMENT:
        case FLIM_MSG.SKIP_DATA: {
          messages.push({ type });
          break;
        }
        case FLIM_MSG.LASER_PERIOD: {
          messages.push({ type, laserPeriod: f64(), frequency: f64() });
          break;
        }
        case FLIM_MSG.CHANNELS_DETECTION: {
          const smaChannels = u32Array(u32());
          const usbChannels = u32Array(u32());
          const flags = new Uint8Array(arrayBuffer.slice(o, o + 9));
          o += 9;
          messages.push({ type, smaChannels, usbChannels, flags });
          break;
        }
        case FLIM_MSG.CPS: {
          messages.push({ type, step: u8(), frame: u32(), channel: u32(), cps: u64() });
          break;
        }
        case FLIM_MSG.SBR: {
          messages.push({ type, step: u8(), frame: u32(), channel: u32(), sbr: f64() });
          break;
        }
        case FLIM_MSG.INTENSITY_LIMITS: {
          messages.push({ type, step: u8(), frame: u32(), channel: u32(), max: u64(), min: u64() });
          break;
        }
        case FLIM_MSG.INTENSITY_HISTOGRAM: {
          const step = u8();
          const frame = u32();
          const channel = u32();
          const labels = u32Array(u32());
          const counts = u32Array(u32());
          messages.push({ type, step, frame, channel, labels, counts });
          break;
        }
        case FLIM_MSG.LINE_INSTA_FLIM: {
          const step = u8();
          const frame = u32();
          const line = u32();
          const channel = u32();
          const { intensities, lifetimes } = instaFlimPairs(u32());
          messages.push({ type, step, frame, line, channel, intensities, lifetimes });
          break;
        }
        case FLIM_MSG.INSTA_FLIM_LIMITS: {
          messages.push({ type, step: u8(), frame: u32(), channel: u32(), max: f32(), min: f32() });
          break;
        }
        case FLIM_MSG.INSTA_FLIM_HISTOGRAM: {
          const step = u8();
          const frame = u32();
          const channel = u32();
          const labels = f32Array(u32());
          const counts = u32Array(u32());
          messages.push({ type, step, frame, channel, labels, counts });
          break;
        }
        case FLIM_MSG.INTENSITY_TRACE: {
          const step = u8();
          const frame = u32();
          const channel = u32();
          const binWidthMicros = u32();
          const bins = f32Array(u32());
          messages.push({ type, step, frame, channel, binWidthMicros, bins });
          break;
        }
        case FLIM_MSG.PEAK_KCPS_MATRIX: {
          const step = u8();
          const frame = u32();
          const channel = u32();
          const data = f32Array(u32());
          messages.push({ type, step, frame, channel, data });
          break;
        }
        case FLIM_MSG.COMPLETED_FRAME_LINES: {
          const step = u8();
          const frame = u32();
          const channel = u32();
          const nLines = u32();
          const lines = [];
          for (let i = 0; i < nLines; i++) lines.push(u32Array(u32()));
          messages.push({ type, step, frame, channel, lines });
          break;
        }
        case FLIM_MSG.COMPLETED_FRAME_INSTA_FLIM_LINES: {
          const step = u8();
          const frame = u32();
          const channel = u32();
          const nLines = u32();
          const lines = [];
          for (let i = 0; i < nLines; i++) lines.push(instaFlimPairs(u32()));
          messages.push({ type, step, frame, channel, lines });
          break;
        }
        case FLIM_MSG.IMAGE_DIAGNOSTIC: {
          const frame = u32();
          const nCps = u32();
          const cpsPerChannel = [];
          for (let i = 0; i < nCps; i++) cpsPerChannel.push({ channel: u32(), cps: u64() });
          const pixelDwellTime = u8() ? f64() : null;
          const pixelCountPerLine = u8() ? u32() : null;
          const lineTime = u8() ? f64() : null;
          const lineCount = u8() ? u32() : null;
          const frameTime = f64();
          const detectedWidth = u8() ? u32() : null;
          const detectedHeight = u8() ? u32() : null;
          const pixelDetected = u8() !== 0;
          const lineDetected = u8() !== 0;
          const frameDetected = u8() !== 0;
          messages.push({
            type, frame, cpsPerChannel, pixelDwellTime, pixelCountPerLine,
            lineTime, lineCount, frameTime, detectedWidth, detectedHeight,
            pixelDetected, lineDetected, frameDetected,
          });
          break;
        }
        case FLIM_MSG.IMAGE_DIAGNOSTIC_END: {
          messages.push({ type, lastFrame: u32() });
          break;
        }
        case FLIM_MSG.PIXEL_DWELL_TIME: {
          messages.push({ type, step: u8(), dwellTime: f64() });
          break;
        }
        case FLIM_MSG.LASER_FREQUENCY: {
          messages.push({ type, frequency: f64(), laserPeriod: f64() });
          break;
        }
        default:
          // Unknown tag: cannot know its length, abort this chunk
          console.warn(`flimBinaryParser: unknown message type ${type}, dropping rest of chunk`);
          return messages;
      }
    }
  } catch (e) {
    // Truncated message at chunk end - keep what we parsed
    console.warn('flimBinaryParser: truncated chunk', e);
  }
  return messages;
}

export default parseFlimChunk;
