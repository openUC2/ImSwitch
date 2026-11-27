// src/components/holo/holoMath.js
// Wave optics for the hologram apps, all in the browser: FFT, angular-spectrum
// propagation, in-line and off-axis forward models, sideband demodulation and
// fringe counting. Fields are pairs of Float64Arrays (re, im), row-major,
// power-of-two square or rectangular grids of w x h.
//
// Sign convention follows the backend controllers: a positive reconstruction
// distance dz propagates from the sensor BACK towards the object.

// ---------------------------------------------------------------- FFT
function fft1d(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const xr = re[b] * cr - im[b] * ci;
        const xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
}

export const isPow2 = (n) => n > 0 && (n & (n - 1)) === 0;

// In-place 2D FFT (unshifted: DC at index 0).
export function fft2(re, im, w, h, inverse = false) {
  for (let y = 0; y < h; y++) {
    fft1d(re.subarray(y * w, (y + 1) * w), im.subarray(y * w, (y + 1) * w), inverse);
  }
  const cr = new Float64Array(h);
  const ci = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) { cr[y] = re[y * w + x]; ci[y] = im[y * w + x]; }
    fft1d(cr, ci, inverse);
    for (let y = 0; y < h; y++) { re[y * w + x] = cr[y]; im[y * w + x] = ci[y]; }
  }
}

// Swap quadrants (fftshift == ifftshift for even sizes).
export function fftshift(a, w, h) {
  const out = new a.constructor(a.length);
  const hw = w >> 1;
  const hh = h >> 1;
  for (let y = 0; y < h; y++) {
    const ys = (y + hh) % h;
    for (let x = 0; x < w; x++) out[ys * w + ((x + hw) % w)] = a[y * w + x];
  }
  return out;
}

// Spatial frequency of FFT bin k on an n-point grid with pitch dx (1/m).
const freq = (k, n, dx) => (k < n / 2 ? k : k - n) / (n * dx);

// ---------------------------------------------------------------- propagation
// Angular-spectrum free-space propagation by z (m, may be negative), in place.
// Evanescent components are dropped. The constant phase exp(ikz) is removed.
export function propagate(re, im, w, h, z, lambda, dx) {
  if (!z) return;
  fft2(re, im, w, h);
  const k0 = 1 / lambda;
  for (let y = 0; y < h; y++) {
    const fy = freq(y, h, dx);
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const fx = freq(x, w, dx);
      const s = k0 * k0 - fx * fx - fy * fy;
      if (s <= 0) { re[i] = 0; im[i] = 0; continue; }
      const ph = 2 * Math.PI * z * (Math.sqrt(s) - k0);
      const c = Math.cos(ph);
      const sn = Math.sin(ph);
      const r = re[i] * c - im[i] * sn;
      im[i] = re[i] * sn + im[i] * c;
      re[i] = r;
    }
  }
  fft2(re, im, w, h, true);
}

// ---------------------------------------------------------------- helpers
export const intensity = (re, im) => {
  const out = new Float64Array(re.length);
  for (let i = 0; i < re.length; i++) out[i] = re[i] * re[i] + im[i] * im[i];
  return out;
};
export const amplitude = (re, im) => intensity(re, im).map(Math.sqrt);
export const phase = (re, im) => {
  const out = new Float64Array(re.length);
  for (let i = 0; i < re.length; i++) out[i] = Math.atan2(im[i], re[i]);
  return out;
};

// Seeded RNG so a sample looks the same on every render.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Separable Gaussian blur, sigma in pixels, clamped edges.
export function gaussianBlur(a, w, h, sigma) {
  if (!(sigma > 0.3)) return Float64Array.from(a);
  const r = Math.ceil(3 * sigma);
  const k = new Float64Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp((-i * i) / (2 * sigma * sigma)); sum += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  const tmp = new Float64Array(a.length);
  const out = new Float64Array(a.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -r; i <= r; i++) s += k[i + r] * a[y * w + Math.min(w - 1, Math.max(0, x + i))];
      tmp[y * w + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -r; i <= r; i++) s += k[i + r] * tmp[Math.min(h - 1, Math.max(0, y + i)) * w + x];
      out[y * w + x] = s;
    }
  }
  return out;
}

// Box mean of radius r via a summed-area table: cost independent of r, used for
// the flat-field (illumination envelope) of real holograms.
export function boxBlur(a, w, h, r) {
  const W = w + 1;
  const sat = new Float64Array(W * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += a[y * w + x];
      sat[(y + 1) * W + x + 1] = sat[y * W + x + 1] + row;
    }
  }
  const out = new Float64Array(a.length);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w, x + r + 1);
      const s = sat[y1 * W + x1] - sat[y0 * W + x1] - sat[y1 * W + x0] + sat[y0 * W + x0];
      out[y * w + x] = s / ((x1 - x0) * (y1 - y0));
    }
  }
  return out;
}

// Divide by the smooth illumination envelope so free space sits at 1.
export function flatField(a, w, h) {
  const r = Math.max(4, Math.round(Math.min(w, h) / 16));
  const env = boxBlur(boxBlur(a, w, h, r), w, h, r);
  return a.map((v, i) => v / Math.max(env[i], 1e-6));
}

// ---------------------------------------------------------------- samples
// A sample is a complex transmission t = amp * exp(i*phase) on an n x n grid
// with pitch dx (m). Objects are discs/ellipses with soft edges.
export function makeSample(kind, n, dx, { seed = 7, absorption = 0.8, phaseShift = 1.0 } = {}) {
  const amp = new Float64Array(n * n).fill(1);
  const ph = new Float64Array(n * n);
  const rand = rng(seed);
  const fov = n * dx;
  const blob = (cx, cy, rx, ry, a, p) => {
    const x0 = Math.max(0, Math.floor((cx - rx) / dx) - 1);
    const x1 = Math.min(n - 1, Math.ceil((cx + rx) / dx) + 1);
    const y0 = Math.max(0, Math.floor((cy - ry) / dx) - 1);
    const y1 = Math.min(n - 1, Math.ceil((cy + ry) / dx) + 1);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot((x * dx - cx) / rx, (y * dx - cy) / ry);
        const m = Math.min(1, Math.max(0, (1 - d) * Math.min(rx, ry) / dx + 0.5));
        if (m <= 0) continue;
        const i = y * n + x;
        amp[i] *= 1 - a * m;
        ph[i] += p * m;
      }
    }
  };
  if (kind === "dust") {
    // Opaque specks, 2..8 um (at least 3 px wide): a dusty coverslip.
    for (let k = 0; k < 14; k++) {
      const r = Math.max(1.5 * dx, (1 + 3 * rand()) * 1e-6);
      blob(fov * (0.15 + 0.7 * rand()), fov * (0.15 + 0.7 * rand()), r, r, absorption, 0);
    }
  } else if (kind === "cells") {
    // Nearly transparent cells: mostly phase, a little absorption at the nucleus.
    for (let k = 0; k < 6; k++) {
      const cx = fov * (0.2 + 0.6 * rand());
      const cy = fov * (0.2 + 0.6 * rand());
      const rx = (6 + 6 * rand()) * Math.max(dx, 1e-6); // 6..12 px, >= 6 um
      blob(cx, cy, rx, rx * (0.6 + 0.4 * rand()), absorption * 0.15, phaseShift);
      blob(cx, cy, rx * 0.3, rx * 0.3, absorption * 0.3, phaseShift * 0.5);
    }
  } else {
    // "point": one small scatterer in the centre - the textbook Gabor zone plate.
    blob(fov / 2, fov / 2, 2 * dx, 2 * dx, absorption, phaseShift);
  }
  return { amp, phase: ph };
}

export function sampleField({ amp, phase: ph }) {
  const re = new Float64Array(amp.length);
  const im = new Float64Array(amp.length);
  for (let i = 0; i < amp.length; i++) { re[i] = amp[i] * Math.cos(ph[i]); im[i] = amp[i] * Math.sin(ph[i]); }
  return { re, im };
}

// ---------------------------------------------------------------- in-line model
// Point source at z1 before the sample, sensor at z2 behind it. The Fresnel
// scaling theorem turns the spherical wave into a plane wave over an effective
// distance zEff = z1 z2 / (z1 + z2) with magnification M = (z1 + z2) / z1, so
// the grid is laid out in the sample plane with pitch pixel / M.
export function inlineGeometry({ z1, z2, pixel }) {
  const M = (z1 + z2) / z1;
  return { M, zEff: (z1 * z2) / (z1 + z2), dxObj: pixel / M };
}

// sourceSize (m): an extended source (LED) is a sum of mutually incoherent
// points; each shifts the hologram, which blurs it by sourceSize * z2 / z1.
export function simulateInline({ sample, n, lambda, z1, z2, pixel, sourceSize = 0 }) {
  const { zEff, dxObj } = inlineGeometry({ z1, z2, pixel });
  const { re, im } = sampleField(sample);
  propagate(re, im, n, n, zEff, lambda, dxObj);
  const scattered = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) scattered[i] = (re[i] - 1) ** 2 + im[i] ** 2;
  const hologram = gaussianBlur(intensity(re, im), n, n, (sourceSize * z2) / z1 / pixel);
  return { hologram, scattered, zEff, dxObj };
}

// Back-propagate a recorded intensity by dz (amplitude only, phase lost:
// this is where the twin image comes from).
export function reconstructInline(hologram, w, h, dz, lambda, dx) {
  const re = hologram.map((v) => Math.sqrt(Math.max(0, v)));
  const im = new Float64Array(re.length);
  propagate(re, im, w, h, -dz, lambda, dx);
  return { re, im };
}

// ---------------------------------------------------------------- off-axis
// Object wave = sample defocused by `defocus`, reference = tilted plane wave
// with spatial frequency (sin thetaX / lambda, sin thetaY / lambda).
export function simulateOffAxis({ sample, n, lambda, pixel, defocus, thetaX, thetaY, refAmp = 1 }) {
  const { re, im } = sampleField(sample);
  propagate(re, im, n, n, defocus, lambda, pixel);
  // Carrier snapped to the FFT grid, so the demo reconstruction has no
  // residual tilt (a real hologram has one unless the peak is hit exactly).
  const snap = (k) => Math.round(k * n * pixel) / (n * pixel);
  const kx = snap(Math.sin(thetaX) / lambda);
  const ky = snap(Math.sin(thetaY) / lambda);
  const hologram = new Float64Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = y * n + x;
      const p = 2 * Math.PI * (kx * x + ky * y) * pixel;
      const rr = refAmp * Math.cos(p);
      const ri = refAmp * Math.sin(p);
      hologram[i] = (re[i] + rr) ** 2 + (im[i] + ri) ** 2;
    }
  }
  // Carrier position in the SHIFTED spectrum (pixels from the left/top).
  const carrier = { cx: n / 2 - kx * n * pixel, cy: n / 2 - ky * n * pixel };
  return { hologram, carrier, period: 1 / Math.max(Math.hypot(kx, ky) * pixel, 1e-12) };
}

// log|FFT| of a real image, shifted so DC is in the centre (for display).
export function logSpectrum(img, w, h) {
  const re = Float64Array.from(img);
  const im = new Float64Array(img.length);
  fft2(re, im, w, h);
  const out = new Float64Array(img.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.log1p(Math.hypot(re[i], im[i]));
  return fftshift(out, w, h);
}

// Crop a circular sideband (centre cx, cy in shifted-spectrum pixels, radius r),
// move it to DC, inverse FFT, then refocus by dz. Returns the complex field.
export function demodulateOffAxis(img, w, h, { cx, cy, r }, dz, lambda, dx) {
  const re = Float64Array.from(img);
  const im = new Float64Array(img.length);
  fft2(re, im, w, h);
  const sre = fftshift(re, w, h);
  const sim = fftshift(im, w, h);
  const ore = new Float64Array(img.length);
  const oim = new Float64Array(img.length);
  const icx = Math.round(cx);
  const icy = Math.round(cy);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 > r * r) continue;
      const tx = x - icx + w / 2;
      const ty = y - icy + h / 2;
      if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
      ore[ty * w + tx] = sre[y * w + x];
      oim[ty * w + tx] = sim[y * w + x];
    }
  }
  const fre = fftshift(ore, w, h);
  const fim = fftshift(oim, w, h);
  fft2(fre, fim, w, h, true);
  propagate(fre, fim, w, h, -dz, lambda, dx);
  return { re: fre, im: fim };
}

// Brightest spectral peak outside the DC region, right half-plane (the other
// order is its mirror). Same rule as the backend's find_sideband.
export function findSideband(img, w, h, dcFraction = 0.05) {
  const win = new Float64Array(img.length);
  let mean = 0;
  for (let i = 0; i < img.length; i++) mean += img[i];
  mean /= img.length;
  for (let y = 0; y < h; y++) {
    const wy = 0.5 - 0.5 * Math.cos((2 * Math.PI * y) / (h - 1));
    for (let x = 0; x < w; x++) {
      const wx = 0.5 - 0.5 * Math.cos((2 * Math.PI * x) / (w - 1));
      win[y * w + x] = (img[y * w + x] - mean) * wx * wy;
    }
  }
  const spec = logSpectrum(win, w, h);
  let best = -1;
  let bx = w / 2;
  let by = h / 2;
  for (let y = 0; y < h; y++) {
    for (let x = w / 2; x < w; x++) {
      if (x === w / 2 && y > h / 2) continue;
      if (Math.hypot((x - w / 2) / w, (y - h / 2) / h) < dcFraction) continue;
      if (spec[y * w + x] > best) { best = spec[y * w + x]; bx = x; by = y; }
    }
  }
  const dist = Math.hypot(bx - w / 2, by - h / 2);
  return { cx: bx, cy: by, r: Math.max(3, Math.round(dist / 3)) };
}

// ---------------------------------------------------------------- fringe counting
// Count fringes in an intensity trace with a hysteresis comparator around the
// mid level: each bright->dark->bright cycle is one fringe. The threshold is
// `hysteresis` x the half peak-to-peak swing (5th..95th percentile), but never
// below 3x the noise, estimated from the sample-to-sample jumps - otherwise a
// still interferometer reports its own noise as fringes.
// ponytail: the noise estimate assumes >= ~20 samples per fringe; faster fringes
// raise it and are undercounted. One detector cannot tell the direction.
export function countFringes(values, hysteresis = 0.3) {
  if (!values || values.length < 3) return { fringes: 0, mean: 0, threshold: 0, noise: 0 };
  const pct = (arr, q) => arr[Math.floor(q * (arr.length - 1))];
  const sorted = Array.from(values).sort((a, b) => a - b);
  const lo = pct(sorted, 0.05);
  const hi = pct(sorted, 0.95);
  const mean = (lo + hi) / 2;
  const jumps = [];
  for (let i = 1; i < values.length; i++) jumps.push(Math.abs(values[i] - values[i - 1]));
  jumps.sort((a, b) => a - b);
  const noise = (1.4826 * pct(jumps, 0.5)) / Math.SQRT2; // median jump -> sigma
  const threshold = Math.max(((hi - lo) / 2) * hysteresis, 3 * noise);
  let state = 0;
  let halves = 0;
  for (const v of values) {
    const d = v - mean;
    if (d > threshold && state !== 1) { if (state === -1) halves++; state = 1; }
    else if (d < -threshold && state !== -1) { if (state === 1) halves++; state = -1; }
  }
  return { fringes: halves / 2, mean, threshold, noise };
}

// ---------------------------------------------------------------- display
// Map a float image to RGBA. "phase" uses a cyclic colour map over [-pi, pi].
export function toRGBA(values, { min, max, colormap = "gray", gamma = 1 } = {}) {
  const n = values.length;
  let lo = min;
  let hi = max;
  if (lo === undefined || hi === undefined) {
    lo = Infinity; hi = -Infinity;
    for (let i = 0; i < n; i++) { const v = values[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  const span = hi > lo ? hi - lo : 1;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    let t = (values[i] - lo) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    if (colormap === "phase") {
      const p = t * 2 * Math.PI;
      out[i * 4] = 255 * (0.5 + 0.5 * Math.cos(p));
      out[i * 4 + 1] = 255 * (0.5 + 0.5 * Math.cos(p - (2 * Math.PI) / 3));
      out[i * 4 + 2] = 255 * (0.5 + 0.5 * Math.cos(p + (2 * Math.PI) / 3));
    } else {
      const g = 255 * (gamma === 1 ? t : t ** gamma);
      out[i * 4] = g; out[i * 4 + 1] = g; out[i * 4 + 2] = g;
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}
