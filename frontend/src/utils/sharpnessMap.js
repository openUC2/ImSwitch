// Per-cell sharpness of a live frame, for aligning the optics by eye.
//
// A tilted sample, objective or objective mount leaves only a band of the
// field in focus. Splitting the frame into cols x rows cells and scoring each
// (gradient energy / mean², so brightness does not count) shows where the
// sharp part is. Even scores across the field = no tilt.

/** Sharpness per cell (row-major, rows*cols numbers) of RGBA pixel data. */
export function sharpnessGrid(data, width, height, cols = 8, rows = 6, stride = 2) {
  const sums = new Float64Array(cols * rows);
  const means = new Float64Array(cols * rows);
  const counts = new Float64Array(cols * rows);
  const lum = (x, y) => data[(y * width + x) * 4 + 1]; // green ≈ luminance
  for (let y = 0; y < height - stride; y += stride) {
    const r = Math.min(rows - 1, Math.floor((y * rows) / height));
    for (let x = 0; x < width - stride; x += stride) {
      const c = Math.min(cols - 1, Math.floor((x * cols) / width));
      const v = lum(x, y);
      const gx = lum(x + stride, y) - v;
      const gy = lum(x, y + stride) - v;
      const i = r * cols + c;
      sums[i] += gx * gx + gy * gy;
      means[i] += v;
      counts[i] += 1;
    }
  }
  return Array.from(sums, (s, i) => {
    if (!counts[i]) return 0;
    const mean = means[i] / counts[i];
    return s / counts[i] / Math.max(1, mean * mean);
  });
}

/**
 * Left/right and top/bottom balance of a grid in [-1, 1]: 0 = even, +lr =
 * right half sharper, +tb = bottom half sharper. Middle row/column ignored.
 */
export function sharpnessBalance(values, cols, rows) {
  let left = 0, right = 0, top = 0, bottom = 0;
  values.forEach((v, i) => {
    const r = Math.floor(i / cols);
    const c = i % cols;
    if (c < cols / 2 - 0.5) left += v;
    if (c > cols / 2 - 0.5 && !(cols % 2 && c === (cols - 1) / 2)) right += v;
    if (r < rows / 2 - 0.5) top += v;
    if (r > rows / 2 - 0.5 && !(rows % 2 && r === (rows - 1) / 2)) bottom += v;
  });
  const norm = (a, b) => (a + b > 0 ? (b - a) / (a + b) : 0);
  return { lr: norm(left, right), tb: norm(top, bottom) };
}
