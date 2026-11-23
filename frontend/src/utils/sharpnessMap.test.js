import { sharpnessBalance, sharpnessGrid } from "./sharpnessMap";

// A 64x48 RGBA frame: a fine checkerboard (sharp) on the right half, flat grey on the left.
const frame = () => {
  const w = 64, h = 48, data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = x >= w / 2 ? (((x >> 1) + (y >> 1)) % 2 ? 200 : 60) : 128;
      data.set([v, v, v, 255], (y * w + x) * 4);
    }
  return { data, w, h };
};

test("the sharp half scores higher and tips the balance", () => {
  const { data, w, h } = frame();
  const grid = sharpnessGrid(data, w, h, 4, 3);
  expect(grid).toHaveLength(12);
  expect(grid[0]).toBe(0); // flat
  expect(grid[3]).toBeGreaterThan(0.1); // checkerboard
  const { lr, tb } = sharpnessBalance(grid, 4, 3);
  expect(lr).toBeGreaterThan(0.95); // the seam between the halves leaks a little into the left
  expect(Math.abs(tb)).toBeLessThan(0.02); // the last row samples a few lines less
});

test("an even field is balanced", () => {
  expect(sharpnessBalance([1, 1, 1, 1, 1, 1], 3, 2)).toEqual({ lr: 0, tb: 0 });
});
