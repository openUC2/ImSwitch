import { autoRange, rangeMaxFor } from "./DisplayRangeControl";

test("raw range follows the brightest bin up to the next 2^n - 1", () => {
  expect(rangeMaxFor(true, [], [])).toBe(255);
  expect(rangeMaxFor(false, [0, 1000, 3000], [5, 5, 1])).toBe(4095); // 12-bit
  expect(rangeMaxFor(false, [0, 1000, 70000], [5, 5, 0])).toBe(1023);
  expect(rangeMaxFor(false, [], [])).toBe(65535);
  expect(rangeMaxFor(false, [0, 1000, 70000], [5, 5, 1], 12)).toBe(4095); // reported depth wins
  expect(rangeMaxFor(true, [], [], 12)).toBe(255); // JPEG stays 8-bit
});

test("auto range clips the darkest and brightest 0.1 %", () => {
  const x = [...Array(100).keys()];
  const y = x.map(() => 100);
  expect(autoRange(x, y)).toEqual([0, 99]);
  expect(autoRange([0, 10, 20], [0, 0, 0])).toBeNull();
});
