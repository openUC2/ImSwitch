// The browser-side wave optics behind the hologram simulator, snapshot
// reconstruction and fringe counter. Each test fails if the physics breaks:
// a wrong FFT sign, a propagation that does not refocus, a sideband in the
// wrong place, or a fringe counter that counts noise.
import {
  fft2,
  propagate,
  makeSample,
  sampleField,
  simulateInline,
  reconstructInline,
  inlineGeometry,
  simulateOffAxis,
  demodulateOffAxis,
  findSideband,
  countFringes,
  intensity,
} from "../components/holo/holoMath";

const N = 64;
const LAMBDA = 520e-9;
const PIX = 2e-6;

const rms = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / a.length);
};

describe("fft2", () => {
  it("round-trips and puts a constant into the DC bin", () => {
    const re = new Float64Array(N * N).fill(2);
    const im = new Float64Array(N * N);
    fft2(re, im, N, N);
    expect(re[0]).toBeCloseTo(2 * N * N);
    expect(Math.abs(re[5])).toBeLessThan(1e-9);
    fft2(re, im, N, N, true);
    expect(re[123]).toBeCloseTo(2);
  });
});

describe("propagate", () => {
  it("is undone by propagating back", () => {
    const s = makeSample("dust", N, PIX, { seed: 3 });
    const { re, im } = sampleField(s);
    propagate(re, im, N, N, 500e-6, LAMBDA, PIX);
    expect(rms(intensity(re, im), s.amp.map((a) => a * a))).toBeGreaterThan(0.01);
    propagate(re, im, N, N, -500e-6, LAMBDA, PIX);
    expect(rms(intensity(re, im), s.amp.map((a) => a * a))).toBeLessThan(1e-6);
  });
});

describe("in-line model", () => {
  it("refocuses best at the effective distance", () => {
    const z1 = 20e-3;
    const z2 = 1e-3;
    const { zEff, dxObj } = inlineGeometry({ z1, z2, pixel: PIX });
    const sample = makeSample("dust", N, dxObj, { seed: 5, absorption: 1 });
    const { hologram } = simulateInline({ sample, n: N, lambda: LAMBDA, z1, z2, pixel: PIX });
    const target = sample.amp;
    const err = (dz) => {
      const { re, im } = reconstructInline(hologram, N, N, dz, LAMBDA, dxObj);
      return rms(intensity(re, im).map(Math.sqrt), target);
    };
    expect(err(zEff)).toBeLessThan(err(0));
    expect(err(zEff)).toBeLessThan(err(2 * zEff));
  });
});

describe("off-axis model", () => {
  const sample = makeSample("cells", N, PIX, { seed: 2, phaseShift: 1 });
  const theta = Math.asin(LAMBDA / (4 * PIX)); // fringe period 4 px
  const sim = simulateOffAxis({
    sample, n: N, lambda: LAMBDA, pixel: PIX, defocus: 500e-6, thetaX: theta, thetaY: 0,
  });

  it("puts fringes of the requested period and the carrier where expected", () => {
    expect(sim.period).toBeCloseTo(4);
    expect(sim.carrier).toEqual({ cx: N / 2 - N / 4, cy: N / 2 });
  });

  it("finds the mirror sideband in the right half-plane", () => {
    const sb = findSideband(sim.hologram, N, N);
    expect(sb.cx).toBe(N / 2 + N / 4);
    expect(sb.cy).toBe(N / 2);
  });

  it("recovers the sample phase after refocusing", () => {
    const { re, im } = demodulateOffAxis(sim.hologram, N, N, { ...sim.carrier, r: 10 }, 500e-6, LAMBDA, PIX);
    // phase relative to the free-space background, rms over the whole field
    let br = 0;
    let bi = 0;
    for (let i = 0; i < N * N; i++) if (sample.phase[i] === 0) { br += re[i]; bi += im[i]; }
    const bg = Math.atan2(bi, br);
    let sq = 0;
    for (let i = 0; i < N * N; i++) {
      const e = Math.atan2(im[i], re[i]) - bg - sample.phase[i];
      sq += Math.atan2(Math.sin(e), Math.cos(e)) ** 2;
    }
    expect(Math.sqrt(sq / (N * N))).toBeLessThan(0.18); // unfocused: 0.27, twin: 0.9
  });
});

describe("countFringes", () => {
  it("counts full cycles and ignores noise", () => {
    const v = [];
    let seed = 1;
    const noise = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
    for (let i = 0; i < 2000; i++) v.push(100 + 40 * Math.cos((2 * Math.PI * 7.25 * i) / 2000) + 8 * noise());
    expect(countFringes(v).fringes).toBe(7);
    expect(countFringes(v.map(() => 100 + 8 * noise())).fringes).toBeLessThanOrEqual(1);
  });
});
