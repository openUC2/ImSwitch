"""Benchmark behind docs/DECONVOLUTION.md: synthetic 3D stacks, the numpy solvers, tables, figures.

    python benchmark.py data    OUT NZ NY NX     # truth/measured/psf as raw float32 (z, y, x)
    python benchmark.py run     OUT NZ NY NX     # time the numpy solvers; prints one RESULT line each
    python benchmark.py table   OUT              # markdown table and the port-vs-Julia comparison
    python benchmark.py figures OUT FIGDIR       # the figures of the document

Redirect `run` into OUT/py_<shape>.log; benchmark.jl runs DeconvOptim.jl on the same files and logs
to OUT/jl_<shape>.log. Julia reads a (z, y, x) C-order file as an (x, y, z) array: same memory.
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

# The recipe lives in ImSwitch; put a copy next to this file to run it without ImSwitch (e.g. on a Pi).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "imswitch/imcontrol/model/aufgaben"))
import deconvolution as dn  # noqa: E402

dn.WORKERS = 4                              # as Julia: 4 threads, comparable timings
FRAME = dn.Optics(pixel_um=0.12, z_step_um=0.5, na=0.75)   # 20x/0.75, unbinned IMX178
PX_UM, DZ_UM = FRAME.pixel_um, FRAME.z_step_um
BACKGROUND = 5.0                            # counts per voxel; the object peaks at 300 after blurring
PADDING = dn.PADDING                        # wide-field light leaves the stack along z
GAMMA = 1e-4                                # Good's roughness weight (counts); the sparse test object wants little smoothing
ITER = 30
SOLVERS = {"py_rl": "Richardson-Lucy, numpy", "jl_rl": "Richardson-Lucy, Julia",
           "py_map_lean": "map_lean, numpy port", "jl_map_lean": "map_cj_lean, Julia",
           "jl_admm_tv": "ADMM + TV (50 it.), Julia", "jl_lbfgs_tv": "L-BFGS + TV (20 it.), Julia"}


def tag(shape):
    return "x".join(str(n) for n in shape)


def voxels(s):
    return int(np.prod([int(n) for n in s.split("x")]))


def maxrss_mb():
    """Peak memory of this process: ru_maxrss is bytes on macOS and KiB on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2 ** (20 if sys.platform == "darwin" else 10))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum(dtype=np.float64) / np.sqrt((a * a).sum(dtype=np.float64) * (b * b).sum(dtype=np.float64)))


def load(out, name, shape):
    return np.fromfile(out / f"{tag(shape)}_{name}.f32", np.float32).reshape(shape)


def results(out):
    """{(shape, solver): row} from all logs; a later log overrides an earlier one."""
    rows = [json.loads(line.split(" ", 1)[1]) for f in sorted(out.glob("*.log")) for line in f.read_text().splitlines()
            if line.startswith("RESULT {")]
    return {(r["shape"], r["solver"]): r for r in rows}


def cmd_data(out, shape):
    truth, measured = dn.synthetic(shape, FRAME, seed=1, background=BACKGROUND)
    for name, a in (("truth", truth), ("measured", measured), ("psf", dn.psf(FRAME, shape))):
        a.astype(np.float32).tofile(out / f"{tag(shape)}_{name}.f32")
    print("data", tag(shape), "peak counts", float(measured.max()))


def cmd_run(out, shape):
    truth, measured, psf = (load(out, n, shape) for n in ("truth", "measured", "psf"))
    solvers = {
        "py_rl": lambda cb=None: dn.richardson_lucy(measured, psf, iterations=ITER, background=BACKGROUND, padding=PADDING, callback=cb),
        "py_map_lean": lambda cb=None: dn.map_lean(measured, psf, gamma=GAMMA, iterations=ITER, background=BACKGROUND, padding=PADDING, callback=cb),
    }
    for name, solve in solvers.items():
        t0 = time.perf_counter()
        result = solve()
        seconds = time.perf_counter() - t0
        result.astype(np.float32).tofile(out / f"{tag(shape)}_{name}.f32")
        row = {"shape": tag(shape), "solver": name, "seconds": round(seconds, 2), "ncc": round(ncc(result, truth), 4),
               "ncc_measured": round(ncc(measured, truth), 4),
               "maxrss_mb": maxrss_mb()}
        if shape[1] <= 256:                                          # convergence curve on the small stack only
            curve = []
            solve(lambda it, obj: curve.append(round(ncc(obj, truth), 4)) and False)
            row["curve"] = curve
        print("RESULT", json.dumps(row), flush=True)


def cmd_table(out):
    get = results(out)
    shapes = sorted({s for s, _ in get}, key=voxels)
    print("| Stack (z x y x x) | Mvoxel | " + " | ".join(SOLVERS.values()) + " |")
    print("|---|---|" + "---|" * len(SOLVERS))
    for s in shapes:
        cells = [f"{get[(s, k)]['seconds']:.1f} s · {get[(s, k)]['maxrss_mb'] / 1024:.1f} GB" if (s, k) in get else "–"
                 for k in SOLVERS]
        print(f"| {s.replace('x', ' x ')} | {voxels(s) / 1e6:.1f} | " + " | ".join(cells) + " |")
    print("\ncorrelation with the object, and numpy port against Julia map_cj_lean:")
    for s in shapes:
        shape = tuple(int(n) for n in s.split("x"))
        truth = load(out, "truth", shape)
        q = {k: round(ncc(load(out, k, shape), truth), 3) for k in SOLVERS if (out / f"{s}_{k}.f32").exists()}
        line = f"{s}: measured {get[(s, 'py_rl')]['ncc_measured']} {q}"
        if (out / f"{s}_jl_map_lean.f32").exists():
            py, jl = load(out, "py_map_lean", shape), load(out, "jl_map_lean", shape)
            line += f" | port vs Julia: correlation {ncc(py, jl):.4f}"
        print(line)


def _panels(panels, shape, px_um, dz_um, path, row_y):
    """xy (central plane) and xz (through row_y) of each (title, volume)."""
    import matplotlib.pyplot as plt
    nz, ny, nx = shape
    fig, axes = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 2.6 * (1 + nz * dz_um / (ny * px_um)) + 0.7),
                             gridspec_kw={"height_ratios": [ny * px_um, nz * dz_um]})
    for (title, a), (top, bottom) in zip(panels, axes.T):
        vmax = np.percentile(a, 99.95)
        top.imshow(a[nz // 2], cmap="gray", vmin=0, vmax=vmax, extent=(0, nx * px_um, ny * px_um, 0))
        top.set_title(title, fontsize=8)
        bottom.imshow(a[:, row_y], cmap="gray", vmin=0, vmax=vmax, aspect="auto", extent=(0, nx * px_um, nz * dz_um, 0))
        for ax in (top, bottom):
            ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("xy, central plane (µm)", fontsize=7)
    axes[1, 0].set_ylabel("xz through the shell (µm)", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def cmd_figures(out, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. wide field: what each solver returns
    shape = (32, 256, 256)
    truth, measured = load(out, "truth", shape), load(out, "measured", shape)
    panels = [("Object (ground truth)", truth), ("Measured wide-field stack", measured - BACKGROUND)]
    for name, title in (("py_rl", "Richardson-Lucy, 30 it. (numpy)"), ("jl_map_lean", "map_cj_lean, 30 it. (Julia)"),
                        ("py_map_lean", "the same solver in numpy"), ("jl_lbfgs_tv60", "L-BFGS + TV, 60 it. (Julia)")):
        if (out / f"{tag(shape)}_{name}.f32").exists():
            panels.append((f"{title}\ncorrelation {ncc(load(out, name, shape), truth):.2f}", load(out, name, shape)))
    _panels(panels, shape, PX_UM, DZ_UM, figdir / "deconvolution_result.png", int(0.3 * shape[1]))

    # 2. light sheet: same solver, the PSF is the detection PSF times the sheet profile
    sheet = dn.Optics(pixel_um=0.33, z_step_um=1.0, na=0.3, modality="lightsheet", sheet_um=4.0)
    px, dz = sheet.pixel_um, sheet.z_step_um                         # 10x/0.3 detection, 4 um thick sheet
    truth, measured = dn.synthetic(shape, sheet, seed=2, background=BACKGROUND)
    psf = dn.psf(sheet, shape)
    t0 = time.perf_counter()
    deconvolved = dn.map_lean(measured, psf, gamma=GAMMA, iterations=ITER, background=BACKGROUND, padding=(0.1, 0, 0))
    print(f"light sheet: measured {ncc(measured, truth):.3f}, deconvolved {ncc(deconvolved, truth):.3f}, "
          f"{time.perf_counter() - t0:.1f} s")
    _panels([("Object (ground truth)", truth), (f"Measured light-sheet stack\ncorrelation {ncc(measured, truth):.2f}", measured - BACKGROUND),
             (f"map_lean, 30 it. (numpy)\ncorrelation {ncc(deconvolved, truth):.2f}", deconvolved)],
            shape, px, dz, figdir / "deconvolution_lightsheet.png", int(0.3 * shape[1]))

    # 3. run time and memory against stack size
    get = results(out)
    shapes = sorted({s for s, _ in get}, key=voxels)
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for key, label in SOLVERS.items():
        mine = [s for s in shapes if (s, key) in get]
        style = "o-" if key.startswith("jl") else "s--"
        ax_t.loglog([voxels(s) / 1e6 for s in mine], [get[(s, key)]["seconds"] for s in mine], style, label=label, ms=4)
        ax_m.loglog([voxels(s) / 1e6 for s in mine], [get[(s, key)]["maxrss_mb"] / 1024 for s in mine], style, label=label, ms=4)
    ax_t.set(xlabel="stack size (million voxels)", ylabel="seconds, warm (Apple M1 Pro, 4 threads)", title="Run time")
    ax_m.set(xlabel="stack size (million voxels)", ylabel="peak memory of the process (GB)", title="Memory")
    ax_m.axhline(8, color="gray", lw=0.8, ls=":")
    ax_m.text(2.2, 8.5, "all the RAM of a Raspberry Pi 5 (8 GB)", fontsize=7, color="gray")
    ax_t.legend(fontsize=7)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for ax, yticks in ((ax_t, [1, 3, 10, 30, 100, 300]), (ax_m, [0.25, 0.5, 1, 2, 4, 8])):
        ax.xaxis.set_major_locator(FixedLocator([2, 8, 17, 34, 67]))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(plain)
        ax.yaxis.set_major_formatter(plain)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "deconvolution_benchmark.png", dpi=170)
    plt.close(fig)

    # 4. convergence of the two numpy solvers
    curves = [r for r in get.values() if "curve" in r]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for r in curves:
        ax.plot(range(1, len(r["curve"]) + 1), r["curve"], label=SOLVERS[r["solver"]])
    ax.axhline(curves[0]["ncc_measured"], color="gray", lw=0.8, ls=":", label="measured stack")
    ax.set(xlabel="iteration", ylabel="correlation with the object", title="Convergence, 32 x 256 x 256")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "deconvolution_convergence.png", dpi=170)
    print("figures written to", figdir)


if __name__ == "__main__":
    command, out = sys.argv[1], Path(sys.argv[2])
    if command == "figures":
        cmd_figures(out, Path(sys.argv[3]))
    elif command == "table":
        cmd_table(out)
    else:
        {"data": cmd_data, "run": cmd_run}[command](out, tuple(int(n) for n in sys.argv[3:6]))
