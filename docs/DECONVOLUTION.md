# 3D deconvolution for FRAME: DeconvOptim.jl, where to run it, what to build around it

Status 2026-11-22. An exploration, not a feature: nothing here is wired into ImSwitch yet.
It covers what the `new_methods` branch of [DeconvOptim.jl](https://github.com/beniroquai/DeconvOptim.jl/tree/new_methods)
can do, how fast it is, whether to keep it in Julia or port it, where it should run (Raspberry Pi,
worker, Arkitekt), what the button in ImSwitch / newswitch should look like, and which other
computational features are within reach. All timings are from one Apple M1 Pro laptop on synthetic
stacks. **Nothing was run on a Raspberry Pi or on real FRAME data.**

## 1. In short

- **The package is a full deconvolution toolbox**, and the branch adds a lot: three conjugate-gradient
  MAP solvers, ADMM, Chambolle–Pock, TGV and L1 regularisers, a per-pixel noise model and padding per axis.
- **It works and it is fast enough.** A 32 × 512 × 512 stack takes 14 s with the new `map_cj_lean`
  solver (4 CPU threads); a 64 × 1024 × 1024 stack takes 3 min and 4 GB.
- **Port one solver to Python, keep Julia as the reference.** A numpy port of `map_cj_lean`
  (`scripts/deconvolution/deconv_numpy.py`, 200 lines with the PSF models, no new dependency) gives the same result
  (correlation 0.99–0.9995 with the Julia output) and is 1.3–2.1 times slower. Rust would not be
  faster: the time goes into FFTs, which are native code already.
- **Raspberry Pi: light version only.** Preview on a crop and small stacks (up to about
  32 × 512 × 512) fit. Larger stacks need more memory and time than a Pi that is also running the
  microscope has. They belong on a worker: a Docker container on a laptop, GPU box or NAS.
- **Arkitekt + mikro is the second way to reach the same worker**, for labs that run an Arkitekt
  server. The worker should not depend on it.
- **The button**: a "Process" panel in the dataset viewer. It reads the optics from the setup, shows
  a before/after preview on a crop, then runs the whole stack as a job and stores the result next to
  the original.
- **Cheapest next wins**: flat-field correction, haze removal, full focus from a z-stack, a measured
  PSF from beads. Section 7 lists these and the advanced ones.

## 2. What DeconvOptim.jl does

DeconvOptim.jl (F. Wechsler, R. Heintzmann, [JuliaCon Proceedings 2023](https://doi.org/10.21105/jcon.00099),
MIT licence) treats deconvolution as an optimisation problem: find the non-negative object whose
blurred version explains the data best, under a noise model and a smoothness prior.

```mermaid
flowchart LR
    M["Measured stack"] --> N["Normalise and pad<br/>border left free"]
    P["PSF<br/>model or measured"] --> O["OTF = FFT of the PSF"]
    N --> F
    O --> F
    subgraph LOOP["Solver loop"]
        F["Forward model<br/>object * PSF + background"] --> L["Loss<br/>Poisson, Gauss, Anscombe"]
        L --> U["Update step"]
        R["Regulariser<br/>Good, TV, Hessian, TGV, L1"] --> U
        C["Positivity<br/>mapping, projection or penalty"] --> U
        U --> F
    end
    U --> X["Crop and rescale"] --> D["Deconvolved stack"]
```

### Building blocks

| Block | Choices (**bold** = new on `new_methods`) |
|---|---|
| Noise model | `Poisson`, `Gauss`, `ScaledGauss`, `Anscombe`, **`WeightedGauss`** (one weight per pixel, e.g. an sCMOS noise map) |
| Regulariser | `Tikhonov` (three modes), `GR` (Good's roughness), `TV`, `TH` (total Hessian), `HS` (Hessian–Schatten), **`L1`** (sparsity), **`TGV`** (no staircasing) |
| Positivity | mappings `Non_negative` (x²), `Map_0_1`, `Piecewise_positive`, **`Pow4_positive`**, **`Abs_positive`**, **`Continuously_positive`**; **`Positivity`** penalty; exact projection in ADMM / Chambolle–Pock |
| Borders | **padding per axis**, e.g. `(0.05, 0.05, 0.25)`: the border is not constrained by the data, which removes FFT wrap-around. A wide-field stack needs it along z |
| Forward model | convolution over chosen dimensions; `invert` accepts any differentiable forward model |
| Hardware | CPU (FFTW, threads); NVIDIA GPU through CUDA.jl; the new solvers use KernelAbstractions kernels on the array's backend |
| Not included | a 3D PSF model (`generate_psf` is a 2D approximation), file I/O, tiling for stacks larger than memory, deskew, blind (PSF-estimating) deconvolution |

### Solvers

The branch (20 commits, 13 October to 17 November 2026, +4,700 lines) adds five solvers that need
no automatic differentiation.

| Solver | Call | Idea | Per iteration | Notes |
|---|---|---|---|---|
| L-BFGS | `deconvolution(...)` | automatic differentiation (Zygote) + Optim.jl | several convolutions, history arrays | most flexible: every loss, regulariser and custom forward model; most memory |
| Richardson–Lucy | `richardson_lucy_iterative` | multiplicative update | 2 convolutions | simple, slow to converge |
| **MAP-CG** | `map_cj` | object = x², conjugate gradients, analytic Newton line search | 3 convolutions | Poisson or Anscombe; Good or Hessian penalty; saddle escape for voxels stuck at zero |
| **MAP-CG lean** | `map_cj_lean` | same, one Newton step + backtracking, stencils computed on the fly | 3 convolutions, **5 arrays** | adds Gauss loss, per-pixel noise weights, a strictly positive mapping. The one to port |
| **MAP-CG exact** | `map_cj_exact` | exact line search after Verveer and Jovin (MAPGR / MAPPR); closed form for Gauss + Good | 3 convolutions, 6 arrays | fewest iterations per result |
| **ADMM** | `opt=ADMM()` | splitting; every sub-step solved exactly, linear step is one division in Fourier space | about 2 convolutions + proximal steps | exact TV / TH / HS / L1, exact positivity, result depends on λ rather than on the iteration count |
| **Chambolle–Pock** | `opt=ChambollePock()` | primal–dual, no linear solve | same | everything ADMM has, plus `TGV` |

### Wide field and light sheet

The solver is the same; only the PSF changes.

| Modality | PSF | What to watch |
|---|---|---|
| Wide field | detection PSF from NA, wavelength, immersion, pixel size and z step | out-of-focus light leaves the stack: pad along z. The missing cone limits how much axial resolution comes back |
| Light sheet | detection PSF × sheet profile (thickness along z) | much better conditioned, converges in few iterations. The sheet thickens away from its waist; a single PSF ignores that |
| Oblique plane (OPM) | as light sheet, in the deskewed frame | deskew first (`OPMController` has it), or shear the PSF and deconvolve the raw stack |

## 3. What it does on a laptop

Test object: beads, filaments, a hollow shell and line pairs; simulated as a 20x / 0.75 wide-field
stack (0.12 µm pixels, 0.5 µm planes, peak 300 photons, background 5), Poisson noise.
"Correlation" is the normalised cross-correlation with the object (measured stack: 0.22).

![Object, measured stack and four deconvolutions, xy and xz](images/deconvolution_result.png)

The same numpy solver on a light-sheet stack (10x / 0.3 detection, 4 µm sheet): correlation 0.40
before, 0.97 after, 4 s.

![Light-sheet example](images/deconvolution_lightsheet.png)

### Run time and memory

Warm run, 4 threads, float32, padding 25 % along z. Iterations: Richardson–Lucy and MAP-CG 30,
ADMM 50, L-BFGS 20. Each cell: seconds · peak memory of the process.

| Stack (z × y × x) | Mvoxel | Richardson–Lucy, numpy | Richardson–Lucy, Julia ¹ | `map_lean`, numpy port | `map_cj_lean`, Julia | ADMM + TV, Julia | L-BFGS + TV, Julia |
|---|---|---|---|---|---|---|---|
| 32 × 256 × 256 | 2.1 | 1.3 s · 0.2 GB | 0.6 s | 4.9 s · 0.5 GB | 3.9 s · 1.0 GB | 3.5 s · 1.5 GB | 6.8 s · 1.6 GB |
| 32 × 512 × 512 | 8.4 | 3.2 s · 0.6 GB | 1.7 s | 23.4 s · 1.1 GB | 13.9 s · 1.3 GB | 9.2 s · 2.4 GB | 22.1 s · 3.2 GB |
| 64 × 512 × 512 | 16.8 | 6.2 s · 1.0 GB | 3.6 s | 54.9 s · 1.9 GB | 33.2 s · 1.8 GB | 21.8 s · 3.7 GB | 42.6 s · 4.9 GB |
| 32 × 1024 × 1024 | 33.6 | 17.1 s · 1.7 GB | 8.0 s | 149 s · 3.0 GB | 106 s · 3.0 GB | 91.3 s · 5.8 GB | not run |
| 64 × 1024 × 1024 | 67.1 | 42.7 s · 3.2 GB | 18.0 s | 366 s · 5.2 GB | 174 s · 4.0 GB | not run | not run |

¹ Julia's Richardson–Lucy has no padding and no background, so its grid is a third smaller. Julia
memory is the peak of one process that ran `map_cj_lean`, Richardson–Lucy, ADMM and L-BFGS in that
order, so Richardson–Lucy has no number of its own.

![Run time and memory against stack size](images/deconvolution_benchmark.png)

### Quality at equal effort (32 × 256 × 256)

| Solver | Iterations | Correlation | Time |
|---|---|---|---|
| Richardson–Lucy (numpy) | 30 / 100 / 300 | 0.54 / 0.81 / 0.94 | 1.3 s / ≈ 4 s / ≈ 13 s ² |
| `map_cj_lean` (Julia) | 30 | 0.93 | 3.9 s |
| `map_lean` (numpy port) | 30 / 100 | 0.90 / 0.97 | 4.9 s / ≈ 16 s ² |
| `map_cj_exact`, `map_cj` (Julia) | 30 | 0.95 | not timed warm |
| L-BFGS + TV, λ = 10⁻⁴ (Julia) | 20 / 60 | 0.87 / 0.98 | 6.7 s / 17 s |
| ADMM + TV (Julia) | 50 / 200 / 1000 | 0.27 / 0.34 / 0.49 | 3.5 s / ≈ 9 s / not timed |
| Chambolle–Pock + TV (Julia) | 300 | 0.28 | not timed |

² Scaled from the time for 30 iterations; the correlation was measured.

![Convergence of the two numpy solvers](images/deconvolution_convergence.png)

What the numbers say:

- **Conjugate gradients beat Richardson–Lucy by about ten times in iterations.** 30 MAP-CG
  iterations do what 300 Richardson–Lucy iterations do.
- **The numpy port matches Julia.** Its output correlates 0.990–0.9995 with Julia's `map_cj_lean`
  on all five stacks. It is 1.3–2.1 times slower and uses about the same memory. It is not optimised.
- **`map_cj_lean` needs the least memory** of the regularised solvers: about 40 % of L-BFGS and half of ADMM.
- **ADMM and Chambolle–Pock were slow on this object** (sparse, high dynamic range), with and
  without regulariser, background or padding. A smaller fixed penalty helped (ρ = 0.01: 0.46 after
  200 iterations). Their documentation reports fast convergence on its own test, so this is worth
  reporting upstream; the benchmark files reproduce it. Timings above are per iteration count, not
  per quality.
- **Regularisation weights are data dependent.** The test object wanted almost none (Good's
  roughness γ = 10⁻⁴, TV λ = 10⁻⁴). Real, dense samples will want more. The UI needs a preview.

### What Julia costs to deploy

| Step | Measured (M1 Pro) |
|---|---|
| Install and precompile the 185 dependencies | 140 s, 1.4 GB peak memory, 633 MB on disk |
| Load the package in a fresh process | 2.2 s |
| First call of a solver (compilation) on top of the warm time | about 5 s (`map_cj_lean`), 7–22 s (ADMM), 1–20 s (L-BFGS) |

So Julia wants a long-lived warm process. A fresh process per job pays the compile time each
time. GitHub issue #125 reports the same need ("a button to warm up the Julia core"), and that an
earlier Julia GUI ran on a Raspberry Pi 4B; it also gives 2–3 min on a laptop CPU and 10–20 s on an
RTX 3060 for a 128 × 512 × 512 stack with the older solver.

## 4. Julia, Python or Rust

| Option | Work | Speed | Deployment | Verdict |
|---|---|---|---|---|
| **Julia as it is**, in a container, called from Python (`juliacall`) or over HTTP | none on the algorithms | fastest on CPU; NVIDIA GPU through CUDA.jl | 0.6 GB of packages plus the runtime, compile time on first use, a warm process to keep alive | on the worker: yes, as the full-feature backend. In the ImSwitch process on the Pi: no |
| **numpy / scipy port of one solver** (`map_cj_lean` + Richardson–Lucy) | prototype exists, 200 lines | 1.3–2.1 times slower than Julia | nothing new: numpy and scipy are ImSwitch dependencies | default on Pi and worker |
| Same code on CuPy or PyTorch | swap the array module | GPU on the worker | one optional dependency on the worker | later, when a GPU worker exists |
| **Rust** | weeks: FFT, stencils, line search, Python bindings | no gain: FFT-bound, and FFTW / pocketfft are native already | one static binary; could compile to WebAssembly / WebGPU | not now. Only if the browser runner of the facility roadmap needs it |

Keep the Julia package as the reference: the port is tested against its output on the benchmark
stacks. New solvers land in Julia first and get ported only when they prove useful.

## 5. Where it should run

A Raspberry Pi 5 has four Cortex-A76 cores, 8 GB of RAM shared with ImSwitch and the camera
buffers, and no GPU that runs FFTs. Expect it to be 5–10 times slower than the laptop above. That
factor is an estimate; work package DEC-0 measures it.

| Stack | numpy port on the laptop | Estimate for a Pi 5 | Verdict for the Pi |
|---|---|---|---|
| crop 32 × 256 × 256 | 5 s · 0.5 GB | 25–50 s | yes: this is the preview |
| 32 × 512 × 512 | 23 s · 1.1 GB | 2–4 min | yes, when no scan is running |
| 64 × 512 × 512 | 55 s · 1.9 GB | 5–9 min | borderline |
| 32 × 1024 × 1024 | 2.5 min · 3.0 GB | 12–25 min | no |
| 64 × 1024 × 1024 | 6 min · 5.2 GB | 30–60 min | no |
| full sensor, 32 × 2048 × 3072 | not run; about 15 GB by scaling | – | no; the worker needs tiles or a GPU |

```mermaid
flowchart LR
    subgraph FR["FRAME, Raspberry Pi 5"]
        UI["Dataset viewer<br/>Process panel"]
        API["Processing endpoints<br/>recipes, preview, jobs"]
        LIGHT["Light runner<br/>numpy: preview, small stacks"]
        ZARR[("OME-Zarr store")]
        UI --> API --> LIGHT
        LIGHT --> ZARR
    end
    subgraph WK["Worker: Docker on a laptop, GPU box or NAS"]
        CORE["Recipe core<br/>numpy or CuPy port"]
        JL["DeconvOptim.jl<br/>warm Julia process"]
        CORE --- JL
    end
    subgraph AK["Arkitekt server, optional"]
        RK["rekuest<br/>action: deconvolve"]
        MK[("mikro<br/>images and provenance")]
    end
    API -- "job over HTTP" --> CORE
    ZARR -- "chunks over HTTP" --> CORE
    CORE -- "result stack" --> ZARR
    API -. "assign action" .-> RK
    RK -.-> CORE
    CORE -. "image in, image out" .-> MK
```

```mermaid
flowchart TD
    Q1{"More than about<br/>10 million voxels?"} -- no --> PI["Run on the Pi<br/>when no scan is running"]
    Q1 -- yes --> Q2{"Worker registered?"}
    Q2 -- yes --> W["Send to the worker"]
    Q2 -- no --> Q3{"Arkitekt bound?"}
    Q3 -- yes --> AKT["Assign the Arkitekt action"]
    Q3 -- no --> T["Offer: crop, bin 2x, or<br/>queue on the Pi overnight"]
```

- **One recipe, several runners.** This is the "one recipe, three runners" plan of
  `FRAME_FACILITY_ROADMAP.md` part B with deconvolution as its first heavy recipe: a pure function
  from a stack and typed parameters to a stack, no ImSwitch imports. In newswitch the recipe becomes
  an action and the worker a sidecar or a remote Arkitekt app (`NEWSWITCH_PLUGIN_ARCHITECTURE.md`).
- **Plain HTTP worker first.** A stand-alone FRAME plus a laptop needs no server: the worker reads
  OME-Zarr chunks from the instrument's `/data` mount and writes the result back.
- **Arkitekt as a second door.** ImSwitch already uploads images to mikro and declares actions
  (`ArkitektController`). The same worker registers a `deconvolve` action; the result is a new mikro
  image linked to the original. An earlier Arkitekt integration called a remote deconvolution after
  a snap; the call is still in `RecordingController.snapImageToPath`, commented out.
- **Julia stays on the worker**, behind the same recipe signature, for the solvers that are not
  ported (TGV, Hessian–Schatten, weighted noise).

## 6. The button

```mermaid
sequenceDiagram
    actor U as User
    participant V as Dataset viewer
    participant A as ImSwitch API
    participant R as Runner, Pi or worker
    U->>V: open a stack, choose Process, Deconvolve
    V->>A: list recipes
    A-->>V: parameter schema, presets from the setup
    U->>V: draw a crop, press Preview
    V->>A: preview(crop, parameters)
    A->>R: run on the crop
    R-->>V: before and after
    U->>V: run on the whole stack
    V->>A: start job
    A->>R: stack reference and parameters
    loop until done
        R-->>V: progress and current result
    end
    R->>A: deconvolved stack and recipe.json
    A-->>V: result opens next to the original
```

| Piece | Proposal |
|---|---|
| Where | "Process" in the dataset viewer (`VizarrViewer`), and on a finished volume in the light-sheet app. Not in Live View: deconvolution works on stored stacks |
| Presets | PSF from the setup: objective NA and magnification, pixel size, z step, emission wavelength per channel, modality (wide field, light sheet + sheet thickness). All editable |
| Parameters shown | method (fast = Richardson–Lucy, standard = MAP-CG), iterations, regularisation (one slider), background (auto from the stack), padding (auto per modality). The rest behind "advanced" |
| Preview | a crop of about 32 × 256 × 256 on the Pi, before/after slider, xy and xz. This is where the user finds the regularisation weight |
| Job | queued, cancellable, refused on the Pi while a scan runs (`controller/busy.py`); progress with the current iterate |
| Result | `deconvolved.ome.zarr` next to the source through `model/aufgaben/storage.py`, with `recipe.json` (recipe name and version, parameters, PSF, runner). The same file reruns the job from the UI, an Aufgaben step or a notebook |
| API sketch | `listProcessingRecipes`, `previewProcessing` (POST), `startProcessingJob` (POST), `getProcessingStatus`, `cancelProcessingJob`. The panel is built from the recipe's JSON schema, so the next recipe needs no new UI |

## 7. Beyond deconvolution

What commercial systems offer (names as marketed) and the open route with UC2 hardware.

| Users get elsewhere | Sold as | Open route | Effort | Runs on |
|---|---|---|---|---|
| Flat-field / shading correction | every vendor | calibration flat field (board row QCAL-flatfield) or a BaSiC-type estimate from the scan | low | Pi |
| Haze removal for wide field | Leica THUNDER (Computational Clearing), Keyence Haze Reduction | subtract a background estimated at a feature scale; live and 2D. The rigorous version is 3D deconvolution | low | Pi, live |
| Full focus / extended depth of field | Keyence Full Focus, Nikon EDF, Zeiss | sharpest plane per pixel from a z-stack (`focus_value` exists) | low | Pi, while the stack is taken |
| 3D deconvolution | Zeiss ZEN, Nikon NIS-Elements, Evident TruSight, Andor ClearView-GPU, Huygens, Microvolution | this document | medium | worker; preview on the Pi |
| Measured PSF | Huygens PSF Distiller | average bead images from a bead slide (QCAL-psf) | medium | worker |
| Resolution and quality number | few | image decorrelation or FRC: a stop criterion for the iterations and a QC value per run | low | Pi or worker |
| Denoising | Nikon Denoise.ai, camera-side filters | Noise2Void / CARE models exported to ONNX; non-local means as the classical fallback | medium | worker or browser |
| AI deblurring and restoration | Nikon Clarify.ai and Enhance.ai, Zeiss, Leica Aivia, Evident TruAI | CARE with training pairs the FRAME can record itself (short and long exposure, 4x and 20x of the same field) | advanced | worker with GPU |
| Optical sectioning by structured light | Zeiss Apotome, Keyence Sectioning | DMD / SLM patterns with three-phase or HiLo reconstruction (`SLMController`, PR #179) | medium | Pi or worker |
| Fluctuation super-resolution | Andor SRRF-Stream+ | eSRRF or MSSR on a fast series (NanoPyx is Python) | medium | worker |
| Pixel reassignment, joint deconvolution | Zeiss Airyscan Joint Deconvolution | image-scanning microscopy with the galvo and a camera, multi-view deconvolution | advanced | worker |
| Light sheet: deskew, deconvolve, fuse | Zeiss Lattice Lightsheet 7, Leica | deskew (`OPMController`), deconvolve with the sheet PSF, fuse with multiview-stitcher | medium | worker |
| Label-free contrast | specialised vendors | DPC is in ImSwitch; Fourier ptychography with the LED matrix resolves beyond the objective NA and is the UC2-specific one | advanced | worker |
| Processing during acquisition | Zeiss Direct Processing | an Aufgaben step hands each finished stack to the runner | medium | worker |

Existing open implementations to compare against or reuse instead of writing: scikit-image
`richardson_lucy` (already installed), RedLionfish (Richardson–Lucy on OpenCL, block-wise),
pycudadecon (CUDA, with deskew), `psfmodels` (Gibson–Lanni and vectorial PSFs), PointSpreadFunctions.jl.

## 8. Work packages

Not on the roadmap board yet. Sizes: S up to 2 days, M up to 2 weeks.

| ID | What | Done when | Size |
|---|---|---|---|
| DEC-0 | Run `scripts/deconvolution/benchmark.py` on a Pi 5, and on one real FRAME stack with beads | the estimate column of section 5 is replaced by numbers | S |
| DEC-1 | Recipe: move the numpy solvers into a module without ImSwitch imports, PSF from optics metadata, test against stored Julia outputs | `deconvolve(stack, optics, parameters)` passes the comparison test | S |
| DEC-2 | Preview endpoint on the Pi and the crop + before/after view | a user finds the regularisation weight on a crop without leaving the viewer | M |
| DEC-3 | Worker container: recipe core, job API, reads and writes OME-Zarr over HTTP; optional Julia backend with a warm process | a 64 × 1024 × 1024 stack is deconvolved from the UI with the Pi idle | M |
| DEC-4 | Arkitekt app around the worker: `deconvolve` action on mikro images | the action shows in the Arkitekt UI and returns a linked image | S |
| DEC-5 | Process panel built from the recipe schema; job list; `recipe.json` | a second recipe (flat field) appears without UI work | M |
| DEC-6 | PSF: theoretical per objective and channel; measured from beads | deconvolution with the measured PSF beats the theoretical one on a bead slide | M |
| DEC-7 | Light sheet: sheet thickness in the PSF, deskew for OPM stacks | the virtual light-sheet volume deconvolves end to end | M |
| DEC-8 | GPU on the worker (CuPy / PyTorch backend, or CUDA.jl) | the full-sensor stack runs in minutes | S |

Order: DEC-0 → DEC-1 → DEC-2 → DEC-3 → DEC-5 → DEC-6, then DEC-4, DEC-7, DEC-8 as needed.

## 9. Risks and open points

- **The PSF decides the result.** FRAME's tube lens is unknown in detail, objectives corrected for
  0.17 mm cover glass are used through 1 mm plastic, and the field is not fully lit in the corners.
  A theoretical PSF will be optimistic; a measured one (DEC-6) is the fix.
- **Wide-field stacks have a missing cone.** Deconvolution removes haze and sharpens laterally; it
  does not turn a wide-field stack into a confocal one. Say so in the UI.
- **Quantitative use.** A regularised result is not linear in the object. Keep the raw data, store
  the parameters, and label deconvolved images as such.
- **Memory on the worker.** A full-sensor stack does not fit a laptop in one piece. Tiling with
  overlap is not in DeconvOptim.jl and is part of DEC-3.
- **Pi numbers are estimates** until DEC-0.
- **ADMM / Chambolle–Pock convergence** on sparse objects needs a look upstream before these
  solvers are offered in a UI.

## 10. How the numbers were made

| File | Purpose |
|---|---|
| `scripts/deconvolution/deconv_numpy.py` | PSF models, Richardson–Lucy and the `map_cj_lean` port; `python deconv_numpy.py` runs a self-check |
| `scripts/deconvolution/benchmark.py` | test stacks (`data`), numpy timings (`run`), table (`table`), figures (`figures`) |
| `scripts/deconvolution/benchmark.jl` | DeconvOptim.jl on the same stacks; every solver twice (first call, warm) |

Machine: Apple M1 Pro, 16 GB, macOS; Julia 1.13.3; DeconvOptim.jl `new_methods` at `c72a2c8`;
numpy 2.1, scipy 1.16; 4 threads for FFTs and Julia.
