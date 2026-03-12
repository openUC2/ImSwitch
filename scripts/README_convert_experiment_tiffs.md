# convert_experiment_tiffs.py

Post-processing tool for ImSwitch ExperimentController data. Converts individual
TIFF tiles produced during scanning experiments into viewer-friendly formats for
**napari** and **Fiji/ImageJ**.

## Input format

The tool expects the directory layout produced by `OMEWriter._write_individual_tiff`:

```
<base_dir>/
├── tiles/
│   ├── timepoint_0000/
│   │   ├── t20260309_075340_x53362097_y24965231_z-933125_c0_405_i0_p1023.tif
│   │   ├── t20260309_075340_x53362097_y24965231_z-933125_c1_532_i0_p1023.tif
│   │   └── ...
│   ├── timepoint_0001/
│   │   └── ...
```

### Filename convention

```
t{YYYYMMDD_HHMMSS}_x{X}_y{Y}_z{Z}_c{cIdx}_{channelName}_i{iter}_p{power}.tif
```

| Field         | Description                                       |
|---------------|---------------------------------------------------|
| `timestamp`   | Acquisition timestamp (`YYYYMMDD_HHMMSS`)         |
| `x`, `y`, `z` | Stage position in µm × 1000 (integer)            |
| `c_idx`       | Channel index (0-based)                            |
| `channelName` | Illumination source name (e.g. `405`, `532`, `LED`) |
| `iter`        | Tile iterator (running index within the scan)      |
| `power`       | Illumination power value                           |

## Output modes

All outputs are written to `<tiles_dir>/../converted/` by default (override with `-o`).

### `composite` — Per-position hyperstack (napari)

For each unique **(x, y)** position, builds a **TZCYX** ImageJ-compatible
hyperstack containing all timepoints, Z-planes, and channels. Open directly in
napari as a multi-channel composite.

Output: `converted/composite/composite_x{X}_y{Y}.ome.tif`

### `stitch` — XY-stitched canvas (Fiji)

For each **channel × timepoint × Z-plane**, stitches all XY tiles into a single
large canvas image. Simple grid placement (no sub-pixel registration).

Output: `converted/stitched/stitched_{channel}_t{TTTT}_z{Z}.ome.tif`

### `mip` — Max intensity projection, stitched (Fiji)

For each **channel × timepoint**, computes a max intensity projection (MIP) over
Z at each XY position, then stitches the MIPs into a single canvas.

Output: `converted/mip_stitched/mip_stitched_{channel}_t{TTTT}.ome.tif`

### `focus` — Best-focus plane selection, stitched (Fiji / napari)

For each **channel × timepoint**, selects the sharpest Z-plane at each XY
position using a normalised Laplacian-variance focus measure, then stitches the
selected frames into a single canvas.  This is equivalent to a "post-processing
autofocus": unlike MIP (which takes the brightest projection), it selects the
plane that is actually in focus.

Focus metric: $\frac{\text{Var}(\nabla^2 I)}{\langle I \rangle^2}$  
(normalised Laplacian variance — higher = sharper).

Output: `converted/best_focus/bestfocus_stitched_{channel}_t{TTTT}.ome.tif`

### `mip-composite` — MIP composite hyperstack (napari)

Computes per-channel MIP-stitched canvases and merges them into a single
**TZCYX** multi-channel composite stack for napari.

Output: `converted/mip_composite/mip_composite.ome.tif`

### `tile-config` — Fiji TileConfiguration.txt

Writes a `TileConfiguration.txt` file per channel and timepoint, mapping each
individual TIFF to its approximate pixel position. Use with Fiji's
**Plugins → Stitching → Grid/Collection Stitching** for sub-pixel overlap
registration.

Output: `converted/tile_config/TileConfiguration_{channel}_t{TTTT}.txt`

## Usage

```bash
# Install dependencies
pip install tifffile numpy

# Run all modes (default)
python convert_experiment_tiffs.py /path/to/experiment/tiles

# Only composite stacks
python convert_experiment_tiffs.py /path/to/experiment/tiles --mode composite

# Only MIP stitched
python convert_experiment_tiffs.py /path/to/experiment/tiles --mode mip

# Multiple modes at once
python convert_experiment_tiffs.py /path/to/experiment/tiles --mode composite mip tile-config

# Custom output directory
python convert_experiment_tiffs.py /path/to/experiment/tiles -o /path/to/output
```

### Available modes

| Mode            | Best for  | Description                              |
|-----------------|-----------|------------------------------------------|
| `composite`     | napari    | Per-position TZCYX hyperstack            |
| `stitch`        | Fiji      | Per-channel/Z stitched canvas            |
| `mip`           | Fiji      | Z-MIP then stitch per channel            |
| `focus`         | Fiji      | Best-focus Z-plane (Laplacian var) stitch |
| `mip-composite` | napari    | All-channel MIP stitched composite       |
| `tile-config`   | Fiji      | TileConfiguration.txt for precise stitch |

## Example workflow

### Quick overview with MIP

```bash
python convert_experiment_tiffs.py ./tiles --mode mip
# Open mip_stitched_405_t0000.ome.tif in Fiji for a quick overview
```

### Multi-channel viewing in napari

```bash
python convert_experiment_tiffs.py ./tiles --mode composite
# In napari: File → Open → select composite_x*_y*.ome.tif
# Channels appear as separate layers, Z-slider for focal planes
```

### High-quality stitching in Fiji

```bash
python convert_experiment_tiffs.py ./tiles --mode tile-config
# In Fiji: Plugins → Stitching → Grid/Collection Stitching
#   Type: Positions from file
#   Layout file: TileConfiguration_405_t0000.txt
#   Enable "Compute overlap" for sub-pixel registration
```

## Protocol JSON

ImSwitch saves a protocol JSON file alongside the experiment data:
```
20260309_075340_experiment_t0000_protocol.json
```

This file records the **planned** experiment configuration:
- `snake_tiles`: All scan positions with stage coordinates
- `z_positions`: Z-stack planes
- `illumination_sources` / `illumination_intensities`: Channel configuration
- `workflow_steps`: Every step the controller intended to execute (move, illuminate, acquire, save)
- `autofocus`: Autofocus settings
- `exposures` / `gains`: Per-channel camera settings

**Important caveat**: The protocol is written *before* execution begins, so it
reflects the **intended** experiment plan, not what actually happened. If the
experiment was interrupted (e.g. user abort, hardware error), the protocol still
contains all planned steps. To verify what was actually acquired, check the
tiles directory — `convert_experiment_tiffs.py` works only with files that
actually exist on disk.

## Notes

- Coordinate values in filenames are in µm × 1000 (integer) to preserve
  sub-micron precision without floating-point issues.
- The "stitch" and "mip" modes use simple grid placement. For sub-pixel
  accurate stitching, use the `tile-config` mode with Fiji's Grid/Collection
  plugin.
- Large experiments may produce very large stitched images. Consider using
  `mip` mode first for a quick overview before running `stitch` on all Z-planes.
- ImageJ has a ~4 GB file size limit. Very large composite stacks may need to
  be opened in napari instead.
