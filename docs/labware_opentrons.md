# Opentrons-style Labware in ImSwitch

ImSwitch ships an Opentrons-compatible labware layer that replaces the older
hard-coded `wellplate_layouts` module. Plates are described by JSON files
that follow the [Opentrons labware schema v2](https://github.com/Opentrons/opentrons/blob/edge/shared-data/labware/schemas/2.json),
loaded once at controller startup and exposed to the frontend through a small
set of REST endpoints.

This document covers:

1. Where labware definitions live and how they are discovered
2. The on-disk JSON schema (mm) and the in-memory model (µm)
3. Adding a new plate — both built-in (Python generator) and via dropping in
   an Opentrons community JSON
4. The HTTP endpoints
5. The frontend `LabwareSelectionPanel`
6. The OME-NGFF `plate_metadata.json` sidecar emitted by experiments

---

## 1. Layout on disk

```
imswitch/imcontrol/model/labware/
├── __init__.py                # public API: LabwareManager, load_labware_*
├── models.py                  # Pydantic models (Opentrons mm + ImSwitch µm)
├── loader.py                  # JSON → LabwareDefinition (mm → µm)
├── manager.py                 # LabwareManager: discovery, caching, offsets
├── selector.py                # WellSelectionPattern → resolved wells/points
├── generators.py              # Built-in SBS plate generator (Python → JSON)
├── schema_v2.json             # Vendored Opentrons schema (relaxed `format`)
└── definitions/
    ├── openuc2/
    │   ├── corning_6_wellplate_16.8ml_flat/1.json
    │   ├── corning_12_wellplate_6.9ml_flat/1.json
    │   ├── corning_24_wellplate_3.4ml_flat/1.json
    │   ├── corning_48_wellplate_1.6ml_flat/1.json
    │   ├── corning_96_wellplate_360ul_flat/1.json
    │   ├── corning_384_wellplate_112ul_flat/1.json
    │   ├── greiner_96_wellplate_650ul_uclear/1.json
    │   ├── ibidi_8well_chambered_coverslip/1.json
    │   ├── slide_4x_histosample_heidstar/1.json
    │   ├── ropod_2slides_uc2/1.json
    │   └── dep_chip_8x6_uc2/1.json
    └── <your-namespace>/<your_load_name>/<version>.json
```

`LabwareManager` recursively scans `definitions/` on first use. The directory
layout follows Opentrons' convention: `<namespace>/<loadName>/<version>.json`.
The `loadName` field inside the JSON is what the rest of ImSwitch (and the
frontend) uses to refer to the plate.

## 2. Units

| Layer                    | Units | Purpose                                     |
|--------------------------|-------|---------------------------------------------|
| JSON on disk             | mm    | Opentrons-native, drag-drop community defs  |
| `OpentronsLabwareV2`     | mm    | Loader-internal Pydantic mirror             |
| `LabwareDefinition` (RAM)| µm    | What every other ImSwitch module sees       |

Conversion happens in exactly one place: `loader.py` (`MM_TO_UM = 1000.0`).

## 3. Adding a new plate

### 3a. Drop in an Opentrons community JSON

The fastest way: download a JSON from the
[Opentrons labware library](https://labware.opentrons.com/) (or write your own),
place it at:

```
imswitch/imcontrol/model/labware/definitions/<namespace>/<loadName>/<version>.json
```

then restart ImSwitch. The new plate appears in `getLabwareList` and in the
frontend's labware dropdown automatically.

The schema requirements are minimal:

- `schemaVersion: 2`
- `parameters.loadName` — must match the folder name
- `parameters.format` — any string (`"96Standard"`, `"384Standard"`,
  `"24Standard"`, `"irregular"`, …); ImSwitch does not gate behaviour on it
- `ordering` — list of columns of well IDs (must agree with `wells` keys)
- `wells` — dict keyed by well ID (`"A1"`, `"H12"`, …) with `x`, `y`, `z`,
  `depth`, `totalLiquidVolume`, `shape` (`"circular"` or `"rectangular"`),
  and `diameter` *or* `xDimension`/`yDimension`
- `dimensions` — `xDimension`/`yDimension`/`zDimension` of the outer plate
- `metadata.displayName`, `metadata.displayCategory`,
  `metadata.displayVolumeUnits` (`"µL"` | `"mL"` | `"L"`)
- `brand.brand`
- `cornerOffsetFromSlot` — vector `{x, y, z}`

Any additional Opentrons fields are tolerated (`extra="allow"`).

A bad file is logged with a `LabwareValidationError` and skipped — it never
blocks controller startup.

### 3b. Add a built-in SBS plate via the generator

Built-in plates live in `generators.py` (`_BUILTIN_SPECS`). Append a new entry
and run:

```bash
python -m imswitch.imcontrol.model.labware.generators
```

This regenerates the JSONs under `definitions/openuc2/`. Example spec:

```python
dict(
    rows=4, cols=6,
    well_diameter_mm=16.26,
    well_depth_mm=17.4,
    total_volume_uL=3400.0,
    well_spacing_x_mm=19.3,
    well_spacing_y_mm=19.3,
    a1_x_offset_mm=17.48,
    a1_y_offset_mm=13.67,
    load_name="corning_24_wellplate_3.4ml_flat",
    display_name="Corning 24 Well Plate 3.4 mL Flat",
    brand="Corning",
    format_override="24Standard",
    tags=["SBS", "wellPlate", "_generated"],
),
```

The generator follows the SBS convention: A1 at the back-left of the plate.

## 4. HTTP endpoints

All endpoints hang off `ExperimentController`.

### New (preferred)

- `GET /ExperimentController/getLabwareList`
  → `[{load_name, display_name, format, rows, cols, well_count, tags, ...}]`
- `GET /ExperimentController/getLabwareDefinition?load_name=...&offset_x_um=...&offset_y_um=...`
  → full `LabwareDefinition` (µm) with optional rigid offset baked in
- `POST /ExperimentController/selectWellsByPattern`
  body: `{ loadName, pattern: WellSelectionPattern, offsetXUm, offsetYUm }`
  → resolved wells + their µm coordinates
- `POST /ExperimentController/applyWellSelectionToExperiment`
  body: `{ loadName, pattern, offsetXUm, offsetYUm, conditionLabels, pointNameTemplate }`
  → `{ points: [...] }` ready to push into the experiment's `pointList`

### `WellSelectionPattern` shape

```json
{
  "wells":   ["A1", "B5"],          // explicit IDs (optional)
  "ranges":  ["A1:C3", "H12"],      // Excel-style ranges (optional)
  "rows":    ["A", "C"],            // entire rows (optional)
  "columns": [1, 6, 12],            // entire columns (optional)
  "all":     false                  // shorthand for the whole plate
}
```

Pattern fields are unioned. `pointNameTemplate` understands `{well_id}`,
`{row}`, `{column}`, `{label}`.

### Compatibility wrappers (legacy clients)

- `GET /ExperimentController/getAvailableWellplateLayouts`
- `GET /ExperimentController/getWellplateLayout?layout_name=...`

Both delegate to the labware manager and return the legacy dict shape the
old frontend canvas understands (`{name, wells: [...], width, height,
unit: "um", ...}`). The `generateCustomWellplateLayout` endpoint and the
`wellplate_layouts` module have been removed — define plates as JSON instead.

## 5. Frontend: `LabwareSelectionPanel`

`frontend/src/components/LabwareSelectionPanel.jsx` mounts inside
`WellSelectorComponent` and exposes:

- **Labware picker** — populated from `getLabwareList`; switching it clears
  the current well selection and (after confirmation) the experiment's
  point list
- **Well chip grid** — click chips to toggle, *or* type a pattern
  (`A1:C3, B5`) and hit *Resolve* to expand it into chips
- **Condition labels** — tag selected wells with a free-text label that
  travels through to the OME-NGFF metadata
- **Sub-position generator** — emit one point per well at the well centre,
  *or* an `Nx × Ny` rectangular grid inside each well with user-set µm
  spacing; a small SVG preview shows the dot pattern in the well outline
- **Apply** — append or replace the experiment's `pointList`

The panel uses the new endpoints exclusively. Layout switching (the legacy
`Layout` dropdown in `WellSelectorComponent`) shows a confirmation dialog
when the existing `pointList` is non-empty so users do not lose their
selections by accident.

## 6. OME-NGFF plate metadata sidecar

When an experiment runs against a labware definition, ImSwitch writes a
`plate_metadata.json` sidecar next to the OME-Zarr store. The file follows
the [OME-NGFF 0.4 plate spec](https://ngff.openmicroscopy.org/0.4/#plate-md)
plus an `imswitch_well` block per well that carries:

- `labwareLoadName`, `wellRow`, `wellColumn`
- `conditionLabel` (if set)
- the resolved acquisition `points` for that well

See `imswitch/imcontrol/model/io/ome_writers/plate_metadata.py` for the
exact schema.
