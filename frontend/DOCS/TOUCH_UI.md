# Touch UI, compact layouts and the on-screen number pad

The full SPA (not only the `#/mobile` kiosk shell) adapts to touchscreens —
phones, tablets and the 800×480 Raspberry Pi display running chromium in kiosk
mode, which has no OS keyboard.

## Detection — `src/hooks/useDeviceProfile.js`

One hook decides what kind of device this is. Do not add new
`useMediaQuery("(max-width: …)")` checks for touch behaviour; use the hook.

| Flag | Meaning |
|---|---|
| `touchUI` | Primary input is a finger (or forced). The app theme gets ~40 px hit areas (`themes/touchTheme.js`). |
| `compactLayout` | Phone, narrow window, short screen (<560 px), or a mid-size touch screen. Stream-first layouts; the nav drawer becomes an overlay opened from a dense 48 px top bar. |
| `phoneLayout` | Width < 600 px. |
| `keypad` | Show the on-screen number pad (touch UI on a device without its own soft keyboard, or forced). |
| `isPortrait` | Dock controls below instead of beside the stream. |

Both behaviours can be forced in **Settings → Touch UI / Number pad**
(`Auto` / `On` / `Off`, persisted in `UIPreferencesSlice`). A kiosk with a mouse
plugged in reports a fine pointer — set *Touch UI: On* there.

## Number pad — `src/components/touch/TouchKeypadHost.jsx`

Mounted once in `App.jsx`. While enabled it attaches to every focused
`<input type="number">` and every input with `inputmode="numeric|decimal"`
(e.g. `FreeNumberField`), so existing fields need no changes.

- The value is buffered and written on **OK** like a physical keyboard would:
  value + `input`/`change` (React `onChange`), Enter, then blur — so fields
  that commit on change, on Enter or on blur all work, and hardware-backed
  fields get one write instead of `2`, `20`, `200`, ….
- `min` / `max` / `step` attributes are honoured (range check, ± step keys).
- Optional per-field hints via `inputProps`:

```jsx
<TextField
  type="number"
  inputProps={{ min: 1, step: 1000, "data-keypad-presets": "5000,10000,20000" }}
/>
```

- Opt a field or a whole subtree out with `data-no-keypad`.
- A USB keyboard keeps working while the pad is open (digits, `.`, `-`,
  Backspace, Enter/Tab = OK, Escape = cancel, ↑/↓ = step).

## Compact layouts

- **Live View** (`components/LiveView.js`): stream + Live/Snap on one side,
  `TouchControlDock` with Stage / Light / Camera / Focus / Lens / More on the
  other (below in portrait). The Stage tab is `TouchStagePanel`: XY D-pad with
  STOP, Z column, step sizes. XY hold = constant-velocity move until release;
  Z hold only repeats single steps (`hooks/useStageJog.js`).
- **WellPlate** (`axon/wellplate2/WellPlateWorkspace.jsx`): a Viewport /
  Experiment switch replaces the side-by-side split; both panes stay mounted.
- **Plate map** (`axon/WellSelectorCanvas.js`): tap = click, long press =
  context menu (remove point, "We are here"), two fingers = pinch-zoom and pan,
  plus on-map zoom buttons.

## Testing without a touchscreen

Use the browser's device emulation (or Settings → Touch UI: On) against the
virtual microscope (`example_virtual_microscope.json`). Tests:
`src/__tests__/deviceProfile.test.js`, `src/components/touch/TouchKeypadHost.test.js`.
