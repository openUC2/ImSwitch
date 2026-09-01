# Extending ImSwitch with plugins

The entry point for anyone building on ImSwitch from outside openUC2.

A plugin adds a backend controller and a React widget to a running ImSwitch
instance. You write two files, run `make build`, and drop the result into a
directory the container already watches. No rebuild of ImSwitch, no `pip
install`, no fork.

| | |
|---|---|
| **Start here** | [When to write a plugin](#1-when-to-write-a-plugin) |
| **Build one** | [imswitch-plugin-template](https://github.com/openUC2/imswitch-plugin-template) → [WRITING_A_PLUGIN.md](https://github.com/openUC2/imswitch-plugin-template/blob/main/docs/WRITING_A_PLUGIN.md) |
| **Deploy one** | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **Develop without Docker** | [DEPLOYMENT.md §8](DEPLOYMENT.md#8-running-without-docker-native-development) — Windows / macOS / Linux setup |
| **Build without `make`** | [DEPLOYMENT.md §9](DEPLOYMENT.md#9-building-a-plugin-without-make) |
| **Why it is built this way** | [DECISIONS.md](DECISIONS.md) |
| **Reference plugin** | [imswitch-plugin-goniometer](https://github.com/openUC2/imswitch-plugin-goniometer) |

---

## 1. When to write a plugin

Most integrations should **not** be plugins. ImSwitch already exposes every
controller method decorated with `@APIExport` over HTTP, and a REST client is
safer, simpler and independently deployable.

| Your situation | Use |
|---|---|
| Need in-process access to camera frames without a round trip | **Plugin** |
| Need a control loop tighter than ~100 ms against hardware | **Plugin** |
| Need a UI that lives inside the ImSwitch window, sharing its theme and state | **Plugin** |
| Need to react to hardware events with sub-frame latency | **Plugin** |
| Driving acquisitions, moving stages, reading results | **REST API** |
| Batch processing, analysis, reporting on saved data | **REST API** |
| Integrating a LIMS, a scheduler, or another instrument | **REST API** |
| Anything you want to run on a different machine | **REST API** |
| Contributing a new *device type* (a new camera or stage driver) | **Neither — upstream it.** See [ADR-002](DECISIONS.md#adr-002--plugins-are-controller-only) |

### The honest downside

**A plugin runs inside the microscope process.** Your Python executes in the same
interpreter as the acquisition loop, and your JavaScript in the operator's
browser. There is no sandbox and no signing.

What that means in practice:

- A plugin that blocks for 30 seconds occupies a FastAPI worker thread the whole
  time. A plugin that deadlocks can wedge the instrument.
- A plugin that imports a second copy of NumPy does not crash — it silently
  disagrees with the host, and you get wrong numbers off a microscope. This is
  why the dependency contract below is not negotiable.
- The frontend half is isolated by an error boundary, so a crashing widget shows
  an error card rather than blanking the UI. **The backend half has no such
  protection.**

If a REST client would do the job, write a REST client.

### Using the REST API instead

The full surface is self-documenting on any running instance:

```
http://<host>:8001/imswitch/api/docs        Swagger UI
http://<host>:8001/imswitch/openapi.json    machine-readable
```

Endpoints follow `/imswitch/api/<Controller>/<method>`. There is also a Python
client, `imswitchclient`, on PyPI.

---

## 2. Quickstart

```bash
git clone https://github.com/openUC2/imswitch-plugin-template my-plugin
cd my-plugin
make install-ui
make build check
```

Three files to edit:

| File | What goes in it |
|---|---|
| `imswitch_plugin_example/plugin.toml` | name, menu entry, which hardware you need |
| `imswitch_plugin_example/controller.py` | your endpoints and events |
| `ui-src/src/Widget.jsx` | your UI |

Then ship it:

```bash
make dist
rsync -a dist/example pi@microscope:/home/pi/ImSwitchPlugins/
```

Add the plugin's `name` to `availableWidgets` in the instrument's setup file,
`docker compose restart server`, and check:

```bash
curl http://microscope:8001/imswitch/api/plugins
```

Full walkthrough, including the rename checklist:
[WRITING_A_PLUGIN.md](https://github.com/openUC2/imswitch-plugin-template/blob/main/docs/WRITING_A_PLUGIN.md).

---

## 3. The three contracts

Everything the template's `make check` enforces reduces to these. Each has a
stability marker; the full per-surface table is in
[DECISIONS.md §2](DECISIONS.md#2-stable-surface).

### 3.1 Python: `imswitch.plugin_sdk` only — **frozen**

```python
from imswitch.plugin_sdk import (
    PluginController, PluginContext, PluginRegistration,
    APIExport, Event, load_manifest,
)
```

That module is the entire contract. It will not change incompatibly within a
major version.

Everything else — `imswitch.imcontrol`, `imswitch.imcommon`, `MasterController`,
`SetupInfo`, `MultiManager`, the setup-file schema — is **private**. Importing
any of it voids every guarantee here, and those modules do move between minor
releases.

> The old `imswitch.implugins` entry-point mechanism has been **removed**. If you
> have a plugin built on it, it no longer loads; see
> [ADR-001](DECISIONS.md#adr-001--v2-pluginmanager-is-the-only-plugin-mechanism)
> for the migration.

### 3.2 Dependencies: the host provides them — **frozen**

`[project].dependencies` in your `pyproject.toml` **must stay empty**.

A plugin is bind-mounted and imported from `sys.path`; there is no install step,
so nothing you declare gets installed. And if someone does `pip install` your
plugin, each declared dependency risks resolving to a second copy of a library
the host already has loaded in-process.

Provided by the host, import freely:

> numpy · scipy · pydantic · fastapi · starlette · uvicorn · opencv (`cv2`) ·
> tifffile · zarr · h5py · Pillow · requests · python-socketio · imswitch

Need something else? Vendor it (small and pure-Python), or open an issue to add
it to the host image. Adding it to `dependencies` fails CI.

**Adding a package to that host-provided list is a breaking change to the
deployment model**, so it goes through a release note, not a patch.

### 3.3 Frontend: shared singletons — **provisional**

React, MUI, Redux, Emotion, socket.io-client and notistack come from the host at
runtime through the Module Federation share scope. In your `package.json` they
are **`peerDependencies`**, never `dependencies`.

Two settings make this structural rather than a rule you have to remember:

- **`eager: false`** — the host is the eager provider. `eager: true` in a remote
  pulls a second React into your bundle.
- **`import: false`** (the template's `fallback: false`) — otherwise webpack also
  emits a *local fallback copy* of each shared package, which is the same bug
  deferred to runtime. With it off, a missing host module fails loudly at load.

The canonical list is [`frontend/shared-deps.js`](../../frontend/shared-deps.js).
Plugins vendor it **byte-identically** and CI diffs the two. Adding a package is
safe; removing one is announced a release ahead.

What you get for it: `useTheme()`, `useSelector()` and `useDispatch()` work in
your widget with no props and no bridge object, because it renders inside the
host's own `<Provider>` and `<ThemeProvider>`.

---

## 4. Lifecycle

What happens between `docker compose up` and your widget appearing:

```
container start
  │
  ├─ entrypoint.sh: PLUGIN_PATH → export IMSWITCH_PLUGIN_DIR
  │                 log the directory + a listing of its children
  │
  ├─ ImSwitchServer.run()
  │    ├─ createAPI()            → registers /api/plugins on the router
  │    ├─ include_router()       → routes become live
  │    └─ PluginManager.discover()
  │         │
  │         for each subdirectory of the plugin directory:
  │         ├─ import the package, call register(ctx)
  │         ├─ parse plugin.toml           → PluginManifest   ─┐ malformed
  │         ├─ availableWidgets gate       → status "disabled" ─┤ → reported in
  │         ├─ resolve required_hardware   → unmet = error     ─┤   the response,
  │         ├─ construct the controller                        ─┘   never raised
  │         └─ build the APIRouter, bind Event → socket namespace
  │
  ├─ PluginManager.attach_to_app(app)
  │    ├─ mount router      at /imswitch/plugin/<name>/api
  │    └─ mount ui/dist     at /imswitch/plugin/<name>/ui   (StaticFiles)
  │
  └─ set_plugin_manager()   → /imswitch/api/plugins starts answering

browser
  ├─ GET /imswitch/api/plugins            → manifests + errors
  ├─ merge into the app registry          → sidebar entry appears
  ├─ user clicks it
  ├─ <script src=…/ui/remoteEntry.js>     → federation container registers
  ├─ container.init(shareScope)           → host React/MUI/Redux handed over
  └─ widget renders inside <Provider> + <ThemeProvider>
```

Two properties worth noting, because they are what make the model work:

- **The gate happens before the controller is constructed.** A plugin that is not
  in `availableWidgets` never runs its own `__init__` and never claims hardware.
- **The bundle is served from the mounted directory, same-origin.** Nothing is
  copied into the image and no CORS is involved.

---

## 5. What we guarantee

| Surface | Marker | What that means |
|---|---|---|
| `imswitch.plugin_sdk` | **Frozen** | No incompatible change within a major version |
| `plugin.toml` core fields | **Frozen** | New optional fields may appear; existing ones keep their meaning |
| `/imswitch/plugin/<n>/api`, `/imswitch/plugin/<n>/ui` | **Frozen** | Mount layout will not move |
| `GET /imswitch/api/plugins` response shape | **Frozen** | Keys may be added, never removed or retyped |
| Host-provided dependency list | **Frozen** | Additions are a release-note event |
| Plugin directory layout, `main(plugin_dir=...)`, `$IMSWITCH_PLUGIN_DIR` | **Frozen** | Directory scanning is the *only* discovery mechanism — there is no `pip install` path and no entry-point group. |
| Socket.IO namespace `/plugin/<n>` | **Provisional** | Layout settled; host-side plumbing still being tightened |
| `frontend/shared-deps.js` | **Provisional** | Removals announced one minor release ahead |
| `host_app` exposes (`./store`, `./contexts`) | **Provisional** | The set will grow before it freezes |
| `store.injectReducer` | **Provisional** | `persist: true` is accepted but **not implemented** |
| `plugin.toml` `sdk_min` / `imswitch_min` / `permissions` | **Provisional** | Parsed and validated but **not yet enforced** — do not rely on the host rejecting a mismatch |
| First-available hardware fallback | **Provisional** | Convenient in dev, surprising on multi-device rigs; likely to be gated behind a flag |
| `hostIP` / `hostPort` widget props | **Deprecated** | Use the `apiBase` prop and `useSelector(s => s.connectionSettingsState)` |
| Everything else in `imswitch.*` | **Private** | No guarantee at all |

### What we explicitly do not guarantee

- **Plugin state does not persist across page reloads.** Redux slices you inject
  are not in the persist whitelist, and cannot be — the whitelist is fixed before
  the store is created. Keep durable state on your backend.
- **Duplicate plugin names are not resolved.** First loaded wins; the second is
  reported as an error and skipped ([ADR-003](DECISIONS.md#adr-003--duplicate-plugin-names-are-not-resolved)).
- **No isolation.** See [the honest downside](#the-honest-downside).
- **No plugin installation from the UI.** The plugin directory is a trust
  boundary; mount it read-only. Any future install-from-UI feature would need
  signing first.

---

## 6. Troubleshooting

Every diagnosis starts in the same place:

```bash
curl -s http://<host>:8001/imswitch/api/plugins | python3 -m json.tool
```

| Symptom | Command / field that tells you | Cause |
|---|---|---|
| Absent from `/api/plugins` | `docker compose exec server ls -la /opt/imswitch/plugins` | Mount missing, or the directory holds no importable Python package |
| Present, `"status": "disabled"` | the entry's `reason` field | Its **manifest** name is not in `availableWidgets` |
| In the `errors` array | the `error` string | Malformed `plugin.toml`, unmet non-optional hardware, or a duplicate name |
| Loaded, `remote_entry: null` | shown in App Manager as *no widget* | The frontend was never built into `<package>/ui/dist` |
| Loaded but not in the sidebar | App Manager | Toggled off there |
| Sidebar entry shows an error card | the card names the URL and reason | See the four rows below |
| …"script could not be fetched (404)" | `curl http://host:8001<remote_entry>` | `dist_dir` disagrees with where the bundle actually is |
| …"did not register federation scope" | compare `plugin.toml` ↔ `webpack.config.js` | `[plugin.ui].scope` ≠ `ModuleFederationPlugin.name` |
| …"does not expose ./Widget" | `exposes` in `webpack.config.js` | `[plugin.ui].exposed` ≠ the key in `exposes` |
| …"timed out after 10s" | browser network tab | Bundle served but never finishes loading |
| Invalid hook call on mount | `make check` in your plugin | You bundled React — check `dependencies` and `eager`/`fallback` |
| `useSelector` throws "could not find store" | as above | `react-redux` is not being shared |
| Theme wrong (light inside a dark app) | `useTheme().palette.mode` | `@mui/material` is not being shared |
| Widget loads, endpoints 404 | `curl <api_base>/status` | You built the URL by hand and dropped the `/imswitch` prefix — use the `apiBase` prop |
| Wrong camera on a multi-camera rig | `plugin_bindings` in the setup file | The first-available fallback picked for you; pin it explicitly |

Operator-side detail, including the exact log lines to grep:
[DEPLOYMENT.md §6](DEPLOYMENT.md#6-troubleshooting).

---

## 7. Getting help

- Plugin API questions and bug reports:
  [ImSwitch issues](https://github.com/openuc2/ImSwitch/issues)
- Template problems:
  [template issues](https://github.com/openUC2/imswitch-plugin-template/issues)

When reporting, include the output of `curl .../imswitch/api/plugins` and the
plugin lines from `docker compose logs server`. Between them they explain most
failures without a round trip.
