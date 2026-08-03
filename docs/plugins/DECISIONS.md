# ImSwitch plugin architecture — decision record

Status: **accepted** · Applies to: `feature/pluginsystemV2` and later · Last reviewed: 2026-08-03

This document is the architecture decision record for the ImSwitch plugin system. It first
inventories what is actually in the tree today, then records the decisions that follow from
that inventory, and closes with a stability table an external plugin author can rely on.

`file:line` references are against the current tree. Code that has since been deleted is
quoted against the last commit that contained it: `bb4e04f1` for the two dead v1 paths removed
in WP0, `894fc018` for the three live v1 paths removed afterwards.

---

## 1. Inventory — the two plugin mechanisms

Two independent plugin mechanisms *used to* coexist. **v1 has since been removed in full**;
this section is kept as the record of what it was and why it went, because that is the
evidence ADR-001 rests on.

### 1.1 v1 — `imswitch.implugins` entry points (REMOVED)

An untyped, undeclared set of `importlib.metadata` entry-point hooks. There was no manifest:
the host inferred a plugin's purpose from the *suffix of the entry-point name*
(`_controller`, `_widget`, `_manager`, `_info`). Nothing validated the loaded object.

ImSwitch itself only ever declared the three *sub*-groups, and declared them empty
(`pyproject.toml:127`-131 and `setup.py:115`-117 @ `894fc018`):

```toml
[project.entry-points."imswitch.implugins.detectors"]
[project.entry-points."imswitch.implugins.lasers"]
[project.entry-points."imswitch.implugins.positioner"]
```

The bare `imswitch.implugins` group that four of the five consumers below queried was never
declared by the host at all — it existed only if an external package declared it.

| Consumer | Group queried | Entry-point name | Was it reachable? |
|---|---|---|---|
| `ImConMainView._addWidgetNoQt` (`ImConMainView.py:49` @ 894fc018) | `imswitch.implugins` | `<widgetKey>_widget` | Yes — fallback when no built-in React widget module existed |
| `MasterController.__init__` (`MasterController.py:203` @ 894fc018) | `imswitch.implugins` | `*_manager`, `*_info` | Yes — but constructed every manager with `moduleInfo = None`, carrying a TODO saying the setup info was never wired up |
| `MultiManager.__init__` (`MultiManager.py:38` @ 894fc018) | `imswitch.implugins.<subpkg>` | any | Yes — only in the `except` branch of a failed built-in manager import |
| `SetupInfo.add_attribute` (`SetupInfo.py:923` @ bb4e04f1) | `imswitch.implugins` | `<attr>_info` | No — dead. See 1.1.1 |
| `ImConMainController.loadPlugin` (`ImConMainController.py:279` @ bb4e04f1) | `imswitch.implugins` | `<widgetKey>_controller` | No — unreachable. See 1.1.2 |

#### 1.1.1 `SetupInfo.add_attribute` was dead

`SetupInfo.add_attribute()` was defined at `SetupInfo.py:915` (@ bb4e04f1). Its only mention
anywhere in the tree was a commented-out line in `MasterController.__init__`:

```python
# self.__setupInfo.add_attribute(attr_name=entry_point.name.split("_manager")[0], attr_value={})
```

The method also could not work as written: it called `setattr(self, ..., field(default_factory=...))`
on an *instance*, which stores a `dataclasses.Field` object rather than a value. Both the
method and the commented-out call were removed in the change that introduced this ADR.

#### 1.1.2 `ImConMainController.loadPlugin` was unreachable

The controller construction loop resolves a controller class by dynamic import
(`ImConMainController.py:98` @ bb4e04f1):

```python
try:
    module = importlib.import_module(f"...controllers.{controller_name}")
    controller_class = getattr(module, controller_name)
except Exception as e:
    self.__logger.warning(f"Could not dynamically import (1) {controller_name}: {e}")
    continue                      # ← line 108
if controller_class is not None:
    ...
else:                             # ← line 122: unreachable
    mPlugin = self.loadPlugin(widgetKey)
```

`controller_class` was initialised to `None` at line 94 and only ever reassigned inside the
`try`. Every path that leaves it `None` raises, and the `except` handler `continue`s past the
`else`. The `else` branch — the sole caller of `loadPlugin()` — could therefore never execute.
Both the branch and the method were removed in the change that introduced this ADR. Widget
controllers contributed by v1 plugins have consequently never loaded on this branch.

### 1.2 v2 — `PluginManager` + `imswitch.plugin_sdk`

A manifest-driven system. `plugin.toml` is parsed into a validated pydantic
[`PluginManifest`](../../imswitch/plugin_sdk/__init__.py#L98); the plugin exposes exactly one
`register(ctx) -> PluginRegistration` function; the host resolves hardware by *role* and
mounts backend routes and the built frontend bundle.

| Element | Location | Reachable? |
|---|---|---|
| Plugin directory — `main(plugin_dir=...)` → `$IMSWITCH_PLUGIN_DIR` → `DEFAULT_DROPIN` (`/opt/imswitch/plugins`) | [plugin_manager.py:54](../../imswitch/plugin_manager.py#L54) | **Yes** — the only discovery mechanism |
| `PluginManager.discover()` | [plugin_manager.py:134](../../imswitch/plugin_manager.py#L134) | **Yes** — [ImSwitchServer.py:703](../../imswitch/imcontrol/controller/server/ImSwitchServer.py#L703) |
| `PluginManager.attach_to_app()` — mounts `/plugin/<n>/api` and `/plugin/<n>/ui` | [plugin_manager.py:161](../../imswitch/plugin_manager.py#L161) | **Yes** — [ImSwitchServer.py:704](../../imswitch/imcontrol/controller/server/ImSwitchServer.py#L704) |
| `PluginManager.manifest_list()` | [plugin_manager.py:203](../../imswitch/plugin_manager.py#L203) | **Yes** as of WP1 — served by [`list_plugins`, ImSwitchServer.py:94](../../imswitch/imcontrol/controller/server/ImSwitchServer.py#L94) at `GET /imswitch/api/plugins`, which [App.jsx:526](../../frontend/src/App.jsx#L526) fetches. Before WP1 it had no caller and that route 404'd. |
| `PluginManager.shutdown()` | [plugin_manager.py:249](../../imswitch/plugin_manager.py#L249) | **Yes** — [ImSwitchServer.py:734](../../imswitch/imcontrol/controller/server/ImSwitchServer.py#L734) |

### 1.3 Summary of the inventory

v1 had five consumers. Two were dead on arrival. The other three were *fallback* paths that
fired only after a built-in lookup had already failed, and one of those three
(`MasterController`) could only ever construct a manager with `moduleInfo = None` — it carried
a TODO admitting the setup info was never wired up, so no v1 manager plugin could have been
configured from a setup file at all. v1 never had a manifest, a version contract, a dependency
contract or a frontend story. v2 has all four, and the whole chain from drop-in directory to
browser is now connected.

---

## ADR-001 — v2 `PluginManager` is the only plugin mechanism

**Decision.** Plugins target `imswitch.plugin_sdk` and the v2 `PluginManager`. The v1
`imswitch.implugins` entry-point hooks are **removed**, not deprecated.

**Consequences.**

- All five v1 consumers are gone from the tree, along with the empty `imswitch.implugins.*`
  entry-point group declarations in `pyproject.toml` and `setup.py`. Nothing in ImSwitch reads
  that group any more.
- Any external package that declared `imswitch.implugins` entry points stops being loaded. This
  is a **breaking change** and belongs in the release notes. The migration is to port to
  `imswitch.plugin_sdk`: a `plugin.toml`, one `register(ctx)`, and a `PluginController`.
- Documentation, the template repository and partner-facing material describe only v2.

**Rationale.** Keeping two mechanisms doubles the surface we document, test and support, and
the two differ on every axis that matters — validation, versioning, error reporting, frontend
delivery. v1 could not be grown into v2 because it had no manifest to hang any of those
properties on.

**Why removed outright rather than deprecated for a release.** The original plan was to keep v1
until the next major version. That was reconsidered once the inventory showed what "keeping it"
actually bought: two of the five paths were already dead, and of the three live ones, none could
carry a fully-configured plugin (see 1.3). Keeping a mechanism that cannot work costs
documentation, test surface and reader confusion while protecting nothing. The removal is
recorded here rather than silently done, so that a partner who reports "my plugin stopped
loading" gets a straight answer and a migration path.

---

## ADR-002 — Plugins are controller-only

**Decision.** A plugin contributes a **controller** and a **frontend widget**. It declares
which *existing* hardware it needs by role in `plugin.toml` and receives a live handle through
`ctx.hardware`. A plugin may **not** register a new `Manager` subclass, i.e. it may not
introduce a new device *type*.

**What is in scope.**

- `PluginController` subclasses with `@APIExport` endpoints mounted at `/plugin/<name>/api`.
- `Event` declarations pushed over the per-plugin Socket.IO namespace.
- `required_hardware` entries of kind `detector`, `positioner`, `laser`, `recording`, `custom`
  ([plugin_sdk/__init__.py:73](../../imswitch/plugin_sdk/__init__.py#L73)-78), resolved to
  concrete devices by `PluginManager._resolve_hardware`
  ([plugin_manager.py:428](../../imswitch/plugin_manager.py#L428)).
- A Module-Federation frontend bundle served from `/plugin/<name>/ui`.

**What is out of scope.**

- Registering a new `Manager` class, a new device kind, or a new sub-manager package.

**Rationale.** A new device type is not a controller concern. It requires (a) a new field or
schema extension in the setup file, parsed by `SetupInfo`, and (b) instantiation through
`MultiManager.__init__`, which imports sub-managers by name from a fixed package path
([MultiManager.py:23](../../imswitch/imcontrol/model/managers/MultiManager.py#L23)-29). Both
are host-private surfaces we are not ready to freeze: the setup-file schema still changes
between minor releases, and `MultiManager`'s import convention is an implementation detail.
Publishing them now would lock in a contract we would immediately want to break.

This is the decision most likely to be contested. The escape hatch for a partner who genuinely
needs a new device type is to contribute the `Manager` upstream — the driver lives in the host
tree, the *logic* on top of it lives in their plugin.

**Revisit when** the setup-file schema has been stable across two minor releases and there is
a concrete second requester. Not before.

---

## ADR-003 — Duplicate plugin names are not resolved

**Decision.** Plugin names are a flat global namespace keyed on `manifest.name`. The first
plugin to load with a given name wins. Any subsequent registration of the same name is
rejected, recorded in the error list, and skipped. No namespacing, no versioned coexistence,
no user-facing conflict resolution.

**Consequences.**

- `_activate()` raises `ValueError` on a name collision
  ([plugin_manager.py:341](../../imswitch/plugin_manager.py#L341)-344). `discover()` catches
  it per plugin ([plugin_manager.py:141](../../imswitch/plugin_manager.py#L141)) so one
  collision cannot abort the discovery loop.
- The collision surfaces in the `errors` array of `GET /imswitch/api/plugins`, with the source
  of the plugin that already holds the name, so an operator can tell *which two* collided.
- Load order is the plugin directory sorted by path. That is deliberate but **not** a
  stability guarantee — do not rely on one plugin shadowing another by naming.

**Rationale.** The name is used as a URL path segment, a Socket.IO namespace, a webpack
federation scope and a menu key. Making it non-unique means disambiguating in four places for
a situation that only arises from a packaging mistake. Reporting the mistake clearly is worth
more than resolving it.

---

## ADR-004 — Plugins are directories, never pip packages

**Decision.** A plugin is discovered by scanning one directory. There is no entry-point group,
no `pip install` path, and no second discovery mechanism. The directory is chosen by
`main(plugin_dir=...)`, else `$IMSWITCH_PLUGIN_DIR`, else `/opt/imswitch/plugins`.

**Consequences.**

- `imswitch/plugin_manager.py` does not import `importlib.metadata` and never consults entry
  points. Guarded by `test_pip_installed_plugins_are_not_discovered`.
- A plugin's `pyproject.toml` declares **no** `[project.entry-points]` table. The template's
  `make check` fails if one appears.
- Deployment is: build → copy or bind-mount the directory → restart. Nothing is installed into
  the host's Python environment, ever.
- A plugin's `pyproject.toml` remains useful for editors, linters and `pip install -e .` during
  local test runs. It is not a deployment mechanism.

**Rationale.** The dependency contract (ADR: none — see the stable-surface table) is the whole
reason this system is safe: a plugin must not be able to introduce a second copy of a library
the host already has loaded, because two NumPys in one process produce wrong numbers rather
than a clean crash. `pip install` is precisely the mechanism that would let that happen, and
supporting it would mean policing dependency resolution rather than simply not resolving
anything. Removing the path removes the failure mode.

It also halves the surface: one discovery order, one set of docs, one thing to explain, and a
`source` field in the manifest list that always means the same thing.

**Cost, stated plainly.** A plugin cannot be published to PyPI and installed with one command.
Distribution is a directory — copied, `rsync`-ed, or delivered as a `FROM scratch` image used
as a volume source (see [DEPLOYMENT.md §5](DEPLOYMENT.md#5-delivering-plugins-as-images-instead)).
For a plugin that must run in-process on a specific instrument, that is the appropriate
distribution model anyway.

---

## 2. Stable surface

What an external plugin author may depend on.

- **Frozen** — will not change incompatibly within a major version. Breaking changes require
  a major version bump and a migration note.
- **Provisional** — intended to become frozen, but may still change in a minor release. Changes
  are announced in release notes.
- **Private** — no guarantee whatsoever. May change or disappear in any release. Do not depend
  on it; if you need something here, open an issue instead.

| Surface | Status | Notes |
|---|---|---|
| `imswitch.plugin_sdk` — `PluginController`, `PluginContext`, `APIExport`, `Event`, `PluginRegistration`, `load_manifest` | **Frozen** | The only ImSwitch module a plugin may import. Versioned by `SDK_VERSION` ([plugin_sdk/__init__.py:62](../../imswitch/plugin_sdk/__init__.py#L62)). |
| `PluginManifest` / `plugin.toml` schema — `name`, `display_name`, `version`, `ui`, `required_hardware` | **Frozen** | New optional fields may be added; existing fields keep their meaning. |
| `plugin.toml` — `sdk_min`, `imswitch_min`, `permissions` | **Provisional** | Declared and validated, but **not yet enforced by the host**. Set them honestly; do not assume the host will reject a mismatch. |
| `ctx.hardware` — `detector()`, `positioner()`, `laser()` role accessors | **Frozen** | Role names come from your own manifest. |
| `ctx.hardware` — the fallback that binds the first available device when no explicit binding exists (`_first_available`, [plugin_manager.py:460](../../imswitch/plugin_manager.py#L460)) | **Provisional** | Convenient in development, surprising on a multi-device instrument. Likely to be gated behind a host flag. Declare explicit bindings in the setup file for production. |
| `/plugin/<name>/api/*` — mount layout and prefix | **Frozen** | Your router, mounted verbatim under this prefix. |
| `/plugin/<name>/ui/*` — static mount of the built bundle | **Frozen** | Same-origin, so `remoteEntry.js` needs no CORS. |
| The host's `root_path` (`/imswitch`) prefixing browser-facing plugin URLs | **Frozen** | Never build plugin URLs by hand — use `api_base` / `remote_entry` from the manifest. Starlette matches an `APIRouter` route with *or* without the prefix but matches a `StaticFiles` mount **only with** it, so an unprefixed bundle URL 404s. `socket_ns` is exempt: it is a Socket.IO namespace, not a URL. |
| Plugins are always filed under the **Plugins** category in the sidebar and App Manager | **Frozen** | `menu_group` is display metadata (`menuGroup`), not routing. Filing a plugin into a built-in category hides it among ~35 built-in apps, which is how people conclude the plugin system is broken. |
| `GET /imswitch/api/plugins` — manifest list response shape (`{"plugins": [...], "errors": [...]}`, and the per-plugin keys `name`, `display_name`, `version`, `status`, `menu`, `remote_entry`, `scope`, `exposed`, `api_base`, `socket_ns`) | **Frozen** | Consumed by the host frontend. Keys may be added, not removed or retyped. `status` is one of `"loaded"` / `"disabled"`; load failures appear in the sibling `errors` array instead. A `"disabled"` entry carries `reason` and has `remote_entry` / `api_base` / `socket_ns` set to `null`, because nothing is mounted for it. |
| Socket.IO namespace `/plugin/<name>` and the event names declared via `Event` | **Provisional** | The namespace layout is settled; how the host exposes the underlying server to `PluginContext._get_socketio` ([plugin_sdk/__init__.py:376](../../imswitch/plugin_sdk/__init__.py#L376)) is still probing several attribute names and will be tightened. |
| Frontend federation shared-module list ([frontend/shared-deps.js](../../frontend/shared-deps.js)) | **Provisional** | Adding a package is safe. Removing one breaks every plugin that imports it, so removals are announced one minor release ahead. Not yet published as an npm package; the template vendors a copy and CI diffs it. **Host declares these `eager: true`; a remote must not** — `eager` in a remote pulls a second React into its bundle. |
| Host federation remote `host_app` and its exposes — `./store`, `./contexts`, `./sharedDeps`, served at `/imswitch/ui/remoteEntry.js` | **Provisional** | The set of exposed modules will grow before it freezes. Plugins declare `remotes: { host_app: "host_app@/imswitch/ui/remoteEntry.js" }`. |
| `store.injectReducer(key, reducer, { persist })` | **Provisional** | Returns `false` if the key is taken (first registration wins). `persist: true` is accepted but **not implemented** — it warns and the slice does not survive a reload. |
| `host_app/contexts` exports — `useWebSocket`, `useWidgetContext`, `LiveWidgetContext`, `useJupyter`, `usePWA`, `baseTheme` and their providers | **Provisional** | The Redux store and MUI theme do *not* come from here: `useSelector` and `useTheme` work directly via the shared singletons. |
| Plugin widget props `hostIP` / `hostPort` | **Deprecated** | Kept for the existing goniometer plugin. Read `useSelector((s) => s.connectionSettingsState)` instead. |
| Directory scanning as the only discovery mechanism | **Frozen** | There is no `pip install` path and no entry-point group. See ADR-004. |
| `$IMSWITCH_PLUGIN_DIR` and the drop-in directory layout | **Frozen** | Default `/opt/imswitch/plugins`. |
| `imswitch.implugins` (all sub-groups) | **Removed** | See ADR-001. Nothing reads this group. |
| Everything else — `imswitch.imcontrol`, `imswitch.imcommon`, `MasterController`, `SetupInfo`, `MultiManager`, the setup-file schema | **Private** | Importing any of these from a plugin voids every guarantee above. |

---

## 3. Open items this ADR deliberately does not decide

- **Plugin state persistence.** Redux slices injected by a plugin are not in the
  `redux-persist` whitelist and will not survive a page reload. This is deliberate, not an
  oversight: the whitelist is fixed before `persistStore` runs, which is impossible for a
  reducer that arrives at runtime. `injectReducer(..., { persist: true })` warns rather than
  half-implementing it. A plugin that needs persistence should own a nested `persistReducer`
  with its own storage key.
- **Plugin directory as a trust boundary.** Anything that can write to `$IMSWITCH_PLUGIN_DIR`
  executes arbitrary Python in the host process and arbitrary JavaScript in the operator's
  browser. Acceptable for a bind mount controlled by the device owner; mount read-only. Any
  future "install a plugin from the UI" feature needs signing and is out of scope here.
- **Version skew between host and plugin.** `sdk_min` is declared but unenforced (see the
  table above), and the frontend shared-dependency list is not checked at load time at all.
