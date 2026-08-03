# ImSwitch plugin architecture — decision record

Status: **accepted** · Applies to: `feature/pluginsystemV2` and later · Last reviewed: 2026-08-03

This document is the architecture decision record for the ImSwitch plugin system. It first
inventories what is actually in the tree today, then records the decisions that follow from
that inventory, and closes with a stability table an external plugin author can rely on.

`file:line` references are against the tree **after** the WP0/WP1 change that introduced this
document. Code that this change deleted is quoted against its last commit, `bb4e04f1`.

---

## 1. Inventory — the two plugin mechanisms

Two independent plugin mechanisms exist in the tree. They share no code, no manifest format
and no naming convention. Every claim below is quoted with `file:line` as of this document's
last review.

### 1.1 v1 — `imswitch.implugins` entry points

An untyped, undeclared set of `importlib.metadata` entry-point hooks. There is no manifest:
the host infers a plugin's purpose from the *suffix of the entry-point name*
(`_controller`, `_widget`, `_manager`, `_info`). Nothing validates the loaded object.

Note that ImSwitch itself only ever declares the three *sub*-groups, and declares them empty
— [pyproject.toml:127](../../pyproject.toml#L127)-131 and
[setup.py:115](../../setup.py#L115)-117:

```toml
[project.entry-points."imswitch.implugins.detectors"]
[project.entry-points."imswitch.implugins.lasers"]
[project.entry-points."imswitch.implugins.positioner"]
```

The bare `imswitch.implugins` group that four of the five consumers below query is never
declared by the host at all. It exists only if an external package declares it.

| Consumer | Group queried | Entry-point name it looks for | Reachable? |
|---|---|---|---|
| [ImConMainView.py:49](../../imswitch/imcontrol/view/ImConMainView.py#L49) | `imswitch.implugins` | `<widgetKey>_widget` | **Yes** |
| [MasterController.py:203](../../imswitch/imcontrol/controller/MasterController.py#L203) | `imswitch.implugins` | `*_manager`, `*_info` | **Yes** |
| [MultiManager.py:38](../../imswitch/imcontrol/model/managers/MultiManager.py#L38) | `imswitch.implugins.<subpkg>` | any | **Yes**, but only in the `except` branch of a failed built-in manager import |
| `SetupInfo.add_attribute` (`SetupInfo.py:923` @ bb4e04f1) | `imswitch.implugins` | `<attr>_info` | **No** — dead, removed. See 1.1.1 |
| `ImConMainController.loadPlugin` (`ImConMainController.py:279` @ bb4e04f1) | `imswitch.implugins` | `<widgetKey>_controller` | **No** — unreachable, removed. See 1.1.2 |

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
| Entry-point group `imswitch.plugins` | [plugin_manager.py:46](../../imswitch/plugin_manager.py#L46) | **Yes** |
| Drop-in dir `$IMSWITCH_PLUGIN_DIR` (default `/opt/imswitch/plugins`) | [plugin_manager.py:50](../../imswitch/plugin_manager.py#L50)-51 | **Yes** |
| `PluginManager.discover()` | [plugin_manager.py:90](../../imswitch/plugin_manager.py#L90) | **Yes** — [ImSwitchServer.py:703](../../imswitch/imcontrol/controller/server/ImSwitchServer.py#L703) |
| `PluginManager.attach_to_app()` — mounts `/plugin/<n>/api` and `/plugin/<n>/ui` | [plugin_manager.py:99](../../imswitch/plugin_manager.py#L99) | **Yes** — [ImSwitchServer.py:704](../../imswitch/imcontrol/controller/server/ImSwitchServer.py#L704) |
| `PluginManager.manifest_list()` | [plugin_manager.py:129](../../imswitch/plugin_manager.py#L129) | **Yes** as of WP1 — served by [`list_plugins`, ImSwitchServer.py:94](../../imswitch/imcontrol/controller/server/ImSwitchServer.py#L94) at `GET /imswitch/api/plugins`, which [App.jsx:526](../../frontend/src/App.jsx#L526) fetches. Before WP1 it had no caller and that route 404'd. |
| `PluginManager.shutdown()` | [plugin_manager.py:156](../../imswitch/plugin_manager.py#L156) | **Yes** — [ImSwitchServer.py:734](../../imswitch/imcontrol/controller/server/ImSwitchServer.py#L734) |

### 1.3 Summary of the inventory

v1 has five consumers, of which three are reachable and all three are *fallback* paths: they
fire only when a built-in lookup has already failed. v1 has never had a manifest, a version
contract, a dependency contract or a frontend story. v2 has all four, and after WP1 the whole
chain from drop-in directory to browser is connected.

---

## ADR-001 — v2 `PluginManager` is the only supported plugin mechanism

**Decision.** New plugins target `imswitch.plugin_sdk` and the v2 `PluginManager`. The v1
`imswitch.implugins` entry-point hooks are deprecated.

**Consequences.**

- No new host code may consume `imswitch.implugins`. The three reachable consumers listed in
  1.1 are frozen: bugs are fixed, features are not added.
- v1 hooks are kept only for already-shipped external packages that declare them. They are
  removed no earlier than the next **major** ImSwitch version, and that removal must be
  announced in the release notes of the preceding minor release.
- Documentation, the template repository and partner-facing material describe only v2. v1 is
  mentioned only in a deprecation note.
- Dead v1 code is removed immediately rather than waiting for the major version: dead code
  carries the maintenance and documentation cost of a supported path without any of the
  benefit. Removed under this ADR: `SetupInfo.add_attribute` (1.1.1) and
  `ImConMainController.loadPlugin` plus its unreachable caller (1.1.2).

**Rationale.** Keeping two mechanisms doubles the surface we document, test and support, and
the two differ on every axis that matters — validation, versioning, error reporting, frontend
delivery. v1 cannot be fixed incrementally into v2 because it has no manifest to hang any of
those properties on.

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
  ([plugin_manager.py:259](../../imswitch/plugin_manager.py#L259)).
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
  ([plugin_manager.py:230](../../imswitch/plugin_manager.py#L230)-233). `discover()` catches
  it per plugin ([plugin_manager.py:96](../../imswitch/plugin_manager.py#L96)) so one
  collision cannot abort the discovery loop.
- The collision surfaces in the `errors` array of `GET /imswitch/api/plugins`, with the source
  of the plugin that already holds the name, so an operator can tell *which two* collided.
- Load order (entry points first, then the drop-in directory sorted by path) determines the
  winner. That order is deliberate but **not** a stability guarantee — do not rely on a
  drop-in shadowing a pip-installed plugin of the same name.

**Rationale.** The name is used as a URL path segment, a Socket.IO namespace, a webpack
federation scope and a menu key. Making it non-unique means disambiguating in four places for
a situation that only arises from a packaging mistake. Reporting the mistake clearly is worth
more than resolving it.

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
| `ctx.hardware` — the fallback that binds the first available device when no explicit binding exists (`_first_available`, [plugin_manager.py:291](../../imswitch/plugin_manager.py#L291)) | **Provisional** | Convenient in development, surprising on a multi-device instrument. Likely to be gated behind a host flag. Declare explicit bindings in the setup file for production. |
| `/plugin/<name>/api/*` — mount layout and prefix | **Frozen** | Your router, mounted verbatim under this prefix. |
| `/plugin/<name>/ui/*` — static mount of the built bundle | **Frozen** | Same-origin, so `remoteEntry.js` needs no CORS. |
| `GET /imswitch/api/plugins` — manifest list response shape (`{"plugins": [...], "errors": [...]}`, and the per-plugin keys `name`, `display_name`, `version`, `status`, `menu`, `remote_entry`, `scope`, `exposed`, `api_base`, `socket_ns`) | **Frozen** | Consumed by the host frontend. Keys may be added, not removed or retyped. Exception: the *values* of `status` are provisional — today it is always `"ok"`; WP2 replaces that with `"loaded"` / `"disabled"` / `"error"`. |
| Socket.IO namespace `/plugin/<name>` and the event names declared via `Event` | **Provisional** | The namespace layout is settled; how the host exposes the underlying server to `PluginContext._get_socketio` ([plugin_sdk/__init__.py:376](../../imswitch/plugin_sdk/__init__.py#L376)) is still probing several attribute names and will be tightened. |
| Frontend federation shared-module list (`frontend/shared-deps.js`, WP3) | **Provisional** | Adding a package is safe. Removing one breaks every plugin that imports it, so removals are announced one minor release ahead. Not yet published as an npm package; the template vendors a copy and CI diffs it. |
| Host federation remote `host_app` and its exposes (`./store`, `./contexts`, `./sharedDeps`, WP3) | **Provisional** | The set of exposed modules will grow before it freezes. |
| `imswitch.plugins` entry-point group name | **Frozen** | |
| `$IMSWITCH_PLUGIN_DIR` and the drop-in directory layout | **Frozen** | Default `/opt/imswitch/plugins`. |
| `imswitch.implugins` (all sub-groups) | **Deprecated** | See ADR-001. No new use. |
| Everything else — `imswitch.imcontrol`, `imswitch.imcommon`, `MasterController`, `SetupInfo`, `MultiManager`, the setup-file schema | **Private** | Importing any of these from a plugin voids every guarantee above. |

---

## 3. Open items this ADR deliberately does not decide

- **Plugin state persistence.** Redux slices injected by a plugin are not in the
  `redux-persist` whitelist and will not survive a page reload. Decided in WP3; noted here so
  it is not rediscovered.
- **Plugin directory as a trust boundary.** Anything that can write to `$IMSWITCH_PLUGIN_DIR`
  executes arbitrary Python in the host process and arbitrary JavaScript in the operator's
  browser. Acceptable for a bind mount controlled by the device owner; mount read-only. Any
  future "install a plugin from the UI" feature needs signing and is out of scope here.
- **Version skew between host and plugin.** `sdk_min` is declared but unenforced (see the
  table above), and the frontend shared-dependency list is not checked at load time at all.
