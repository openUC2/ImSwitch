# ImSwitch plugin architecture — work plan to a shippable template

**Target state**

> An external developer clones a template repo, writes one Python controller and one React
> widget, runs `make build`, and gets a directory (or an OCI image) that we bind-mount into
> the stock ImSwitch container. On restart the plugin's endpoints appear under
> `/plugin/<name>/api`, its widget appears in the sidebar, and that widget uses the host's
> React, the host's MUI theme and the host's Redux store — no rebuild of ImSwitch, no
> `pip install`, no second copy of any library.

**Basis:** `openUC2/ImSwitch` branch `feature/pluginsystemV2`, plus
`openUC2/imswitch-plugin-goniometer` as the reference plugin and
`openUC2/os-rpi` `deployments/imswitch.pkg/deployment.compose.yml` as the deployment target.

**Scope decision:** multiple plugins registering the same name is out of scope. First one
wins, second is reported as an error in the manifest list. Do not build conflict resolution.

---

## 0. Where the branch actually stands

The v2 design is sound and roughly 80% implemented. What is missing is not architecture —
it is a short list of unfinished wires, each of which independently prevents the system from
working end to end. Nothing below requires a rewrite.

### Blocking defects (verified against the branch)

| # | File | Problem |
|---|------|---------|
| ~~B1~~ | `imswitch/imcontrol/controller/server/ImSwitchServer.py` | **Fixed in WP1.** `PluginManager.manifest_list()` is fully implemented and **never routed**. `App.jsx` fetches `/imswitch/api/plugins`, which 404s. No plugin can ever appear. |
| ~~B2~~ | same | **Fixed in WP1** (module-level `_PLUGIN_MANAGER` + `set_plugin_manager()`). `PluginManager` is stored on `self._plugin_manager` inside `run()`. Route handlers are module-level closures over `app`/`api_router` and cannot reach it. |
| ~~B3~~ | same | **Fixed in WP1** (`register_plugin_routes(api_router)` called from `createAPI()`). `createAPI()` → `app.include_router(api_router)` → `PluginManager.discover()`. Any route added to `api_router` after `include_router` is ignored. `/api/plugins` must be registered *before* discovery but must *read* state that only exists after it. |
| B4 | `frontend/src/components/navigation/NavigationDrawer.jsx` | Accepts a `plugins` prop (line 37) and **never renders it**. The menu is built from the static `APP_REGISTRY`. `selectedPlugin` can therefore never equal a plugin's name, so the render block in `App.jsx` (~line 600) is unreachable. |
| B5 | `frontend/craco.config.js` | `shared` contains only `react`, `react-dom`, `react/jsx-runtime`. Redux, MUI and Emotion are not shared, so a plugin gets its own `ReactReduxContext` and its own MUI theme context. `useSelector` throws; `useTheme` returns the default theme. |
| B6 | `frontend/src/App.jsx`, `loadRemote` | On a missing federation scope the function `console.error`s and `return`s without calling `reject`. The promise never settles and `<Suspense>` hangs forever with no error surfaced. |
| B7 | `imswitch/plugin_manager.py` | `discover()` loads every plugin it finds. There is no gating on `availableWidgets`, which the deployment model requires. |
| ~~B8~~ | `imswitch/imcontrol/controller/ImConMainController.py` ~line 98 | **Resolved in WP0** by deleting the dead branch and `loadPlugin()` — see `docs/plugins/DECISIONS.md` §1.1.2. Legacy v1 path: `continue` in the `except` block makes the `else: loadPlugin(...)` branch unreachable. |

### Design gaps (not defects, but required for the target state)

- No mechanism for a plugin to contribute a Redux slice (`store.js` uses static `combineReducers`).
- Host does not `expose` anything, so a plugin cannot import host contexts (`WebSocketContext`, `WidgetContext`) or the store object.
- `APP_REGISTRY` is a compile-time constant; runtime plugins cannot participate in the App Manager enable/disable flow.
- No template repo, no CI check that a plugin bundles no second React and imports no extra Python dependency.

### What already works and should not be touched

- `imswitch/plugin_sdk/__init__.py` — the SDK surface is well designed. `PluginController`, `Event`, `PluginManifest`, role-based `ctx.hardware`. Keep it.
- `PluginManager.attach_to_app()` — mounts `plugin.router` at `/plugin/<n>/api` and `ui_dir` as `StaticFiles` at `/plugin/<n>/ui`. **This means a bind-mounted plugin ships its frontend automatically**; the browser fetches `remoteEntry.js` same-origin. This is the key property that makes the whole plan work.
- `_load_dropin()` — imports a package from `$IMSWITCH_PLUGIN_DIR` with `sys.path.insert`. Works without `pip install`.
- `build_router()` uses `router.add_api_route(path, fn)`, so a plain `def` handler goes to FastAPI's threadpool. Plugin routes are *safer* than core `@APIExport` GET routes, which wrap sync functions in `async def` and block the event loop.

---

## How to use this document

Each work package has:

- **Why** — the human-readable rationale, suitable for pasting into an issue or a partner email.
- **Files** — what gets touched.
- **Prompt** — a self-contained Claude Code session. Run them in order; each assumes the previous is merged.
- **Done when** — the acceptance check. Do not move on until it passes.

Run each prompt in a fresh Claude Code session from the repo root. Prompts assume the working
tree is on a branch off `feature/pluginsystemV2`.

---

## WP0 — Baseline audit and decisions ✅ done

**Delivered.** `docs/plugins/DECISIONS.md`. Beyond the ADR, the two dead v1 code paths the
audit found were deleted (`ImConMainController.loadPlugin` + its unreachable caller,
`SetupInfo.add_attribute`).

**ADR-001 subsequently strengthened: v1 is now removed outright, not deprecated.** The
inventory is what changed the decision — of the five `imswitch.implugins` consumers, two were
already dead, and of the three live ones none could carry a fully-configured plugin
(`MasterController` constructed every v1 manager with `moduleInfo = None`, under a TODO
admitting the setup info was never wired up). Keeping a mechanism that cannot work costs
documentation and test surface while protecting nothing. All five consumers and the empty
`imswitch.implugins.*` entry-point groups in `pyproject.toml`/`setup.py` are gone. This is a
**breaking change for any external package that declared those entry points** and belongs in
the release notes; the migration path is in ADR-001.

**Why.** Two plugin systems currently coexist: the v1 entry-point hooks (`imswitch.implugins`,
consumed by `ImConMainController.loadPlugin` and `MultiManager`) and the v2 `PluginManager`.
Keeping both doubles the surface we have to document and test, and the v1 controller path is
already dead code. Decide now, in writing, that v2 is the only supported path for new work,
and that v1 stays only as long as an existing deployment depends on it.

Second decision to record: plugins are **controller-only**. A plugin declares which existing
hardware it needs by role and gets a live manager handle. Contributing new *device types*
(a new `Manager` subclass) is explicitly out of scope for v1 of the template — that path
stays internal because it requires setup-file schema changes.

**Files.** `docs/plugins/DECISIONS.md` (new).

**Prompt.**

```
Read imswitch/plugin_manager.py, imswitch/plugin_sdk/__init__.py,
imswitch/imcontrol/controller/ImConMainController.py (the loadPlugin method and the
controller construction loop), and imswitch/imcontrol/model/managers/MultiManager.py.

Write docs/plugins/DECISIONS.md as an architecture decision record covering:

1. Inventory of the two plugin mechanisms currently in the tree. For each: the entry-point
   group name, which host code consumes it, and whether that code path is currently
   reachable. Quote file:line for each claim.
2. ADR-001: v2 PluginManager is the only supported mechanism for new plugins. v1
   (imswitch.implugins) is deprecated, kept only for already-shipped plugins, removed no
   earlier than the next major version.
3. ADR-002: plugins are controller-only. They may claim existing hardware by role via
   ctx.hardware. They may not register new Manager classes. State the reason: a new device
   type requires the setup-file schema and MultiManager device instantiation, which is host
   surface we are not ready to make public.
4. ADR-003: duplicate plugin names are not resolved. First loaded wins; subsequent
   registrations of the same name are reported as errors in the manifest list and skipped.
5. A "stable surface" table: for each of imswitch.plugin_sdk, /plugin/<n>/api,
   /plugin/<n>/ui, the socket namespace, and the frontend federation shared-module list —
   state whether it is frozen, provisional, or private.

Do not change any code in this work package.
```

**Done when.** The ADR is merged and the team agrees with ADR-002 in particular — it is the
decision most likely to be contested later.

---

## WP1 — Make plugins visible to the frontend ✅ done

**Delivered.** `GET /imswitch/api/plugins` and `GET /imswitch/api/plugins/{name}`, registered by
`register_plugin_routes()` in `ImSwitchServer.py` from `createAPI()`; `set_plugin_manager()`
called after `discover()`/`attach_to_app()`, which also clears `app.openapi_schema`. Tests in
`imswitch/imcontrol/_test/test_plugin_api.py` (9 cases, incl. a B3 registration-order guard).

**Why.** This is the single highest-leverage fix. `manifest_list()` already returns exactly
the payload `App.jsx` expects (`name`, `remote_entry`, `scope`, `exposed`, `api_base`,
`socket_ns`, `menu`) and nothing serves it. Until this route exists, every other piece of
the system is untestable.

Two subtleties make this more than a five-line change. The `PluginManager` is created inside
`ImSwitchServer.run()`, but route handlers are module-level closures — they need a module-level
reference. And routes must be registered on `api_router` *before* `app.include_router(api_router)`,
which happens before `discover()`. So register the route early and have it read a mutable
module-level holder that discovery fills in later.

**Files.** `imswitch/imcontrol/controller/server/ImSwitchServer.py`, `imswitch/plugin_manager.py`.

**Prompt.**

```
Goal: expose the v2 plugin manifest over HTTP at GET /imswitch/api/plugins so the React
frontend can discover and load plugins.

Context you must read first:
- imswitch/imcontrol/controller/server/ImSwitchServer.py — note the module-level `app` and
  `api_router` objects, the createAPI() method, and ImSwitchServer.run() where
  app.include_router(api_router) is called and where PluginManager is constructed.
- imswitch/plugin_manager.py — PluginManager.manifest_list() and PluginManager.errors().
- frontend/src/App.jsx, function usePluginWidgets() — the exact URL and response shape the
  frontend expects.

Implement:

1. In ImSwitchServer.py, add a module-level holder near the `app`/`api_router` definitions:
       _PLUGIN_MANAGER = None
   with a module-level setter function. Do not use a class attribute; the route closure
   needs module scope.

2. Register the route inside createAPI(), so it lands on api_router BEFORE
   app.include_router(api_router) runs. The handler must read _PLUGIN_MANAGER at call time,
   not at registration time. Response shape, matching what manifest_list() already returns:
       {"plugins": [...], "errors": [...]}
   If _PLUGIN_MANAGER is None (discovery failed or has not run), return empty lists with
   HTTP 200 — never 500. The frontend must degrade to "no plugins" rather than break.

3. In ImSwitchServer.run(), after self._plugin_manager.discover() and attach_to_app(),
   call the setter. Keep the existing try/except so a plugin failure cannot prevent the
   server from starting.

4. Add GET /imswitch/api/plugins/<name> returning the single manifest entry, 404 if absent.
   The frontend does not need it yet but it makes debugging a specific plugin much easier.

5. After attach_to_app() mounts new routes on `app`, set app.openapi_schema = None so the
   OpenAPI document regenerates and includes plugin routes.

Add a test at imswitch/imcontrol/_test/test_plugin_api.py using fastapi.testclient that
asserts: the route exists, returns 200 with both keys present when no plugin manager is
set, and returns a loaded plugin's manifest when one is.
```

**Done when.** `curl http://<host>:8001/imswitch/api/plugins` returns `{"plugins": [], "errors": []}`
on a clean install and does not 404.

---

## WP2 — Drop-in loading from a bind mount, gated by `availableWidgets` ✅ done

**Delivered.** `_is_enabled()` gating before hardware resolution and controller construction;
`DisabledPlugin` records surfaced in `manifest_list()` with `status: "disabled"` and a `reason`;
`ui_dir` fallback to `<plugin_root>/<dist_dir>` via `__imswitch_plugin_root__`; `dropin_root()`
resolved per call; per-plugin startup log line. 16 tests in
`imswitch/imcontrol/_test/test_plugin_manager.py`.

**Also fixed (not in the original defect list).** Browser-facing plugin URLs were missing the
host's `root_path`. Starlette matches an `APIRouter` route with *or* without the `/imswitch`
prefix, but matches a `StaticFiles` **Mount only with it** — so `remote_entry` as published
would have 404'd every plugin bundle in production while the mount looked correct. The manager
now captures `app.root_path` in `attach_to_app()` and prefixes `remote_entry` / `api_base`
(`socket_ns` stays unprefixed — it is a Socket.IO namespace, not a URL). Guarded by
`test_advertised_urls_actually_resolve_on_the_host_app`, which fetches the published URL rather
than asserting a string shape.

**Why.** Two requirements meet here. First, a plugin must load from a read-only bind mount
with no `pip install` — that is what makes the Docker story work and what guarantees no
second NumPy can appear. `_load_dropin()` already does the import; what needs verifying is
that `register()`'s `ui_dir` resolution works for a drop-in package (the goniometer uses
`importlib.resources.files(__package__)`, which behaves differently for a path-imported
package than for an installed one).

Second, the deployment model says a plugin loads only if it is named in `availableWidgets`
in the setup file. Today `discover()` loads everything it finds. Gating serves a real purpose:
one plugin directory can be mounted across a fleet of microscopes while each instrument's
setup file decides which plugins are active.

Note that gating must happen *after* the manifest is read (we need the name) but *before*
the controller is constructed (we do not want to claim hardware for an inactive plugin).

**Files.** `imswitch/plugin_manager.py`, `imswitch/imcontrol/controller/server/ImSwitchServer.py`.

**Prompt.**

```
Goal: (a) make drop-in plugin loading from a bind-mounted directory work reliably, and
(b) gate plugin activation on the setup file's availableWidgets list.

Read first:
- imswitch/plugin_manager.py — _iter_register_fns(), _load_dropin(), discover(), _activate(),
  and the constants ENTRY_POINT_GROUP / DROPIN_ENV_VAR / DEFAULT_DROPIN.
- imswitch/plugin_sdk/__init__.py — PluginRegistration (manifest, controller_factory, ui_dir)
  and load_manifest().
- imswitch/imcontrol/view/guitools/ViewSetupInfo.py — availableWidgets and the hasWidget()
  helper. Note availableWidgets may be `True` (meaning all) or a list.

Implement:

1. Drop-in ui_dir robustness. In _load_dropin, after importing the package, record the
   resolved filesystem directory of that package and attach it to the returned register
   function (e.g. fn.__imswitch_plugin_root__ = str(src_dir / pkg_name)). In _activate,
   if reg.ui_dir is None or does not resolve to an existing directory, fall back to
   <plugin_root>/<manifest.ui.dist_dir>. This makes a plugin work identically whether it was
   pip-installed or bind-mounted, without the plugin author having to care.

2. availableWidgets gating. PluginManager.__init__ already receives setup_info. Add a
   private helper _is_enabled(name) implementing:
       - availableWidgets is True  -> everything enabled
       - name in availableWidgets  -> enabled (case-sensitive, matches manifest.name)
       - otherwise                 -> not enabled
   Call it in _activate() immediately after the manifest is parsed and BEFORE
   _resolve_hardware() and before controller instantiation. A disabled plugin must not
   claim hardware and must not construct its controller.
   Record disabled plugins in a new self._disabled list and include them in manifest_list()
   with status "disabled" so the UI can show them greyed out rather than hiding them —
   a silently missing plugin is the worst possible debugging experience.

3. Duplicate names: keep the existing ValueError in _activate but make sure it is caught and
   recorded via _record_error rather than aborting the whole discovery loop. Confirm the
   caller in discover() already wraps _activate in try/except; if not, add it.

4. Startup log line: after discovery, log one line per plugin at INFO with name, version,
   status (loaded/disabled/error) and source. Operators debugging a bind mount need this.

5. Make DEFAULT_DROPIN honour the env var at call time, not import time, so tests can
   monkeypatch it.

Add tests in imswitch/imcontrol/_test/test_plugin_manager.py using tmp_path to build a fake
drop-in plugin directory:
   - loads when its name is in availableWidgets
   - is reported as "disabled", with no controller constructed, when it is not
   - resolves ui_dir from the package root when register() returns ui_dir=None
   - a second plugin with a duplicate name is recorded as an error and does not prevent the
     first from loading
```

**Done when.** A minimal plugin dropped into `$IMSWITCH_PLUGIN_DIR` appears in
`/imswitch/api/plugins` with `status: "loaded"` when listed in `availableWidgets`, and
`status: "disabled"` when not.

---

## WP3 — Frontend shared runtime ✅ done

**Delivered.** `frontend/shared-deps.js` (`SHARED_DEPS` + `makeShared({ eager })`);
`craco.config.js` now uses `makeShared({ eager: true })` and exposes `./store`, `./contexts`,
`./sharedDeps` under `filename: "remoteEntry.js"`; `src/context/index.js` barrel;
`store.injectReducer(key, reducer, { persist })`.

**Deviation from the prompt, point 4.** The prompt's `store.replaceReducer(buildRootReducer({...}))`
would have broken persistence app-wide. A fresh `persistReducer` instance starts with a null
persistoid and only gets one from the `PERSIST` action, which `persistStore` already dispatched
at startup — so re-wrapping stops persisting *every* whitelisted slice the first time any plugin
injects a reducer. Instead the combined reducer is a mutable binding behind a stable proxy, so
the persist layer and the cross-tab sync wrapper keep their identity for the life of the page.

**Why.** This is the work package that answers "can the plugin use the same Redux store".
The answer is yes, and the React tree is already correct: `index.js` wraps `<App/>` in
`<Provider store={store}>`, `App.jsx` wraps its content in `<ThemeProvider>`, and the
federated plugin renders inside both. Context flows down the tree regardless of which webpack
bundle a component was compiled in.

The only thing breaking it is module identity. `react-redux` is not in the host's `shared`
block, so a plugin importing it gets a second copy carrying a second `ReactReduxContext`
object. React resolves context by object identity, finds nothing, and throws. Making it a
singleton collapses the two copies into one and `useSelector` starts reading the host store
with no imports, no props and no bridge object.

Sharing npm packages does not cover *our* modules — `WebSocketContext`, `store.js`,
`backendapi` — so the host also has to become a federation remote and expose them. And
because `store.js` uses a static `combineReducers`, a plugin cannot add its own slice without
a reducer-injection API.

One caveat to decide explicitly: an injected reducer is not in the `redux-persist` whitelist,
so plugin state will not survive a page reload unless registered. Default to not persisting;
make it opt-in.

**Files.** `frontend/craco.config.js`, `frontend/src/state/store.js`, `frontend/src/context/index.js` (new), `frontend/src/App.jsx`.

**Prompt.**

```
Goal: let a Module-Federation plugin widget share the host's React, MUI theme, Redux store
and socket client, and let it register its own Redux slice.

Read first:
- frontend/craco.config.js — the current ModuleFederationPlugin config, name "host_app",
  shared block with only react / react-dom / react/jsx-runtime.
- frontend/src/index.js — <Provider store={store}> wrapping <App/>.
- frontend/src/App.jsx — the ThemeProvider at ~line 447 and the plugins.map render block at
  ~line 600. Confirm the plugin component renders inside both providers.
- frontend/src/state/store.js — configureStore with a static combineReducers and
  rootReducerWithSync.
- frontend/src/context/ — JupyterContext, LiveWidgetContext, ThemeContext, WebSocketContext,
  WidgetContext.

Implement:

1. Extract the shared-module list into frontend/shared-deps.js exporting a plain array of
   package names, so the host config and the plugin template cannot drift:
     react, react-dom, react/jsx-runtime, react-redux, @reduxjs/toolkit,
     @mui/material, @mui/icons-material, @emotion/react, @emotion/styled,
     socket.io-client, notistack
   Export a helper makeShared({ eager }) that maps each name to
   { singleton: true, eager, requiredVersion: false }.

2. In craco.config.js use makeShared({ eager: true }) — the host is the provider and must
   load these eagerly. Also add to ModuleFederationPlugin:
       filename: "remoteEntry.js",
       exposes: {
         "./store":     "./src/state/store.js",
         "./contexts":  "./src/context/index.js",
         "./sharedDeps": "./shared-deps.js",
       }
   so the host is a bidirectional federation container.

3. Create frontend/src/context/index.js re-exporting every context and provider from the
   files in that directory, as the single public surface for plugins.

4. In state/store.js add dynamic reducer injection. Keep the existing static reducer map in
   a named const. Add:
       const injectedReducers = {};
       store.injectReducer = (key, reducer) => {
         if (injectedReducers[key]) return false;
         injectedReducers[key] = reducer;
         store.replaceReducer(buildRootReducer({ ...staticReducers, ...injectedReducers }));
         return true;
       };
   where buildRootReducer is a factory extracted from the existing rootReducerWithSync
   construction so the cross-tab sync wrapper is preserved. Injected reducers are NOT added
   to the redux-persist whitelist by default; accept an optional third argument
   { persist: false } and leave persist support as a TODO comment rather than half-implementing it.

5. Verify — and fix if needed — that the plugins.map render block in App.jsx sits inside
   both <Provider> and <ThemeProvider>. Do not pass theme or store as props to plugin
   components; they must come from context. Keep passing hostIP/hostPort for now for
   backwards compatibility with the existing goniometer plugin, but add a code comment
   marking them deprecated in favour of
   useSelector(s => s.connectionSettingsState).

Do not change any plugin repo in this work package. Confirm `npm run build` succeeds and
the existing app still runs with no console errors.
```

**Done when.** The host builds, serves `/imswitch/ui/remoteEntry.js`, and the app behaves
exactly as before. Nothing user-visible changes in this WP — it is enabling work.

---

## WP4 — Dynamic app registry, navigation, and a loader that fails loudly ✅ done

**Delivered.** `loadRemote` now settles on every path (missing scope, missing expose, script
404, thrown init) plus a 10s timeout, each message naming the plugin and its URL;
`PluginErrorBoundary` around each widget; `makeRegistryEntryFromManifest` + `resolveMuiIcon` +
`APP_CATEGORIES.PLUGINS`; `appManager.dynamicApps` / `seenDynamicApps` / `pluginErrors` with
union selectors; `NavigationDrawer` renders plugins (sorted by `menu.order`) instead of ignoring
them; a "Plugins not available" panel in the App Manager.

**Extra hole closed.** A plugin that loads a backend but ships no built frontend bundle
(`remote_entry: null`) was going to be filtered out of both the sidebar and the registry — i.e.
disappear silently, the exact failure mode point 6 exists to prevent. It is now reported as
"no widget" with its working `api_base`.

**Verified live**, not with a stub — real backend + real browser, two drop-in plugins:
- Sidebar shows `APPS › Demo Widget` (manifest group `apps` → built-in category) and
  `PLUGINS › Goniometer` (group `Measurement` → fallback bucket). Both clickable.
- Demo widget rendered `host React identity shared: YES`, `useTheme().palette.mode = dark`
  (host theme, not MUI's default) and `useSelector(connectionSettingsState).ip` — with no
  props. That is WP3's central claim, confirmed empirically.
- Goniometer (no built bundle) produced the error card naming the plugin and the 404 URL
  instead of an endless spinner, and the rest of the app kept working; switching away and back
  recovered.
- With `goniometer` removed from `availableWidgets` it vanished from the sidebar and appeared
  in the App Manager panel as *disabled*, quoting the reason.

**Why.** With WP1 the manifest is served and with WP3 the runtime is shared, but a plugin
still cannot be *reached*: `NavigationDrawer` takes a `plugins` prop and ignores it, and the
menu is built from the static `APP_REGISTRY`. Nothing can set `selectedPlugin` to a plugin's
name, so the render block never fires.

The right fix is to merge runtime manifests into the app registry rather than bolting on a
second menu. That way plugins participate in the App Manager enable/disable flow and category
grouping like any built-in app, and `menu_label` / `menu_icon` / `menu_group` / `order` from
`plugin.toml` do what they claim to do.

The loader bug is small but vicious: when a federation scope is missing, `loadRemote` logs
and returns without settling the promise, so `<Suspense>` spins forever with no error. Any
plugin author's first typo produces a hang with no diagnostic.

**Files.** `frontend/src/App.jsx`, `frontend/src/components/navigation/NavigationDrawer.jsx`, `frontend/src/constants/appRegistry.js`, `frontend/src/state/slices/appManagerSlice.js`.

**Prompt.**

```
Goal: make runtime-discovered plugins appear in the sidebar and be selectable, and make the
federated module loader fail visibly instead of hanging.

Read first:
- frontend/src/App.jsx — usePluginWidgets(), loadRemote(), the plugins.map render block, and
  where `plugins` is passed to NavigationDrawer.
- frontend/src/components/navigation/NavigationDrawer.jsx — it accepts `plugins = []` at
  line ~37 and never uses it. Note how it builds entries from APP_REGISTRY filtered by
  selectEnabledApps.
- frontend/src/constants/appRegistry.js — APP_REGISTRY shape: id, name, description,
  category, icon, enabled, essential, keywords, pluginId.
- frontend/src/state/slices/appManagerSlice.js — enabledApps and the selectors.

Implement:

1. Fix loadRemote:
   - reject(new Error(...)) on a missing window[scope] instead of returning.
   - wrap the whole init() body so any throw rejects.
   - add a timeout (10s) that rejects with a message naming the plugin and its remote_entry URL.
   - keep the existing data-mf script dedupe.

2. Add an error boundary component around each plugin render so a crashing widget shows an
   inline error card with the plugin name and the error message, and does not blank the app.

3. Make the app registry extensible at runtime. Add a function
   makeRegistryEntryFromManifest(manifest) mapping a plugin manifest to the APP_REGISTRY
   shape:
       id: `plugin:${manifest.name}`
       name: manifest.menu.label || manifest.display_name
       category: manifest.menu.group (fall back to a new APP_CATEGORIES.PLUGINS)
       icon: resolve manifest.menu.icon by name from @mui/icons-material, falling back to
             ExtensionIcon if the name does not resolve
       essential: false
       pluginId: manifest.name
       isDynamic: true
   Do NOT mutate the exported APP_REGISTRY object. Instead add a Redux slice field
   (appManager.dynamicApps) populated when the manifest fetch resolves, and change the
   selectors used by NavigationDrawer and AppManager to read the union of APP_REGISTRY and
   dynamicApps.

4. Default enable state for a dynamic app: enabled. Rationale — the backend already gated on
   availableWidgets, so if a plugin reached the frontend the operator meant to have it. The
   AppManager toggle then controls visibility only. Add a comment saying so.

5. NavigationDrawer: render dynamic plugin entries alongside built-ins, sorted by
   manifest.menu.order within their group, and wire onClick to handlePluginChange(pluginId)
   so selectedPlugin matches the plugins.map key in App.jsx.

6. Show failures. Plugins returned in the `errors` array of /imswitch/api/plugins, and those
   with status "disabled", must be visible somewhere in the UI — a section in AppManager is
   fine. A plugin that silently does not appear is the failure mode we most need to avoid.

Verify with a stub: temporarily hardcode one fake manifest in usePluginWidgets pointing at a
non-existent remoteEntry, confirm it appears in the sidebar, is clickable, and shows the
error card rather than hanging. Remove the stub before committing.
```

**Done when.** A fake manifest produces a clickable sidebar entry, and a broken remote entry
produces a visible error rather than an infinite spinner.

---

## WP5 — Template repository ✅ done

**Delivered** at `../imswitch-plugin-template` (sibling of this repo, like the goniometer).
Backend (`register()` + `PluginController` with the thread-and-event pattern), `ui-src/` with a
widget that reads theme/store/dispatch from context and injects its own Redux slice, `Makefile`
(`build`/`check`/`dist`), `FROM scratch` Dockerfile, `.github/workflows/ci.yml`,
`docs/WRITING_A_PLUGIN.md`.

**Deviation from the prompt's layout.** The package is `imswitch_plugin_example/`, not
`plugin/`. A drop-in plugin is imported by directory name off `sys.path`, so a package literally
named `plugin` would be a landmine — it would shadow anything else called `plugin` in the
process. The template's rename checklist covers it and `make check` verifies the five names agree.

**Two design calls worth knowing.**
- Remotes declare shared modules with webpack's `import: false`. Without it webpack *also* emits
  a local fallback copy of React "just in case the host doesn't provide it" — which is the
  duplicate-React bug merely deferred to runtime, and would have made the CI "no second React"
  check impossible to pass. With it, a missing host module fails loudly at load instead.
- `shared-deps.js` is now canonical in `frontend/shared-deps.js` and copied **byte-identically**
  into each plugin, so `make check` can diff them. It gained a `fallback` option so one file
  serves both sides.

**Verified.** 14 contract checks pass on a clean build; deliberately breaking three of them
(a runtime dependency, React moved to `dependencies`, a scope typo) produces exactly 4 failures,
and reverting restores green. Built bundle is 20 KiB with no React runtime in it.

**Why.** Everything up to here is host-side. This is the deliverable an external developer
actually touches. The goniometer plugin is close to the right shape but was written against
a host that did not share enough, so it defensively passes `apiBase`, `socket` and `theme`
as props and hand-rolls what should come from context. It also has three concrete problems
to avoid repeating: `eager: true` on `react` in a *remote* (only the host should be eager),
`@mui/material` declared singleton on a side that does not share it, and `numpy` /
`opencv-python-headless` / `pydantic` listed as hard `dependencies`, which would pull a
second NumPy if anyone ever `pip install`s it.

The template must make the dependency contract structural rather than documented — it should
be difficult to get wrong, and CI should catch it when someone does.

**Files.** New repo `openUC2/imswitch-plugin-template`.

**Prompt.**

```
Create a new repository skeleton at ./imswitch-plugin-template implementing the ImSwitch v2
plugin contract. This is a template external developers will clone.

Read first, and treat as the reference for what the host expects:
- imswitch/plugin_sdk/__init__.py (PluginController, APIExport, Event, PluginManifest,
  PluginRegistration, load_manifest)
- imswitch/plugin_manager.py (_load_dropin, _activate, attach_to_app mount layout)
- frontend/shared-deps.js (created in WP3)
- the goniometer plugin at ../gonio if available — copy its good structure, but fix the
  issues listed below.

Layout:

  imswitch-plugin-template/
    README.md
    Makefile
    LICENSE
    plugin/
      __init__.py            register(ctx) -> PluginRegistration
      controller.py          ExampleController(PluginController)
      plugin.toml            manifest
      ui/                    built bundle lands here (gitignored)
    ui-src/
      package.json
      webpack.config.js
      src/index.js
      src/Widget.jsx
    Dockerfile
    .github/workflows/ci.yml
    docs/WRITING_A_PLUGIN.md

Requirements:

BACKEND
- controller.py subclasses PluginController, declares one Event, and exposes three endpoints
  via the SDK's @APIExport: a GET status, a POST that does work, and a GET that returns the
  last result. Show the correct pattern for a long-running operation: run it in a thread and
  emit an Event on completion, never block the endpoint. Add a comment explaining that plugin
  routes go through FastAPI's threadpool for sync defs but that blocking for tens of seconds
  still holds a worker.
- plugin.toml declares one required_hardware entry (detector, role "camera", optional=true)
  so the template loads on a system with no camera.
- __init__.py's register() must return ui_dir=None and rely on the WP2 fallback, OR resolve
  it via importlib.resources — pick the importlib.resources path and add a comment that the
  host falls back to <package>/<dist_dir> if it fails, so both install modes work.
- pyproject.toml must list NO runtime dependencies. Add a comment block naming the packages
  the host provides (numpy, pydantic, opencv, fastapi, imswitch) and stating that adding
  anything to dependencies breaks the bind-mount deployment model. Put dev/test deps under
  [project.optional-dependencies].

FRONTEND
- webpack.config.js: ModuleFederationPlugin with name matching plugin.toml [plugin.ui].scope,
  filename remoteEntry.js, exposes { "./Widget": "./src/Widget.jsx" },
  publicPath "auto", and shared built from the SAME list as frontend/shared-deps.js but with
  eager:false and singleton:true. Add a prominent comment: the host is eager, remotes are
  never eager; eager:true in a remote pulls a second React into the bundle.
- package.json: every shared package goes in peerDependencies, NOT dependencies.
  devDependencies may install them for local dev only.
- Widget.jsx must demonstrate the shared runtime:
    useTheme() from @mui/material for the host theme
    useSelector(s => s.connectionSettingsState) for the backend URL — no hostIP prop
    useDispatch() for a host action
    store.injectReducer for a plugin-owned slice, imported from "host_app/store"
  and declare remotes: { host_app: "host_app@/imswitch/ui/remoteEntry.js" } in webpack.
- src/index.js keeps the async boundary: import("./Widget").

BUILD
- Makefile: `make build` runs the UI build and copies ui-src/dist into plugin/ui/dist;
  `make dist` produces a directory tree ready to bind-mount; `make check` runs the CI checks.
- Dockerfile: multi-stage, final stage FROM scratch containing only the plugin tree, so the
  image can be used as a volume source. Document both usages in comments: (a) docker build
  --output to get a directory, (b) mount the image as a volume.

CI (.github/workflows/ci.yml) — these checks are the point of the template:
- assert pyproject [project].dependencies is empty
- import the plugin package with only the stdlib plus a stub imswitch.plugin_sdk on the path,
  and fail on any ImportError naming a package the host does not provide
- assert no shared package appears in package.json dependencies (only peerDependencies)
- build the UI and assert the output bundle contains no second React: fail if
  "react.production.min" or an equivalent marker appears in the emitted chunks
- validate plugin.toml against the PluginManifest schema
- assert webpack shared list matches frontend/shared-deps.js (vendor a copy and diff it)

docs/WRITING_A_PLUGIN.md: quickstart, the hardware role model, the event model, the
dependency contract and why it exists, how to develop against a running ImSwitch with
webpack dev server, and a troubleshooting table (plugin not in /api/plugins, not in sidebar,
widget hangs, hooks error, theme wrong).
```

**Done when.** `make build && make check` passes in a clean clone, and the produced `dist/`
tree loads in a running ImSwitch container.

---

## WP6 — Docker image and compose integration ✅ done (one part not applicable)

**Delivered.** `Dockerfile` creates `/opt/imswitch/plugins` and sets `IMSWITCH_PLUGIN_DIR`;
`docker/entrypoint.sh` gains `PLUGIN_PATH`, exports `IMSWITCH_PLUGIN_DIR`, and logs the
directory plus a listing of its children (non-fatal when missing — a plugin problem must never
stop a microscope booting); `docker/docker-compose.yml` gets the read-only bind mount,
`PLUGIN_PATH`, the `volume-setup` chown, and the commented-out image-as-volume-source pattern;
`docs/plugins/DEPLOYMENT.md` covers layout, enabling, verification and a symptom-keyed
troubleshooting table.

**Not done: the os-rpi compose file.** That lives in a different repository which is not checked
out here, so I could not edit it without guessing at its current contents — and the prompt
requires everything else in it to stay byte-identical, which is exactly the kind of promise you
cannot keep while editing blind. `docs/plugins/DEPLOYMENT.md` §2 states precisely what to add
(the `:ro` mount, `PLUGIN_PATH`, the `volume-setup` chown entry) and what not to touch
(`device_cgroup_rules`, `group_add`, `extra_hosts`, restart policy, pinned digest, and no
`ports:` section). Applying it is a two-minute change once that repo is available.

**Why.** The deployment target is `os-rpi`'s `deployment.compose.yml`, which currently
bind-mounts config, datasets and `/media` but has no plugin directory. Two delivery modes
are worth supporting: a host directory for development and self-hosting, and a plugin
container used as a volume source for versioned distribution.

Note the compose file has no `ports:` section — networking is handled by the surrounding
OS layer, so do not add port mappings. Also note `volume-setup` chowns bind-mounted
directories because Docker creates them root-owned; the plugin directory needs the same
treatment if it is writable, or should be mounted read-only, which is preferable.

**Files.** `deployments/imswitch.pkg/deployment.compose.yml` in `os-rpi`, `docker/entrypoint.sh` and `Dockerfile` in ImSwitch.

**Prompt.**

```
Goal: make the stock ImSwitch container load bind-mounted plugins, and update the os-rpi
deployment compose file accordingly.

Read first:
- docker/entrypoint.sh — how env vars become CLI params, and the CONFIG_PATH/DATA_PATH
  validation pattern.
- Dockerfile — the uv venv at /opt/imswitch/.venv and how the app is launched.
- imswitch/plugin_manager.py — DROPIN_ENV_VAR ("IMSWITCH_PLUGIN_DIR") and DEFAULT_DROPIN
  ("/opt/imswitch/plugins").
- The os-rpi compose file (deployments/imswitch.pkg/deployment.compose.yml) — note it has
  no ports section and uses a volume-setup service to fix bind-mount ownership.

Implement:

1. Dockerfile: create /opt/imswitch/plugins in the image so the mount point always exists,
   even when nothing is mounted.

2. entrypoint.sh:
   - add PLUGIN_PATH env var, defaulting to /opt/imswitch/plugins
   - export IMSWITCH_PLUGIN_DIR="$PLUGIN_PATH"
   - log the directory and a listing of its immediate children at startup, mirroring the
     existing "Available configuration files" block. When a bind mount is wrong, this log
     line is what tells the operator.
   - do NOT fail startup if the directory is missing or empty; log and continue.

3. Write deployments/imswitch.pkg/deployment.compose.yml as an updated version adding:
   - a read-only bind mount /home/pi/ImSwitchPlugins:/opt/imswitch/plugins:ro
   - PLUGIN_PATH=/opt/imswitch/plugins in environment
   - the plugin directory added to the volume-setup service's chown list so the directory
     exists with the right ownership before the server starts
   Keep everything else — device_cgroup_rules, group_add, extra_hosts, restart policy,
   the pinned image digest — byte-identical. Do not add a ports section.

4. In the same file, add a commented-out second pattern showing plugin delivery via a
   container image used as a volume source:
       plugin-goniometer:
         image: ghcr.io/openuc2/imswitch-plugin-goniometer:0.2.0
         volumes: [ plugins:/out ]
         command: sh -c "cp -a /plugin /out/goniometer"
   with server depends_on service_completed_successfully and a named `plugins` volume.
   Explain in comments when to use which pattern.

5. Add docs/plugins/DEPLOYMENT.md: directory layout the operator must create, how to add a
   plugin, how to verify it loaded (curl /imswitch/api/plugins), and a troubleshooting
   section covering: directory not mounted, plugin name absent from availableWidgets,
   manifest parse failure, and hardware role unmet.
```

**Done when.** On a Pi, dropping the template's `dist/` into `/home/pi/ImSwitchPlugins/` and
adding its name to `availableWidgets` makes the widget appear in the browser after
`docker compose restart server`.

---

## WP7 — End-to-end acceptance test ✅ done

**Delivered.** `imswitch/imcontrol/_test/test_plugin_e2e.py` (7 tests) and
`.github/workflows/plugin-e2e.yml`. Hermetic: no Docker, no hardware, no network, and no wait
on the frontend build artifact, so it fails within a minute if WP1–WP4 regress. The workflow
also asserts the host/remote halves of the shared-deps contract in Node — the host must share
React as an eager singleton, a remote must be non-eager with no fallback copy.

All six ordered assertions from the prompt are covered, plus "an absent plugin directory is
normal, not an error". Total plugin test count: **32** across the three files.

**Why.** This system has many moving parts across two languages and a container boundary.
Without one automated test that exercises the whole chain, it will silently regress the first
time someone touches `craco.config.js` or reorders startup in `ImSwitchServer.run()`. The
test doubles as the definition of "working".

**Files.** `imswitch/imcontrol/_test/test_plugin_e2e.py`, `.github/workflows/plugin-e2e.yml`.

**Prompt.**

```
Goal: one automated test proving the full plugin chain works, from bind-mounted directory to
served frontend bundle.

Build a pytest fixture that:
- creates a temp directory containing a minimal valid plugin (package with register(), a
  plugin.toml, and a fake ui/dist/remoteEntry.js containing a recognisable marker string)
- sets IMSWITCH_PLUGIN_DIR to it
- constructs a PluginManager with a stubbed master and a setup_info whose availableWidgets
  contains the plugin name
- runs discover() and attach_to_app() against a real FastAPI app
- drives it with fastapi.testclient

Assert, in order:
1. GET /imswitch/api/plugins lists the plugin with status "loaded" and the manifest fields
   the frontend consumes: name, remote_entry, scope, exposed, api_base, socket_ns, menu.
2. The plugin's own endpoint under /plugin/<name>/api/... responds.
3. GET /plugin/<name>/ui/remoteEntry.js returns 200 and contains the marker string —
   this is the assertion that proves a bind-mounted frontend is actually served.
4. With the plugin's name removed from availableWidgets, it appears with status "disabled",
   its API route is absent (404), and its controller was never constructed.
5. A plugin whose plugin.toml is malformed appears in the errors array with a readable
   message, and a second valid plugin in the same directory still loads.
6. A plugin declaring a non-optional hardware role that the stub master cannot satisfy is
   reported as an error rather than raising.

Add .github/workflows/plugin-e2e.yml running this on push. Keep it hermetic — no Docker, no
real hardware, no network.
```

**Done when.** The test passes in CI and fails if any of WP1–WP4 is reverted.

---

## WP8 — Partner-facing documentation ✅ done

**Delivered.** `docs/plugins/README.md` (plugin-vs-REST decision table, quickstart, the three
contracts with stability markers, a text lifecycle diagram, a per-surface guarantee table
including an explicit "what we do not guarantee", and a symptom-keyed troubleshooting table);
`docs/INTEGRATION.md` (new — routes a partner to REST client / plugin / upstream contribution,
and replaces fork-the-core with a table of supported alternatives); plus an "Extending ImSwitch"
section in the root `README.md`, without which neither doc is discoverable.

**Two things the docs state plainly rather than glossing.** A plugin runs unsandboxed in the
microscope process — a blocking plugin holds a FastAPI worker, and a duplicated NumPy gives
wrong numbers rather than a crash. And `sdk_min` / `imswitch_min` / `permissions` are parsed but
**not enforced**, so no one should rely on the host rejecting a mismatched plugin.

All internal links and heading anchors across the four plugin docs were verified to resolve.

**Follow-up: developer setup added to DEPLOYMENT.md.** §8 covers running ImSwitch natively
(no Docker) on Windows/macOS/Linux — why `IMSWITCH_PLUGIN_DIR` must be set (the default
`/opt/imswitch/plugins` is a container path), how to set it per-OS, the symlink/junction loop
that lets ImSwitch import straight out of a git checkout, and the Windows-only `_data/static/imswitch`
symlink problem. §9 documents what each Makefile target actually does, with verified bash and
PowerShell equivalents for machines without `make`.

While writing §8 the drop-in layout rules were checked empirically, which surfaced a papercut:
the scanner accepted `<root>/<package>/__init__.py` but the loader then rejected it with a bare
"no python package in ...". That error now names both working layouts and the fix, guarded by
`test_package_directly_under_the_root_gets_an_actionable_error`.

**Why.** This closes the loop on the original question we were asked ("what SDK/API should we
use, and how do we extend the software?"). It also decides the support model: what an external
developer may rely on, and what we reserve the right to change.

**Files.** `docs/plugins/` in ImSwitch, plus the integration guide already drafted for partners.

**Prompt.**

```
Write docs/plugins/README.md as the single entry point for external plugin developers,
linking to the other plugin docs.

Cover:
1. When to write a plugin vs. when to just use the REST API. Decision table: needs in-process
   frame access or sub-100ms hardware loops -> plugin; everything else -> REST client. Be
   honest that a plugin runs in the host process and a plugin bug can take down the
   microscope.
2. Quickstart: clone the template, three files to edit, build, mount, restart.
3. The three contracts, each with a stability marker:
   - Python: imswitch.plugin_sdk only. Everything else is private.
   - Dependencies: host-provided list; adding any is a breaking change to the deployment model.
   - Frontend: the federation shared-module list; peerDependencies only; host is eager,
     remote is not.
4. Lifecycle diagram in text: container start -> entrypoint sets IMSWITCH_PLUGIN_DIR ->
   PluginManager.discover -> manifest parse -> availableWidgets gate -> hardware role
   resolution -> controller construction -> router mount -> UI static mount -> browser fetches
   /imswitch/api/plugins -> federation load -> widget renders inside host Provider/ThemeProvider.
5. What we guarantee and what we do not, per surface.
6. Troubleshooting table keyed by symptom, with the exact command or log line that diagnoses
   each: plugin absent from /api/plugins; present but disabled; present but not in sidebar;
   sidebar entry hangs; hooks error on mount; theme does not match; useSelector throws.

Then update the partner integration guide (docs/INTEGRATION.md if present, otherwise create
it) so the "extending the software" section points at the template repo and describes the
plugin path as the supported extension mechanism, replacing any advice to fork the core.
```

**Done when.** A developer outside openUC2 can go from zero to a rendered widget using only
these docs.

---

## Post-WP8 corrections

Three changes after the first end-to-end use by someone other than the author, where a
correctly-loaded plugin did not appear in the UI.

**1. Selector identity bug (the actual cause).** `selectDynamicApps` and `selectPluginErrors`
returned `state.appManager.dynamicApps || []` — a **fresh array on every call** whenever the
field was absent. `useSelector` compares by reference, so every consumer re-rendered on every
dispatched action and the `useMemo` chain in AppManager was invalidated continuously. The field
is absent exactly when redux-persist rehydrates an `appManager` saved by a build without plugin
support — i.e. on every upgrade, which is the normal path rather than an edge case. Both
selectors now return a single frozen module-level constant.

**2. Plugins were filed into built-in categories.** `makeRegistryEntryFromManifest` used the
manifest's `menu_group` as the category when it named a built-in one. The template ships
`menu_group = "apps"`, so the example plugin landed among the 19 built-in Applications with
nothing marking it as a plugin — and anyone hunting under "Plugins" found an empty tab reading
"No apps found". Plugins now always get `category: "plugins"`; `menu_group` survives as
`menuGroup` display metadata. Recorded in the stable-surface table.

**3. pip-install support removed (ADR-004).** Entry-point discovery is gone: no
`ENTRY_POINT_GROUP`, no `importlib.metadata` import, no entry-point table in the template's
`pyproject.toml`, and `_plugin_root_for` no longer falls back to a module's `__file__`. Directory
scanning is the only mechanism. Guarded by `test_pip_installed_plugins_are_not_discovered`,
which fails if anything consults `entry_points()`.

**Also:** `main(plugin_dir=...)` / `--plugin-dir` now set the plugin directory explicitly,
taking precedence over `$IMSWITCH_PLUGIN_DIR` and then `DEFAULT_DROPIN`. ImSwitch logs the
resolved path at startup (`[main] Plugin folder: ...`). This exists because the default is a
container path, so every native developer previously had to configure a shell variable.

Verified live against the user's own `~/Documents/ImSwitchPlugins/imswitch-plugin-template`:
sidebar shows `PLUGINS › Example`, the App Manager shows a `PLUGINS 1/1` tab and 35 cards with
no empty state, and the widget renders reporting `theme: dark` and its injected Redux slice.

---

## Status summary

| WP | State |
|---|---|
| WP0 audit + ADRs | done — v1 removed entirely (stronger than planned) |
| WP1 `/api/plugins` route | done, 9 tests |
| WP2 drop-in + gating | done, 16 tests; also fixed a `root_path` URL bug not in the defect list |
| WP3 frontend shared runtime | done; deviated from prompt point 4 to avoid breaking redux-persist |
| WP4 registry + navigation | done; verified live in a browser |
| WP5 template repo | done at `../imswitch-plugin-template`, 14 contract checks, negative-tested |
| WP6 docker + compose | done for this repo; the os-rpi compose file is documented, not edited (different repo) |
| WP7 e2e test | done, 7 tests + `plugin-e2e.yml` |
| WP8 documentation | done — `docs/plugins/README.md`, `docs/INTEGRATION.md`, root README section |

32 plugin tests total. Full `imswitch/imcontrol/_test` run: same 26 pre-existing failures as the
`bb4e04f1` baseline, no new ones.

---

## Sequencing and effort

```
WP0  audit + ADRs           ── 0.5 d   no dependencies
WP1  /api/plugins route     ── 0.5 d   unblocks everything
WP2  drop-in + gating       ── 1.5 d   depends on WP1
WP3  frontend shared runtime── 2 d     independent of WP1/WP2, can run in parallel
WP4  registry + navigation  ── 2 d     depends on WP1 and WP3
WP5  template repo          ── 2.5 d   depends on WP3 (shared-deps list)
WP6  docker + compose       ── 1 d     depends on WP2 and WP5
WP7  e2e test               ── 1 d     depends on WP1, WP2, WP6
WP8  documentation          ── 1 d     last
```

Roughly two developer-weeks, or about one week with the backend track (WP1, WP2, WP6, WP7)
and the frontend track (WP3, WP4, WP5) running in parallel. The critical path is
WP1 → WP2 → WP6 → WP7.

**Minimum demo.** WP1 + WP3 + WP4 alone is enough to render the existing goniometer plugin
in the sidebar with the host's theme and store, from a bind mount. That is the milestone
worth showing to a partner before the template repo exists.

**Status: done, and verified live.** The goniometer's `ui/webpack.config.js` was rebuilt on the
shared-deps contract (vendored `shared-deps.js`, `eager: false`, `fallback: false`, `host_app`
remote), its shared packages moved from `dependencies` to `peerDependencies`, and its widget
now takes `apiBase` from the manifest instead of assembling a URL that would have missed the
`/imswitch` prefix. The resulting bundle is 24 KiB with no React in it.

Running against a real backend, the widget renders inside the host shell with **no console
errors** and its calls resolve at `/imswitch/plugin/goniometer/api/get_focus_metric` → 200.

The bundle is built into `src/imswitch_plugin_goniometer/ui/dist/` on this machine but is
**gitignored**, as build artifacts should be — so a fresh clone still needs
`cd ui && npm install && npm run build`, then copy `ui/dist` into the package (the template
automates exactly this as `make build`). Worth adding the same Makefile to the goniometer repo
so the step is not folklore.

---

## Risks

**The two-Reacts failure is invisible until mount time.** It presents as an incomprehensible
hooks error with no indication that federation is involved. The CI bundle check in WP5 is
the mitigation; without it, expect to lose a day to this at least once.

**`redux-persist` and injected reducers.** Plugin slices will not persist. Decide deliberately
(WP3 point 4) rather than discovering it in the field.

**No authentication on plugin loading.** Anything that can write to the mounted plugin
directory can execute arbitrary Python in the ImSwitch process and arbitrary JavaScript in
the operator's browser. Mount read-only, and state plainly in the deployment doc that the
plugin directory is a trust boundary. This is acceptable for a bind mount controlled by the
device owner; it is not acceptable for any future "install a plugin from the UI" feature,
which would need signing.

**`_first_available` hardware fallback.** `_resolve_hardware` silently binds the first
detector when no explicit binding exists. Convenient in development, surprising in production
on a multi-camera system. Consider gating it behind a flag once the template is stable — not
urgent, but note it before someone depends on the behaviour.

**Version skew between host and plugin.** A plugin built against a host with a different
shared-dependency list will fail confusingly. Neither guard exists today: `sdk_min` is parsed
into the manifest but never compared against `SDK_VERSION` (verified in WP0 — see the stable
surface table in `docs/plugins/DECISIONS.md`), and the frontend shared list is not checked at
all. Vendoring `shared-deps.js` into the template and diffing it in
CI (WP5) is a stopgap; publishing it as a small npm package is the real fix if the plugin
ecosystem grows beyond a handful.