# Deploying ImSwitch plugins

A plugin is a directory. You put it somewhere ImSwitch can see, name it in the
instrument's setup file, and restart. There is no `pip install`, no rebuild of
the ImSwitch image, and no second copy of any library.

- **§1–7 — deploying into the Docker container.** For operators.
- **[§8 — running without Docker](#8-running-without-docker-native-development).**
  Plain Python on Windows, macOS or Linux. This is the loop you want while
  *developing* a plugin, and the default plugin directory is a container path,
  so it needs one environment variable set.
- **[§9 — building without `make`](#9-building-a-plugin-without-make).** What the
  template's Makefile actually does, as bash and PowerShell.

If you are writing a plugin, also read the template's
[WRITING_A_PLUGIN.md](https://github.com/openUC2/imswitch-plugin-template/blob/main/docs/WRITING_A_PLUGIN.md).

---

## 1. Directory layout

Create the plugin directory on the host:

```bash
mkdir -p /home/pi/ImSwitchPlugins
```

One subdirectory per plugin. Each must contain a Python package — a directory
with `__init__.py` that defines `register()`:

```
/home/pi/ImSwitchPlugins/
├── goniometer/
│   └── src/
│       └── imswitch_plugin_goniometer/
│           ├── __init__.py          register(ctx)
│           ├── controller.py
│           ├── plugin.toml          the manifest
│           └── ui/dist/
│               └── remoteEntry.js   the built widget
└── example/
    └── imswitch_plugin_example/
        ├── __init__.py
        ├── plugin.toml
        └── ui/dist/remoteEntry.js
```

Both shapes work: the host looks for `<plugin>/src/<package>/` first, then
`<plugin>/<package>/`. `make dist` in the template produces the second.

## 2. Wiring the mount

Already present in `docker/docker-compose.yml`:

```yaml
    volumes:
      - /home/pi/ImSwitchPlugins:/opt/imswitch/plugins:ro
    environment:
      - PLUGIN_PATH=/opt/imswitch/plugins
```

`PLUGIN_PATH` is exported to the app as `IMSWITCH_PLUGIN_DIR`. The default is
already `/opt/imswitch/plugins`, so you only need the variable if you mount
somewhere else.

> **`:ro` is not optional.** Anything that can write to this directory executes
> arbitrary Python in the microscope process and arbitrary JavaScript in the
> operator's browser. This directory is a trust boundary. It is fine for a mount
> controlled by the device owner; it is not a mechanism for installing plugins
> from the internet.

### os-rpi deployments

`deployments/imswitch.pkg/deployment.compose.yml` in the `os-rpi` repo needs the
same two additions — the read-only bind mount and `PLUGIN_PATH` — plus
`/home/pi/ImSwitchPlugins` added to the `volume-setup` service's `chown` list so
the directory exists with sane ownership before the server starts.

Leave everything else in that file byte-identical: `device_cgroup_rules`,
`group_add`, `extra_hosts`, the restart policy and the pinned image digest.
**Do not add a `ports:` section** — networking is handled by the surrounding OS
layer.

## 3. Enable the plugin

A mounted plugin does not load until the instrument's setup file says so. Edit
`/home/pi/ImSwitchConfig/imcontrol_setups/<your-setup>.json` and add the
plugin's `name` (from its `plugin.toml`, *not* the directory name) to
`availableWidgets`:

```json
{
  "availableWidgets": ["Settings", "View", "Positioner", "goniometer"]
}
```

This gate is deliberate: one plugin directory can be mounted across a fleet
while each instrument decides which plugins are live. `"availableWidgets": true`
enables everything.

## 4. Restart and verify

```bash
docker compose restart server
```

```bash
curl -s http://localhost:8001/imswitch/api/plugins | python3 -m json.tool
```

A healthy plugin:

```json
{
  "name": "goniometer",
  "status": "loaded",
  "remote_entry": "/imswitch/plugin/goniometer/ui/remoteEntry.js",
  "api_base": "/imswitch/plugin/goniometer/api",
  "reason": ""
}
```

The startup log prints one line per plugin, which is usually faster:

```bash
docker compose logs server | grep -iE "plugin|PLUGIN_PATH"
```

```
[2026-08-03 09:12:04] Using PLUGIN_PATH: /opt/imswitch/plugins
[2026-08-03 09:12:04] Available plugins:
drwxr-xr-x 3 1000 1000 4096 Aug  3 09:10 goniometer
[PluginManager]   plugin goniometer   v0.2.0   status=loaded   source=dropin:/opt/imswitch/plugins/goniometer
```

## 5. Delivering plugins as images instead

For a fleet, pinning the plugin version in the compose file beats rsync. The
plugin ships as a `FROM scratch` image containing only its tree, and copies
itself into a named volume before the server starts:

```yaml
services:
  plugin-goniometer:
    image: ghcr.io/openuc2/imswitch-plugin-goniometer:0.2.0
    volumes:
      - plugins:/out
    command: sh -c "cp -a /plugin /out/goniometer"

  server:
    depends_on:
      plugin-goniometer:
        condition: service_completed_successfully
    volumes:
      - plugins:/opt/imswitch/plugins:ro

volumes:
  plugins:
```

| | Bind mount | Image as volume source |
|---|---|---|
| Update | `rsync` + restart | change the tag, `docker compose up -d` |
| Version recorded | nowhere | in the compose file |
| Needs a registry | no | yes |
| Best for | development, one instrument | a fleet |

Do not use both for the same plugin — a bind mount over `/opt/imswitch/plugins`
hides the named volume.

---

## 6. Troubleshooting

Start here every time:

```bash
curl -s http://localhost:8001/imswitch/api/plugins | python3 -m json.tool
```

### The plugin is not in the list at all

The host never saw it. Check the mount landed:

```bash
docker compose exec server ls -la /opt/imswitch/plugins
```

Empty or missing → the bind mount is wrong. Check the host path exists and
matches the compose file exactly.

Present but still not listed → the directory does not contain an importable
package. The host needs `<plugin>/src/<pkg>/__init__.py` or
`<plugin>/<pkg>/__init__.py`:

```bash
docker compose exec server find /opt/imswitch/plugins -name '__init__.py'
```

### It is listed with `"status": "disabled"`

Working as designed — read the `reason` field. Almost always: its `name` is not
in `availableWidgets`. Note it is the **manifest name**, not the directory name:

```bash
docker compose exec server \
  grep -A2 '^\[plugin\]' /opt/imswitch/plugins/*/src/*/plugin.toml
```

### It appears in the `errors` array

The `error` string names the cause.

- `ValidationError` / `TypeError` from parsing — `plugin.toml` is malformed or
  missing a required key. Validate it against the schema:
  ```bash
  docker compose exec server python -c \
    "from imswitch.plugin_sdk import load_manifest; print(load_manifest('/opt/imswitch/plugins/<dir>/src/<pkg>/plugin.toml'))"
  ```
- `required hardware not available: detector:camera` — the plugin declared a
  non-optional role the setup file cannot fill. Either add the device, or add an
  explicit binding, or ask the plugin author to mark the role optional.
- `plugin 'x' already loaded from ...` — two plugins claim the same name. First
  one wins; the second is skipped. Rename one.
- `ModuleNotFoundError: No module named 'foo'` — the plugin has an undeclared
  dependency the host image does not provide. This is a plugin bug: report it.
  Do not `pip install` into the container, which breaks on the next image pull.

### `/api/plugins` shows it as loaded, but the UI shows no plugin

The backend is fine; the problem is in the browser. In order of likelihood:

1. **Stale frontend.** The plugin UI arrived in a specific ImSwitch version. If
   the served bundle predates it, nothing will ever render plugins. Rebuild
   (`npm run build` in `frontend/`) or pull a newer image.
2. **Stale persisted Redux state.** The App Manager's state is saved in the
   browser's `localStorage`. An entry saved by an older build lacks the plugin
   fields. Hard-reload, and if it persists, clear the entry:
   ```js
   // browser devtools console
   localStorage.removeItem("persist:root"); location.reload();
   ```
3. **Looking in the wrong category.** Plugins are always filed under
   **Plugins**, regardless of the `menu_group` in their manifest. If the App
   Manager is filtered to another category you will see "No apps found".
   Select "All Apps".

### It loads but there is no sidebar entry

Check `remote_entry` in the manifest output. If it is `null`, the plugin has a
backend but no built frontend — nothing was found in its `ui/dist`. The App
Manager lists it under "Plugins not available" as *no widget*. Ask the author to
ship a built bundle.

If `remote_entry` is set but the entry still does not appear, check the App
Manager: the plugin may simply be toggled off there.

### The sidebar entry shows an error card

The card names the plugin, the URL and the reason. Common ones:

| Message | Cause |
|---|---|
| `script could not be fetched (network/404)` | `remoteEntry.js` is not where the manifest says. `curl http://host:8001<remote_entry>` |
| `did not register federation scope "x"` | The plugin's `plugin.toml` scope and its webpack config disagree — a plugin bug |
| `does not expose "./Widget"` | Same, for `exposed` |
| `timed out after 10s` | The bundle is being served but never finishes loading; check the network tab |

### Hardware binding is not what you expected

When no explicit binding exists, the host binds the **first available device** of
the right kind. On a multi-camera instrument that is rarely what you want. Pin
it in the setup file:

```json
{
  "plugin_bindings": { "detector:camera": "WidefieldCamera" }
}
```

---

## 7. Security summary

- The plugin directory is a **trust boundary**. Mount it `:ro`.
- A plugin runs in the microscope process. A plugin that crashes badly enough
  can take the instrument down; the frontend is isolated by an error boundary,
  the backend is not.
- There is no signing and no sandbox. Install plugins you would be willing to
  run as arbitrary code on that machine, because that is what they are.

---

## 8. Running without Docker (native development)

This is the loop you want while writing a plugin: no container, no image
rebuild, no `rsync` — edit the file, restart ImSwitch, reload the browser.

### 8.1 Why you must set the plugin directory

It defaults to a **container path**:

```python
# imswitch/plugin_manager.py
DEFAULT_DROPIN = "/opt/imswitch/plugins"
```

`/opt/imswitch/plugins` does not exist on a Windows or macOS workstation, and
almost certainly not on a Linux one outside the container. Discovery then finds
nothing and — deliberately — says nothing louder than a log line, because an
absent plugin directory is the normal case for most installs.

So when running natively, **always say where your plugins are**. Three ways,
highest precedence first:

| | How | Use when |
|---|---|---|
| 1 | `main(plugin_dir=...)` or `--plugin-dir` | **Preferred.** Everything in one place, no shell state |
| 2 | `$IMSWITCH_PLUGIN_DIR` | Docker, CI, or a shell you have already configured |
| 3 | `DEFAULT_DROPIN` | The container default; leave it alone |

**The explicit parameter is the simplest:**

```python
from imswitch.__main__ import main

main(
    default_config=r"C:\Users\me\Documents\ImSwitchConfig\imcontrol_setups\example_virtual_microscope.json",
    plugin_dir=r"C:\Users\me\Documents\ImSwitchPlugins",
    ssl=0,
)
```

or from the command line:

```bash
python -m imswitch --plugin-dir ~/ImSwitchPlugins --http-port 8001 --no-ssl
```

ImSwitch logs the resolved directory at startup, so you can confirm it took:

```
[main] Plugin folder: C:\Users\me\Documents\ImSwitchPlugins
```

> **Do not edit `DEFAULT_DROPIN` in the source** to point at your own machine.
> It will end up committed, break the Docker image (whose entrypoint expects the
> container path), and silently override everyone else's setup.

### 8.2 Pick a directory

Anywhere you can write. Suggested:

| OS | Suggested path |
|---|---|
| Windows | `%USERPROFILE%\ImSwitchPlugins` → `C:\Users\<you>\ImSwitchPlugins` |
| macOS | `~/ImSwitchPlugins` |
| Linux | `~/ImSwitchPlugins` |

```bash
mkdir -p ~/ImSwitchPlugins                       # macOS / Linux
```
```powershell
New-Item -ItemType Directory "$env:USERPROFILE\ImSwitchPlugins" -Force   # Windows
```

### 8.3 Setting it via the environment instead

Only needed if you prefer it to `plugin_dir=` — for a shell you configure once,
or for Docker and CI where there is no `main()` call to edit.

**Per session:**

```bash
export IMSWITCH_PLUGIN_DIR=~/ImSwitchPlugins        # bash / zsh
```
```powershell
$env:IMSWITCH_PLUGIN_DIR = "$env:USERPROFILE\ImSwitchPlugins"   # PowerShell
```
```bat
set IMSWITCH_PLUGIN_DIR=%USERPROFILE%\ImSwitchPlugins           :: cmd.exe
```

**Persistently:**

```bash
echo 'export IMSWITCH_PLUGIN_DIR=~/ImSwitchPlugins' >> ~/.zshrc   # macOS
echo 'export IMSWITCH_PLUGIN_DIR=~/ImSwitchPlugins' >> ~/.bashrc  # Linux
```
```powershell
# Windows, current user, survives reboots. Open a NEW terminal afterwards.
[Environment]::SetEnvironmentVariable(
  "IMSWITCH_PLUGIN_DIR", "$env:USERPROFILE\ImSwitchPlugins", "User")
```

(Prefer `main(plugin_dir=...)` from §8.1 over any of these — it needs no shell
configuration and is visible in the code that starts the server.)

### 8.4 The directory layout that actually works

The host scans the **immediate children** of `$IMSWITCH_PLUGIN_DIR`, and inside
each looks for a Python package **one level down**. Both of these load:

```
$IMSWITCH_PLUGIN_DIR/
├── my-plugin/                        ← any name, dashes fine
│   └── imswitch_plugin_mine/         ← the package: valid Python identifier
│       ├── __init__.py               ← must define register(ctx)
│       ├── plugin.toml
│       └── ui/dist/remoteEntry.js
└── goniometer/
    └── src/                          ← a src/ layout works too
        └── imswitch_plugin_goniometer/
            ├── __init__.py
            └── plugin.toml
```

This one **does not** load:

```
$IMSWITCH_PLUGIN_DIR/
└── imswitch_plugin_mine/             ← package placed directly under the root
    ├── __init__.py
    └── plugin.toml
```

```
FileNotFoundError: no python package in .../imswitch_plugin_mine: expected
imswitch_plugin_mine/<package>/__init__.py or imswitch_plugin_mine/src/<package>/__init__.py
```

The extra level is not ceremony: the outer directory is named by whoever
deploys it (often with dashes, which are not importable), while the package name
has to be a valid Python identifier and unique across every plugin in the
process.

### 8.5 Develop in place with a symlink

Do not copy your plugin into the directory on every change — link it, and the
running ImSwitch imports straight out of your git checkout.

```bash
# macOS / Linux
ln -s ~/code/my-plugin ~/ImSwitchPlugins/my-plugin
```
```powershell
# Windows. A junction needs no admin rights and no developer mode.
New-Item -ItemType Junction `
  -Path "$env:USERPROFILE\ImSwitchPlugins\my-plugin" `
  -Target "$env:USERPROFILE\code\my-plugin"
```

Verified working: with a junction pointing at a goniometer checkout, discovery
resolves the package, the manifest and `ui/dist` inside the live repo.

Python does not hot-reload, so **restart ImSwitch after a backend change**. A
frontend-only change needs a rebuild of the plugin bundle plus a browser reload
(hard-reload, or the old `remoteEntry.js` may be cached).

### 8.6 Run it

```bash
uv run python main.py --headless --http-port 8001
```

Then check the plugin was seen:

```bash
curl -s http://localhost:8001/imswitch/api/plugins | python -m json.tool
```

Remember the [`availableWidgets` gate](#3-enable-the-plugin): a discovered
plugin reports `"status": "disabled"` until its **manifest name** is listed in
the setup file you launched with. Setup files live in
`~/ImSwitchConfig/imcontrol_setups/` (`%USERPROFILE%\ImSwitchConfig\...` on
Windows).

### 8.7 Serving the frontend natively

`/imswitch/ui` is served from `imswitch/_data/static/imswitch`, which in the git
repo is a **symlink** to `frontend/build`. On Windows a default clone checks that
symlink out as a 24-byte text file, so the mount silently fails and you get no
UI at all:

```
Could not mount /imcontrol ui static files since directory is missing/a symlink.
```

Fix it once, either by enabling symlinks:

```bash
git config core.symlinks true    # then re-checkout that path
```

or by replacing it with a junction:

```powershell
$link = "<repo>\imswitch\_data\static\imswitch"
if (Test-Path $link) { (Get-Item $link).Delete() }
New-Item -ItemType Junction -Path $link -Target "<repo>\frontend\build"
```

Either way `frontend/build` must exist — build it with `npm run build` in
`frontend/` (on Windows run it as `npx craco build`, since the repo's `build`
script uses Unix-style `VAR=value` prefixing that `cmd.exe` cannot parse).

---

## 9. Building a plugin without `make`

`make build` is three commands in a trench coat. Windows has no `make` by
default, so here is exactly what each target does and how to run it by hand.
Paths assume the template's names — substitute your own package.

| Target | What it actually does |
|---|---|
| `make install-ui` | `npm install` in `ui-src/` |
| `make build` | `npm run build` in `ui-src/`, then replace `<package>/ui/dist` with `ui-src/dist` |
| `make check` | `python scripts/check_contract.py` |
| `make dist` | `make build`, then copy `<package>` into `dist/<plugin-name>/` and strip `__pycache__` |
| `make clean` | delete `dist/`, `build/`, `ui-src/dist/`, `<package>/ui/dist/`, `__pycache__` |

**bash / zsh** (macOS, Linux, Git Bash):

```bash
PACKAGE=imswitch_plugin_example
PLUGIN_NAME=example

# make install-ui
(cd ui-src && npm install)

# make build
(cd ui-src && npm run build)
rm -rf "$PACKAGE/ui/dist"
mkdir -p "$PACKAGE/ui"
cp -r ui-src/dist "$PACKAGE/ui/dist"

# make check
python scripts/check_contract.py

# make dist
rm -rf dist
mkdir -p "dist/$PLUGIN_NAME"
cp -r "$PACKAGE" "dist/$PLUGIN_NAME/$PACKAGE"
find dist -name '__pycache__' -type d -prune -exec rm -rf {} +
```

**PowerShell** (Windows):

```powershell
$PACKAGE = "imswitch_plugin_example"
$PLUGIN_NAME = "example"

# make install-ui
Push-Location ui-src; npm install; Pop-Location

# make build
Push-Location ui-src; npm run build; Pop-Location
Remove-Item -Recurse -Force "$PACKAGE\ui\dist" -ErrorAction SilentlyContinue
New-Item -ItemType Directory "$PACKAGE\ui" -Force | Out-Null
Copy-Item -Recurse "ui-src\dist" "$PACKAGE\ui\dist"

# make check
python scripts\check_contract.py

# make dist
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
New-Item -ItemType Directory "dist\$PLUGIN_NAME" -Force | Out-Null
Copy-Item -Recurse $PACKAGE "dist\$PLUGIN_NAME\$PACKAGE"
Get-ChildItem dist -Recurse -Directory -Filter __pycache__ |
  Remove-Item -Recurse -Force
```

The only step that genuinely matters is **build**: the manifest's
`[plugin.ui].dist_dir` (default `ui/dist`) is resolved relative to the *package*
directory, so the built bundle has to end up at `<package>/ui/dist/`. Everything
else is packaging convenience.

Skipping `make dist` is fine during development — point
`$IMSWITCH_PLUGIN_DIR` at a directory containing a symlink to your repo
(§8.5) and the plugin loads from `<repo>/<package>/` directly.
