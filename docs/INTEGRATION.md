# Integrating with ImSwitch

For partners and external developers building on top of ImSwitch.

This document answers one question: **given what you want to build, which
extension point should you use?** Each option below links to its own reference.

---

## Pick an integration path

```
Does your code need to run inside the microscope process —
in-process frame access, or a control loop tighter than ~100 ms?
│
├─ No ─→ Are you adding a UI that must live inside the ImSwitch window?
│        │
│        ├─ No ──→  (1) REST / Socket.IO client        ← start here
│        └─ Yes ─→  (2) Plugin
│
└─ Yes ─→ Are you adding a new DEVICE TYPE (camera, stage, laser driver)?
          │
          ├─ No ──→  (2) Plugin
          └─ Yes ─→  (3) Upstream contribution
```

Most integrations are (1). Reach for (2) when latency or UI placement forces it.

---

## 1. REST / Socket.IO client — the default

Every controller method ImSwitch decorates with `@APIExport` is already an HTTP
endpoint. Nothing to install on the microscope, nothing to keep in version step,
and your code can run anywhere.

```
http://<host>:8001/imswitch/api/docs        Swagger UI, live on any instance
http://<host>:8001/imswitch/openapi.json    machine-readable schema
```

Endpoints follow `/imswitch/api/<Controller>/<method>`. Live data (camera
frames, stage positions, experiment progress) is pushed over Socket.IO on the
same origin.

A Python client is published as [`imswitchclient`](https://pypi.org/project/imswitchclient/).

**Choose this when:** driving acquisitions, moving stages, pulling results,
batch analysis, LIMS or scheduler integration, or anything that should survive
an ImSwitch upgrade untouched.

**Trade-off:** every frame crosses a network boundary. If you need pixels in
process, see below.

### Related surfaces

ImSwitch also speaks several standard protocols, if one already fits your
ecosystem: **SiLA 2** and **Arkitekt** (both surfaced through the setup file as
`availableWidgets` entries), an **OMERO** exporter, and an embedded **Jupyter**
server for scripting against the running instrument.

---

## 2. Plugin — for in-process work and embedded UI

A plugin adds a backend controller and a React widget to a running ImSwitch. You
write two files, build, and drop the result into a directory the container
already watches: **no rebuild of ImSwitch, no `pip install`, no fork.**

Your endpoints appear under `/imswitch/plugin/<name>/api`, your widget in the
sidebar — rendering inside the host's React tree, so it uses the host's MUI
theme, Redux store and socket connection with no props and no bridge object.

**Choose this when:** you need in-process frame access, a sub-100 ms hardware
loop, or a UI that belongs inside the ImSwitch window.

**Trade-off — read this before committing:** a plugin runs in the microscope
process. There is no sandbox. A plugin that blocks holds a worker thread; a
plugin that bundles a second copy of NumPy produces wrong numbers rather than a
clean crash. That is why the plugin contract is strict about dependencies.

| | |
|---|---|
| **Start here** | [docs/plugins/README.md](plugins/README.md) |
| Template repo | [imswitch-plugin-template](https://github.com/openUC2/imswitch-plugin-template) |
| Writing guide | [WRITING_A_PLUGIN.md](https://github.com/openUC2/imswitch-plugin-template/blob/main/docs/WRITING_A_PLUGIN.md) |
| Deploying | [docs/plugins/DEPLOYMENT.md](plugins/DEPLOYMENT.md) |
| Design rationale, stability guarantees | [docs/plugins/DECISIONS.md](plugins/DECISIONS.md) |
| Reference implementation | [imswitch-plugin-goniometer](https://github.com/openUC2/imswitch-plugin-goniometer) |

```bash
git clone https://github.com/openUC2/imswitch-plugin-template my-plugin
cd my-plugin && make install-ui && make build check
```

> **Note for existing integrations.** The older `imswitch.implugins` entry-point
> mechanism has been **removed**. Packages built on it no longer load. The
> migration is a `plugin.toml`, one `register(ctx)` function and a
> `PluginController` — see
> [ADR-001](plugins/DECISIONS.md#adr-001--v2-pluginmanager-is-the-only-plugin-mechanism).

---

## 3. Upstream contribution — for new device types

A plugin deliberately **cannot** register a new `Manager` class. Supporting a new
camera, stage or laser is not a controller concern: it also requires a setup-file
schema change and device instantiation through `MultiManager`, both of which are
host-private surfaces that still move between minor releases. Publishing them
now would freeze a contract we would immediately want to break.

So the driver goes upstream, into the ImSwitch tree, where we maintain it
alongside the rest. The *logic built on top of it* can still live in your plugin.

Open an issue at [ImSwitch issues](https://github.com/openuc2/ImSwitch/issues)
describing the device and we will point you at the right manager base class.

Rationale in full:
[ADR-002](plugins/DECISIONS.md#adr-002--plugins-are-controller-only).

---

## Do not fork the core

Forking ImSwitch to add functionality is not a supported integration path, and
we would rather you did not: a fork stops receiving hardware support, bug fixes
and security updates, and every ImSwitch release widens the gap.

Everything a fork was previously needed for now has a supported route:

| You forked to… | Do this instead |
|---|---|
| add a control panel or analysis view | Plugin (§2) |
| add an endpoint for your own tooling | Plugin (§2), or a REST client (§1) |
| script an acquisition | REST client (§1) or the embedded Jupyter server |
| support a new camera or stage | Upstream contribution (§3) |
| change a default or a setup layout | Setup file — no code change needed |

If you find something none of these covers, that is a gap in the extension
surface and we want to hear about it. Open an issue rather than a fork.

---

## Version compatibility

Plugins declare `imswitch_min` and `sdk_min` in their manifest. **Be aware that
neither is enforced by the host today** — they are recorded but not checked, so
do not rely on ImSwitch rejecting a mismatched plugin. The per-surface stability
table, including what is frozen and what is still provisional, is in
[DECISIONS.md §2](plugins/DECISIONS.md#2-stable-surface).

The REST API surface is generated from the running instance, so the OpenAPI
document at `/imswitch/openapi.json` is always the accurate answer for a given
deployment.
