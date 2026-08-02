# newswitch: Docker-installable "Apps" — architecture options & difficulties

*Status: 2026-07 options analysis (not a committed plan). Question: how to ship newswitch as a
Docker-Compose stack where "Apps" (the newswitch analog of ImSwitch's controller/manager/widget
bundle) can be added — via a volume-mounted Python file or a sidecar container — such that the App
adds backend logic **and** renders its UI inside the main frontend, ideally without rebuilding the
base image. Companion to `NEWSWITCH_MIGRATION.md`.*

---

## 0. TL;DR

- The federated model you're describing — **each container = one agent; a central point aggregates
  their actions/states/UI; one frontend** — is the **native design of rekuest/arkitekt**, the
  framework newswitch is built on (same author). It is *already implemented* in the library, but
  the aggregation point is a **central Arkitekt broker** (a separate deployment: Rekuest/Fluss/Lok/
  Kabinet), **not** the base FastAPI app. newswitch today uses only the library's *single-agent,
  self-contained* `contrib.fastapi` mode.
- **Server-driven UI ("bloks") exists in the library and works on the backend** (newswitch even
  registers one, `"jonda"`), but **newswitch has no frontend renderer for bloks** — the blok is
  effectively dead. This is the single most important gap to close, because a blok renderer is what
  lets an App ship UI **without a frontend rebuild**.
- The frontend's **transport layer is already multi-app / multi-backend** (per-app endpoints, one
  WebSocket per app). But the **hooks/UI are code-generated at build time**, and the page layout is
  **hardcoded** — so "drop in an App and its panel appears" needs a runtime path that does not exist
  yet.
- **The hard constraints are physical, not framework:** only one process can own the serial/CAN bus,
  and rekuest **locks are process-local** (a `stage_position` lock is *not* shared across
  containers). Both push you to one rule: **the base app owns the hardware; every App calls base
  actions over the network.** That single rule dissolves most of the difficulty (no cross-process
  DI, no cross-process locks) and makes an "MDA sidecar" clean.
- Recommended target: a **tiered** system — boot-time in-process plugins for trusted first-party
  hardware-adjacent Apps; sidecar-agent containers for isolated/heavy/third-party Apps (browser is
  the aggregator, via the existing multi-app transport); and a **blok renderer + generic
  schema-form fallback** as the no-rebuild UI path, with module-federation reserved for the few
  Apps that need custom visualization.

---

## 1. The two irreducible tensions

Every option below is shaped by two facts that no amount of framework cleverness removes:

**T1 — "Consume base libraries in-process" and "isolated sidecar container" are opposites.**
A volume-mounted `.py` that does `from newswitch.managers import StageManager` and gets the *real,
live* manager injected must run **inside the base interpreter**. A sidecar in its own container
**cannot** import the base's in-memory objects — it can only talk to the base over the network. You
pick one *per App*; you cannot have both for the same App. In-process buys zero-latency direct
access at the cost of isolation; sidecar buys isolation at the cost of an RPC hop.

**T2 — The hardware is a singleton, and locks don't cross processes.**
Exactly one process can hold the UC2 serial port / CAN bus open. And rekuest's `locks=[...]` are
process-local `ContextVar`s (confirmed: `rekuest_next/state/lock.py`; `FastApiAgent.alock` only
*broadcasts* lock state to the UI, it does not coordinate across processes). So two containers both
"holding" `stage_position` would **not** actually exclude each other. Consequence: **hardware must
live in one process (the base app), and Apps that need hardware must call the base app's actions**,
where the base's local locks correctly serialize them. An "MDA App" therefore should contain *no*
hardware driver — it orchestrates by calling `move_stage`, `snap_image`, `set_illumination` on the
base. This is also exactly what rekuest's cross-agent call model (`afind`/`acall`, declared
dependencies) is built to do.

---

## 2. Two axes of choice

An "App" = backend code + frontend UI. Each half has independent options.

### Axis A — where the App's BACKEND runs

| | A1 · In-process (volume-mounted `.py`) | A2 · Sidecar, base-as-gateway | A3 · Sidecar, Arkitekt-broker |
|---|---|---|---|
| Runs in | base container interpreter | own container | own container |
| Gets base libs / live managers | **yes, directly injected** | no — RPC to base | no — RPC via broker |
| Dependency isolation | ✗ (shares base's deps/py-version) | ✓ | ✓ |
| Crash/security isolation | ✗ | ✓ | ✓ |
| Hardware access | direct (in the hardware process) | call base actions | call base actions |
| Cross-App locks | shared (same process) | must route through base | broker-coordinated |
| Aggregation needed | none (same registry) | **build a fan-in gateway** | **already exists** (broker) |
| Compose story | mount a dir, restart | `docker compose up app` | `up` + broker stack |
| Maturity in newswitch | must build (plugins/ is a placeholder) | must build | configure existing (heavy) |

### Axis B — how the App's FRONTEND UI reaches the main UI

| | What it is | Rebuild-free? | Expressiveness | Build cost |
|---|---|---|---|---|
| B1 · Generic schema forms | render any action as a form + any state as a readout, from JSON schema fetched at runtime | ✓ | low (forms/readouts only) | medium — build a `DynamicForm`/`DynamicPanel` + runtime hook materializer |
| B2 · **Bloks** (server-driven UI) | backend ships a JSX-ish component tree (`jsx("<Card>…")`); frontend interprets it against a whitelist, binding `$state.x` / `@action(...)` | ✓ | medium (layout, lists via `<foreach>`, wired to real actions/states) | medium — **build the renderer + component whitelist** (DSL & backend already exist) |
| B3 · Module federation / remote ESM | App ships a *separately-built* React bundle; host loads it and mounts into a slot | ✓ (loads a prebuilt remote) | **full** (custom viz, own npm deps) | **high** — shared-singleton SDK contract, per-App build, versioning, CSP |
| B4 · iframe / web component | sidecar serves its own mini-frontend; host embeds it, bridges theme/auth/ws via postMessage | ✓ | full but *siloed* | medium — clunky integration, double runtime |
| B5 · Rebuild codegen on install | installing an App re-runs `just generate` + frontend build against the new schema | **✗** | full (hand-written panels) | low to build (already works) — but not "drop-in" |

**Prerequisite for all of B1–B4:** a **dynamic panel/slot registry** in the frontend. Today
`src/pages/IndexPage.tsx` is a hardcoded `ResizablePanelGroup` with literal `<StageControl/>` etc.,
and there is no component registry. Apps need somewhere to mount.

---

## 3. What already exists vs. what must be built

Grounded in the code (venv = `rekuest_next`/`arkitekt_next`; repo = newswitch):

**Already there (reusable):**
- Backend *produces* bloks: `rekuest_next/blok/parser.py` (`jsx()`, `BlokParser`), `blok/registry.py`
  (`build_declared_bloks`), `AppRegistry.register_blok`; served at `/schemas/bloks`. newswitch calls
  it once (`app.py` `register_blok("jonda", jsx(...))`).
- Cross-agent RPC & discovery: `rekuest_next/remote.py` (`afind`/`acall`), `postmans/graphql.py`,
  `agents/caller.py` (`AgentPostman`, `AssignRequest`), declared dependencies (`declare.py`,
  `agents/dependency.py`), `AgentMode.ORCHESTRATOR`.
- Agent-dials-broker transport: `RekuestAgent` + `WebsocketAgentTransport` +
  `contrib/arkitekt/ArkitektWebsocketAgentTransport` (Fakts discovery, Herre auth).
- Per-App packaging as a Docker image + deployment registry: `arkitekt_next/cli/commands/plugin/`
  (`build`/`publish`), Kabinet; and a full compose-generated broker in `arkitekt_next/server/`.
- Frontend multi-app transport: `src/lib/rekuest/transport/TransportProvider.tsx` +
  `createScopedProvider.tsx` — **per-app `apiEndpoint`/`wsEndpoint`, one WS per app**. The generator
  (`frontend/plugins/generate-app.ts`) already takes an **array** of apps; only `default` is wired.

**Must be built (absent in newswitch):**
- **Frontend blok renderer + component whitelist** (maps blok tag names → React components, binds
  `$state`/`@action` to runtime hooks, implements `<foreach>`). Nothing in `frontend/src` renders
  bloks; codegen never even fetches `/schemas/bloks`.
- **Runtime (in-browser) schema → hooks/forms** materialization (B1). Codegen is a build-time
  `buildStart` Vite hook writing `.ts` files; there is no runtime path.
- **Dynamic panel/slot registry + non-hardcoded layout** (`IndexPage` is fixed).
- **Backend plugin loading**: dynamic import / directory scan / entry points. `plugins/` is a
  one-line placeholder; `provide_managers` is one-shot at boot; `configure_fastapi` snapshots one
  registry to build routes, so post-boot registration produces no routes.
- **Base-as-gateway fan-in** (A2): ingest sidecar schemas + relay their `/ws` state-patches &
  task-events into the base's single frontend socket. Not provided (`configure_fastapi` = one agent).
- **Cross-process lock coordination** (only needed if you *don't* adopt the "base owns hardware"
  rule; see §1/T2).

---

## 4. Recommended architecture (tiered, self-contained, Pi-friendly)

Chosen to (a) avoid standing up the heavy Arkitekt broker on constrained hardware, (b) honor T1/T2,
(c) reuse the multi-app transport and blok DSL that already exist. Three App tiers:

### Tier 1 — In-process plugin, discovered at boot *(first-party, hardware-adjacent)*
- A mounted `/plugins` directory (compose `volumes:`). At startup, **before** `configure_fastapi`,
  the base scans and imports each module; their `@register`/`@context`/`@state` populate the
  registry; new managers are appended to what `provide_managers` returns.
- Gets base libraries and **live manager injection** for free (T1: this is the whole point of
  in-process). Shares the hardware process, so hardware locks Just Work.
- **Add/remove an App = edit the mounted dir + `docker compose restart base`.** Not hot, but
  compose-native and honest. (Hot registration is possible but fights the boot-time route snapshot —
  defer.)
- Constraint: the plugin must be compatible with the base's Python + rekuest + pydantic versions,
  and any extra dependency must already be in the base image. Curate this tier to first-party.

### Tier 2 — Sidecar agent, browser-aggregated *(isolated / heavy-dep / third-party; e.g. MDA)*
- The App is its own container running its own `contrib.fastapi` agent (own image, own deps). It
  exposes its own `/schemas/*` + `/ws`.
- **The browser is the aggregator** — no backend gateway to build. The frontend registers a second
  app key (the transport already supports per-app endpoints/WS) pointed at the sidecar's URL; its
  panel mounts alongside the base's. This is the lowest-effort path to "UI inside the main UI."
- The sidecar owns **no hardware**. To do an acquisition it **calls the base app's actions** (rekuest
  cross-agent call, or plain HTTP to the base's action routes). The base's local `stage_position`
  lock serializes it against everything else (T2 solved without cross-process locks).
- Compose: `docker compose up mda-app`. Independent versioning and crash isolation.

### Tier 3 — Federated remote UI *(the few Apps needing custom visualization)*
- Only when an App needs real custom viz (a wellplate designer, a 3D stage map) that generic
  forms/bloks can't express. The App ships a **module-federation remote** built against a published
  **frontend SDK** (React/zustand singletons + the runtime transport hooks + theme). Host loads the
  remote into a slot.
- Highest cost (SDK contract, per-App build, CSP). Keep the count tiny; everything else uses
  bloks/forms.

### The UI substrate that makes Tiers 1–2 "no-rebuild"
Build, in order:
1. **Dynamic panel registry + flexible layout** to replace the hardcoded `IndexPage` — Apps declare
   panels; layout is data.
2. **Blok renderer + whitelist** (B2) — the primary way Apps ship UI. A first-party App author writes
   its panel as a `jsx("<Card>…")` string in Python; it renders in the main UI with no frontend PR.
   Works identically for Tier 1 and Tier 2.
3. **Generic schema-form fallback** (B1) — any action/state with no blok still gets an
   auto-generated form/readout. B1+B2 together cover the large majority of Apps with zero frontend
   rebuild.

### End-to-end: an "MDA App" under this design
1. Ship `mda-app` container (Tier 2): a rekuest agent with `@register def run_mda(plan) -> ...` that
   loops channels/z/positions, calling the **base** `move_stage` / `set_illumination` / `snap_image`
   actions and yielding progress via `progress()`/`pausepoint()`. Its acquisition state is a `@state`.
2. It registers a **blok** describing its panel (channel table, z-range, Run button bound to
   `@run_mda(...)`, progress bound to `$MdaState`).
3. Compose adds the service; the frontend is told there's an app at `mda-app:8099`; the blok renderer
   draws the panel inside the main UI. No base image rebuild, no frontend rebuild.
4. On Run, `run_mda` drives the shared stage/camera through the base app, whose `stage_position` lock
   prevents collisions with manual jogging.

---

## 5. The alternative: go fully Arkitekt-native

Instead of a self-contained base app, adopt the framework's intended topology: deploy the **Arkitekt
broker stack** (`arkitekt_next/server/` generates the compose: Rekuest, Fluss, Lok auth, Kabinet
deployment registry, Mikro data). Every App — base hardware included — becomes an agent that dials
into the broker; the broker aggregates schemas, routes cross-agent calls, coordinates reservations,
and (being central) can coordinate locks; bloks are a first-class broker concept; `arkitekt-next
plugin build` packages each App as an image and Kabinet is the "app store."

- **Upside:** you inherit a whole federation + deployment + auth + app-registry system instead of
  building one. Cross-process locks and multi-agent aggregation stop being your problem.
- **Downside:** large operational surface (several new services, a GraphQL broker, Fakts/Herre auth),
  heavier runtime — a real concern on Raspberry-Pi-class deployments — and it inverts newswitch's
  current self-contained, single-`/ws` design. The frontend would target the broker, not the app.
- **When it wins:** multi-instrument fleets, multi-tenant/lab-server deployments, or if you want a
  managed app-store with signed/published plugins. **When it loses:** a single microscope on a Pi in
  a box, which is the openUC2 bread-and-butter — there the tiered self-contained design (§4) is far
  lighter.

A reasonable long-run stance: build §4 now (it's the incremental, Pi-safe path and reuses the same
blok/transport primitives), and keep the door open to §5 by *not* diverging from rekuest's agent/
blok abstractions — so an App written for the self-contained base can later dial a broker unchanged.

---

## 6. Difficulty / risk register

- **Dependency & ABI coupling (Tier 1).** In-process plugins must match the base's exact
  Python/rekuest/pydantic. This fragility is already real in this project (a rekuest
  `Union.__getitem__` bug makes the backend refuse to import on Python 3.14 vs 3.13). Curate Tier 1
  to first-party; push anything with its own/conflicting deps to Tier 2.
- **The frontend build-time wall.** "Drop in without rebuilding" is *only* achievable through a
  **runtime** UI path (bloks or generic forms). Anything relying on the current codegen implies a
  rebuild (B5). Building the blok renderer is the pivot that unlocks the whole vision — prioritize it.
- **Cross-process locks & the hardware singleton (T2).** The safe rule "base owns hardware, Apps call
  it" must be enforced by convention and code review; a sidecar that opens its own serial port would
  silently break mutual exclusion. Consider having the base refuse to share the bus and expose *only*
  action-level hardware access.
- **Latency for tight loops.** Tier-2 Apps pay an RPC hop per hardware op — fine for MDA
  (move/settle/snap), wrong for fast closed-loop control (autofocus PID, focus lock). Keep real-time
  control in-process (Tier 1) or in the base.
- **Publishing an SDK / API contract.** Tier 3 (federation) and any third-party backend turn the
  manager protocols, action schemas, blok component vocabulary, and frontend hook library into a
  **public API** requiring semver discipline — which newswitch's lockstep monorepo versioning
  currently works against. Version the plugin contract separately from the app.
- **Security/trust.** In-process Python and federated JS both execute arbitrary code in the host
  (process / browser page). Acceptable for a trusted first-party lab appliance; for untrusted Apps
  use the sidecar + iframe boundary and a curated blok whitelist (never an unrestricted component
  interpreter).
- **Blok expressiveness ceiling.** The DSL gives layout + lists + action/state binding, not arbitrary
  logic or custom canvases. Know early which Apps exceed it (they go to Tier 3) so you don't try to
  force rich viz through bloks.
- **State/task/lock relay if you build a base-gateway (A2).** Only incur this if you reject the
  browser-as-aggregator approach; the browser-aggregator (Tier 2) avoids the fan-in entirely and is
  strongly preferred as the starting point.

---

## 7. Suggested build order (incremental, each step demoable)

1. **Dynamic panel registry + de-hardcode `IndexPage`.** (Prereq for any App UI.)
2. **Blok renderer + whitelist**, and make codegen/runtime fetch `/schemas/bloks`. Revive the dead
   `"jonda"` blok as the first end-to-end proof.
3. **Boot-time in-process plugin discovery** (`/plugins` dir → import → register before
   `configure_fastapi`). Port one small first-party App (e.g. an objective or utility panel) as a
   blok-only plugin.
4. **Generic schema-form fallback** for action/state with no blok.
5. **Tier-2 sidecar template**: a second container = its own agent, wired as a second frontend app,
   calling base hardware actions. Build the **MDA App** here as the flagship.
6. **(Optional, later)** Module-federation slot for one custom-viz App; and/or evaluate the
   Arkitekt-broker path (§5) if fleet/multi-tenant needs emerge.
