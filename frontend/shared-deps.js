// ─────────────────────────────────────────────────────────────────────────────
//  Module Federation shared-module list — the single source of truth.
//
//  CANONICAL COPY: ImSwitch, frontend/shared-deps.js.
//  Plugins vendor this file verbatim into their ui-src/ and their CI diffs the
//  two. Keep it byte-identical on both sides; if they drift, a plugin either
//  bundles a duplicate of something the host already provides (hooks error at
//  mount time) or asks for something the host does not share (load failure).
//
//  Every package here is loaded ONCE for the whole page and shared between the
//  host and every plugin. That is what lets a plugin call useSelector, useTheme
//  or useDispatch with no props, no bridge object and no import from the host:
//  React resolves context by object identity, so both sides have to be holding
//  the *same* module instance.
//
//  Get it wrong and the failure is invisible until mount time — a plugin that
//  bundles its own react-redux gets its own ReactReduxContext, useSelector finds
//  no Provider, and you get an incomprehensible hooks error with nothing
//  pointing at federation as the cause.
//
//  Plain CommonJS on purpose: this file is required by a webpack config *and*
//  exposed as a federated module.
// ─────────────────────────────────────────────────────────────────────────────

const SHARED_DEPS = [
  // React itself. jsx-runtime is separate and just as important — the automatic
  // JSX transform imports from it directly, so leaving it unshared smuggles in a
  // second React.
  "react",
  "react-dom",
  "react/jsx-runtime",

  // State. Without react-redux here, a plugin's useSelector throws.
  "react-redux",
  "@reduxjs/toolkit",

  // Theme. Without these, a plugin's useTheme() silently returns MUI's default
  // theme instead of the host's — the widget renders, just wrong.
  "@mui/material",
  "@mui/icons-material",
  "@emotion/react",
  "@emotion/styled",

  // Transport and notifications, so a plugin reuses the host's live connection
  // and snackbar stack rather than opening a second socket.
  "socket.io-client",
  "notistack",
];

/**
 * Build a webpack `shared` block from SHARED_DEPS.
 *
 * @param {{eager?: boolean, fallback?: boolean}} options
 *   `eager`    — true ONLY for the host, which is the provider and must have
 *                these in its own bundle at startup. A REMOTE with eager:true
 *                pulls a second React into its bundle: the exact bug this file
 *                exists to prevent.
 *   `fallback` — whether webpack also emits a local copy of each package to use
 *                when the host does not provide it. True for the host. FALSE
 *                for a plugin: there, the fallback *is* the duplicate-React bug,
 *                merely deferred to runtime. With fallback:false a plugin fails
 *                loudly at load if the host is missing a shared module, and the
 *                shell renders that error instead of a subtly broken widget.
 * @returns {Record<string, object>} webpack ModuleFederationPlugin `shared` block
 */
function makeShared({ eager = false, fallback = true } = {}) {
  return SHARED_DEPS.reduce((shared, name) => {
    shared[name] = {
      // singleton: collapse host and plugin copies into one module instance.
      // This is the setting that makes context work across the bundle boundary.
      singleton: true,
      eager,
      // The host's version always wins. Plugins declare these as
      // peerDependencies and must not pin a range, or a harmless minor bump in
      // the host turns into a runtime warning storm.
      requiredVersion: false,
      // undefined leaves webpack's default (emit a fallback); false makes the
      // module consume-only.
      import: fallback ? undefined : false,
    };
    return shared;
  }, {});
}

module.exports = { SHARED_DEPS, makeShared };
