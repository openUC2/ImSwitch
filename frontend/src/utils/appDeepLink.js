// src/utils/appDeepLink.js
// Deep link straight into one app, so a QR code stuck on the instrument opens it:
//
//   http://192.168.4.1/imswitch/ui/index.html?app=holo
//
// The SPA is served by a plain StaticFiles mount (no SPA fallback), so a real
// path like /app/holo would 404 on a hard reload — only index.html plus a query
// string or hash survives. The kiosk UI already uses the hash (#/mobile), so the
// query string is what is left for us.
//
// The value is matched loosely against id / pluginId / name / keywords, ignoring
// case and separators, so a printed QR code keeps working when we rename things:
// ?app=holo, ?app=inline, ?app=holoController and ?app=Hologram%20Processing all
// land on the same app.

import { APP_REGISTRY } from "../constants/appRegistry";

const normalize = (value) => String(value ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");

export function resolveApp(value) {
  const want = normalize(value);
  if (!want) return null;

  const apps = Object.values(APP_REGISTRY);
  // Most specific match first; keywords are shared between related apps
  // (e.g. "holography"), so they lose against an explicit id/pluginId/name.
  return (
    apps.find((app) => normalize(app.id) === want) ||
    apps.find((app) => normalize(app.pluginId) === want) ||
    apps.find((app) => normalize(app.name) === want) ||
    apps.find((app) => (app.keywords || []).some((k) => normalize(k) === want)) ||
    null
  );
}

// Reads ?app= from the query string, falling back to a query inside the hash
// (".../index.html#/?app=holo") so the parameter survives either URL shape.
export function readAppParam(href = window.location.href) {
  const [, search = "", hash = ""] = /^[^?#]*(\?[^#]*)?(#.*)?$/.exec(href) || [];
  const fromSearch = new URLSearchParams(search).get("app");
  if (fromSearch) return fromSearch;
  const hashQuery = hash.slice(hash.indexOf("?") + 1);
  return hash.includes("?") ? new URLSearchParams(hashQuery).get("app") : null;
}

// The app requested by the current URL, or null.
export function getDeepLinkApp(href = window.location.href) {
  return resolveApp(readAppParam(href));
}
