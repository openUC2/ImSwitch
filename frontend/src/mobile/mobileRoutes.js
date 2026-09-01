// Hash-based routing for the kiosk UI. The SPA is served by a plain
// StaticFiles mount (ImSwitchServer.py mounts /ui with no SPA fallback), so a
// real /mobile path would 404 on a hard reload. The hash survives reloads
// everywhere and is what the chromium kiosk service should be pointed at:
//
//   http://<host>:<port>/imswitch/ui/index.html#/mobile
//
// Sub-pages are #/mobile/<page>; anything unknown falls back to the home page.
import { useCallback, useEffect, useState } from "react";

export const MOBILE_HASH = "#/mobile";

export const MOBILE_PAGES = {
  HOME: "home",
  STAGE: "stage",
  LASERS: "lasers",
  LEDS: "leds",
  CAMERA: "camera",
  OBJECTIVE: "objective",
  WIFI: "wifi",
  SYSTEM: "system",
};

// Returns the active kiosk page id, or null when the hash is not a kiosk URL
// (i.e. the regular desktop SPA should render).
export function parseMobilePage() {
  const hash = window.location.hash || "";
  if (hash !== MOBILE_HASH && !hash.startsWith(`${MOBILE_HASH}/`)) {
    return null;
  }
  const rest = hash.slice(MOBILE_HASH.length).replace(/^\//, "");
  const page = rest.split(/[/?#]/)[0];
  return Object.values(MOBILE_PAGES).includes(page) ? page : MOBILE_PAGES.HOME;
}

export function navigateToMobilePage(page) {
  window.location.hash =
    !page || page === MOBILE_PAGES.HOME ? MOBILE_HASH : `${MOBILE_HASH}/${page}`;
}

// Leave the kiosk and return to the full desktop SPA.
export function exitMobileUI() {
  window.location.hash = "#/";
}

export function useMobileRoute() {
  const [page, setPage] = useState(parseMobilePage);

  useEffect(() => {
    const onHashChange = () => setPage(parseMobilePage());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = useCallback((p) => navigateToMobilePage(p), []);

  return { page, navigate };
}
