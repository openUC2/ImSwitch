// src/utils/snapPreview.js
//
// Push a single captured frame into the live viewport.
//
// When the live stream is stopped — which is the normal state for long
// exposures, where streaming is disabled — the viewport canvas has nothing to
// draw and stays blank after a snap. The backend can return the captured frame
// as a PNG data URL (`snapImageToPath?returnPreview=true`), and this helper
// hands it to the viewer through the same imperative CustomEvent channel the
// live JPEG frames use, so the image appears exactly where the stream would be.
//
// Kept out of the Redux path on purpose: like the live frames, the payload is a
// full image and would bloat every persisted state snapshot.

export const SNAP_PREVIEW_EVENT = "uc2:snap-preview";

/**
 * Display a captured frame in the live viewport.
 *
 * @param {string} dataUrl - `data:image/png;base64,...` from the snap response
 * @returns {boolean} true if the frame was dispatched
 */
export function showSnapPreview(dataUrl) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:image/")) {
    return false;
  }
  if (typeof window === "undefined" || typeof window.dispatchEvent !== "function") {
    return false;
  }
  window.dispatchEvent(
    new CustomEvent(SNAP_PREVIEW_EVENT, { detail: { dataUrl } }),
  );
  return true;
}

export default showSnapPreview;
