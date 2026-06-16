/**
 * frameBus — a lightweight, Redux-independent channel for live preview
 * frames.
 *
 * WHY THIS EXISTS
 * Dispatching every JPEG frame into the global Redux store
 * (`setLiveViewImage`) creates a new `liveStreamState` object 16×/sec,
 * which re-renders EVERY component subscribed to that slice (~20 of
 * them, including the react-zoom-pan-pinch wrapper). That re-render
 * cascade saturates the browser main thread, so the live <img>'s
 * `onload` callback is delayed by ~180 ms even though the frame itself
 * is a tiny 640×480 JPEG that decodes in ~1 ms. MJPEG is smooth purely
 * because it's a raw <img> that never touches Redux/React per frame.
 *
 * frameBus restores that property for the socket.io JPEG path: the
 * viewer subscribes here and paints imperatively (no React render, no
 * Redux dispatch), so the 16 fps stream stays entirely off the React
 * commit path. Redux is still updated, but at a throttled rate, only
 * for the secondary thumbnail consumers and the "stream has an image"
 * UI state.
 */

let latestUrl = null;
const subscribers = new Set();

export const frameBus = {
  /** Publish the newest frame URL (blob: or data:) to all subscribers. */
  publish(url) {
    latestUrl = url;
    for (const cb of subscribers) {
      try {
        cb(url);
      } catch (e) {
        // A misbehaving subscriber must not break the stream.
        // eslint-disable-next-line no-console
        console.error("frameBus subscriber error:", e);
      }
    }
  },

  /** Most recent frame URL, or null if none yet (for late subscribers). */
  getLatest() {
    return latestUrl;
  },

  /** Subscribe to frames. Returns an unsubscribe function. */
  subscribe(cb) {
    subscribers.add(cb);
    return () => subscribers.delete(cb);
  },
};

export default frameBus;
