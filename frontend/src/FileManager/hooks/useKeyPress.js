import { useEffect, useMemo, useRef, useCallback } from "react";

const normalizeKey = (key) => {
  return key.toLowerCase();
};

export const useKeyPress = (keys, callback, disable = false) => {
  const lastKeyPressed = useRef(new Set([]));
  const keysSet = useMemo(() => {
    return new Set(keys.map((key) => normalizeKey(key)));
  }, [keys]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.repeat) return; // To prevent this function from triggering on key hold e.g. Ctrl hold

      // Don't prevent default for browser zoom shortcuts (Cmd/Ctrl + Plus/Minus/0)
      const isZoomShortcut =
        (e.metaKey || e.ctrlKey) &&
        (e.key === "+" || e.key === "-" || e.key === "=" || e.key === "0");

      if (isZoomShortcut) {
        return; // Let browser handle zoom shortcuts
      }

      lastKeyPressed.current.add(normalizeKey(e.key));

      if (keysSet.isSubsetOf(lastKeyPressed.current) && !disable) {
        e.preventDefault();
        callback(e);
        return;
      }
    },
    [keysSet, callback, disable]
  );

  const handleKeyUp = useCallback((e) => {
    lastKeyPressed.current.delete(normalizeKey(e.key));
  }, []);

  const handleBlur = useCallback(() => {
    lastKeyPressed.current.clear();
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
    };
  }, [handleKeyDown, handleKeyUp, handleBlur]);
};
