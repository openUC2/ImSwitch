import { createContext, useContext, useState, useEffect } from "react";

const PWAContext = createContext(null);

export function PWAProvider({ children }) {
  const [installPromptEvent, setInstallPromptEvent] = useState(null);

  useEffect(() => {
    const handleBeforeInstallPrompt = (event) => {
      console.log("[PWA] beforeinstallprompt event received, saving...");
      event.preventDefault(); // Prevent the mini infobar from appearing on mobile
      setInstallPromptEvent(event);
    };

    const handleAppInstalled = () => {
      console.log("[PWA] App installed, clearing prompt event");
      setInstallPromptEvent(null);
    };

    window.addEventListener("beforeinstallprompt", handleBeforeInstallPrompt);
    window.addEventListener("appinstalled", handleAppInstalled);

    return () => {
      window.removeEventListener(
        "beforeinstallprompt",
        handleBeforeInstallPrompt,
      );
      window.removeEventListener("appinstalled", handleAppInstalled);
    };
  }, []);

  return (
    <PWAContext.Provider value={{ installPromptEvent, setInstallPromptEvent }}>
      {children}
    </PWAContext.Provider>
  );
}

export function usePWA() {
  const context = useContext(PWAContext);
  if (!context) {
    console.warn(
      "usePWA must be used within a PWAProvider - installPromptEvent will be unavailable",
    );
    return { installPromptEvent: null, setInstallPromptEvent: () => {} };
  }
  return context;
}
