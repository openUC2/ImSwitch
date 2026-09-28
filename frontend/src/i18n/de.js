// src/i18n/de.js
// German catalog. Keys are the English source strings exactly as they appear in
// the JSX — anything missing here falls back to English, so partial coverage is
// fine and a new string never renders as a broken key.
//
// {placeholders} must survive translation verbatim; they are filled by t().

const de = {
  // --- shared UI chrome ---
  "Dark Mode": "Dunkles Design",
  Language: "Sprache",

  // --- illumination colours (also used as dynamic keys, hence lower case) ---
  Red: "Rot",
  Green: "Grün",
  Blue: "Blau",
  "White (mean)": "Weiß (Mittelwert)",
  red: "Rot",
  green: "Grün",
  blue: "Blau",
  white: "Weiß",

  // --- header / documentation / overexposure ---
  "Inline Hologram Processing": "Inline-Hologramm-Verarbeitung",
  Detector: "Detektor",
  Documentation: "Dokumentation",
  "Open the HoloBox documentation in a new tab.":
    "Öffnet die HoloBox-Dokumentation in einem neuen Tab.",
  "Reduce by 30%": "Um 30 % verringern",
  "Overexposed: {percent}% of the ROI is clipped. Clipped pixels carry no fringe information — reduce the exposure time (or the illumination) until this clears.":
    "Überbelichtet: {percent} % des ROI sind übersteuert. Übersteuerte Pixel enthalten keine Interferenzinformation — verringern Sie die Belichtungszeit (oder die Beleuchtung), bis die Warnung verschwindet.",

  // --- processing controls ---
  Start: "Start",
  "Start streaming frames from the detector into the hologram reconstruction pipeline.":
    "Beginnt, Bilder vom Detektor in die Hologramm-Rekonstruktion zu streamen.",
  Resume: "Fortsetzen",
  "Resume processing live frames (restores previous binning).":
    "Setzt die Verarbeitung von Live-Bildern fort (stellt das vorherige Binning wieder her).",
  Pause: "Pause",
  "Pause processing — re-reconstructs only the last frame (binning=1) so you can scrub dz/ROI cheaply.":
    "Pausiert die Verarbeitung — nur das letzte Bild wird neu rekonstruiert (Binning=1), damit dz/ROI günstig durchgefahren werden können.",
  Stop: "Stopp",
  "Stop processing entirely and close the worker.":
    "Beendet die Verarbeitung vollständig und schließt den Worker.",
  Refresh: "Aktualisieren",
  "Re-read all hologram parameters from the backend.":
    "Liest alle Hologramm-Parameter erneut vom Backend.",
  Processing: "Verarbeitung läuft",
  Stopped: "Gestoppt",
  Paused: "Pausiert",
  Frames: "Bilder",
  Processed: "Verarbeitet",
  "Stream clients": "Stream-Clients",
  "BG divide ON": "Hintergrunddivision AN",

  // --- tabs ---
  Live: "Live",
  Background: "Hintergrund",
  "Refine (HQ)": "Verfeinern (HQ)",

  // --- camera stream ---
  "Camera Stream": "Kamerabild",
  "Stream not active": "Stream nicht aktiv",
  "Live preview from the detector. Click to set the hologram ROI.":
    "Live-Vorschau des Detektors. Klicken, um den Hologramm-ROI zu setzen.",
  "Click to set ROI center (auto-applies).":
    "Klicken setzt den ROI-Mittelpunkt (wird sofort übernommen).",
  Preview: "Vorschau",
  "Full-frame mode (ROI disabled)": "Vollbildmodus (ROI deaktiviert)",

  // --- processed stream ---
  "Processed Hologram": "Verarbeitetes Hologramm",
  "Reconstructed intensity at the current propagation distance dz. When dz=0 this is just the selected colour channel of the ROI.":
    "Rekonstruierte Intensität bei der aktuellen Propagationsdistanz dz. Bei dz=0 ist das nur der gewählte Farbkanal des ROI.",
  "Show the raw, in-focus hologram (reconstruct at dz=0) instead of the dz set with the slider":
    "Zeigt das rohe, fokussierte Hologramm (Rekonstruktion bei dz=0) statt des per Regler gesetzten dz",
  "Raw (dz=0)": "Roh (dz=0)",
  "Re-open the MJPEG stream (use if the processed view freezes).":
    "Öffnet den MJPEG-Stream neu (falls die verarbeitete Ansicht einfriert).",
  Restart: "Neu starten",
  "Processed stream stalled — no frames for more than {seconds} s.":
    "Verarbeiteter Stream steht — seit über {seconds} s keine Bilder.",
  "dz = 0: showing the extracted {channel} channel of the ROI (no propagation).":
    "dz = 0: zeigt den extrahierten Kanal {channel} des ROI (keine Propagation).",

  // --- dz ---
  "Propagation Distance (dz)": "Propagationsdistanz (dz)",
  "Distance from sensor to virtual image plane. Larger values reconstruct objects farther from the sensor.":
    "Abstand vom Sensor zur virtuellen Bildebene. Größere Werte rekonstruieren Objekte weiter weg vom Sensor.",
  "Max dz (mm)": "Max. dz (mm)",
  "Upper bound of the slider in millimeters.":
    "Obere Grenze des Reglers in Millimetern.",
  "Step (µm)": "Schrittweite (µm)",
  "Slider step size in micrometers.":
    "Schrittweite des Reglers in Mikrometern.",

  // --- detector panel ---
  "Illumination colour": "Beleuchtungsfarbe",
  "Illumination colour. Sets both the RGB channel that is reconstructed and the wavelength used for propagation. 'White' averages all channels.":
    "Beleuchtungsfarbe. Legt sowohl den rekonstruierten RGB-Kanal als auch die Wellenlänge für die Propagation fest. „Weiß“ mittelt alle Kanäle.",
  "Reconstructing at {nm} nm": "Rekonstruktion bei {nm} nm",
  custom: "abweichend",
  "Exposure (ms)": "Belichtung (ms)",
  "Sensor integration time in milliseconds.":
    "Integrationszeit des Sensors in Millisekunden.",
  Gain: "Verstärkung",
  "Analog gain (sensor-dependent units).":
    "Analoge Verstärkung (sensorabhängige Einheiten).",
  "Exposure mode": "Belichtungsmodus",
  "Manual: fixed exposure. Auto: camera adapts exposure each frame.":
    "Manuell: feste Belichtung. Auto: Kamera passt die Belichtung pro Bild an.",
  Manual: "Manuell",
  Auto: "Auto",
  "Exposure Auto Once": "Belichtung einmalig automatisch",
  "Run a single auto-exposure pass, then return to manual.":
    "Führt einen einzelnen Auto-Belichtungsdurchlauf aus und schaltet dann zurück auf manuell.",

  // --- white balance ---
  "White balance": "Weißabgleich",
  "Tip: under a monochromatic laser, leave AWB on Manual with neutral (1.0/1.0) gains. AWB tries to balance the scene to white and pushes the opposite-channel gain way up under a single-colour source, which is what makes a red laser look blue.":
    "Tipp: Bei monochromatischem Laser den Weißabgleich auf Manuell mit neutralen Verstärkungen (1,0/1,0) lassen. Der automatische Weißabgleich versucht, die Szene auf Weiß zu bringen, und dreht bei einfarbiger Beleuchtung die Verstärkung des Gegenkanals stark hoch — deshalb wirkt ein roter Laser dann blau.",
  "AWB mode": "Weißabgleich-Modus",
  "Auto: continuous (bad under laser). Manual: fixed gains. Once: measure now and lock.":
    "Auto: fortlaufend (ungeeignet unter Laser). Manuell: feste Verstärkungen. Einmalig: jetzt messen und festhalten.",
  "Once (lock now)": "Einmalig (jetzt festhalten)",
  "Red gain": "Rot-Verstärkung",
  "Red channel gain. Neutral = 1.0.": "Verstärkung des Rotkanals. Neutral = 1,0.",
  "Blue gain": "Blau-Verstärkung",
  "Blue channel gain. Neutral = 1.0.":
    "Verstärkung des Blaukanals. Neutral = 1,0.",
  "AWB Once": "Weißabgleich einmalig",
  "Run AWB once, lock the resulting gains. Point camera at a white target first.":
    "Führt den Weißabgleich einmal aus und hält die Verstärkungen fest. Zuvor die Kamera auf ein weißes Ziel richten.",
  "Neutral gains (1.0)": "Neutrale Verstärkungen (1,0)",
  "Reset both gains to 1.0 — neutral, no per-channel correction.":
    "Setzt beide Verstärkungen auf 1,0 zurück — neutral, keine kanalweise Korrektur.",

  // --- focus sweep ---
  "Focus Sweep (auto dz)": "Fokus-Sweep (automatisches dz)",
  Running: "Läuft",
  "Start dz (µm)": "Start-dz (µm)",
  "End dz (µm)": "End-dz (µm)",
  "Steps (< 20)": "Schritte (< 20)",
  "Start Sweep": "Sweep starten",
  "Steps dz from start to end (1 step/second) and loops until stopped. Stop keeps the currently active dz. Maximum 19 steps.":
    "Fährt dz von Start bis Ende (1 Schritt/Sekunde) und wiederholt, bis gestoppt wird. Stopp behält das gerade aktive dz. Maximal 19 Schritte.",

  // --- ROI ---
  "ROI Selection": "ROI-Auswahl",
  "Square crop in sensor pixels that gets propagated. Set to full-frame to skip cropping entirely.":
    "Quadratischer Ausschnitt in Sensorpixeln, der propagiert wird. Vollbild überspringt den Zuschnitt komplett.",
  "Full frame": "Vollbild",
  "Bypass the ROI crop and reconstruct the full sensor (with software binning applied).":
    "Umgeht den ROI-Zuschnitt und rekonstruiert den gesamten Sensor (mit Software-Binning).",
  "Reset ROI to image center, size 256px.":
    "Setzt den ROI auf die Bildmitte zurück, Größe 256 px.",
  "Center X (relative to center)": "Mitte X (relativ zur Bildmitte)",
  "Center Y (relative to center)": "Mitte Y (relativ zur Bildmitte)",
  "ROI Size": "ROI-Größe",
  backend: "Backend",
  preview: "Vorschau",
  Scaling: "Skalierung",
  subsampling: "Unterabtastung",
  binning: "Binning",
  "Apply ROI": "ROI übernehmen",

  // --- developer options ---
  "Developer Options": "Entwickleroptionen",
  "Pixel Size (µm)": "Pixelgröße (µm)",
  "Effective sensor pixel size before binning. Binning factor is applied automatically by the propagator.":
    "Effektive Sensor-Pixelgröße vor dem Binning. Der Binning-Faktor wird vom Propagator automatisch berücksichtigt.",
  "Wavelength (nm)": "Wellenlänge (nm)",
  "Illumination wavelength in nanometers. Normally set by the Illumination colour dropdown; override it here to match a measured LED peak.":
    "Beleuchtungswellenlänge in Nanometern. Wird normalerweise über die Auswahl „Beleuchtungsfarbe“ gesetzt; hier überschreiben, um einer gemessenen LED-Wellenlänge zu entsprechen.",
  "Numerical Aperture (NA)": "Numerische Apertur (NA)",
  "Reserved for future band-limiting; currently informational only.":
    "Für eine spätere Bandbegrenzung reserviert; derzeit nur informativ.",
  Binning: "Binning",
  "Software binning factor applied before propagation. Larger = faster, lower resolution.":
    "Software-Binning-Faktor vor der Propagation. Größer = schneller, geringere Auflösung.",
  "Update Frequency (Hz)": "Aktualisierungsrate (Hz)",
  "Target processing rate. Higher = more CPU. The actual rate is bounded by camera fps and reconstruction cost.":
    "Angestrebte Verarbeitungsrate. Höher = mehr CPU-Last. Die tatsächliche Rate ist durch Kamera-Bildrate und Rekonstruktionsaufwand begrenzt.",
  "Flip X": "X spiegeln",
  "Mirror image horizontally before reconstruction.":
    "Spiegelt das Bild vor der Rekonstruktion horizontal.",
  "Flip Y": "Y spiegeln",
  "Mirror image vertically before reconstruction.":
    "Spiegelt das Bild vor der Rekonstruktion vertikal.",
  Rotation: "Drehung",
  "Rotate image counter-clockwise before reconstruction (in degrees).":
    "Dreht das Bild vor der Rekonstruktion gegen den Uhrzeigersinn (in Grad).",

  // --- background tab ---
  "Background Normalization": "Hintergrund-Normalisierung",
  "Divides every live frame by a stored background image. Removes the static illumination envelope, fixed-pattern speckle (dust on fiber tip / slide / sensor glass) and the |R|² pedestal — multiplicative artifacts, so we divide, not subtract. The single biggest cheap win.":
    "Teilt jedes Live-Bild durch ein gespeichertes Hintergrundbild. Entfernt die statische Beleuchtungsverteilung, ortsfeste Speckle (Staub auf Faserende, Objektträger oder Sensorglas) und den |R|²-Sockel — multiplikative Artefakte, deshalb wird geteilt und nicht subtrahiert. Der größte Gewinn bei geringstem Aufwand.",
  "Median burst": "Median-Serie",
  "— capture with the sample in view; moving objects wash out, leaving the static illumination/speckle.":
    "— mit der Probe im Bild aufnehmen; bewegte Objekte mitteln sich heraus, übrig bleiben Beleuchtung und Speckle.",
  Snapshot: "Einzelbild",
  "— for static samples: move the sample out of the FOV first, then capture.":
    "— für statische Proben: die Probe zuerst aus dem Sichtfeld fahren, dann aufnehmen.",
  "Burst frames": "Bilder pro Serie",
  "Acquire (median burst)": "Aufnehmen (Median-Serie)",
  "Acquire (snapshot)": "Aufnehmen (Einzelbild)",
  Clear: "Löschen",
  "Remove the stored background and turn off live division.":
    "Entfernt den gespeicherten Hintergrund und schaltet die Live-Division ab.",
  "Divide out background (live)": "Hintergrund herausrechnen (live)",
  "Divide every live frame by this background. Disabled until a background is acquired.":
    "Teilt jedes Live-Bild durch diesen Hintergrund. Erst verfügbar, wenn ein Hintergrund aufgenommen wurde.",
  "Background stored": "Hintergrund gespeichert",
  "No background": "Kein Hintergrund",
  frames: "Bilder",
  "Background Preview": "Hintergrund-Vorschau",
  "The stored background (current colour channel), downsampled for display.":
    "Der gespeicherte Hintergrund (aktueller Farbkanal), zur Anzeige verkleinert.",
  "No background acquired yet.": "Noch kein Hintergrund aufgenommen.",

  // --- refine tab ---
  "High-Quality Reconstruction": "Hochwertige Rekonstruktion",
  "Iterative single-shot reconstruction. Uses the current dz and (if enabled) the background normalization. Phase retrieval suppresses the twin image; TV-regularized additionally smooths speckle while preserving edges. Takes a few seconds.":
    "Iterative Einzelbild-Rekonstruktion. Nutzt das aktuelle dz und — falls aktiviert — die Hintergrund-Normalisierung. Die Phasenrückgewinnung unterdrückt das Zwillingsbild; die TV-Regularisierung glättet zusätzlich Speckle bei erhaltenen Kanten. Dauert einige Sekunden.",
  "Reconstructs the latest frame at the current dz ({dz} µm)":
    "Rekonstruiert das letzte Bild beim aktuellen dz ({dz} µm)",
  "with background division.": "mit Hintergrunddivision.",
  "Focus dz on the Live tab first.":
    "dz zuerst im Tab „Live“ scharfstellen.",
  Method: "Methode",
  "Phase retrieval (twin-image removal)":
    "Phasenrückgewinnung (Zwillingsbild-Entfernung)",
  "TV-regularized": "TV-reguliert",
  "Phase retrieval": "Phasenrückgewinnung",
  Iterations: "Iterationen",
  iterations: "Iterationen",
  "Reconstructing...": "Rekonstruiere …",
  "Reconstruct (high quality)": "Rekonstruieren (hohe Qualität)",
  Advanced: "Erweitert",
  "Support threshold": "Support-Schwelle",
  "Object-support threshold (0–1). Higher = tighter support (less of the field is treated as object). Used by both methods.":
    "Schwelle für den Objekt-Support (0–1). Höher = engerer Support (weniger des Feldes gilt als Objekt). Wird von beiden Methoden genutzt.",
  "TV weight": "TV-Gewicht",
  "Total-variation regularization strength (TV-regularized method only). Higher = smoother, more speckle suppression, softer edges.":
    "Stärke der Total-Variation-Regularisierung (nur bei der TV-regulierten Methode). Höher = glatter, mehr Speckle-Unterdrückung, weichere Kanten.",
  Reconstruction: "Rekonstruktion",
  Amplitude: "Amplitude",
  Phase: "Phase",
  'Press "Reconstruct (high quality)" to compute.':
    "Auf „Rekonstruieren (hohe Qualität)“ drücken, um zu berechnen.",
};

export default de;
