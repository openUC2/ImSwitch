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
  "Touch UI": "Touch-Bedienung",
  "Touchscreen detected": "Touchscreen erkannt",
  "No touchscreen detected": "Kein Touchscreen erkannt",
  "Number pad": "Ziffernblock",

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
  "Slider step size in micrometers, at most a tenth of Max dz.":
    "Schrittweite des Reglers in Mikrometern, höchstens ein Zehntel von Max dz.",

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

  // ---- Hologram apps: snapshot, simulation, build guides, fringe trace
  "Off-axis controller not reachable: {msg}": "Off-Axis-Controller nicht erreichbar: {msg}",
  "Off-Axis Holography": "Off-Axis-Holografie",
  "Fringe trace": "Streifenverlauf",
  Simulate: "Simulieren",
  Learn: "Lernen",
  "Freeze the current frame; dz and sideband changes still apply to it.":
    "Friert das aktuelle Bild ein; Änderungen an dz und Seitenband wirken weiterhin darauf.",
  "Find the brightest side peak in the spectrum and select it.":
    "Sucht den hellsten Seitenpeak im Spektrum und wählt ihn aus.",
  "Find sideband": "Seitenband finden",
  "Re-read all parameters from the backend.": "Alle Parameter neu vom Backend lesen.",
  "Fringe period {p} px": "Streifenperiode {p} px",
  "1. Click the camera image to place the square on the fringes. 2. Click one of the two side peaks in the spectrum (or Find sideband). 3. Drag dz to refocus.":
    "1. Ins Kamerabild klicken, um das Quadrat auf die Streifen zu setzen. 2. Einen der beiden Seitenpeaks im Spektrum anklicken (oder „Seitenband finden“). 3. dz ziehen, um nachzufokussieren.",
  "Camera - click to place the square": "Kamera – klicken, um das Quadrat zu setzen",
  "Spectrum - click a side peak": "Spektrum – einen Seitenpeak anklicken",
  "Refocus dz": "Nachfokussieren dz",
  "dz (µm)": "dz (µm)",
  "Unwrap phase": "Phase entfalten",
  "Show the cropped spectrum": "Ausgeschnittenes Spektrum zeigen",
  "Settings - optics, region, sideband, apodization": "Einstellungen – Optik, Bereich, Seitenband, Apodisierung",
  "Pixel size (µm)": "Pixelgröße (µm)",
  "Colour channel": "Farbkanal",
  "Square size (px)": "Quadratgröße (px)",
  "Sideband X": "Seitenband X",
  "Sideband Y": "Seitenband Y",
  "Crop width": "Ausschnittbreite",
  "Crop height": "Ausschnitthöhe",
  FFT: "FFT",
  "Apodization (soft crop edge)": "Apodisierung (weicher Ausschnittrand)",
  Window: "Fenster",
  "Reading the trace failed: {msg}": "Lesen des Verlaufs fehlgeschlagen: {msg}",
  "Could not start the trace: {msg}": "Verlauf konnte nicht gestartet werden: {msg}",
  "Put the camera where the fringes are (instead of the screen). Click a pixel on a fringe flank, start, then warm one arm with your hand or push the table lightly: every fringe that passes the pixel is one bright-dark-bright cycle.":
    "Die Kamera dorthin stellen, wo die Streifen sind (statt des Schirms). Ein Pixel auf einer Streifenflanke anklicken, starten, dann einen Arm mit der Hand erwärmen oder leicht gegen den Tisch drücken: Jeder Streifen, der am Pixel vorbeiläuft, ist ein Hell-Dunkel-Hell-Zyklus.",
  "Pixel at {x}% / {y}% of the frame": "Pixel bei {x} % / {y} % des Bildes",
  "Click the live image to choose the pixel.": "Ins Livebild klicken, um das Pixel zu wählen.",
  "sensor pixel": "Sensorpixel",
  "Average over": "Mittelung über",
  "Start trace": "Verlauf starten",
  "Intensity change at the pixel": "Intensitätsänderung am Pixel",
  "time (s)": "Zeit (s)",
  "I − I(0) (counts)": "I − I(0) (Counts)",
  "Dotted lines: the counting thresholds. Export the plot with the camera icon above it.":
    "Gepunktete Linien: die Zählschwellen. Den Plot über das Kamerasymbol darüber exportieren.",
  Interferometer: "Interferometer",
  "Michelson (mirror moves λ/2 per fringe)": "Michelson (Spiegel bewegt sich λ/2 pro Streifen)",
  "Mach-Zehnder (path changes λ per fringe)": "Mach-Zehnder (Weg ändert sich um λ pro Streifen)",
  Hysteresis: "Hysterese",
  "Fringes counted": "Gezählte Streifen",
  "Mirror moved": "Spiegel bewegt um",
  "Path length changed": "Weglänge geändert um",
  Samples: "Messpunkte",
  "per s": "pro s",
  "N fringes = N × {f} × λ. One pixel cannot tell the direction: a mirror that moves forth and back counts both ways. Fringes faster than ~1 per 20 frames are missed - slow down or shorten the exposure.":
    "N Streifen = N × {f} × λ. Ein Pixel kennt die Richtung nicht: Ein Spiegel, der hin und zurück fährt, wird in beide Richtungen gezählt. Streifen schneller als ~1 pro 20 Bilder werden verpasst – langsamer bewegen oder die Belichtung verkürzen.",
  "Save as PNG": "Als PNG speichern",
  source: "Quelle",
  sample: "Probe",
  sensor: "Sensor",
  "primary wave": "Primärwelle",
  "scattered wave": "Streuwelle",
  "object wave (through the sample)": "Objektwelle (durch die Probe)",
  "reference wave": "Referenzwelle",
  "drawn larger": "vergrößert gezeichnet",
  "In-line hologram": "In-Line-Hologramm",
  "Off-axis hologram": "Off-Axis-Hologramm",
  "Everything here is computed in your browser - no microscope needed.":
    "Alles hier wird in deinem Browser berechnet – kein Mikroskop nötig.",
  Sample: "Probe",
  "One point scatterer": "Ein punktförmiger Streuer",
  "Dust particles": "Staubteilchen",
  "Transparent cells": "Durchsichtige Zellen",
  "Letters (UC2)": "Buchstaben (UC2)",
  Absorption: "Absorption",
  "Phase delay": "Phasenverzögerung",
  "Light and camera": "Licht und Kamera",
  Wavelength: "Wellenlänge",
  "Pixel size": "Pixelgröße",
  Geometry: "Geometrie",
  "Source to sample z1": "Quelle bis Probe z1",
  "Sample to sensor z2": "Probe bis Sensor z2",
  "Source size": "Quellgröße",
  "A fibre-coupled laser is a few µm wide, an LED chip 100-300 µm. A bigger source washes out the outer rings - that is spatial coherence.":
    "Ein fasergekoppelter Laser ist wenige µm breit, ein LED-Chip 100–300 µm. Eine größere Quelle verwischt die äußeren Ringe – das ist räumliche Kohärenz.",
  "Reference wave": "Referenzwelle",
  "Angle between the waves": "Winkel zwischen den Wellen",
  "Fringe direction": "Streifenrichtung",
  "Sample out of focus by": "Probe defokussiert um",
  "Sideband crop radius": "Radius des Seitenbandausschnitts",
  "Magnification M = {m} · the wave travels like a plane wave over z_eff = {zeff} µm · field of view {fov} µm":
    "Vergrößerung M = {m} · die Welle läuft wie eine ebene Welle über z_eff = {zeff} µm · Bildfeld {fov} µm",
  "1 · Sample (transmission)": "1 · Probe (Transmission)",
  "What we want to see.": "Das wollen wir sehen.",
  "2 · Scattered wave at the sensor": "2 · Streuwelle am Sensor",
  "Only the light the sample scattered.": "Nur das Licht, das die Probe gestreut hat.",
  "3 · Hologram = |primary + scattered|²": "3 · Hologramm = |Primär- + Streuwelle|²",
  "What the camera records: rings, no phase.": "Das nimmt die Kamera auf: Ringe, keine Phase.",
  "4 · Reconstruction at dz": "4 · Rekonstruktion bei dz",
  "Back-propagated by the slider below.": "Zurückpropagiert um den Wert des Schiebers unten.",
  "Reconstruction distance dz": "Rekonstruktionsabstand dz",
  "Jump to focus": "In den Fokus springen",
  "Drag dz: the rings collapse into the sample at dz = z_eff. The faint halo that stays is the twin image - the camera lost the phase, so the conjugate wave is reconstructed too. At dz = -z_eff you focus the twin instead.":
    "dz ziehen: Bei dz = z_eff fallen die Ringe zur Probe zusammen. Der schwache Halo, der bleibt, ist das Zwillingsbild – die Kamera hat die Phase verloren, deshalb wird auch die konjugierte Welle rekonstruiert. Bei dz = -z_eff fokussierst du stattdessen den Zwilling.",
  "Fringe period {p} px. The camera needs at least 2 px per fringe (better 3-4); bigger angles alias.":
    "Streifenperiode {p} px. Die Kamera braucht mindestens 2 px pro Streifen (besser 3–4); größere Winkel erzeugen Aliasing.",
  "The fringes are finer than the pixels can sample - reduce the angle.":
    "Die Streifen sind feiner, als die Pixel abtasten können – den Winkel verkleinern.",
  "1 · Sample phase": "1 · Phase der Probe",
  "Dust only absorbs. Pick cells to see a phase object.":
    "Staub absorbiert nur. Wähle Zellen, um ein Phasenobjekt zu sehen.",
  "A transparent sample only delays the light.": "Eine durchsichtige Probe verzögert das Licht nur.",
  "2 · Hologram (zoom in on the fringes)": "2 · Hologramm (in die Streifen hineinzoomen)",
  "Straight fringes, bent where the sample delays the light.":
    "Gerade Streifen, verbogen dort, wo die Probe das Licht verzögert.",
  "3 · Spectrum - click a sideband": "3 · Spektrum – ein Seitenband anklicken",
  "Centre: the plain intensities. The two side peaks carry the object wave and its twin.":
    "Mitte: die reinen Intensitäten. Die beiden Seitenpeaks tragen die Objektwelle und ihren Zwilling.",
  "4 · Reconstructed amplitude": "4 · Rekonstruierte Amplitude",
  "5 · Reconstructed phase": "5 · Rekonstruierte Phase",
  "Back to the true sideband": "Zurück zum echten Seitenband",
  "One sideband gives amplitude AND phase from a single image - no twin image. Set dz to the defocus to refocus. The other sideband is the twin: its phase is inverted and it focuses at -dz. A crop that touches the centre lets the plain intensity leak in.":
    "Ein Seitenband liefert Amplitude UND Phase aus einem einzigen Bild – ohne Zwillingsbild. dz auf die Defokussierung setzen, um nachzufokussieren. Das andere Seitenband ist der Zwilling: Seine Phase ist invertiert und er fokussiert bei -dz. Ein Ausschnitt, der die Mitte berührt, lässt die reine Intensität einsickern.",
  "Could not read this image ({msg}). Use PNG or JPG.":
    "Dieses Bild konnte nicht gelesen werden ({msg}). PNG oder JPG verwenden.",
  "Snap failed: {msg}": "Aufnahme fehlgeschlagen: {msg}",
  "Snap from camera": "Von der Kamera aufnehmen",
  "Upload image": "Bild hochladen",
  "{name} · {w} × {h} px · click the image to move the square":
    "{name} · {w} × {h} px · ins Bild klicken, um das Quadrat zu verschieben",
  "Snap a frame or upload a PNG/JPG hologram. The reconstruction runs in this browser.":
    "Ein Bild aufnehmen oder ein PNG/JPG-Hologramm hochladen. Die Rekonstruktion läuft in diesem Browser.",
  "Mean of R, G, B": "Mittelwert aus R, G, B",
  "Divide by smooth background": "Durch glatten Hintergrund teilen",
  "Crop radius": "Ausschnittradius",
  "Full image - click to choose the region": "Ganzes Bild – klicken, um den Bereich zu wählen",
  "Spectrum - click a sideband": "Spektrum – ein Seitenband anklicken",
  "Reconstructed phase": "Rekonstruierte Phase",
  "Reconstructed amplitude": "Rekonstruierte Amplitude",
  "Distance dz": "Abstand dz",
  "Slider max (µm)": "Schieber-Maximum (µm)",
  "The phase is shown wrapped to ±π. A sideband centre that is slightly off adds a tilt; click the peak more precisely or use Find sideband.":
    "Die Phase wird auf ±π gefaltet gezeigt. Ein leicht verschobenes Seitenbandzentrum fügt eine Verkippung hinzu; den Peak genauer anklicken oder „Seitenband finden“ nutzen.",
  "Drag dz until the rings collapse into sharp objects. dz is the sample-to-sensor distance (for a point source far away). Use the pixel size of the image as saved - a binned or resized image has bigger pixels.":
    "dz ziehen, bis die Ringe zu scharfen Objekten zusammenfallen. dz ist der Abstand Probe–Sensor (bei weit entfernter Punktquelle). Die Pixelgröße des gespeicherten Bildes verwenden – ein gebinntes oder verkleinertes Bild hat größere Pixel.",
  Michelson: "Michelson",
  "Mach-Zehnder (off-axis)": "Mach-Zehnder (Off-Axis)",
  "Full guide in the documentation": "Vollständige Anleitung in der Dokumentation",
  "Cube layout (top view)": "Würfelanordnung (Draufsicht)",
  "You need": "Du brauchst",
  "LED / laser": "LED / Laser",
  empty: "leer",
  "sample + camera": "Probe + Kamera",
  "fibre (laser)": "Faser (Laser)",
  "beam splitter": "Strahlteiler",
  "mirror (kinematic)": "Spiegel (kinematisch)",
  mirror: "Spiegel",
  "screen / camera": "Schirm / Kamera",
  "beam splitter 1": "Strahlteiler 1",
  "beam splitter 2": "Strahlteiler 2",
  camera: "Kamera",
  "In-line holography: a lensless microscope": "In-Line-Holografie: ein Mikroskop ohne Linse",
  "A point of light, a sparse sample and a camera - no lens. The light that passes the sample unchanged (primary wave) interferes with the light the sample scatters. The camera records the rings; the computer refocuses them.":
    "Ein Lichtpunkt, eine dünn besetzte Probe und eine Kamera – keine Linse. Das Licht, das unverändert an der Probe vorbeigeht (Primärwelle), interferiert mit dem Licht, das die Probe streut. Die Kamera nimmt die Ringe auf, der Computer fokussiert sie.",
  "LED cube (Holo target) or fibre-coupled red laser in its fibre holder":
    "LED-Würfel (Holo-Target) oder fasergekoppelter roter Laser im Faserhalter",
  "Raspberry Pi camera module with the sample holder": "Raspberry-Pi-Kameramodul mit Probenhalter",
  "a sparse sample: dust, pollen or fine sand on a coverslip":
    "eine dünn besetzte Probe: Staub, Pollen oder feiner Sand auf einem Deckglas",
  "base plate and puzzle pieces, 2 empty cubes as spacers":
    "Grundplatte und Puzzleteile, 2 leere Würfel als Abstandshalter",
  "Start with an empty base plate.": "Beginne mit einer leeren Grundplatte.",
  "Put the light source cube at one end. The light has to travel straight along the row.":
    "Setze den Lichtquellenwürfel an ein Ende. Das Licht muss gerade entlang der Reihe laufen.",
  "Add the camera cube at the other end. Fix every cube with puzzle pieces on top and bottom so the row cannot bend.":
    "Setze den Kamerawürfel ans andere Ende. Befestige jeden Würfel oben und unten mit Puzzleteilen, damit sich die Reihe nicht verbiegt.",
  "Put the sample into the camera cube, as close to the sensor as you can - almost touching. A short sample-to-sensor distance gives clear rings even with the partially coherent LED.":
    "Setze die Probe in den Kamerawürfel, so nah wie möglich an den Sensor – fast berührend. Ein kurzer Abstand Probe–Sensor gibt klare Ringe, auch mit der teilkohärenten LED.",
  "Switch the light on, cover the setup against room light and set the exposure to about 1 ms. You should see shadows with rings around them.":
    "Schalte das Licht ein, decke den Aufbau gegen Raumlicht ab und stelle die Belichtung auf etwa 1 ms. Du solltest Schatten mit Ringen darum sehen.",
  "Dust is the best first sample: small, sparse, high contrast.":
    "Staub ist die beste erste Probe: klein, dünn verteilt, kontrastreich.",
  "Washed-out image: stray light. Cover the setup with a box.":
    "Flaues Bild: Streulicht. Decke den Aufbau mit einer Schachtel ab.",
  "Saturated (white or coloured) spots carry no information: reduce the exposure.":
    "Übersteuerte (weiße oder farbige) Stellen tragen keine Information: Belichtung verringern.",
  "Then open the Live tab, choose the colour of your light and drag dz until the rings collapse into sharp objects.":
    "Öffne dann den Live-Tab, wähle die Farbe deines Lichts und ziehe dz, bis die Ringe zu scharfen Objekten zusammenfallen.",
  "Michelson interferometer: watch light cancel out": "Michelson-Interferometer: Licht löscht sich aus",
  "A beam splitter sends the laser to two mirrors; both beams come back and overlap. Where they arrive in step they add up, out of step they cancel. A change of the mirror distance by half a wavelength moves the pattern by one fringe - you measure distances of 0.3 µm.":
    "Ein Strahlteiler schickt den Laser zu zwei Spiegeln; beide Strahlen kommen zurück und überlagern sich. Wo sie im Gleichtakt ankommen, addieren sie sich, im Gegentakt löschen sie sich aus. Ändert sich der Spiegelabstand um eine halbe Wellenlänge, wandert das Muster um einen Streifen – du misst Abstände von 0,3 µm.",
  "red laser (650 nm) with fibre and fibre holder cube": "roter Laser (650 nm) mit Faser und Faserhalterwürfel",
  "beam splitter cube": "Strahlteilerwürfel",
  "two mirror cubes, at least one kinematic (adjustable)":
    "zwei Spiegelwürfel, mindestens einer kinematisch (justierbar)",
  "screen, or the Raspberry Pi camera instead of the screen":
    "Schirm, oder die Raspberry-Pi-Kamera statt des Schirms",
  "1.5 mm hex screwdriver, base plate, puzzle pieces": "1,5-mm-Inbusschlüssel, Grundplatte, Puzzleteile",
  "Put the fibre holder cube on the plate. Do not switch the laser on yet.":
    "Setze den Faserhalterwürfel auf die Platte. Den Laser noch nicht einschalten.",
  "Place the beam splitter right after it. It splits the beam into two at 90°.":
    "Setze den Strahlteiler direkt dahinter. Er teilt den Strahl in zwei im 90°-Winkel.",
  "Add a mirror at each output of the beam splitter. Check that both mirrors sit at 90° to the beam (open the cube and turn the insert if not).":
    "Setze an jeden Ausgang des Strahlteilers einen Spiegel. Prüfe, dass beide Spiegel im 90°-Winkel zum Strahl stehen (sonst den Würfel öffnen und den Einsatz drehen).",
  "Put the screen on the fourth side of the beam splitter.":
    "Setze den Schirm an die vierte Seite des Strahlteilers.",
  "Switch the laser on. You will see two dots, one per mirror. Turn the screws of the kinematic mirror, slowly, until the dots lie exactly on top of each other.":
    "Schalte den Laser ein. Du siehst zwei Punkte, einen pro Spiegel. Drehe langsam an den Schrauben des kinematischen Spiegels, bis die Punkte genau übereinanderliegen.",
  "Add a lens in front of the beam splitter (or replace the screen by the camera). The overlapping spots turn into rings or stripes.":
    "Setze eine Linse vor den Strahlteiler (oder ersetze den Schirm durch die Kamera). Die überlagerten Flecken werden zu Ringen oder Streifen.",
  "Done. Touch the table or warm one arm with your hand and watch the fringes move. Count them with the Fringe trace tab.":
    "Fertig. Berühre den Tisch oder erwärme einen Arm mit der Hand und beobachte, wie die Streifen wandern. Zähle sie im Tab „Streifenverlauf“.",
  "Laser safety: never look into the beam or its reflections; keep it below eye level.":
    "Lasersicherheit: nie in den Strahl oder seine Reflexe schauen; den Strahl unter Augenhöhe halten.",
  "One screw at a time, small turns.": "Immer nur eine Schraube, kleine Drehungen.",
  "With the camera: few wide fringes = mirrors almost parallel. Tilt one mirror a little for straight, evenly spaced stripes.":
    "Mit der Kamera: wenige breite Streifen = Spiegel fast parallel. Einen Spiegel leicht kippen für gerade, gleichmäßige Streifen.",
  "Mach-Zehnder: the off-axis hologram setup": "Mach-Zehnder: der Aufbau für Off-Axis-Hologramme",
  "Two separate paths that meet again at a second beam splitter. One path carries the sample (object wave), the other is the reference. Meeting under a small angle they write straight fringes onto the camera; the sample bends them. One image then gives amplitude and phase.":
    "Zwei getrennte Wege, die sich an einem zweiten Strahlteiler wieder treffen. Ein Weg führt durch die Probe (Objektwelle), der andere ist die Referenz. Treffen sie sich unter einem kleinen Winkel, schreiben sie gerade Streifen auf die Kamera; die Probe verbiegt sie. Ein Bild liefert dann Amplitude und Phase.",
  "two beam splitter cubes": "zwei Strahlteilerwürfel",
  "two mirror cubes set to 45°, one kinematic": "zwei Spiegelwürfel auf 45°, einer kinematisch",
  "Raspberry Pi camera (or a screen to start)": "Raspberry-Pi-Kamera (oder zu Beginn ein Schirm)",
  "optional: polarizer to equalise the two arms": "optional: Polarisator, um beide Arme anzugleichen",
  "Place the fibre holder cube on the plate.": "Setze den Faserhalterwürfel auf die Platte.",
  "The first beam splitter divides the light into two paths.":
    "Der erste Strahlteiler teilt das Licht auf zwei Wege auf.",
  "Set both mirrors to 45° and place them so each path turns by 90° towards the far corner.":
    "Stelle beide Spiegel auf 45° und setze sie so, dass jeder Weg um 90° zur gegenüberliegenden Ecke abknickt.",
  "The second beam splitter brings both paths together. Put the screen behind it.":
    "Der zweite Strahlteiler führt beide Wege zusammen. Setze den Schirm dahinter.",
  "Turn the kinematic mirror until both spots overlap on the screen. Make both arms equally bright (block one, then the other; a polarizer helps) for the best contrast.":
    "Drehe am kinematischen Spiegel, bis sich beide Flecken auf dem Schirm überlagern. Mache beide Arme gleich hell (einen, dann den anderen abdecken; ein Polarisator hilft) für den besten Kontrast.",
  "Fringes appear. Replace the screen by the camera: now you can see fine stripes of a few pixels and align much faster.":
    "Streifen erscheinen. Ersetze den Schirm durch die Kamera: Jetzt siehst du feine Streifen von wenigen Pixeln und justierst viel schneller.",
  "Put a thin transparent sample into ONE arm. In the Live tab the spectrum shows two side peaks: click one (or Find sideband) to get amplitude and phase.":
    "Bringe eine dünne durchsichtige Probe in EINEN Arm. Im Live-Tab zeigt das Spektrum zwei Seitenpeaks: einen anklicken (oder „Seitenband finden“), um Amplitude und Phase zu erhalten.",
  "Aim for 3-5 pixels per fringe. Finer fringes alias, coarser ones push the side peaks into the centre.":
    "Ziel: 3–5 Pixel pro Streifen. Feinere Streifen erzeugen Aliasing, gröbere schieben die Seitenpeaks in die Mitte.",
  "Keep everything rigid: every cube on puzzle pieces, on a solid table.":
    "Alles steif halten: jeder Würfel auf Puzzleteilen, auf einem stabilen Tisch.",
  "The beam splitters are not polarisation-neutral - rotate the polarizer if the contrast is poor.":
    "Die Strahlteiler sind nicht polarisationsneutral – bei schlechtem Kontrast den Polarisator drehen.",
};

export default de;
