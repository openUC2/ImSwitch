# Redux State Debugging Guide - LiveStream Format

## Problem
`liveStreamState.currentImageFormat` war `undefined`, was dazu führte, dass der Stream nicht mit dem richtigen Protokoll gestartet wurde.

## Lösung
Drei Änderungen wurden vorgenommen:

### 1. Standardformat beim Laden setzen
```javascript
// In loadBackendSettings()
const currentFormat = isJpegActive ? 'jpeg' : 'binary';
console.log('Setting initial format to:', currentFormat);
dispatch(setImageFormat(currentFormat));
```

### 2. Fallback-Format bei Fehlern
```javascript
// Bei Fehler oder fehlenden Daten
if (!liveStreamState.currentImageFormat) {
  console.log('Setting default format to binary');
  dispatch(setImageFormat('binary'));
}
```

### 3. Sofortige State-Aktualisierung beim Format-Wechsel
```javascript
// Im Stream Format Dropdown onChange
const newFormat = isJpeg ? 'jpeg' : 'binary';
console.log('Format changed to:', newFormat);
dispatch(setImageFormat(newFormat));
```

## Debugging-Schritte

### 1. Redux DevTools öffnen
1. Chrome DevTools öffnen (F12)
2. Zum "Redux" Tab wechseln
3. State-Baum expandieren: `liveStreamState` → `currentImageFormat`

### 2. Initial Load prüfen
```javascript
// Erwarteter State nach dem Laden
{
  liveStreamState: {
    currentImageFormat: 'binary',  // oder 'jpeg'
    streamSettings: {
      binary: {
        enabled: true,
        compression: { algorithm: 'lz4', level: 0 },
        // ...
      },
      jpeg: {
        enabled: false,
        // ...
      }
    }
  }
}
```

### 3. Format-Wechsel prüfen
Schritte:
1. StreamControlOverlay öffnen
2. Settings Tab wählen
3. Stream Format Dropdown öffnen
4. Von "Binary" zu "JPEG" wechseln

**In Redux DevTools beobachten**:
- Action: `liveStreamState/setImageFormat`
- Payload: `"jpeg"`
- State Update: `currentImageFormat: "jpeg"`

**In Browser Console**:
```
Format changed to: jpeg
```

### 4. Submit prüfen
Nach dem Klick auf "Submit":

**Redux DevTools**:
- Action 1: `liveStreamState/setImageFormat` (falls Format geändert)
- Action 2: `liveStreamState/setStreamSettings`
- Action 3: `liveStreamState/setBackendCapabilities`

**Browser Console**:
```
Stream parameters updated successfully for protocol: jpeg
```

**Network Tab**:
```
POST /LiveViewController/setStreamParameters?protocol=jpeg
Request Body: {
  "jpeg_quality": 80,
  "subsampling_factor": 1,
  "throttle_ms": 100
}
Response: {
  "status": "success",
  "protocol": "jpeg",
  "params": {...}
}
```

### 5. Stream Start prüfen
Nach dem Klick auf "Play":

**LiveView.js console logs**:
```javascript
// Wenn currentImageFormat = 'jpeg'
Started jpeg stream

// Wenn currentImageFormat = 'binary' oder undefined
Started binary stream
```

**Network Tab**:
```
POST /LiveViewController/startLiveView?protocol=jpeg
oder
POST /LiveViewController/startLiveView?protocol=binary
```

## Häufige Probleme & Lösungen

### Problem 1: currentImageFormat ist undefined
**Symptom**: Stream startet immer als 'binary', auch wenn JPEG gewählt wurde

**Debug**:
```javascript
// In LiveView.js, vor apiLiveViewControllerStartLiveView
console.log('liveStreamState:', liveStreamState);
console.log('currentImageFormat:', liveStreamState.currentImageFormat);
```

**Erwartete Ausgabe**:
```
liveStreamState: { currentImageFormat: 'jpeg', ... }
currentImageFormat: jpeg
```

**Lösung**: 
- Redux DevTools prüfen, ob `setImageFormat` aufgerufen wurde
- Sicherstellen, dass StreamControlOverlay `dispatch(setImageFormat())` aufruft
- Initial Load Logs prüfen: "Setting initial format to: ..."

### Problem 2: Format ändert sich nicht beim Dropdown-Wechsel
**Symptom**: Dropdown ändert sich, aber Redux State bleibt gleich

**Debug**:
```javascript
// Im Stream Format Dropdown onChange
console.log('Dropdown changed to:', e.target.value);
console.log('isJpeg:', isJpeg);
console.log('newFormat:', newFormat);
```

**Lösung**:
- Prüfen, ob `dispatch(setImageFormat(newFormat))` ausgeführt wird
- Redux DevTools Action Log prüfen
- Sicherstellen, dass keine Fehler in der Console sind

### Problem 3: Submit aktualisiert State nicht
**Symptom**: Submit erfolgreich, aber Redux State unverändert

**Debug**:
```javascript
// In handleSubmitSettings, vor den dispatch Aufrufen
console.log('isJpegMode:', isJpegMode);
console.log('newFormat:', newFormat);
console.log('Will dispatch setImageFormat with:', newFormat);
```

**Lösung**:
- Prüfen, ob `dispatch(setImageFormat(newFormat))` vor dem API-Call ist
- Redux DevTools Action Log prüfen
- Sicherstellen, dass kein Error thrown wird

### Problem 4: Backend erhält falsches Protokoll
**Symptom**: Backend-Logs zeigen anderes Protokoll als erwartet

**Debug**:
```javascript
// In LiveView.js toggleStream
console.log('Current format from Redux:', liveStreamState.currentImageFormat);
console.log('Determined protocol:', protocol);
console.log('Calling startLiveView with protocol:', protocol);
```

**Network Tab prüfen**:
- Request URL sollte `?protocol=jpeg` oder `?protocol=binary` enthalten
- Bei Fehler: `currentImageFormat` ist undefined → Fallback auf 'binary'

## Test-Checkliste

### Initial Load Test
- [ ] App starten
- [ ] Redux State prüfen: `currentImageFormat` ist gesetzt
- [ ] Console Log: "Setting initial format to: binary" (oder jpeg)
- [ ] Kein Fehler in Console

### Format Change Test  
- [ ] StreamControlOverlay öffnen
- [ ] Settings Tab
- [ ] Stream Format zu JPEG ändern
- [ ] Redux State prüfen: `currentImageFormat` = 'jpeg'
- [ ] Console Log: "Format changed to: jpeg"
- [ ] Zurück zu Binary wechseln
- [ ] Redux State prüfen: `currentImageFormat` = 'binary'
- [ ] Console Log: "Format changed to: binary"

### Submit Test
- [ ] Format zu JPEG ändern
- [ ] Quality anpassen (z.B. 90)
- [ ] Submit klicken
- [ ] Redux State prüfen: `currentImageFormat` = 'jpeg'
- [ ] Redux State prüfen: `streamSettings.jpeg.quality` = 90
- [ ] Console Log: "Stream parameters updated successfully for protocol: jpeg"
- [ ] Network Tab: POST request zu `/setStreamParameters?protocol=jpeg`
- [ ] Response: `"status": "success"`
- [ ] Success-Message in UI

### Stream Start Test
- [ ] Stream stoppen (falls läuft)
- [ ] Format auf JPEG setzen und submiten
- [ ] Play klicken
- [ ] Console Log: "Started jpeg stream"
- [ ] Network Tab: POST zu `/startLiveView?protocol=jpeg`
- [ ] Stream läuft
- [ ] Stop klicken
- [ ] Format auf Binary setzen und submiten
- [ ] Play klicken
- [ ] Console Log: "Started binary stream"
- [ ] Network Tab: POST zu `/startLiveView?protocol=binary`
- [ ] Stream läuft

## Entwickler-Notizen

### Redux State Struktur
```javascript
liveStreamState: {
  currentImageFormat: 'binary' | 'jpeg',  // WICHTIG: Muss immer gesetzt sein!
  streamSettings: {
    binary: {
      enabled: boolean,
      compression: { algorithm: string, level: number },
      subsampling: { factor: number },
      throttle_ms: number
    },
    jpeg: {
      enabled: boolean,
      quality: number,
      subsampling: { factor: number },
      throttle_ms: number
    }
  },
  backendCapabilities: {
    binaryStreaming: boolean,
    webglSupported: boolean
  },
  // ... andere Felder
}
```

### Wichtige Dispatch-Aufrufe
```javascript
// Format setzen (immer wenn sich Format ändert!)
dispatch(setImageFormat('binary' | 'jpeg'));

// Settings speichern
dispatch(setStreamSettings(settingsObject));

// Backend-Fähigkeiten setzen
dispatch({
  type: 'liveStreamState/setBackendCapabilities',
  payload: { binaryStreaming: true, webglSupported: true }
});
```

### Reihenfolge beim Submit
1. **ZUERST** Redux State aktualisieren (setImageFormat, setStreamSettings)
2. **DANN** Backend API aufrufen
3. **ZULETZT** Success/Error-Meldung anzeigen

Diese Reihenfolge stellt sicher, dass die UI sofort reagiert, auch wenn der Backend-Call fehlschlägt.

## Console Commands für Quick-Debug

```javascript
// Redux State direkt abfragen (im Browser Console)
window.store.getState().liveStreamState.currentImageFormat

// Manuell Format setzen (für Tests)
window.store.dispatch({ type: 'liveStreamState/setImageFormat', payload: 'jpeg' })

// Kompletten State ausgeben
console.log(JSON.stringify(window.store.getState().liveStreamState, null, 2))
```

**Hinweis**: `window.store` muss in der App exportiert werden:
```javascript
// In index.js oder store.js
window.store = store;
```
