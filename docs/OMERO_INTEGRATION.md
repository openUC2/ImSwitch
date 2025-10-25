# OMERO Integration in ImSwitch ExperimentManager

## Übersicht

Der ExperimentManager wurde erweitert, um OMERO-Server-Konfigurationen direkt über die Setup-Konfigurationsdatei zu laden, ähnlich wie andere Manager in ImSwitch (z.B. MCTManager).

## Konfiguration

### Setup-Konfigurationsdatei

Fügen Sie einen `experiment`-Abschnitt zu Ihrer Setup-JSON-Datei hinzu:

```json
{
  "detectors": { ... },
  "lasers": { ... },
  "positioners": { ... },
  "experiment": {
    "omeroServerUrl": "omero.example.com",
    "omeroUsername": "researcher",
    "omeroPassword": "secret123",
    "omeroPort": 4064,
    "omeroGroupId": 1,
    "omeroProjectId": 100,
    "omeroDatasetId": 200,
    "omeroEnabled": true,
    "omeroConnectionTimeout": 30,
    "omeroUploadTimeout": 600
  },
  "availableWidgets": ["Image", "Laser", "Positioner", "Experiment"]
}
```

### OMERO-Parameter

- **omeroServerUrl**: URL des OMERO-Servers
- **omeroUsername**: Benutzername für die Authentifizierung
- **omeroPassword**: Passwort für die Authentifizierung
- **omeroPort**: Server-Port (Standard: 4064)
- **omeroGroupId**: Gruppen-ID für Uploads (-1 für Standard-Gruppe)
- **omeroProjectId**: Projekt-ID für Uploads (-1 für kein spezifisches Projekt)
- **omeroDatasetId**: Dataset-ID für Uploads (-1 für kein spezifisches Dataset)
- **omeroEnabled**: Ob OMERO-Integration aktiviert ist
- **omeroConnectionTimeout**: Verbindungs-Timeout in Sekunden
- **omeroUploadTimeout**: Upload-Timeout in Sekunden

## API-Endpunkte

### GET /experiment/omero/config
Aktuelle OMERO-Konfiguration abrufen.

### POST /experiment/omero/config
OMERO-Konfiguration aktualisieren.

```json
{
  "serverUrl": "omero.new-server.com",
  "username": "new_user",
  "isEnabled": true
}
```

### GET /experiment/omero/enabled
Prüfen, ob OMERO-Integration aktiviert ist.

### GET /experiment/omero/connection-params
OMERO-Verbindungsparameter abrufen (Passwort wird aus Sicherheitsgründen maskiert).

## Verwendung im Code

```python
# OMERO-Konfiguration abrufen
config = experiment_manager.getOmeroConfig()

# Prüfen, ob OMERO aktiviert ist
if experiment_manager.isOmeroEnabled():
    # Verbindungsparameter für OMERO-Client abrufen
    connection_params = experiment_manager.getOmeroConnectionParams()
    
    # Upload-Parameter abrufen
    upload_params = experiment_manager.getOmeroUploadParams()

# Konfiguration zur Laufzeit ändern
experiment_manager.setOmeroConfig({
    "serverUrl": "new-server.com",
    "isEnabled": True
})
```

## Implementierung

Die Implementierung folgt dem Muster des MCTManagers:
- Die Konfiguration wird direkt aus der experimentInfo der Setup-Datei geladen
- Keine separate Konfigurationsdatei erforderlich
- Einfache, direkte Attributzugriffe auf die Konfigurationswerte
- Konsistente API mit anderen ImSwitch-Managern

## Sicherheit

- Das Passwort wird bei API-Abfragen automatisch maskiert
- Die Konfiguration wird nur zur Laufzeit im Speicher gehalten
- Passwörter sollten in produktiven Umgebungen sicher verwaltet werden
