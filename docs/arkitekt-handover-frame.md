# Arkitekt hub on another FRAME, with the pallet and an NVMe

Goal: repeat the hub benchmark on a second FRAME, with the hub's data on an NVMe instead of the SD
card. First run (SD card, FRAME ID10): [arkitekt-hub-pi.md](arkitekt-hub-pi.md). Reference for the
ImSwitch side: [ARKITEKT.md](ARKITEKT.md).

## What you need

- A FRAME (Pi 5) on openUC2 OS with internet, an NVMe it can see (`lsblk` shows `nvme0n1`), and a
  member of organization 7 on go.arkitekt.live to approve the hub and the apps.
- This FRAME's setup file with the `arkitekt` block below.
- A MacBook with Orkestrator 7.x (macOS arm64), on the same network.

## 1. NVMe

Format it once and mount it where the hub keeps its data, so the pallet needs no change:

```bash
sudo mkfs.ext4 -L frame-nvme /dev/nvme0n1          # erases it
echo 'LABEL=frame-nvme /home/pi/arkitekt ext4 defaults,noatime,nofail 0 2' | sudo tee -a /etc/fstab
sudo mkdir -p /home/pi/arkitekt && sudo mount -a && df -h /home/pi/arkitekt
```

For a run with the scan canvases on the NVMe too, mount a second one (or a folder bind) at
`/home/pi/Datasets`. Check that `/etc/fstab` is still there after the first reboot (openUC2 OS
overlays `/etc`). A first run on the SD card is the baseline: skip this step for it.

## 2. Pallet (branch `feat/arkitekt-hub` in openUC2/os-rpi)

```bash
forklift plt switch github.com/openUC2/os-rpi@feat/arkitekt-hub   # replaces this FRAME's local pallet
forklift plt enable-depl arkitekt-hub
forklift plt enable-depl-feat imswitch arkitekt                   # ImSwitch keeps its Arkitekt login
forklift plt stage && sudo reboot
```

- The branch is `edge` plus the hub, the `arkitekt` feature and an ImSwitch image with the arkitekt
  16 code (`sha-3ef0c15`). If this FRAME has deployments that `edge` lacks (FRAME ID10 has
  flim-imager and uc2-flasher), re-enable them with `forklift plt add-depl` before staging.
- At boot `arkitekt-hub-setup.service` creates the hub in `/home/pi/arkitekt/arkitekt-hub` and
  Forklift starts it. rekuest, mikro and the reporter wait for the next step.
- Check: `journalctl -u arkitekt-hub-setup`, `docker ps | grep arkitekt-hub`, and
  `curl -sk https://localhost:2755/rustfs/health` (200).

## 3. Connect the hub (once, in a browser)

```bash
konstruktor hub connect --in /home/pi/arkitekt/arkitekt-hub --server go.arkitekt.live --no-open
```

Open the link it prints, accept the hub into organization 7. The hub is named like the FRAME
(`openuc2-<name>`). Codes last about 10 minutes. Check:
`konstruktor status --in /home/pi/arkitekt/arkitekt-hub` shows 8/8 running and the organization.

## 4. ImSwitch

Setup file, `arkitekt` block (no `redeemToken`, it is gone):

```json
"arkitekt": {"enabled": true, "appName": "frame-<name>", "url": "https://go.arkitekt.live",
             "autoConnect": true, "useMikro": true,
             "uploadExperiments": true, "deleteAfterUpload": false}
```

Panel: **Connect**, open the approval link (it is valid about 5 minutes, and ImSwitch shows a fresh
one for half an hour), choose **this FRAME's hub** on the approval page, approve. The panel shows
"connected". The login is kept in `/home/pi/arkitekt/imswitch`, so a new container does not ask again.

## 5. Look at it in Orkestrator

Log in at go.arkitekt.live, pick this FRAME's hub. If it says "Service mikro not found", the Mac
does not reach the Pi: macOS Settings → Privacy → Local Network → Orkestrator on, and
`curl -k -s -o /dev/null -w '%{http_code}' https://<pi address>:2755/mikro/ht` on the Mac must print 200. First
view: note the time to the first picture, pan and zoom, and whether RGB looks right.

## 6. Benchmark

Pass marks (from the brief): ImSwitch plus the idle hub leave at least 1.5 GB free; a scan is at most
10 % slower than without the hub; ingest takes at most 2 × the scan time; 900 tiles are placed in
Orkestrator.

- **Real scan:** Run Tile Scan from Orkestrator, or an experiment with `uploadExperiments`. Time:
  scan end → image in mikro (`getArkitektUploadQueue` shows seconds per canvas).
- **Synthetic scan** (no stage needed): `scripts/arkitekt_canvas_bench.py make 10 10`, then
  `upload <canvas.ome.zarr> --hub <hub id>`. It is in the ImSwitch image under `/opt/imswitch`
  (not tried from there yet). While it runs: `vmstat 5` (look at `wa`), `free -m`, `vcgencmd get_throttled`.
- **First figures (SD card, 10 × 10 RGB tiles, 3.8 GB raw):** pyramid 128 s, upload 259 s, 6.5 min
  from scan end to mikro, 38 % of the time waiting on I/O. Idle hub 0.5–0.75 GB RAM. Expect the
  NVMe to remove most of the waiting; then the re-encoding on the CPU is the limit.
- Record the same numbers for the SD card and the NVMe in the tracking doc.

## Known gaps

- The pallet's **boot path is not tested end to end** (setup unit at boot, `hub connect` in this
  layout). It was tested on a throwaway hub: up in 11 s, databases and buckets created. Report
  what the first boot does.
- Forklift removes compose projects it does not know at every apply: a hub started by `konstruktor
  up` is gone after the next apply. Use the pallet deployment, not `konstruktor up`.
- The hub's addresses are fixed when it connects. After an IP change: `konstruktor hub connect
  --in … --reach this-network`; reserve the DHCP lease.
- Every scan is stored twice (ImSwitch's canvas and mikro's copy) unless `deleteAfterUpload` is on.
- The images are digest-pinned together with konstruktor 0.38.1: do not bump them one by one.

## Where things are

| | |
|---|---|
| ImSwitch | branch `feat/arkitekt-hub-mikro` (PR #333) |
| Pallet | os-rpi branch `feat/arkitekt-hub`, package README `deployments/arkitekt-hub.pkg/README.md` |
| Hub on FRAME ID10 | `/home/pi/arkitekt/hub`, hub id 111 (konstruktor-started; not on the pallet) |
