"""CAN streaming OTA: host -USB-> master -CAN SDO block transfer-> slave.

uc2rest's canota streams the image; this module adds the image download
(verified against version.json), retries, non-terminal status reporting and
the check that the rebooted node reports the new version.

Status payloads (``emit``, e.g. ImSwitch's sigOTAStatusUpdate):
{canId, status, progress, message, method: "can_streaming", [page,
totalPages, speed]}. Final states are "success" and "error";
"initializing", "uploading", "attempt_failed", "retrying" and "verifying"
are intermediate.
"""
import logging
import threading
import time

from .guard import SerialPortGuard


class CanOta:
    RETRIES = 3
    RETRY_DELAY_S = 5

    def __init__(self, client, server, bus, guard: SerialPortGuard, emit=None, logger=None):
        """:param client: callable returning the uc2rest client (or None)."""
        self._client = client
        self.server = server
        self.bus = bus
        self.guard = guard
        self._emit_signal = emit or (lambda payload: None)
        self._log = logger or logging.getLogger(__name__)
        self.status = {}  # can_id -> last status payload (getOTAStatus)
        self.status_lock = threading.Lock()

    def _emit(self, can_id, status, message, progress=None, **extra):
        with self.status_lock:
            if progress is None:
                progress = self.status.get(can_id, {}).get("progress", 0)
            payload = {"canId": can_id, "status": status, "progress": progress,
                       "message": message, "method": "can_streaming", **extra}
            self.status[can_id] = payload
        self._emit_signal(payload)

    # ── public ──────────────────────────────────────────────────────────────

    def start(self, can_id, baud=None) -> dict:
        """One node, refusing ({"status": "busy"}) while the port is in use."""
        busy = self.guard.claim_can()
        if busy:
            return {"status": "busy", "message": busy}
        try:
            return self.upload(can_id, baud=baud)
        finally:
            self.guard.can.release()

    def start_many(self, can_ids, delay_between=5) -> dict:
        """Several nodes one after another under one claim of the port."""
        busy = self.guard.claim_can()
        if busy:
            return {"status": "busy", "message": busy}
        results = []
        try:
            for index, can_id in enumerate(can_ids):
                results.append({"can_id": can_id, "result": self.upload(can_id)})
                if delay_between > 0 and index < len(can_ids) - 1:
                    time.sleep(delay_between)
        finally:
            self.guard.can.release()
        return {"status": "success", "results": results,
                "message": f"CAN streaming OTA completed for {len(can_ids)} devices"}

    def upload(self, can_id, filename=None, expected_version=None, baud=None, cancel=None) -> dict:
        """Download, stream (with retries) and verify one node. The caller
        holds guard.can. *filename* defaults to the node's image on the server
        (custom id_<N>_*.bin or its fixed role), *expected_version* to that
        image's version in version.json, *baud* to the live link's baud (the
        master never switches). "success" means the rebooted node reports the
        new version — or, without version.json, that the transfer completed."""
        try:
            client = self._client()
            if client is None or not hasattr(client, "canota"):
                raise RuntimeError("CAN OTA module not available in UC2 client")
            if baud is None:
                baud = getattr(getattr(client, "serial", None), "baudrate", None) or 921600
            self._emit(can_id, "initializing", "Starting CAN streaming upload...", 0)
            manifest = self.server.manifest()
            filename = filename or self.server.can_image(can_id)
            path = self.server.download(filename, manifest) if filename else None
            if not path:
                raise RuntimeError("Failed to download or verify firmware")
            if expected_version is None:
                expected_version = self.server.file_version(manifest, path.name)

            self._emit(can_id, "uploading", "Uploading firmware via CAN streaming...", 5)
            error = self._stream(client.canota, can_id, path, baud)
            if error:
                raise RuntimeError(error)

            if expected_version:
                self._emit(can_id, "verifying",
                           f"Waiting for node {can_id} to report {expected_version}...", 98)
                verified, seen = self.bus.wait_for_version(can_id, expected_version, cancel=cancel)
                if not verified:
                    raise RuntimeError(f"Flashed, but node {can_id} reports {seen!r} "
                                       f"instead of {expected_version!r}")
                message = f"Updated and verified: node {can_id} runs {expected_version}"
            else:
                message = ("Firmware uploaded - device rebooting (not verified: the firmware "
                           "server has no version.json)")
            self._emit(can_id, "success", message, 100)
            return {"status": "success", "can_id": can_id, "message": message}
        except Exception as e:
            self._log.error(f"CAN streaming OTA failed for device {can_id}: {e}")
            self._emit(can_id, "error", f"Upload failed: {e}", error=str(e))
            return {"status": "error", "can_id": can_id,
                    "message": f"CAN streaming OTA failed: {e}"}

    def cancel(self) -> dict:
        """Abort the running stream at the next chunk (the node keeps its old firmware)."""
        try:
            self._client().canota.cancel_streaming_ota()
            self._log.info("CAN streaming OTA cancellation requested")
            return {"status": "success", "message": "Cancellation requested"}
        except Exception as e:
            self._log.warning(f"Could not signal canota cancel: {e}")
            return {"status": "error", "message": str(e)}

    # ── internals ───────────────────────────────────────────────────────────

    def _stream(self, canota, can_id, path, baud):
        """Up to RETRIES attempts; None on success, else the error message. A
        failure inside an attempt is reported as "attempt_failed", not as the
        terminal "error" (the UI would count the device as failed)."""
        def progress(page, total, sent, speed):
            self._emit(can_id, "uploading", f"Page {page}/{total} - {speed:.1f} KB/s",
                       int(page / total * 90) + 5, page=page, totalPages=total, speed=speed)

        def attempt_status(message, ok):
            (self._log.info if ok else self._log.warning)(f"CAN OTA: {message}")
            self._emit(can_id, "uploading" if ok else "attempt_failed", message)

        for attempt in range(1, self.RETRIES + 1):
            canota._cancel_event.clear()
            if canota.start_can_streaming_ota_blocking(
                    can_id=can_id, firmware_path=str(path), progress_callback=progress,
                    status_callback=attempt_status, baud=baud):
                return None
            if canota._cancel_event.is_set():
                return "Upload cancelled by user"
            if attempt < self.RETRIES:
                self._emit(can_id, "retrying", f"Attempt {attempt}/{self.RETRIES} failed — "
                                               f"waiting {self.RETRY_DELAY_S} s before retry...")
                time.sleep(self.RETRY_DELAY_S)
                self._emit(can_id, "uploading",
                           f"Retrying... (attempt {attempt + 1}/{self.RETRIES})", 5)
        return f"CAN streaming upload failed after {self.RETRIES} attempts"
