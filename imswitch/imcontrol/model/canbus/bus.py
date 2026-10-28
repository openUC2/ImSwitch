"""The CAN bus behind the USB master: discovery, node ids, restarts.

Talks to the master through a uc2rest client (``client()`` returns it, or
None when no board is connected). A bus scan SDO-reads every routed node's
build, fwVersion (OD 0x2500), fwImage (OD 0x2501) and MAC.
"""
import logging
import time


class CanBus:
    def __init__(self, client, logger=None):
        """:param client: callable returning the uc2rest client (or None)."""
        self._client = client
        self._log = logger or logging.getLogger(__name__)

    def _can(self):
        client = self._client()
        return getattr(client, "can", None) if client is not None else None

    def scan(self, timeout=5, probe_range=False) -> dict:
        """{"master": {...}, "scan": [{canId, deviceTypeStr, statusStr, build,
        fwVersion, fwImage, mac}], "detected_ids": [...]}; {} when there is no
        CAN master or the scan failed."""
        can = self._can()
        if can is None:
            self._log.error("CAN bus module not available in UC2 client")
            return {}
        try:
            try:
                response = can.scan(timeout=timeout, probe_range=probe_range)
            except TypeError:  # uc2rest without probe_range
                response = can.scan(timeout=timeout)
        except Exception as e:
            self._log.error(f"Error scanning CAN bus: {e}")
            return {}
        result = response[-1] if isinstance(response, (list, tuple)) and response else response
        if not isinstance(result, dict):
            return {}
        result["detected_ids"] = [d["canId"] for d in result.get("scan", [])]
        self._log.info(f"Detected CAN devices: {result['detected_ids']}")
        return result

    def reassign(self, new_id, mac=None, target=None, expect_mac=None, timeout=5) -> dict:
        """Move a node to *new_id* over the bus, identified by MAC (preferred;
        the master finds it) or by its current id. The node persists the id and
        reappears there after ~0.3 s."""
        client = self._client()
        if self._can() is None:
            return {"status": "error", "message": "CAN bus module not available"}
        payload = {"task": "/can_act", "setRemoteNodeId": int(new_id)}
        if mac:
            payload["byMac"] = str(mac)
        elif target is not None:
            payload["target"] = int(target)
            if expect_mac:
                payload["expectMac"] = str(expect_mac)
        else:
            return {"status": "error", "message": "provide either 'mac' or 'target'"}
        self._log.info(f"Reassigning CAN node {mac or target} -> {new_id}")
        try:
            response = client.post_json("/can_act", payload, getReturn=True, timeout=timeout,
                                        nResponses=1)
        except Exception as e:
            self._log.error(f"reassignCANId failed: {e}")
            return {"status": "error", "message": str(e)}
        result = response[-1] if isinstance(response, (list, tuple)) and response else response
        return result if isinstance(result, dict) else {"status": "success", "response": result}

    def restart(self, can_id) -> None:
        """Reboot node *can_id* (0 = the master itself)."""
        self._can().reboot_remote(can_address=can_id, isBlocking=True, timeout=1)

    def wait_for_version(self, can_id, expected, cancel=None, settle=4, timeout=40):
        """After an OTA: wait until *can_id* answers the scan with *expected*.
        Returns (ok, last_seen_version). A slave reports its old version until
        its deferred reboot (2 s after the transfer), so polling starts after
        *settle* seconds."""
        time.sleep(settle)
        deadline = time.time() + timeout
        seen = None
        while time.time() < deadline and not (cancel and cancel.is_set()):
            node = next((n for n in self.scan(timeout=5).get("scan", [])
                         if n.get("canId") == can_id), {})
            seen = node.get("fwVersion") or seen
            if node.get("statusStr") != "unreachable" and node.get("fwVersion") == expected:
                return True, seen
            time.sleep(2)
        return False, seen
