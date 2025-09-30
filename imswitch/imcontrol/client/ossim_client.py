"""Client for interacting with the Raspberry Pi FastAPI service that drives
Digital Micromirror Device (DMD) patterns for OSSIM acquisitions.

The client offers a minimal, fault-tolerant wrapper around the REST endpoints
used by the OSSIM dock widget. All responses follow a unified structure so the
UI layer can easily present human readable error messages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
from requests import Response


class OssimClient:
    """HTTP client used to communicate with the OSSIM FastAPI backend."""

    def __init__(
        self,
        base_url: str = "http://192.168.137.2:8000",
        timeout: float = 2.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        """Query the FastAPI health endpoint."""

        return self._request("GET", "/health")

    def display(self, pattern_id: int) -> Dict[str, Any]:
        """Request the DMD to display a given pattern."""

        return self._request("GET", f"/display/{pattern_id}")

    def status(self) -> Dict[str, Any]:
        """Fetch current backend status information."""

        return self._request("GET", "/status")

    def list_patterns(self) -> Dict[str, Any]:
        """Return the available pattern metadata."""

        return self._request("GET", "/patterns")

    def reload(
        self,
        pattern_dir: Optional[str] = None,
        pattern_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Trigger a reload of patterns on the backend."""

        payload: Dict[str, Any] = {}
        if pattern_dir is not None:
            payload["pattern_dir"] = pattern_dir
        if pattern_files is not None:
            payload["pattern_files"] = pattern_files

        kwargs: Dict[str, Any] = {}
        if payload:
            kwargs["json"] = payload

        return self._request("POST", "/reload", **kwargs)

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        """Perform a HTTP request with retry and consistent response format."""

        url = f"{self.base_url}{endpoint}"
        last_error: Optional[str] = None

        for attempt in range(2):
            try:
                response: Response = self._session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as exc:
                    last_error = f"无法解析服务器返回的 JSON：{exc}"
                    continue
                return {"ok": True, "data": data}
            except requests.Timeout:
                last_error = "请求超时，请检查网络连接或目标服务是否在线"
            except requests.ConnectionError as exc:
                last_error = f"无法连接到树莓派服务：{exc}"
            except requests.HTTPError as exc:
                detail = self._extract_error_message(exc.response)
                status_code = exc.response.status_code if exc.response else "未知"
                last_error = f"服务器返回错误状态码 {status_code}：{detail}"
            except requests.RequestException as exc:
                last_error = f"请求失败：{exc}"

            if attempt == 0:
                continue

        return {"ok": False, "error": last_error or "未知错误"}

    @staticmethod
    def _extract_error_message(response: Optional[Response]) -> str:
        if response is None:
            return ""
        try:
            payload = response.json()
        except ValueError:
            return response.text
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str):
                return detail
        return response.text
