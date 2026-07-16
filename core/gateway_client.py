"""HTTP proxy client for edge-ai-gateway (LLM routing)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:3000").rstrip("/")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
GATEWAY_INFER_TIMEOUT_SEC = float(os.environ.get("GATEWAY_INFER_TIMEOUT_SEC", "120"))


def gateway_base_url() -> str:
    return GATEWAY_URL


def llm_base_url() -> str:
    return LLM_BASE_URL


def llm_reachable() -> bool:
    """True if something responds on LLM_BASE_URL (e.g. llama-server :8080)."""
    url = LLM_BASE_URL
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    req = urllib.request.Request(url, method="GET", headers={"Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return 200 <= resp.status < 500
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{GATEWAY_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=GATEWAY_INFER_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            parsed = {"error": detail or exc.reason, "http_status": exc.code}
        parsed.setdefault("error", exc.reason)
        parsed["http_status"] = exc.code
        parsed["gateway_reachable"] = True
        return parsed
    except urllib.error.URLError as exc:
        return {
            "error": str(exc.reason or exc),
            "gateway_reachable": False,
            "gateway_url": GATEWAY_URL,
        }


def infer_prompt(prompt: str) -> dict:
    return _request("POST", "/infer", {"Infer": {"prompt": prompt}})


def get_status() -> dict:
    return _request("GET", "/status")
