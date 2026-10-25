#!/usr/bin/env python3
"""FACADE server health check (stdlib only; run by facade-health.timer every 60 s).

Checks, and prints ONE JSON line to stdout (journald) with the verdict:
  web     GET http://127.0.0.1:3000/ answers 2xx/3xx (the Next.js website is up)
  demo    GET http://127.0.0.1:7860/photoguard/ answers 2xx/3xx (Gradio is up and
          serves the /photoguard sub-path Caddy forwards to it)
  worker  GET http://127.0.0.1:8000/v1/capabilities with the bearer reports the
          exact capabilities the website requires (ready, gpu_smoke_validated
          after the start-up self-test, recipe photoguard-encoder-linf-v1, 512,
          200, 1800, experimental)
  public  the same capabilities through Caddy/TLS at https://$FACADE_HOST/v1/...
          (skipped when FACADE_HOST is unset or --no-public is given)
  gpu     informational nvidia-smi memory/utilisation; never fails the check

Exit 0 when every non-skipped check passes, 1 otherwise. The bearer and
response bodies are never printed; only status codes and mismatched key names.

deploy/bootstrap.sh also runs it once with --no-public --wait, to wait until a
fresh install is fully up. It only uses the Python standard library, so it runs
with the system python3 and does not depend on the server's virtualenv being
healthy.

"Up" for the worker means more than "answers": the website refuses to send
photos unless every capability below has exactly these values, so the health
check asks the same question the website does.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Must match what the website checks before it sends a job (web/lib/processor.ts).
EXPECTED_CAPABILITIES = {
    "ready": True,
    "gpu_smoke_validated": True,
    "recipe": "photoguard-encoder-linf-v1",
    "image_side": 512,
    "total_iterations": 200,
    "retention_seconds": 1800,
    "experimental": True,
}
_BEARER = re.compile(r"[0-9a-f]{64}")
# A plain DNS name: letters, digits, dots and hyphens, at most 253 characters,
# so FACADE_HOST cannot smuggle a path, port or credentials into the URL.
_HOST = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?")
# Read at most 64 KiB of any reply; a capabilities reply is a few hundred bytes.
MAX_BODY = 64 * 1024


def read_bearer(path: Path) -> str:
    """Accept exactly hex64 with at most one trailing newline (the worker's format)."""
    data = Path(path).read_text(encoding="ascii")
    value = data[:-1] if data.endswith("\n") else data
    if _BEARER.fullmatch(value) is None:
        raise ValueError("bearer file is not a single hex64 token")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects. A redirect counts as "up" for a page, and the bearer
    is never sent on to another address."""
    def redirect_request(self, *args, **kwargs):  # report 3xx as-is
        return None


def http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 10.0) -> tuple[int, bytes]:
    """GET `url` and return (status code, first 64 KiB of the body).

    ProxyHandler({}) ignores any proxy set in the environment, so the check
    talks to the real address. An HTTP error status is returned, not raised;
    a connection failure still raises, and _guarded turns it into a result.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "facade-health/1", **(headers or {})})
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_BODY)
    except urllib.error.HTTPError as error:
        return error.code, b""


def evaluate_page(status: int) -> dict:
    """A web page is up when it answers 2xx or 3xx (redirects are not followed)."""
    return {"ok": 200 <= status < 400, "status": status}


def evaluate_capabilities(status: int, body: bytes) -> dict:
    """Compare a /v1/capabilities reply with EXPECTED_CAPABILITIES.

    Each expected key must be present with the same value AND the same type,
    so `"ready": 1` or `"image_side": 512.0` count as mismatches. Only the
    names of mismatched keys are reported, never the reply itself.
    """
    result: dict = {"ok": False, "status": status}
    if status != 200:
        return result
    try:
        data = json.loads(body)
    except ValueError:
        result["error"] = "invalid_json"
        return result
    if not isinstance(data, dict):
        result["error"] = "invalid_json"
        return result
    mismatched = sorted(key for key, value in EXPECTED_CAPABILITIES.items()
                        if key not in data or type(data[key]) is not type(value) or data[key] != value)
    result["ok"] = not mismatched
    if mismatched:
        result["mismatched"] = mismatched
    return result


def _guarded(check) -> dict:
    """Run one check; any exception becomes a failed result naming only the exception's type."""
    try:
        return check()
    except Exception as error:  # connection refused, TLS, timeout, ...
        return {"ok": False, "error": type(error).__name__}


def gpu_info(timeout: float = 5.0) -> dict | None:
    """GPU memory use and load from nvidia-smi, for the log only; None if there is no nvidia-smi."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        output = subprocess.run(
            [binary, "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=True,
        ).stdout.strip().splitlines()[0]
        used, total, util = (int(part.strip()) for part in output.split(","))
        return {"memory_used_mib": used, "memory_total_mib": total, "utilization_pct": util}
    except Exception as error:
        return {"error": type(error).__name__}


def run_checks(args, bearer: str | None) -> dict:
    """Run every check once and return {name: result}; each result has an "ok" field."""
    checks: dict = {}
    checks["web"] = _guarded(lambda: evaluate_page(http_get(args.web_url, timeout=args.timeout)[0]))
    checks["demo"] = _guarded(lambda: evaluate_page(http_get(args.demo_url, timeout=args.timeout)[0]))
    if bearer is None:
        # Without the bearer the worker cannot be asked, so it counts as down.
        checks["worker"] = {"ok": False, "error": "bearer_unreadable"}
    else:
        auth = {"Authorization": "Bearer " + bearer}
        checks["worker"] = _guarded(lambda: evaluate_capabilities(*http_get(args.worker_url, auth, args.timeout)))
        host = "" if args.no_public else (args.public_host or "").strip()
        if host:
            if _HOST.fullmatch(host) is None:
                checks["public"] = {"ok": False, "error": "invalid_host"}
            else:
                url = f"https://{host}/v1/capabilities"
                checks["public"] = _guarded(lambda: evaluate_capabilities(*http_get(url, auth, args.timeout)))
        else:
            # No public host configured: report the check as skipped, not failed.
            checks["public"] = {"ok": True, "skipped": True}
    return checks


def main(argv: list[str] | None = None) -> int:
    """Run the checks (repeatedly with --wait), print one JSON line, return 0 if healthy."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--web-url", default="http://127.0.0.1:3000/")
    parser.add_argument("--demo-url", default="http://127.0.0.1:7860/photoguard/")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8000/v1/capabilities")
    parser.add_argument("--bearer-file", type=Path, default=Path("/etc/facade/worker.bearer"))
    parser.add_argument("--public-host", default=os.environ.get("FACADE_HOST", ""))
    parser.add_argument("--no-public", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--wait", type=float, default=0.0,
                        help="poll every 10 s until healthy or this many seconds pass")
    args = parser.parse_args(argv)
    # A missing or malformed bearer file fails only the worker and public
    # checks; the web and demo checks still run.
    try:
        bearer = read_bearer(args.bearer_file)
    except Exception:
        bearer = None
    deadline = time.monotonic() + max(0.0, args.wait)
    while True:
        checks = run_checks(args, bearer)
        healthy = all(item.get("ok") for item in checks.values())
        if healthy or time.monotonic() >= deadline:
            break
        time.sleep(10)
    report = {"facade_health": "ok" if healthy else "fail", "checks": checks}
    gpu = gpu_info()
    if gpu is not None:
        report["gpu"] = gpu
    print(json.dumps(report, sort_keys=True, separators=(",", ":")), flush=True)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
