#!/usr/bin/env python3
"""Private, memory-only HTTP adapter for the experimental PhotoGuard engine.

This module holds the /v1 API that the FACADE website calls: a strict bearer
check, bounded PNG decoding, per-job access tokens and one retained job. The
always-on server (facade_photoguard_service.py) builds on it.

Localhost is the default; put an authenticated TLS proxy in front of this process.
No service is started on import. This module never stores uploads, outputs, job
metadata, or request logs on disk. TTL/cancel requests stop cooperatively: an
in-flight GPU call must return before its references can be released. This is
reference cleanup, not a promise of forensic memory zeroization.

Running this file directly starts a bare development worker (see --help); the
server runs facade_photoguard_service.py instead.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import io
import ipaddress
import json
import os
import re
import secrets
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Mapping

from PIL import Image, ImageOps

DIFFUSION_RECIPE = "photoguard-diffusion-l2-v1"
ENCODER_RECIPE = "photoguard-encoder-linf-v1"
RECIPE = DIFFUSION_RECIPE  # Backward-compatible imported helper default.
ENGINE_RECIPES = {"encoder": ENCODER_RECIPE, "diffusion": DIFFUSION_RECIPE}
IMAGE_SIDE = 512
ITERATIONS = 200
RETENTION_SECONDS = 1800
MAX_REQUEST_BYTES = 4 * 1024 * 1024
_TOKEN = re.compile(r"[0-9a-f]{64}\Z")
_JOB_PATH = re.compile(r"/v1/jobs/([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(/output)?\Z")


class RequestError(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code


def _decode_png(value: object, *, mask: bool) -> Image.Image:
    if not isinstance(value, str) or not value or len(value) > MAX_REQUEST_BYTES:
        raise ValueError("invalid image")
    raw = base64.b64decode(value, validate=True)
    if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[8:16] != b"\x00\x00\x00\rIHDR":
        raise ValueError("invalid image")
    if struct.unpack(">II", raw[16:24]) != (IMAGE_SIDE, IMAGE_SIDE):
        raise ValueError("invalid image")
    with Image.open(io.BytesIO(raw)) as source:
        if source.format != "PNG" or source.size != (IMAGE_SIDE, IMAGE_SIDE) or getattr(source, "n_frames", 1) != 1:
            raise ValueError("invalid image")
        source.load()
        oriented = ImageOps.exif_transpose(source)
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255) if mask else (255, 255, 255, 255))
        composite = Image.alpha_composite(background, rgba)
        try:
            clean = composite.convert("L" if mask else "RGB")
            clean.info.clear()
            return clean
        finally:
            oriented.close()
            rgba.close()
            background.close()
            composite.close()


def _encode_png(image: Image.Image) -> bytes:
    if not isinstance(image, Image.Image) or image.size != (IMAGE_SIDE, IMAGE_SIDE):
        raise ValueError("invalid engine output")
    converted = image.convert("RGB")
    clean = Image.new("RGB", image.size)
    try:
        clean.paste(converted)
        buffer = io.BytesIO()
        clean.save(buffer, format="PNG")
        output = buffer.getvalue()
        if len(output) > MAX_REQUEST_BYTES:
            raise ValueError("invalid engine output")
        return output
    finally:
        converted.close()
        clean.close()


def _reject_constant(value):
    raise ValueError("invalid JSON constant")


def _unique_json(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate field")
        value[key] = item
    return value


@dataclass
class Job:
    id: str
    token_digest: bytes
    expires: float
    image: Image.Image | None
    mask: Image.Image | None
    status: str = "queued"
    iteration: int = 0
    output: bytes | None = None
    active: bool = False
    cleanup_complete: bool = False
    expired: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)


class JobStore:
    """One process, one retained job. The engine must honor cancel/progress hooks."""
    def __init__(self, engine: Callable | None = None, *, gpu_smoke_validated: bool = False,
                 engine_smoke_validated: bool = False, recipe: str = DIFFUSION_RECIPE,
                 execution_device: str = "cuda:0", clock: Callable[[], float] = time.monotonic):
        if recipe not in ENGINE_RECIPES.values():
            raise ValueError("unsupported fixed recipe")
        if not isinstance(execution_device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?|mps", execution_device):
            raise ValueError("explicit CUDA or MPS execution device required")
        if recipe == DIFFUSION_RECIPE and execution_device == "mps":
            raise ValueError("the diffusion backend has no MPS implementation")
        if gpu_smoke_validated and execution_device == "mps":
            raise ValueError("MPS validation cannot attest the hosted CUDA smoke")
        self.engine = engine
        self._recipe, self._execution_device = recipe, execution_device
        self.gpu_smoke_validated = bool(gpu_smoke_validated)
        self.engine_smoke_validated = bool(engine_smoke_validated or gpu_smoke_validated)
        self.clock = clock
        self._job: Job | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None
        self._worker: threading.Thread | None = None

    @property
    def recipe(self):
        return self._recipe

    @property
    def execution_device(self):
        return self._execution_device

    @property
    def ready(self):
        return self.engine is not None and self.engine_smoke_validated and not self._closed

    def capabilities(self):
        return {"ready": self.ready, "recipe": self.recipe, "image_side": IMAGE_SIDE,
                "total_iterations": ITERATIONS, "retention_seconds": RETENTION_SECONDS,
                "experimental": True, "gpu_smoke_validated": self.gpu_smoke_validated,
                "engine_smoke_validated": self.engine_smoke_validated, "execution_device": self.execution_device}

    @staticmethod
    def _snapshot(job):
        return {"id": job.id, "status": job.status, "iteration": job.iteration,
                "total_iterations": ITERATIONS, "cleanup_complete": job.cleanup_complete}

    @staticmethod
    def _release(job):
        for image in (job.image, job.mask):
            if image is not None:
                image.close()
        job.image = job.mask = None
        job.output = None

    def _cancel_locked(self, job):
        job.cancel.set()
        self._release(job)
        if job.active:
            job.status = "cancelling"
            job.cleanup_complete = False
        else:
            job.status = "cancelled"
            job.cleanup_complete = True

    def reap(self):
        with self._lock:
            job = self._job
            if job and self.clock() >= job.expires:
                job.expired = True
                self._cancel_locked(job)
                if not job.active:
                    self._job = None

    def start_reaper(self):
        with self._lock:
            if self._reaper is not None:
                return
            def loop():
                while not self._stop.wait(0.5):
                    self.reap()
            self._reaper = threading.Thread(target=loop, name="photoguard-expiry", daemon=True)
            self._reaper.start()

    def submit(self, image: Image.Image, mask: Image.Image):
        with self._lock:
            self.reap()
            if not self.ready:
                raise RequestError(503, "not_ready")
            if self._job is not None and not (self._job.cleanup_complete and self._job.status in {"failed", "cancelled"}):
                raise RequestError(409, "busy")
            token = secrets.token_hex(32)
            job = Job(str(uuid.uuid4()), hashlib.sha256(token.encode()).digest(),
                      self.clock() + RETENTION_SECONDS, image, mask)
            self._job = job
            result = dict(self._snapshot(job), access_token=token)
            self._worker = threading.Thread(target=self._execute, args=(job,), name="photoguard-job", daemon=True)
            self._worker.start()
            return result

    def _execute(self, job):
        image = mask = result = None
        encoded = None
        failed = False
        try:
            with self._lock:
                if job.cancel.is_set() or job is not self._job:
                    return
                job.active = True
                job.status = "running"
                image, mask = job.image, job.mask
                job.image = job.mask = None
            def cancelled():
                if self.clock() >= job.expires:
                    with self._lock:
                        job.expired = True
                        self._cancel_locked(job)
                return job.cancel.is_set()
            def progress(iteration, total):
                if type(iteration) is not int or type(total) is not int or total != ITERATIONS or not 0 <= iteration <= ITERATIONS:
                    raise ValueError("invalid engine progress")
                with self._lock:
                    job.iteration = max(job.iteration, iteration)
            result = self.engine(image, mask, cancel=cancelled, progress=progress)
            if not cancelled():
                if job.iteration != ITERATIONS:
                    raise ValueError("engine did not complete the frozen recipe")
                encoded = _encode_png(result)
        except Exception:
            failed = True  # Never retain exception text, input data, or stack traces.
        finally:
            for item in (image, mask, result):
                if isinstance(item, Image.Image):
                    item.close()
            image = mask = result = None
            with self._lock:
                job.active = False
                self._release(job)
                if job.cancel.is_set():
                    job.status = "cancelled"
                    encoded = None
                elif failed or encoded is None:
                    job.status = "failed"
                else:
                    job.status = "completed"
                    job.iteration = ITERATIONS
                    job.output = encoded
                job.cleanup_complete = job.output is None
                if job.expired and self._job is job:
                    self._job = None
            encoded = None

    def _find_locked(self, job_id, token):
        self.reap()
        supplied = token if isinstance(token, str) and _TOKEN.fullmatch(token) else ""
        digest = hashlib.sha256(supplied.encode()).digest()
        job = self._job
        if job is None or job.id != job_id or not hmac.compare_digest(job.token_digest, digest):
            raise RequestError(404, "not_found")
        return job

    def get(self, job_id, token, *, output=False):
        with self._lock:
            job = self._find_locked(job_id, token)
            if output:
                if job.status != "completed" or job.output is None:
                    raise RequestError(409, "not_completed")
                return job.output
            return self._snapshot(job)

    def delete(self, job_id, token):
        with self._lock:
            job = self._find_locked(job_id, token)
            self._cancel_locked(job)
            return self._snapshot(job)

    def close(self):
        self._closed = True
        self._stop.set()
        with self._lock:
            if self._job:
                self._cancel_locked(self._job)
        if self._reaper is not None:
            self._reaper.join(timeout=1)


class WorkerAPI:
    def __init__(self, store: JobStore, bearer_token: str):
        if not isinstance(bearer_token, str) or not 32 <= len(bearer_token) <= 256 or not bearer_token.isascii() or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in bearer_token):
            raise ValueError("a strong printable ASCII API token is required")
        self.store = store
        self._authorization = "Bearer " + bearer_token

    @staticmethod
    def headers(items):
        result = {}
        for key, value in items:
            key = key.lower()
            if key in result:
                raise RequestError(400, "invalid_headers")
            result[key] = value
        return result

    def authorize(self, headers):
        value = headers.get("authorization", "")
        if not isinstance(value, str) or not value.isascii() or not hmac.compare_digest(value, self._authorization):
            raise RequestError(401, "unauthorized")

    @staticmethod
    def json_response(status, value):
        return status, {"Content-Type": "application/json", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}, json.dumps(value, separators=(",", ":"), allow_nan=False).encode()

    def handle(self, method: str, path: str, headers: Mapping[str, str], body: bytes = b""):
        try:
            self.authorize(headers)
            if "transfer-encoding" in headers:
                raise RequestError(400, "invalid_headers")
            if len(body) > MAX_REQUEST_BYTES:
                raise RequestError(413, "request_too_large")
            if method == "GET" and path == "/v1/capabilities":
                return self.json_response(200, self.store.capabilities())
            if method == "POST" and path == "/v1/jobs":
                if headers.get("content-type", "").lower() != "application/json":
                    raise RequestError(415, "unsupported_media_type")
                if not self.store.ready:
                    raise RequestError(503, "not_ready")
                image = mask = None
                try:
                    data = json.loads(body, object_pairs_hook=_unique_json, parse_constant=_reject_constant)
                    if not isinstance(data, dict) or set(data) != {"image_png", "mask_png", "recipe"} or data["recipe"] != self.store.recipe:
                        raise ValueError("invalid recipe or fields")
                    image = _decode_png(data["image_png"], mask=False)
                    mask = _decode_png(data["mask_png"], mask=True)
                    if self.store.recipe == ENCODER_RECIPE:
                        colors = mask.getcolors(maxcolors=3)
                        if colors is None or {value for _, value in colors} != {0, 255}:
                            raise ValueError("encoder mask must contain binary edit and context regions")
                    result = self.store.submit(image, mask)
                    image = mask = None  # Ownership transferred to the single job.
                    return self.json_response(202, result)
                except RequestError:
                    raise
                except (ValueError, TypeError, UnicodeError, OSError, binascii.Error, RecursionError):
                    raise RequestError(400, "invalid_request") from None
                finally:
                    for item in (image, mask):
                        if item is not None:
                            item.close()
            match = _JOB_PATH.fullmatch(path)
            if match and method in {"GET", "DELETE"}:
                job_id, output = match.groups()
                token = headers.get("x-job-token", "")
                if method == "DELETE" and not output:
                    return self.json_response(200, self.store.delete(job_id, token))
                if method == "GET":
                    result = self.store.get(job_id, token, output=bool(output))
                    if output:
                        return 200, {"Content-Type": "image/png", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Content-Disposition": 'attachment; filename="protected.png"'}, result
                    return self.json_response(200, result)
            raise RequestError(404, "not_found")
        except RequestError as error:
            return self.json_response(error.status, {"error": error.code})
        except Exception:
            return self.json_response(500, {"error": "internal_error"})


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    def __init__(self, address, api):
        self.api = api
        self._connections = threading.BoundedSemaphore(8)
        super().__init__(address, WorkerHandler)
    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._connections.release()
            raise
    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()
    def handle_error(self, request, client_address):
        pass  # Never print request data or exception tracebacks.


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "PrivateWorker"
    sys_version = ""
    def setup(self):
        super().setup()
        self.connection.settimeout(10)
    def log_message(self, *args):
        pass
    def send_error(self, code, message=None, explain=None):
        self._write(WorkerAPI.json_response(code, {"error": "invalid_request"}))
    def _write(self, response):
        status, headers, body = response
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True
    def _handle(self):
        try:
            headers = WorkerAPI.headers(self.headers.items())
            self.server.api.authorize(headers)  # Authenticate before reading uploads.
            if "transfer-encoding" in headers:
                raise RequestError(400, "invalid_headers")
            length_text = headers.get("content-length", "0")
            if not re.fullmatch(r"[0-9]{1,10}", length_text):
                raise RequestError(400, "invalid_headers")
            length = int(length_text)
            if length > MAX_REQUEST_BYTES:
                raise RequestError(413, "request_too_large")
            if self.command != "POST" and length:
                raise RequestError(400, "invalid_request")
            body = self.rfile.read(length)
            if len(body) != length:
                raise RequestError(400, "invalid_request")
            response = self.server.api.handle(self.command, self.path, headers, body)
        except RequestError as error:
            response = WorkerAPI.json_response(error.status, {"error": error.code})
        except TimeoutError:
            response = WorkerAPI.json_response(408, {"error": "request_timeout"})
        except Exception:
            response = WorkerAPI.json_response(500, {"error": "internal_error"})
        self._write(response)
    do_GET = do_POST = do_DELETE = _handle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-remote-bind", action="store_true")
    parser.add_argument("--engine", choices=tuple(ENGINE_RECIPES), default="encoder")
    parser.add_argument("--device", default="cuda:0", help="Encoder supports CUDA-FP16 or explicit MPS-FP32; diffusion is CUDA-only")
    parser.add_argument("--engine-smoke-validated", action="store_true", help="Mark the engine as checked: only after this exact recipe/device passed a separate numerical smoke test (this is not a CUDA check)")
    parser.add_argument("--snapshot", help="Already present local model snapshot; never downloaded")
    parser.add_argument("--gpu-smoke-validated", action="store_true", help="Mark the exact recipe as checked on a CUDA GPU by a separate smoke test")
    args = parser.parse_args(argv)
    address = ipaddress.ip_address(args.bind)
    if address.version != 4 or (not address.is_loopback and not args.allow_remote_bind) or not 1 <= args.port <= 65535:
        parser.error("use an IPv4 loopback address; remote binding requires explicit override")
    store = JobStore(recipe=ENGINE_RECIPES[args.engine], execution_device=args.device,
                     gpu_smoke_validated=args.gpu_smoke_validated, engine_smoke_validated=args.engine_smoke_validated)
    api = WorkerAPI(store, os.environ.get("FACADE_PHOTOGUARD_API_TOKEN", ""))
    if args.snapshot:
        from pathlib import Path
        if args.engine == "encoder":
            from facade_photoguard_encoder import load_local_components, protect
        else:
            from facade_photoguard_diffusion import load_local_components, protect
        components = load_local_components(Path(args.snapshot), device=args.device)
        store.engine = lambda image, mask, **hooks: protect(image, mask, components=components, **hooks)
    server = WorkerServer((args.bind, args.port), api)
    store.start_reaper()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("Worker unavailable; check the command-line options and FACADE_PHOTOGUARD_API_TOKEN.") from None
