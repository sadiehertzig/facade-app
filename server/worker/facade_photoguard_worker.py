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

# How one job flows through this module:
#
# 1. The website sends POST /v1/jobs with a JSON body holding the photo and the
#    mask as base64 PNGs plus the recipe name. The request must carry the shared
#    bearer token (Authorization: Bearer ...), which the website holds on its
#    server side; browsers never see it.
# 2. WorkerAPI.handle checks the bearer, the size and the content type, decodes
#    both PNGs in memory, and hands them to JobStore.submit.
# 3. JobStore keeps exactly ONE job. If a job is already held the reply is 409
#    "busy". Otherwise it mints a per-job access token, starts a thread that runs
#    the engine (200 PGD iterations on the GPU), and replies 202 with the job id,
#    the token, and the progress counters.
# 4. The website polls GET /v1/jobs/<id> with the token in X-Job-Token, then
#    downloads GET /v1/jobs/<id>/output (a PNG), then sends DELETE to free the
#    slot. A job that nobody deletes expires on its own after 1800 s.
#
# Two different secrets: the bearer proves "this request comes from the FACADE
# website"; the job token proves "this request comes from whoever created this
# job". Both are compared in constant time and only the token's hash is kept.

# The two engines this process can run. The server only ever runs the encoder
# recipe; the diffusion one stays selectable for local work on a 40 GB GPU.
DIFFUSION_RECIPE = "photoguard-diffusion-l2-v1"
ENCODER_RECIPE = "photoguard-encoder-linf-v1"
RECIPE = DIFFUSION_RECIPE  # Backward-compatible imported helper default.
ENGINE_RECIPES = {"encoder": ENCODER_RECIPE, "diffusion": DIFFUSION_RECIPE}
# Every photo and mask is exactly 512 x 512; the website crops before sending.
IMAGE_SIDE = 512
# The engine must report exactly this many PGD iterations, or the job fails.
ITERATIONS = 200
# A job (its input, its output, its token) lives at most 30 minutes.
RETENTION_SECONDS = 1800
# Hard cap on any request body and on the PNG we send back. A 512 x 512 RGB PNG
# is under 1 MB and base64 adds a third, so 4 MiB leaves room without letting a
# client make this process hold much memory per request.
MAX_REQUEST_BYTES = 4 * 1024 * 1024
# A job token is 32 random bytes written as 64 lowercase hex characters.
_TOKEN = re.compile(r"[0-9a-f]{64}\Z")
# /v1/jobs/<uuid4> or /v1/jobs/<uuid4>/output. The "4" and "[89ab]" positions
# are what make a UUID a version-4 UUID, the only kind this module mints.
_JOB_PATH = re.compile(r"/v1/jobs/([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(/output)?\Z")


class RequestError(Exception):
    """An HTTP error reply: a status code and a short machine-readable code.

    The code (for example "busy" or "not_found") is all the client ever sees.
    Replies never carry free text that could echo part of a request.
    """
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code


def _decode_png(value: object, *, mask: bool) -> Image.Image:
    """Turn one base64 field of the request into a clean 512 x 512 Pillow image.

    The checks run cheapest first: the string length, then the fixed PNG
    signature and the IHDR header (so width and height are read from the raw
    bytes before Pillow parses anything), then Pillow's own decode. Transparent
    pixels are flattened onto black for a mask and white for a photo, the EXIF
    rotation is applied, and every piece of metadata is dropped, so the engine
    gets pixels and nothing else.
    """
    if not isinstance(value, str) or not value or len(value) > MAX_REQUEST_BYTES:
        raise ValueError("invalid image")
    # validate=True refuses characters outside the base64 alphabet instead of
    # silently skipping them.
    raw = base64.b64decode(value, validate=True)
    # 8 bytes of PNG signature, then the IHDR chunk: 4 bytes length (13), 4 bytes
    # type, then width and height as big-endian 32-bit integers at bytes 16-24.
    if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[8:16] != b"\x00\x00\x00\rIHDR":
        raise ValueError("invalid image")
    if struct.unpack(">II", raw[16:24]) != (IMAGE_SIDE, IMAGE_SIDE):
        raise ValueError("invalid image")
    with Image.open(io.BytesIO(raw)) as source:
        # n_frames > 1 means an animated PNG (APNG); only a single frame is accepted.
        if source.format != "PNG" or source.size != (IMAGE_SIDE, IMAGE_SIDE) or getattr(source, "n_frames", 1) != 1:
            raise ValueError("invalid image")
        source.load()
        oriented = ImageOps.exif_transpose(source)
        rgba = oriented.convert("RGBA")
        # Mask polarity: a transparent mask pixel counts as black, which is the
        # protected context. A transparent photo pixel becomes white.
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255) if mask else (255, 255, 255, 255))
        composite = Image.alpha_composite(background, rgba)
        try:
            # "L" is 8-bit grayscale for the mask; "RGB" for the photo.
            clean = composite.convert("L" if mask else "RGB")
            clean.info.clear()
            return clean
        finally:
            # Close every intermediate image now rather than waiting for the
            # garbage collector; each one holds a 512 x 512 pixel buffer.
            oriented.close()
            rgba.close()
            background.close()
            composite.close()


def _encode_png(image: Image.Image) -> bytes:
    """Serialize the engine's output as a plain RGB PNG with no metadata.

    Pasting into a brand-new image guarantees that nothing Pillow may have
    attached to the engine's result (text chunks, ICC profiles) reaches the
    client. The same 4 MiB bound as the input applies on the way out.
    """
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
    """json.loads hook: refuse NaN, Infinity and -Infinity, which are not valid JSON."""
    raise ValueError("invalid JSON constant")


def _unique_json(pairs):
    """json.loads hook: refuse an object that repeats a key.

    Python's default keeps the last value, which would let a client send
    "recipe" twice and have the first copy silently ignored.
    """
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate field")
        value[key] = item
    return value


@dataclass
class Job:
    """Everything the store knows about the one job it holds.

    The photo and mask live here only until the engine thread takes them; the
    output lives here only until the client deletes the job or it expires.
    Nothing in a Job is ever written to disk.
    """
    id: str
    # SHA-256 of the access token; the token itself is not kept
    token_digest: bytes
    # clock() value after which the reaper removes the job
    expires: float
    image: Image.Image | None
    mask: Image.Image | None
    # queued -> running -> completed | failed | cancelling -> cancelled
    status: str = "queued"
    # highest PGD iteration the engine has reported, 0..200
    iteration: int = 0
    # the finished PNG, once status is "completed"
    output: bytes | None = None
    # True while the engine thread is inside the engine call
    active: bool = False
    # True once no input, output or engine reference remains
    cleanup_complete: bool = False
    # set by the reaper; the job vanishes once the engine returns
    expired: bool = False
    # polled by the engine
    cancel: threading.Event = field(default_factory=threading.Event)


class JobStore:
    """One process, one retained job. The engine must honor cancel/progress hooks.

    The store is the single-job queue: `submit` admits a job, `_execute` runs it
    on a background thread, `get`/`delete` serve the client, and `reap` enforces
    the 1800 s expiry. Every method that reads or changes the job takes `_lock`,
    a re-entrant lock because `submit`, `get` and `delete` call `reap` while
    already holding it.

    `engine` is a callable `engine(image, mask, cancel=..., progress=...)` that
    returns a Pillow image. The store never imports torch; the service injects
    the real engine and the tests inject fakes.
    """
    def __init__(self, engine: Callable | None = None, *, gpu_smoke_validated: bool = False,
                 engine_smoke_validated: bool = False, recipe: str = DIFFUSION_RECIPE,
                 execution_device: str = "cuda:0", clock: Callable[[], float] = time.monotonic):
        if recipe not in ENGINE_RECIPES.values():
            raise ValueError("unsupported fixed recipe")
        if not isinstance(execution_device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?|mps", execution_device):
            raise ValueError("explicit CUDA or MPS execution device required")
        if recipe == DIFFUSION_RECIPE and execution_device == "mps":
            raise ValueError("the diffusion backend has no MPS implementation")
        # "gpu_smoke_validated" means the recipe ran end to end on a CUDA GPU in
        # this process; a Mac (MPS) run can never claim that.
        if gpu_smoke_validated and execution_device == "mps":
            raise ValueError("MPS validation cannot attest the hosted CUDA smoke")
        self.engine = engine
        self._recipe, self._execution_device = recipe, execution_device
        self.gpu_smoke_validated = bool(gpu_smoke_validated)
        self.engine_smoke_validated = bool(engine_smoke_validated or gpu_smoke_validated)
        # Injectable clock so the tests can jump 30 minutes ahead instantly.
        self.clock = clock
        self._job: Job | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None
        self._worker: threading.Thread | None = None

    @property
    def recipe(self):
        """The one recipe name this store accepts in POST /v1/jobs (read-only)."""
        return self._recipe

    @property
    def execution_device(self):
        """The torch device the engine runs on, "cuda:0" on the server (read-only)."""
        return self._execution_device

    @property
    def ready(self):
        """True only when an engine is installed, its self-test passed, and the store is open."""
        return self.engine is not None and self.engine_smoke_validated and not self._closed

    def capabilities(self):
        """The JSON for GET /v1/capabilities. The website checks every field before it sends a job."""
        return {"ready": self.ready, "recipe": self.recipe, "image_side": IMAGE_SIDE,
                "total_iterations": ITERATIONS, "retention_seconds": RETENTION_SECONDS,
                "experimental": True, "gpu_smoke_validated": self.gpu_smoke_validated,
                "engine_smoke_validated": self.engine_smoke_validated, "execution_device": self.execution_device}

    @staticmethod
    def _snapshot(job):
        """The public view of a job: status and progress, never the token or any image."""
        return {"id": job.id, "status": job.status, "iteration": job.iteration,
                "total_iterations": ITERATIONS, "cleanup_complete": job.cleanup_complete}

    @staticmethod
    def _release(job):
        """Drop every image the job still holds. Called under the lock."""
        for image in (job.image, job.mask):
            if image is not None:
                image.close()
        job.image = job.mask = None
        job.output = None

    def _cancel_locked(self, job):
        """Mark a job cancelled and free its data. Called under the lock.

        If the engine is still running, the job becomes "cancelling": the engine
        sees the cancel flag at its next check and returns, and only then does
        _execute mark the job "cancelled" with cleanup_complete True.
        """
        job.cancel.set()
        self._release(job)
        if job.active:
            job.status = "cancelling"
            job.cleanup_complete = False
        else:
            job.status = "cancelled"
            job.cleanup_complete = True

    def reap(self):
        """Expire the held job once the clock passes job.expires.

        An idle expired job is dropped at once. A running one is only marked;
        _execute drops it when the engine returns, so the thread never works on
        a job the store has forgotten.
        """
        with self._lock:
            job = self._job
            if job and self.clock() >= job.expires:
                job.expired = True
                self._cancel_locked(job)
                if not job.active:
                    self._job = None

    def start_reaper(self):
        """Start the background thread that calls reap() twice a second."""
        with self._lock:
            if self._reaper is not None:
                return
            def loop():
                while not self._stop.wait(0.5):
                    self.reap()
            self._reaper = threading.Thread(target=loop, name="photoguard-expiry", daemon=True)
            self._reaper.start()

    def submit(self, image: Image.Image, mask: Image.Image):
        """Admit one job, or raise 503 (not ready) / 409 (a job is already held).

        Takes ownership of `image` and `mask`. Returns the public snapshot plus
        the access token, which is the only time the token is ever handed out.
        """
        with self._lock:
            self.reap()
            if not self.ready:
                raise RequestError(503, "not_ready")
            # The slot is free only if the previous job failed or was cancelled
            # AND its cleanup finished. A completed job keeps the slot until the
            # client deletes it or it expires.
            if self._job is not None and not (self._job.cleanup_complete and self._job.status in {"failed", "cancelled"}):
                raise RequestError(409, "busy")
            # 32 random bytes -> 64 hex characters. Only the SHA-256 of the token
            # is stored, so a memory dump of the store does not reveal it.
            token = secrets.token_hex(32)
            job = Job(str(uuid.uuid4()), hashlib.sha256(token.encode()).digest(),
                      self.clock() + RETENTION_SECONDS, image, mask)
            self._job = job
            result = dict(self._snapshot(job), access_token=token)
            self._worker = threading.Thread(target=self._execute, args=(job,), name="photoguard-job", daemon=True)
            self._worker.start()
            return result

    def _execute(self, job):
        """Run the engine for one job on the worker thread and record the outcome.

        The engine gets two callbacks: `cancelled()` so it can stop early, and
        `progress(iteration, total)` so polls can show how far it is. Whatever
        happens, the finally block closes every image and sets the final status.
        """
        image = mask = result = None
        encoded = None
        failed = False
        try:
            with self._lock:
                if job.cancel.is_set() or job is not self._job:
                    return
                job.active = True
                job.status = "running"
                # Take the images out of the Job so a DELETE cannot close them
                # under the engine; this thread owns them from here on.
                image, mask = job.image, job.mask
                job.image = job.mask = None
            def cancelled():
                # Also enforce the expiry from inside the engine, in case the
                # reaper thread is not running (the CLI and some tests).
                if self.clock() >= job.expires:
                    with self._lock:
                        job.expired = True
                        self._cancel_locked(job)
                return job.cancel.is_set()
            def progress(iteration, total):
                # The engine must count 0..200 against a total of exactly 200;
                # anything else is treated as a broken engine, not progress.
                if type(iteration) is not int or type(total) is not int or total != ITERATIONS or not 0 <= iteration <= ITERATIONS:
                    raise ValueError("invalid engine progress")
                with self._lock:
                    job.iteration = max(job.iteration, iteration)
            result = self.engine(image, mask, cancel=cancelled, progress=progress)
            if not cancelled():
                # An engine that returned without reporting iteration 200 did
                # not run the whole recipe; its output is refused.
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
        """Look up the held job by id AND token; a miss on either is a plain 404.

        A wrong token gets the same answer as an unknown id, so a caller cannot
        learn whether a job id exists. The digests are compared with
        hmac.compare_digest, which takes the same time whether or not they match.
        """
        self.reap()
        supplied = token if isinstance(token, str) and _TOKEN.fullmatch(token) else ""
        digest = hashlib.sha256(supplied.encode()).digest()
        job = self._job
        if job is None or job.id != job_id or not hmac.compare_digest(job.token_digest, digest):
            raise RequestError(404, "not_found")
        return job

    def get(self, job_id, token, *, output=False):
        """GET /v1/jobs/<id> (the status snapshot) or .../output (the PNG bytes)."""
        with self._lock:
            job = self._find_locked(job_id, token)
            if output:
                if job.status != "completed" or job.output is None:
                    raise RequestError(409, "not_completed")
                return job.output
            return self._snapshot(job)

    def delete(self, job_id, token):
        """DELETE /v1/jobs/<id>: cancel or discard the job and free its data."""
        with self._lock:
            job = self._find_locked(job_id, token)
            self._cancel_locked(job)
            return self._snapshot(job)

    def close(self):
        """Shut the store: refuse new jobs, cancel the held one, stop the reaper."""
        self._closed = True
        self._stop.set()
        with self._lock:
            if self._job:
                self._cancel_locked(self._job)
        if self._reaper is not None:
            self._reaper.join(timeout=1)


class WorkerAPI:
    """The /v1 request router, independent of any HTTP server.

    `handle(method, path, headers, body)` returns `(status, headers, body)`.
    Keeping it free of sockets lets the tests call it directly, and lets
    WorkerHandler stay a thin adapter.
    """
    def __init__(self, store: JobStore, bearer_token: str):
        # The bearer must be a real secret: 32-256 printable ASCII characters
        # with no whitespace, so it survives an HTTP header unchanged.
        if not isinstance(bearer_token, str) or not 32 <= len(bearer_token) <= 256 or not bearer_token.isascii() or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in bearer_token):
            raise ValueError("a strong printable ASCII API token is required")
        self.store = store
        self._authorization = "Bearer " + bearer_token

    @staticmethod
    def headers(items):
        """Lower-case the header names and refuse a request that repeats one.

        Two Content-Length headers, for example, could make a proxy and this
        server disagree about where the body ends.
        """
        result = {}
        for key, value in items:
            key = key.lower()
            if key in result:
                raise RequestError(400, "invalid_headers")
            result[key] = value
        return result

    def authorize(self, headers):
        """Require the exact "Bearer <token>" header, compared in constant time."""
        value = headers.get("authorization", "")
        if not isinstance(value, str) or not value.isascii() or not hmac.compare_digest(value, self._authorization):
            raise RequestError(401, "unauthorized")

    @staticmethod
    def json_response(status, value):
        """A JSON reply that browsers and proxies must not cache or sniff."""
        return status, {"Content-Type": "application/json", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}, json.dumps(value, separators=(",", ":"), allow_nan=False).encode()

    def handle(self, method: str, path: str, headers: Mapping[str, str], body: bytes = b""):
        """Route one request. Every failure becomes a short JSON error, never a traceback."""
        try:
            self.authorize(headers)
            # Chunked bodies have no Content-Length, so the size cap could not
            # be checked up front; refuse them outright.
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
                    # Exactly these three fields, no more and no fewer, and the
                    # recipe must name the engine this process runs.
                    if not isinstance(data, dict) or set(data) != {"image_png", "mask_png", "recipe"} or data["recipe"] != self.store.recipe:
                        raise ValueError("invalid recipe or fields")
                    image = _decode_png(data["image_png"], mask=False)
                    mask = _decode_png(data["mask_png"], mask=True)
                    if self.store.recipe == ENCODER_RECIPE:
                        # The encoder needs a hard black/white mask with both
                        # colours present: white = the editor may repaint,
                        # black = protect. getcolors returns None above 3 colours.
                        colors = mask.getcolors(maxcolors=3)
                        if colors is None or {value for _, value in colors} != {0, 255}:
                            raise ValueError("encoder mask must contain binary edit and context regions")
                    result = self.store.submit(image, mask)
                    image = mask = None  # Ownership transferred to the single job.
                    return self.json_response(202, result)
                except RequestError:
                    raise
                except (ValueError, TypeError, UnicodeError, OSError, binascii.Error, RecursionError):
                    # Every decoding problem collapses to one code; the message
                    # text is dropped so nothing about the input leaks back.
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
    """The listening socket: one thread per connection, at most 8 at a time.

    Connections beyond the 8th are closed before any bytes are read, so a flood
    of connections cannot pile up threads. Nothing is ever printed.
    """
    daemon_threads = True
    allow_reuse_address = False
    def __init__(self, address, api):
        self.api = api
        self._connections = threading.BoundedSemaphore(8)
        super().__init__(address, WorkerHandler)
    def process_request(self, request, client_address):
        """Take a connection slot without waiting; if none is free, drop the connection."""
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._connections.release()
            raise
    def process_request_thread(self, request, client_address):
        """Give the slot back when the connection's thread finishes."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()
    def handle_error(self, request, client_address):
        pass  # Never print request data or exception tracebacks.


class WorkerHandler(BaseHTTPRequestHandler):
    """Reads one HTTP request, hands it to WorkerAPI, writes the reply, closes.

    The standard library handler would log every request line and print
    tracebacks; both are switched off here because a request line can contain
    a job id and a traceback could contain request data.
    """
    server_version = "PrivateWorker"
    sys_version = ""
    def setup(self):
        """A client that stops sending gets 10 s, then a 408 instead of a stuck thread."""
        super().setup()
        self.connection.settimeout(10)
    def log_message(self, *args):
        pass
    def send_error(self, code, message=None, explain=None):
        """Replace the library's HTML error page with the same JSON shape as every other reply."""
        self._write(WorkerAPI.json_response(code, {"error": "invalid_request"}))
    def _write(self, response):
        """Send (status, headers, body) and close the connection afterwards."""
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
        """Check the bearer and the declared length BEFORE reading the body.

        An unauthenticated client therefore never makes this process read, let
        alone hold, a multi-megabyte upload.
        """
        try:
            headers = WorkerAPI.headers(self.headers.items())
            self.server.api.authorize(headers)  # Authenticate before reading uploads.
            if "transfer-encoding" in headers:
                raise RequestError(400, "invalid_headers")
            length_text = headers.get("content-length", "0")
            # Up to ten digits, so int() cannot be fed something enormous.
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
    # One method serves all three verbs; the router decides what each one means.
    do_GET = do_POST = do_DELETE = _handle


def main(argv=None):
    """Start a bare worker for local development; the server uses facade_photoguard_service.py.

    Without --snapshot no engine is loaded and the worker answers "not ready".
    With one, the chosen engine's weights are loaded from that local directory.
    """
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
    # Listen on 127.0.0.1 unless explicitly told otherwise: this API has no TLS
    # of its own, so anything beyond the local machine must go through a proxy.
    address = ipaddress.ip_address(args.bind)
    if address.version != 4 or (not address.is_loopback and not args.allow_remote_bind) or not 1 <= args.port <= 65535:
        parser.error("use an IPv4 loopback address; remote binding requires explicit override")
    store = JobStore(recipe=ENGINE_RECIPES[args.engine], execution_device=args.device,
                     gpu_smoke_validated=args.gpu_smoke_validated, engine_smoke_validated=args.engine_smoke_validated)
    # The bearer comes from the environment here; WorkerAPI rejects a weak one.
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
