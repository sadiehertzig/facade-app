#!/usr/bin/env python3
"""Always-on FACADE PhotoGuard encoder worker behind one loopback listener.

This is the program the server runs as facade-worker.service:

    python server/worker/facade_photoguard_service.py \\
        --snapshot /srv/facade/models/sd15-inpaint-8a4288 --bind 127.0.0.1 --port 8000 \\
        < /etc/facade/worker.bearer

It is a thin layer over two modules in this directory:

* ``facade_photoguard_worker`` serves the exact /v1 API the website uses: bearer
  check, bounded PNG decoding, per-job capability tokens, one retained job.
* ``facade_photoguard_private_session`` supplies the fresh 200-step analytic
  smoke, the checked engine wrapper (fixed progress, exact white region, byte
  bound, PNG re-validation on every job) and sanitized diagnostics.

What this adds on top of the session layer:

* No job cap and no session deadline. systemd owns the process lifetime.
* A completed job's single slot is released RELEASE_AFTER_OUTPUT_SECONDS after
  its first successful output download, so a user who never presses Clear does
  not hold the slot for 30 minutes. The 1800 s hard expiry counted from
  submission is unchanged, and so is /v1/capabilities (retention_seconds 1800),
  so the website's strict capability check still passes.
* One listener, 127.0.0.1:8000 by default. A non-loopback bind is refused
  unless --allow-remote-bind is given.
* SIGTERM/SIGINT stop gracefully: readiness drops at once, an in-flight job is
  cancelled cooperatively, and the private-session watchdog hard-exits (124)
  if cleanup is still blocked after 20 s.

The bearer is exactly 64 lowercase hex characters on stdin (a single trailing
newline is allowed). It never comes from arguments or the environment. Photos,
masks, outputs, job ids, job tokens and the bearer are never logged or
written; stderr carries one JSON line per lifecycle event, for journald.
Readiness means the engine ran and its output passed the checks; it says nothing
about how well the protection works.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import getpass
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

import facade_photoguard_private_session as private_session
import facade_photoguard_worker as worker

RECIPE = worker.ENCODER_RECIPE
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8000
RELEASE_AFTER_OUTPUT_SECONDS = 120
CLEANUP_GRACE_SECONDS = 20
FIXED_ENCODER_RECIPE = {"side": 512, "iterations": 200, "epsilon": .12, "step_size": .01, "seed": 1234}
_BEARER = re.compile(r"[0-9a-f]{64}")
_BEARER_READ_LIMIT = 130  # Enough to see trailing junk; never an unbounded read.
_SAFE_TEXT = re.compile(r"[A-Za-z0-9:._+-]{1,128}")
require = private_session.require


def log_event(event, **fields):
    """Write one JSON line to stderr. Callers pass lifecycle metadata only."""
    try:
        line = json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False,
                          separators=(",", ":"))
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass  # Logging must never take the worker down.


def read_bearer(stream=None):
    """Return the hex64 bearer from stdin; reject anything else without echoing it."""
    stream = sys.stdin if stream is None else stream
    require(stream is not None, "Expected exactly one hex64 bearer on stdin")
    if stream.isatty():
        value = getpass.getpass("Worker bearer: ")
    else:
        data = stream.read(_BEARER_READ_LIMIT)
        value = data[:-1] if isinstance(data, str) and data.endswith("\n") else data
    require(isinstance(value, str) and _BEARER.fullmatch(value) is not None,
            "Expected exactly one hex64 bearer on stdin")
    return value


def validate_bind(host, port, *, allow_remote_bind=False, allow_ephemeral_port=False):
    """IPv4 loopback only unless explicitly overridden. Port 0 is for tests only."""
    try:
        address = ipaddress.ip_address(host) if isinstance(host, str) else None
    except ValueError:
        address = None
    lowest = 0 if allow_ephemeral_port else 1
    if (address is None or address.version != 4 or type(port) is not int
            or not lowest <= port <= 65535 or (not address.is_loopback and not allow_remote_bind)):
        raise ValueError("use an IPv4 loopback bind; remote binding requires explicit override")
    return str(address)


def load_cuda_engine(snapshot, *, cancel=None):
    """The only production loader: the verified local CUDA-FP16 encoder VAE."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    import facade_photoguard_encoder as encoder
    require(torch.cuda.is_available(), "CUDA required")
    require(encoder.RECIPE_ID == RECIPE and asdict(encoder.DEFAULT_RECIPE) == FIXED_ENCODER_RECIPE,
            "Fixed recipe mismatch")
    components = encoder.load_local_components(Path(snapshot), device="cuda:0", cancel=cancel)
    require(str(components.device) == "cuda:0" and components.dtype == torch.float16,
            "Verified CUDA-FP16 runtime required")

    def engine(image, mask, **hooks):
        return encoder.protect(image, mask, components=components, **hooks)

    return engine, {"device": "cuda:0", "dtype": "float16", "engine_id": encoder.ENGINE_ID,
                    "model_source_digest": components.source_digest}


def _safe_info(info):
    safe = {}
    for key in ("device", "dtype", "engine_id", "model_source_digest"):
        value = info.get(key) if isinstance(info, dict) else None
        if isinstance(value, str) and _SAFE_TEXT.fullmatch(value):
            safe[key] = value
    return safe


class ServiceServer(worker.WorkerServer):
    # Every response closes its connection, so a restart leaves TIME_WAIT
    # sockets on the port. On Linux, SO_REUSEADDR only allows binding over
    # those, never beside a live listener; without it a systemd restart cannot
    # rebind for about 60 s. BSD semantics differ, so elsewhere keep the default.
    allow_reuse_address = sys.platform.startswith("linux")


class ServiceStore(private_session.SessionStore):
    """Session store with the capability shape unchanged and early slot release."""

    def capabilities(self):
        # Skip the session fields; an unbounded service has no cap or deadline.
        with self._lock:
            return dict(worker.JobStore.capabilities(self), efficacy_evaluated=False)

    def get(self, job_id, token, *, output=False):
        with self._lock:
            result = super().get(job_id, token, output=output)
            job = self._job
            if output and job is not None and job.id == job_id:
                # min(): a later download never extends the first one's window,
                # and nothing can outlive the 1800 s expiry set at submission.
                job.expires = min(job.expires, self.clock() + RELEASE_AFTER_OUTPUT_SECONDS)
            return result


class ServiceSession(private_session.PrivateSession):
    """PrivateSession with the finite job cap and deadline lifted."""

    def __init__(self, *, clock=time.monotonic):
        # The parent validates finite limits; satisfy it, then lift both.
        super().__init__(max_jobs=1, deadline=clock() + 60, clock=clock)
        self.max_jobs = math.inf
        self.deadline = math.inf
        self.store = ServiceStore(self)

    def checked_protect(self, image, mask, *, cancel, progress):
        kind = "job" if self.store.gpu_smoke_validated else "smoke"
        started = time.monotonic()
        try:
            result = super().checked_protect(image, mask, cancel=cancel, progress=progress)
        except BaseException as error:
            log_event("engine_call", kind=kind, outcome="raised",
                      seconds=round(time.monotonic() - started, 3),
                      diagnostic=private_session.sanitized_exception(error))
            raise
        log_event("engine_call", kind=kind, outcome="returned",
                  seconds=round(time.monotonic() - started, 3))
        return result

    def summary(self):
        value = super().summary()
        value.update(maximum_session_jobs=None,
                     release_after_output_seconds=RELEASE_AFTER_OUTPUT_SECONDS)
        return value


class WorkerService:
    """Bind first, then load, smoke and serve until a stop is requested."""

    def __init__(self, bearer, *, bind=DEFAULT_BIND, port=DEFAULT_PORT, allow_remote_bind=False,
                 clock=time.monotonic, server_factory=None):
        host = validate_bind(bind, port, allow_remote_bind=allow_remote_bind,
                             allow_ephemeral_port=True)
        require(isinstance(bearer, str) and _BEARER.fullmatch(bearer) is not None,
                "hex64 bearer required")
        self.session = ServiceSession(clock=clock)
        self._stop_requested = threading.Event()
        self._ran = False
        factory = ServiceServer if server_factory is None else server_factory
        self.server = factory((host, port), worker.WorkerAPI(self.session.store, bearer))

    @property
    def address(self):
        return tuple(self.server.server_address[:2])

    def request_stop(self, *_):
        """Signal-handler safe: only sets events."""
        self._stop_requested.set()
        self.session.stop.set()

    def close(self):
        """Release resources of a service whose run() was never called."""
        self.session.stop.set()
        self.server.server_close()
        self.session.close()

    def run(self, snapshot, *, loader=None, grace_seconds=CLEANUP_GRACE_SECONDS, hard_exit=None):
        require(not self._ran, "A service instance runs once")
        self._ran = True
        session, store = self.session, self.session.store
        load = load_cuda_engine if loader is None else loader
        phase, failure, serving = "listening", None, False
        started = time.monotonic()
        watchdog_done, watchdog = private_session.start_deadline_watchdog(
            session, grace_seconds=grace_seconds, hard_exit=hard_exit)
        try:
            threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .2},
                             name="photoguard-http", daemon=True).start()
            serving = True
            store.start_reaper()
            host, port = self.address
            log_event("listening", bind=host, port=port, ready=False)
            phase = "model_load"
            engine, info = load(snapshot, cancel=session.cancelled)
            session.install_engine(engine)
            log_event("model_loaded", **_safe_info(info))
            phase = "fresh_synthetic_smoke"
            smoke_started = time.monotonic()
            session.validate_smoke()
            log_event("ready", smoke_seconds=round(time.monotonic() - smoke_started, 3),
                      fresh_synthetic_iterations=worker.ITERATIONS, capabilities=store.capabilities())
            phase = "serving"
            while not session.stop.wait(1):
                pass
        except BaseException as error:
            if not self._stop_requested.is_set():
                failure = {**private_session.sanitized_exception(error), "phase": phase}
        finally:
            session.stop.set()
            store.close()
            if serving:
                self.server.shutdown()
            self.server.server_close()
            closed = session.close(join_seconds=grace_seconds)
            status = "STOPPED" if failure is None and closed else "FAILED"
            log_event("stopped", status=status, phase=phase, failure=failure,
                      seconds=round(time.monotonic() - started, 3),
                      active_engine_thread_at_exit=not closed, summary=session.summary())
            watchdog_done.set()
            watchdog.join(timeout=1)
        return 0 if status == "STOPPED" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="Verified local SD1.5-inpainting snapshot; never downloaded")
    parser.add_argument("--bind", default=DEFAULT_BIND)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--allow-remote-bind", action="store_true",
                        help="Permit a non-loopback bind; put authenticated TLS in front of it")
    args = parser.parse_args(argv)
    try:
        validate_bind(args.bind, args.port, allow_remote_bind=args.allow_remote_bind)
    except ValueError:
        parser.error("use an IPv4 loopback --bind and a --port in 1..65535; "
                     "remote binding requires --allow-remote-bind")
    try:
        bearer = read_bearer()
    except Exception as error:
        log_event("stopped", status="FAILED", phase="bearer_input",
                  reason="expected exactly one hex64 bearer on stdin",
                  failure=private_session.sanitized_exception(error))
        return 2
    try:
        service = WorkerService(bearer, bind=args.bind, port=args.port,
                                allow_remote_bind=args.allow_remote_bind)
    except Exception as error:
        log_event("stopped", status="FAILED", phase="listener_bind",
                  failure=private_session.sanitized_exception(error))
        return 2
    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, service.request_stop)
        return service.run(args.snapshot)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit("Worker service unavailable; see its sanitized journal events.") from None
