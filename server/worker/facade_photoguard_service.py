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
# The docstring above is printed verbatim by --help, so the longer explanation
# lives here as comments.
#
# Start-up order matters. The listener opens FIRST, before the VAE weights are
# hashed and loaded, so the website can connect at once and simply hears
# "ready": false until the engine is proven. Then:
#
#   1. load_cuda_engine() checks the recipe numbers against FIXED_ENCODER_RECIPE,
#      checks the VAE files' hashes and loads the VAE onto the GPU in FP16;
#   2. validate_smoke() runs one complete 200-iteration job on a synthetic image
#      and checks its output exactly like a user's job;
#   3. only then does /v1/capabilities report "ready": true.
#
# If any step fails, the process logs one sanitized "stopped" event and exits
# with code 2, and systemd restarts it. A failed engine never serves a job.
#
# A "session" in the class names below is the private_session module's word for
# one run of the process. The always-on service is a session with no job cap and
# no deadline.
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
# Loopback only: the website and Caddy, both on this machine, are the only callers.
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8000
# Two minutes after the protected PNG is first downloaded, the job is dropped
# and the single slot frees up for the next user.
RELEASE_AFTER_OUTPUT_SECONDS = 120
# How long a stop may take to clean up before the watchdog force-exits.
# deploy/systemd/facade-worker.service allows 30 s (TimeoutStopSec), so this
# finishes first.
CLEANUP_GRACE_SECONDS = 20
# The one encoder recipe this service will run. Loading refuses an encoder
# module whose defaults differ from these numbers in any way.
FIXED_ENCODER_RECIPE = {"side": 512, "iterations": 200, "epsilon": .12, "step_size": .01, "seed": 1234}
# The bearer is 32 random bytes written as 64 lowercase hex characters.
_BEARER = re.compile(r"[0-9a-f]{64}")
_BEARER_READ_LIMIT = 130  # Enough to see trailing junk; never an unbounded read.
# Only short, plain values from the loader's info dict may reach the log.
_SAFE_TEXT = re.compile(r"[A-Za-z0-9:._+-]{1,128}")
require = private_session.require


def log_event(event, **fields):
    """Write one JSON line to stderr. Callers pass lifecycle metadata only.

    systemd sends stderr to journald, so `journalctl -u facade-worker` shows
    one machine-readable line per event (listening, model_loaded, ready,
    engine_call, stopped). allow_nan=False keeps every line valid JSON.
    """
    try:
        line = json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False,
                          separators=(",", ":"))
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass  # Logging must never take the worker down.


def read_bearer(stream=None):
    """Return the hex64 bearer from stdin; reject anything else without echoing it.

    Why stdin? Command-line arguments are visible to every user through `ps`,
    and environment variables can leak into crash reports and child
    processes. systemd instead feeds the bearer file (readable only by the
    facade user) to stdin
    (StandardInput=file:/etc/facade/worker.bearer).
    """
    stream = sys.stdin if stream is None else stream
    require(stream is not None, "Expected exactly one hex64 bearer on stdin")
    if stream.isatty():
        # Typed by hand in a terminal: getpass does not echo it to the screen.
        value = getpass.getpass("Worker bearer: ")
    else:
        # Read at most 130 characters: 64 for the bearer, one newline, and
        # enough beyond that to notice a file with something extra in it.
        data = stream.read(_BEARER_READ_LIMIT)
        value = data[:-1] if isinstance(data, str) and data.endswith("\n") else data
    require(isinstance(value, str) and _BEARER.fullmatch(value) is not None,
            "Expected exactly one hex64 bearer on stdin")
    return value


def validate_bind(host, port, *, allow_remote_bind=False, allow_ephemeral_port=False):
    """IPv4 loopback only unless explicitly overridden. Port 0 is for tests only.

    Port 0 asks the operating system for any free port, which lets tests run
    side by side. Returns the address as a normalized string.
    """
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
    """The only production loader: the verified local CUDA-FP16 encoder VAE.

    Returns `(engine, info)`. `engine(image, mask, cancel=..., progress=...)`
    protects one photo with the already-loaded VAE; `info` is a few plain
    strings for the "model_loaded" log line. torch and the encoder are
    imported here, not at the top, so the tests can import this module on a
    computer without a GPU.
    """
    # Belt and braces: the Hugging Face libraries must never try the network,
    # even if a file were missing. (The systemd unit sets the same variables.)
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
        """Protect one photo; `hooks` carries the store's cancel and progress callbacks."""
        return encoder.protect(image, mask, components=components, **hooks)

    return engine, {"device": "cuda:0", "dtype": "float16", "engine_id": encoder.ENGINE_ID,
                    "model_source_digest": components.source_digest}


def _safe_info(info):
    """Keep only the four known keys, and only values made of safe characters.

    The loader can be replaced in tests, so its info dict is not trusted to
    be log-safe on its own.
    """
    safe = {}
    for key in ("device", "dtype", "engine_id", "model_source_digest"):
        value = info.get(key) if isinstance(info, dict) else None
        if isinstance(value, str) and _SAFE_TEXT.fullmatch(value):
            safe[key] = value
    return safe


class ServiceServer(worker.WorkerServer):
    """The worker's HTTP server with one change: quick rebinding on Linux after a restart."""
    # Every response closes its connection, so a restart leaves TIME_WAIT
    # sockets on the port. On Linux, SO_REUSEADDR only allows binding over
    # those, never beside a live listener; without it a systemd restart cannot
    # rebind for about 60 s. BSD semantics differ, so elsewhere keep the default.
    allow_reuse_address = sys.platform.startswith("linux")


class ServiceStore(private_session.SessionStore):
    """Session store with the capability shape unchanged and early slot release.

    The website compares /v1/capabilities field by field against what it
    expects, so the reply must look exactly like the plain worker's, plus
    `efficacy_evaluated: false`.
    """

    def capabilities(self):
        """The plain worker's capabilities; the session fields are left out."""
        # Skip the session fields; an unbounded service has no cap or deadline.
        with self._lock:
            return dict(worker.JobStore.capabilities(self), efficacy_evaluated=False)

    def get(self, job_id, token, *, output=False):
        """Serve a GET as usual; after an output download, shorten the job's life to 120 s.

        This is what keeps one user from holding the only slot for the full
        30 minutes after they already have their picture.
        """
        with self._lock:
            result = super().get(job_id, token, output=output)
            job = self._job
            if output and job is not None and job.id == job_id:
                # min(): a later download never extends the first one's window,
                # and nothing can outlive the 1800 s expiry set at submission.
                job.expires = min(job.expires, self.clock() + RELEASE_AFTER_OUTPUT_SECONDS)
            return result


class ServiceSession(private_session.PrivateSession):
    """PrivateSession with the finite job cap and deadline lifted.

    It also logs one "engine_call" line per engine run: whether it was the
    start-up self-test ("smoke") or a user's job, whether it returned or
    raised, and how many seconds it took. Never the image or the job id.
    """

    def __init__(self, *, clock=time.monotonic):
        # The parent validates finite limits; satisfy it, then lift both.
        super().__init__(max_jobs=1, deadline=clock() + 60, clock=clock)
        self.max_jobs = math.inf
        self.deadline = math.inf
        # Replace the parent's store with one that releases slots early.
        self.store = ServiceStore(self)

    def checked_protect(self, image, mask, *, cancel, progress):
        """The parent's checked run, timed and logged."""
        # Until the self-test has passed, the only caller is the self-test.
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
        """The parent's summary, with "no job cap" shown as null and the release delay added."""
        value = super().summary()
        value.update(maximum_session_jobs=None,
                     release_after_output_seconds=RELEASE_AFTER_OUTPUT_SECONDS)
        return value


class WorkerService:
    """Bind first, then load, smoke and serve until a stop is requested.

    The constructor checks the bind address and the bearer and opens the
    socket; `run()` does everything else and returns the process exit code.
    `server_factory` and `run(loader=..., hard_exit=...)` let the tests swap in
    fakes for the socket, the GPU loader and the hard exit.
    """

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
        """(host, port) actually bound; with port 0 this shows which port the OS picked."""
        return tuple(self.server.server_address[:2])

    def request_stop(self, *_):
        """Signal-handler safe: only sets events.

        Called on SIGTERM (systemctl stop) or SIGINT (Ctrl-C). Setting
        session.stop makes the store report not ready at once and the running
        engine see a cancel at its next check; run() then cleans up.
        """
        self._stop_requested.set()
        self.session.stop.set()

    def close(self):
        """Release resources of a service whose run() was never called."""
        self.session.stop.set()
        self.server.server_close()
        self.session.close()

    def run(self, snapshot, *, loader=None, grace_seconds=CLEANUP_GRACE_SECONDS, hard_exit=None):
        """Serve until stopped. Returns 0 after a clean stop, 2 after any failure.

        `phase` names the step in progress, so a failure's log line says
        whether it happened during the model load, the self-test or serving.
        A failure caused by a requested stop is not reported as a failure.
        """
        require(not self._ran, "A service instance runs once")
        self._ran = True
        session, store = self.session, self.session.store
        load = load_cuda_engine if loader is None else loader
        phase, failure, serving = "listening", None, False
        started = time.monotonic()
        # Started before anything that might hang, so even a stuck model load
        # cannot keep the process alive more than grace_seconds after a stop.
        watchdog_done, watchdog = private_session.start_deadline_watchdog(
            session, grace_seconds=grace_seconds, hard_exit=hard_exit)
        try:
            # HTTP runs on its own thread from the start; until the self-test
            # passes it answers POST /v1/jobs with 503 "not_ready".
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
            # The main thread just waits; the HTTP thread and the job threads
            # do the work. wait(1) wakes once a second until a stop is set.
            while not session.stop.wait(1):
                pass
        except BaseException as error:
            if not self._stop_requested.is_set():
                failure = {**private_session.sanitized_exception(error), "phase": phase}
        finally:
            # Shut down in order: refuse new work, stop the HTTP thread, close
            # the socket, wait for the engine, write the last log line, and
            # finally tell the watchdog that cleanup is done.
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
    """Parse the options, read the bearer, open the socket, and run until stopped.

    A bad bearer or a failed bind logs one "stopped" line naming the phase
    and returns 2, so the journal shows why a start did not succeed. (Bad
    options are reported by argparse, which also exits with code 2.)
    """
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
    # Install the stop handlers, and put the old ones back afterwards so that
    # calling main() from a test leaves the process as it found it.
    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, service.request_stop)
        return service.run(args.snapshot)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    # Any unexpected error ends with a fixed one-line message instead of a
    # traceback, which could include request data or file paths.
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit("Worker service unavailable; see its sanitized journal events.") from None
