"""Job session layer for the PhotoGuard encoder worker (a library, not a program).

facade_photoguard_service.py builds the always-on /v1 server from these pieces:

* ``PrivateSession`` owns the engine. Before anything is admitted it runs one
  fresh 200-step job on a synthetic image (``validate_smoke``); readiness opens
  only if that job's output passes the same checks every user job gets:
  exact fixed progress, the white edit region byte-identical, every pixel within
  16 levels of the input, and a clean re-decodable PNG.
* ``SessionStore`` is the worker's one-retained-job store with a job cap and a
  deadline; the service lifts both.
* ``sanitized_exception`` turns an exception into loggable metadata only: type
  names, file names and line numbers, never messages, locals or paths.
* ``start_deadline_watchdog`` hard-exits the process (code 124) if cleanup is
  still blocked some seconds after a stop, because a CUDA call cannot be
  interrupted cooperatively.

Image bytes and per-job tokens stay in memory and are never logged.
"""
from __future__ import annotations

from io import BytesIO
import math
import os
from pathlib import Path
import re
import threading
import time
import traceback
from typing import Callable

import numpy as np
from PIL import Image

import facade_photoguard_worker as worker

RECIPE = worker.ENCODER_RECIPE
MAX_SESSION_SECONDS = 5400
MAX_SESSION_JOBS = 20
MAX_BYTES = worker.MAX_REQUEST_BYTES


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sanitized_exception(error):
    """No messages, locals, source lines, absolute paths, bodies or capabilities."""
    chains, seen = [], set()
    while error is not None and id(error) not in seen and len(chains) < 3:
        seen.add(id(error))
        detail = {
            "type": re.sub(r"[^A-Za-z0-9_]", "_", type(error).__name__)[:80],
            "frames": [
                {"file": re.sub(r"[^A-Za-z0-9_.-]", "_", Path(frame.filename).name)[:100],
                 "function": re.sub(r"[^A-Za-z0-9_<>.-]", "_", frame.name)[:100],
                 "line": frame.lineno}
                for frame in traceback.extract_tb(error.__traceback__)[-12:]
            ],
        }
        if type(getattr(error, "errno", None)) is int:
            detail["errno"] = error.errno
        chains.append(detail)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return {"exception_chain": chains, "messages_omitted": True, "locals_omitted": True}


def analytic_fixture():
    y, x = np.indices((512, 512), dtype=np.int32)
    rgb = np.stack(((3*x + 5*y + 17) % 256, (7*x + 11*y + 29) % 256,
                    (13*x + 17*y + 43) % 256), axis=-1).astype(np.uint8)
    mask = ((x >= 160) & (x < 352) & (y >= 160) & (y < 352)).astype(np.uint8) * 255
    return Image.fromarray(rgb), Image.fromarray(mask)


def validate_pixels(result, image, mask):
    """Check delivered bytes without returning or recording photo-derived data."""
    require(isinstance(result, Image.Image) and result.mode == "RGB" and
            result.size == (512, 512), "Invalid encoder result")
    source = np.asarray(image, dtype=np.uint8)
    output = np.asarray(result, dtype=np.uint8)
    white = np.asarray(mask, dtype=np.uint8) == 255
    require(source.shape == output.shape == (512, 512, 3) and white.shape == (512, 512),
            "Invalid frame geometry")
    require(np.array_equal(output[white], source[white]), "Edit region changed")
    delta = output.astype(np.int16) - source.astype(np.int16)
    require(int(np.abs(delta).max()) <= 16, "Delivered byte bound exceeded")


def validate_png(data, image, mask):
    require(len(data) <= MAX_BYTES, "Invalid output size")
    with Image.open(BytesIO(data)) as result:
        require(result.format == "PNG" and getattr(result, "n_frames", 1) == 1 and
                not result.info, "Invalid delivered PNG")
        result.load()
        validate_pixels(result, image, mask)


class SessionStore(worker.JobStore):
    """One retained job; next admission also requires the prior worker to join."""
    def __init__(self, session):
        self.session = session
        self.admitted = 0
        super().__init__(session.checked_protect, recipe=RECIPE, execution_device="cuda:0",
                         clock=session.clock)

    @property
    def ready(self):
        return super().ready and not self.session.cancelled()

    def capabilities(self):
        with self._lock:
            return dict(super().capabilities(), private_session=True,
                        maximum_session_jobs=self.session.max_jobs,
                        remaining_session_jobs=max(0, self.session.max_jobs - self.admitted),
                        session_seconds_remaining=max(0, int(self.session.deadline - self.clock())),
                        efficacy_evaluated=False)

    def reap(self):
        with self._lock:
            if self.session.cancelled() and self._job:
                self._job.expired = True
                self._cancel_locked(self._job)
                if not self._job.active:
                    self._job = None
            super().reap()

    def submit(self, image, mask):
        with self._lock:
            self.reap()
            if self.session.cancelled():
                raise worker.RequestError(503, "session_ended")
            if self.admitted >= self.session.max_jobs:
                raise worker.RequestError(409, "session_capacity_exhausted")
            previous = self._worker
            if previous is not None:
                if previous.is_alive():
                    raise worker.RequestError(409, "busy")
                previous.join(timeout=0)
            result = super().submit(image, mask)
            # The outer lock prevents the new worker from starting before this cap.
            self._job.expires = min(self._job.expires, self.session.deadline)
            self.admitted += 1
            return result

    def drained(self):
        with self._lock:
            thread = self._worker
            return (self.admitted >= self.session.max_jobs and
                    (thread is None or not thread.is_alive()) and
                    (self._job is None or self._job.cleanup_complete))


class PrivateSession:
    """Owns the engine and the store. The service installs the real CUDA engine;
    tests install small fake engines."""
    def __init__(self, *, max_jobs: int, deadline: float, clock=time.monotonic):
        require(type(max_jobs) is int and 1 <= max_jobs <= MAX_SESSION_JOBS,
                "Finite session job cap required")
        require(math.isfinite(deadline) and 0 < deadline - clock() <= MAX_SESSION_SECONDS,
                "Bounded monotonic deadline required")
        self.max_jobs, self.deadline, self.clock = max_jobs, deadline, clock
        self.stop = threading.Event()
        self._engine: Callable | None = None
        self._smoke_attempted = False
        self._smoke_lock = threading.Lock()
        self._diagnostics_lock = threading.Lock()
        self._failures = []
        self._checked_user_calls = {"returned": 0, "raised": 0}
        self.store = SessionStore(self)

    def cancelled(self):
        return self.stop.is_set() or self.clock() >= self.deadline

    def install_engine(self, engine):
        require(callable(engine) and self._engine is None and not self._smoke_attempted,
                "Engine must be installed once before validation")
        self._engine = engine

    def checked_protect(self, image, mask, *, cancel, progress):
        require(self._engine is not None, "Engine unavailable")
        count = -1
        result = None
        user_call = self.store.gpu_smoke_validated

        def checked_progress(iteration, total):
            nonlocal count
            require(type(iteration) is int and type(total) is int and total == 200 and
                    iteration == count + 1 and 0 <= iteration <= 200, "Invalid fixed progress")
            count = iteration
            progress(iteration, total)

        try:
            require(not self.cancelled() and not cancel(), "Cancelled")
            result = self._engine(image, mask, cancel=lambda: cancel() or self.cancelled(),
                                  progress=checked_progress)
            require(count == 200 and not self.cancelled() and not cancel(), "Incomplete fixed job")
            validate_pixels(result, image, mask)
            png = worker._encode_png(result)
            validate_png(png, image, mask)
            with self._diagnostics_lock:
                if user_call:
                    self._checked_user_calls["returned"] += 1
            return result
        except BaseException as error:
            if result is not None:
                result.close()
            with self._diagnostics_lock:
                if user_call:
                    self._checked_user_calls["raised"] += 1
                if len(self._failures) < 3:
                    self._failures.append(sanitized_exception(error))
            raise

    def validate_smoke(self):
        """Once only; no flag or saved result can bypass this fresh validation."""
        with self._smoke_lock:
            require(not self._smoke_attempted, "Fresh smoke has already been attempted")
            self._smoke_attempted = True
            image, mask = analytic_fixture()
            result = None
            try:
                result = self.checked_protect(image, mask, cancel=self.cancelled,
                                              progress=lambda *_: None)
                with self.store._lock:
                    require(not self.cancelled(), "Session ended during validation")
                    self.store.engine_smoke_validated = True
                    self.store.gpu_smoke_validated = True
            finally:
                for item in (image, mask, result):
                    if item is not None:
                        item.close()

    def close(self, *, join_seconds=20):
        self.stop.set()
        self.store.close()
        thread = self.store._worker
        if thread is not None:
            thread.join(timeout=join_seconds)
        active = thread is not None and thread.is_alive()
        if not active:
            self._engine = None
            self.store.engine = None
        return not active

    def summary(self):
        with self.store._lock, self._diagnostics_lock:
            return {"recipe": RECIPE, "maximum_session_jobs": self.max_jobs,
                    "admitted_jobs": self.store.admitted,
                    "checked_user_engine_calls": dict(self._checked_user_calls),
                    "fresh_smoke_passed": self.store.gpu_smoke_validated,
                    "ready": self.store.ready, "diagnostics": list(self._failures),
                    "image_data_persisted": False, "job_tokens_persisted": False,
                    "efficacy_evaluated": False}


def start_deadline_watchdog(session, *, grace_seconds=20, hard_exit=None):
    """Terminate this process after its deadline/cancellation plus bounded cleanup.

    A CUDA call or synchronous model load may ignore cooperative cancellation.
    Only a real server starts this watchdog; injected exits make its tests harmless.
    """
    require(0 <= grace_seconds <= 20, "Bounded cleanup grace required")
    done = threading.Event()
    exit_process = os._exit if hard_exit is None else hard_exit

    def watch():
        while not done.is_set() and not session.cancelled():
            done.wait(min(.2, max(0, session.deadline - session.clock())))
        if done.is_set():
            return
        session.stop.set()
        if done.wait(grace_seconds):
            return
        exit_process(124)

    thread = threading.Thread(target=watch, name="private-session-deadline", daemon=True)
    thread.start()
    return done, thread
