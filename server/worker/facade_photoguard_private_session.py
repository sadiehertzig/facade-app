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

What "session" means here: one run of the worker process with an optional job
cap and an optional deadline. This module enforces both (at most 20 jobs and
90 minutes) so that a bounded run can never outlive its limits. The always-on
service sets both to infinity and lets systemd decide how long the process
lives, but it keeps everything else in this file: the start-up self-test, the
output checks and the watchdog.

Why check the engine's output at all? The engine is GPU code that could, after
a driver or library problem, quietly return a wrong image. Every job's result
is therefore checked against what the encoder recipe guarantees:
the white region is unchanged, nothing moved by more than 16 of 255 levels,
the progress counter went 0, 1, 2, ... 200 with no gaps, and the PNG that
will actually be sent decodes back to the same pixels. A job that fails any
check fails as a whole; the user never gets a half-checked image.
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
# Upper limits a bounded run may ask for: 90 minutes and 20 jobs.
MAX_SESSION_SECONDS = 5400
MAX_SESSION_JOBS = 20
# A delivered PNG may be no larger than the worker accepts as a request (4 MiB).
MAX_BYTES = worker.MAX_REQUEST_BYTES


def require(value, message):
    """Raise RuntimeError(message) unless `value` is truthy: a one-line assertion that survives `python -O`."""
    if not value:
        raise RuntimeError(message)


def sanitized_exception(error):
    """No messages, locals, source lines, absolute paths, bodies or capabilities.

    An exception's message can quote its input (a file path, a token, part of
    an image), so only safe structure is kept: the exception's type name and,
    for the last 12 stack frames, the bare file name, function name and line
    number. Every string is filtered to a small set of characters and cut to a
    fixed length. Up to three chained exceptions ("raise ... from ...", or one
    raised while handling another) are followed. This is enough to find the
    failing line in the source without logging anything a user sent.
    """
    chains, seen = [], set()
    # `seen` stops a loop if two exceptions point at each other as causes.
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
        # An OSError's errno (a small number such as 28 for "disk full") is safe
        # to keep and often explains the failure on its own.
        if type(getattr(error, "errno", None)) is int:
            detail["errno"] = error.errno
        chains.append(detail)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return {"exception_chain": chains, "messages_omitted": True, "locals_omitted": True}


def analytic_fixture():
    """The synthetic photo and mask the start-up self-test protects.

    The photo is a colour pattern computed from a formula, so the self-test
    never needs a real photo of anyone. The mask is white (editable) on a
    192 x 192 square in the middle and black (protected) everywhere else, so
    both regions are present as the encoder requires.
    """
    y, x = np.indices((512, 512), dtype=np.int32)
    # Each channel is a different diagonal ramp that wraps around at 256, which
    # gives the VAE plenty of edges and colours to work on.
    rgb = np.stack(((3*x + 5*y + 17) % 256, (7*x + 11*y + 29) % 256,
                    (13*x + 17*y + 43) % 256), axis=-1).astype(np.uint8)
    mask = ((x >= 160) & (x < 352) & (y >= 160) & (y < 352)).astype(np.uint8) * 255
    return Image.fromarray(rgb), Image.fromarray(mask)


def validate_pixels(result, image, mask):
    """Check delivered bytes without returning or recording photo-derived data.

    Three promises: the result is a 512 x 512 RGB image; every pixel the mask
    marks white (editable) is byte-for-byte the input; and no colour value of
    any pixel moved by more than 16 levels. 16 is ceil(0.12 * 127.5), the
    encoder's epsilon in 8-bit units. A failure raises; nothing is returned.
    """
    require(isinstance(result, Image.Image) and result.mode == "RGB" and
            result.size == (512, 512), "Invalid encoder result")
    source = np.asarray(image, dtype=np.uint8)
    output = np.asarray(result, dtype=np.uint8)
    white = np.asarray(mask, dtype=np.uint8) == 255
    require(source.shape == output.shape == (512, 512, 3) and white.shape == (512, 512),
            "Invalid frame geometry")
    require(np.array_equal(output[white], source[white]), "Edit region changed")
    # int16 so that a subtraction like 3 - 10 gives -7 instead of wrapping
    # around to 249 as it would in uint8.
    delta = output.astype(np.int16) - source.astype(np.int16)
    require(int(np.abs(delta).max()) <= 16, "Delivered byte bound exceeded")


def validate_png(data, image, mask):
    """Decode the exact PNG bytes that will be sent and check them again.

    validate_pixels already checked the image in memory; this checks what the
    user will really download. It must be a single-frame PNG with no metadata
    at all (`result.info` empty: no text chunks, no ICC profile), at most
    4 MiB, and its decoded pixels must pass the same three promises.
    """
    require(len(data) <= MAX_BYTES, "Invalid output size")
    with Image.open(BytesIO(data)) as result:
        require(result.format == "PNG" and getattr(result, "n_frames", 1) == 1 and
                not result.info, "Invalid delivered PNG")
        result.load()
        validate_pixels(result, image, mask)


class SessionStore(worker.JobStore):
    """One retained job; next admission also requires the prior worker to join.

    This is the worker's JobStore with three additions: it stops admitting
    jobs once the run is cancelled or past its deadline, it counts admitted
    jobs against the cap, and a new job is refused ("busy") while the previous
    job's engine thread is still alive, even if that job was already deleted.
    The engine it runs is always `PrivateSession.checked_protect`, never the
    raw engine, so every job passes through the output checks.
    """
    def __init__(self, session):
        self.session = session
        # jobs accepted so far, counted against session.max_jobs
        self.admitted = 0
        super().__init__(session.checked_protect, recipe=RECIPE, execution_device="cuda:0",
                         clock=session.clock)

    @property
    def ready(self):
        """Ready as in JobStore, and the run has not been stopped or reached its deadline."""
        return super().ready and not self.session.cancelled()

    def capabilities(self):
        """JobStore's capabilities plus the run's remaining jobs and seconds.

        `efficacy_evaluated` is always False: the server says outright that it
        has not measured how well the protection works.
        """
        with self._lock:
            return dict(super().capabilities(), private_session=True,
                        maximum_session_jobs=self.session.max_jobs,
                        remaining_session_jobs=max(0, self.session.max_jobs - self.admitted),
                        session_seconds_remaining=max(0, int(self.session.deadline - self.clock())),
                        efficacy_evaluated=False)

    def reap(self):
        """Expire the held job as JobStore does, and also as soon as the run is stopped."""
        with self._lock:
            if self.session.cancelled() and self._job:
                self._job.expired = True
                self._cancel_locked(self._job)
                if not self._job.active:
                    self._job = None
            super().reap()

    def submit(self, image, mask):
        """Admit a job under the run's limits; the checks run in order, cheapest first."""
        with self._lock:
            self.reap()
            if self.session.cancelled():
                raise worker.RequestError(503, "session_ended")
            if self.admitted >= self.session.max_jobs:
                raise worker.RequestError(409, "session_capacity_exhausted")
            # One job at a time on the GPU: the last job's engine thread must
            # have finished, not just its job record. A deleted job's thread can
            # still be inside a GPU call that ignores the cancel flag.
            previous = self._worker
            if previous is not None:
                if previous.is_alive():
                    raise worker.RequestError(409, "busy")
                # The thread is already finished, so this returns at once; it
                # just collects the thread.
                previous.join(timeout=0)
            result = super().submit(image, mask)
            # The outer lock prevents the new worker from starting before this cap.
            self._job.expires = min(self._job.expires, self.session.deadline)
            self.admitted += 1
            return result

    def drained(self):
        """True once the job cap is used up and the last job has fully finished and been cleaned up."""
        with self._lock:
            thread = self._worker
            return (self.admitted >= self.session.max_jobs and
                    (thread is None or not thread.is_alive()) and
                    (self._job is None or self._job.cleanup_complete))


class PrivateSession:
    """Owns the engine and the store. The service installs the real CUDA engine;
    tests install small fake engines.

    Life of a run: `install_engine` once, then `validate_smoke` once (the store
    stays "not ready" until it passes), then jobs arrive through
    `store.submit`, then `close`. `stop` is a threading.Event that anything
    (a signal handler, the watchdog, close) can set to end the run.
    """
    def __init__(self, *, max_jobs: int, deadline: float, clock=time.monotonic):
        require(type(max_jobs) is int and 1 <= max_jobs <= MAX_SESSION_JOBS,
                "Finite session job cap required")
        # The deadline is a time.monotonic() value, which never jumps when the
        # wall clock is changed, and must lie at most 90 minutes ahead.
        require(math.isfinite(deadline) and 0 < deadline - clock() <= MAX_SESSION_SECONDS,
                "Bounded monotonic deadline required")
        self.max_jobs, self.deadline, self.clock = max_jobs, deadline, clock
        self.stop = threading.Event()
        self._engine: Callable | None = None
        self._smoke_attempted = False
        self._smoke_lock = threading.Lock()
        self._diagnostics_lock = threading.Lock()
        # sanitized_exception() records, at most three kept
        self._failures = []
        # user jobs only, not the self-test
        self._checked_user_calls = {"returned": 0, "raised": 0}
        self.store = SessionStore(self)

    def cancelled(self):
        """True once the run was stopped or its deadline has passed."""
        return self.stop.is_set() or self.clock() >= self.deadline

    def install_engine(self, engine):
        """Set the engine exactly once, before the self-test; it cannot be swapped later."""
        require(callable(engine) and self._engine is None and not self._smoke_attempted,
                "Engine must be installed once before validation")
        self._engine = engine

    def checked_protect(self, image, mask, *, cancel, progress):
        """Run the engine on one job and return its image only if every check passes.

        This is what the store calls for every job, and what the self-test
        calls once at start-up. It watches the progress counter as the engine
        runs, then checks the finished image in memory and again as the PNG
        that will be sent. On any failure the image is closed, a sanitized
        record is kept for the summary, and the exception is re-raised so the
        job fails.
        """
        require(self._engine is not None, "Engine unavailable")
        # the last iteration the engine reported; -1 means none yet
        count = -1
        result = None
        # Before the self-test passes, the only caller is the self-test itself,
        # so only calls after that count as user jobs.
        user_call = self.store.gpu_smoke_validated

        def checked_progress(iteration, total):
            """Accept progress only as 0, 1, 2, ... 200 with no skips or repeats, then pass it on."""
            nonlocal count
            require(type(iteration) is int and type(total) is int and total == 200 and
                    iteration == count + 1 and 0 <= iteration <= 200, "Invalid fixed progress")
            count = iteration
            progress(iteration, total)

        try:
            require(not self.cancelled() and not cancel(), "Cancelled")
            # The engine stops if either the job is cancelled or the whole run ends.
            result = self._engine(image, mask, cancel=lambda: cancel() or self.cancelled(),
                                  progress=checked_progress)
            # All 200 iterations must have been reported. An engine that
            # returned early, or a job cancelled at the last moment, is not delivered.
            require(count == 200 and not self.cancelled() and not cancel(), "Incomplete fixed job")
            validate_pixels(result, image, mask)
            png = worker._encode_png(result)
            validate_png(png, image, mask)
            with self._diagnostics_lock:
                if user_call:
                    self._checked_user_calls["returned"] += 1
            return result
        # BaseException, not Exception, so that even KeyboardInterrupt or
        # SystemExit still closes the image and is recorded before it continues.
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
        """Once only; no flag or saved result can bypass this fresh validation.

        Runs one full 200-iteration job on the synthetic fixture through
        `checked_protect`, on this process's GPU, right now. Only if it passes
        does the store report ready. A second call is refused, so a failed
        self-test cannot simply be run again; the process must restart.
        """
        with self._smoke_lock:
            require(not self._smoke_attempted, "Fresh smoke has already been attempted")
            self._smoke_attempted = True
            image, mask = analytic_fixture()
            result = None
            try:
                result = self.checked_protect(image, mask, cancel=self.cancelled,
                                              progress=lambda *_: None)
                # Taken under the store's lock so no request sees the store
                # half-switched to ready.
                with self.store._lock:
                    require(not self.cancelled(), "Session ended during validation")
                    self.store.engine_smoke_validated = True
                    self.store.gpu_smoke_validated = True
            finally:
                for item in (image, mask, result):
                    if item is not None:
                        item.close()

    def close(self, *, join_seconds=20):
        """Stop the run and wait up to `join_seconds` for the engine thread.

        Returns True if the thread finished, and only then lets go of the
        engine (and with it the VAE on the GPU). False means a GPU call is
        still running; the deadline watchdog is what ends the process then.
        """
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
        """Counts and flags for the final log line; it holds no image data and no tokens."""
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

    The caller sets the returned `done` event once its own cleanup has
    finished. If the run ends (stop or deadline) and `done` is still not set
    `grace_seconds` later, the process exits at once with code 124 (the code
    the `timeout` command uses). os._exit skips Python's normal shutdown, which
    is the point: normal shutdown would wait for the stuck GPU thread.
    """
    require(0 <= grace_seconds <= 20, "Bounded cleanup grace required")
    done = threading.Event()
    exit_process = os._exit if hard_exit is None else hard_exit

    def watch():
        """Poll every 0.2 s (or until the deadline, if sooner) for the end of the run."""
        while not done.is_set() and not session.cancelled():
            done.wait(min(.2, max(0, session.deadline - session.clock())))
        if done.is_set():
            return
        # Make sure every other part of the program sees the run as stopped.
        session.stop.set()
        if done.wait(grace_seconds):
            return
        exit_process(124)

    thread = threading.Thread(target=watch, name="private-session-deadline", daemon=True)
    thread.start()
    return done, thread
