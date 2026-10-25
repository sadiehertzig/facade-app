"""Tests for facade_photoguard_service.py, the always-on /v1 worker program.

Fake engines, synthetic images, loopback HTTP only. No model, CUDA, provider or
public port. Every listener binds 127.0.0.1 with an OS-assigned port. The last
test starts the service as a real child process and stops it with SIGTERM.
"""
import ast
import base64
from contextlib import contextmanager
import http.client
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
from PIL import Image

WORKER_DIR = Path(__file__).resolve().parents[1] / "worker"
import facade_photoguard_private_session as private_session
import facade_photoguard_service as service
import facade_photoguard_worker as worker

TOKEN = "c0ffee" + "7" * 58
# What the website (web/lib/processor.ts) requires before it sends a job: every
# field is compared with === in JavaScript, so types matter too.
WEBSITE_REQUIRED = {"ready": True, "gpu_smoke_validated": True, "recipe": "photoguard-encoder-linf-v1",
                 "image_side": 512, "total_iterations": 200, "retention_seconds": 1800,
                 "experimental": True}
READY_CAPABILITIES = {"ready": True, "recipe": "photoguard-encoder-linf-v1", "image_side": 512,
                      "total_iterations": 200, "retention_seconds": 1800, "experimental": True,
                      "gpu_smoke_validated": True, "engine_smoke_validated": True,
                      "execution_device": "cuda:0", "efficacy_evaluated": False}
FAKE_INFO = {"device": "fake", "dtype": "fake", "engine_id": "fake-engine",
             "model_source_digest": "not-a-model"}


def engine(image, mask, *, cancel, progress):
    """A well-behaved fake engine: reports every iteration 0..200, honours cancel, returns a copy of the photo."""
    for iteration in range(201):
        if cancel():
            raise RuntimeError("cancelled")
        progress(iteration, 200)
    return image.copy()


def phased(job_implementation, smoke_implementation=engine):
    """First engine call is the fresh smoke; every later call is a user job."""
    calls = []
    def implementation(image, mask, **hooks):
        calls.append(1)
        chosen = smoke_implementation if len(calls) == 1 else job_implementation
        return chosen(image, mask, **hooks)
    return implementation


def loader_for(implementation):
    """A fake loader that hands `implementation` to the service instead of loading the VAE."""
    def load(snapshot, *, cancel):
        return implementation, dict(FAKE_INFO)
    return load


def png(image):
    """Encode a Pillow image as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_body():
    """A valid POST /v1/jobs JSON body built from the synthetic fixture, with one pixel changed."""
    image, mask = private_session.analytic_fixture()
    try:
        image.putpixel((0, 0), (20, 40, 60))
        return json.dumps({"image_png": base64.b64encode(png(image)).decode(),
                           "mask_png": base64.b64encode(png(mask)).decode(),
                           "recipe": service.RECIPE}).encode()
    finally:
        image.close(); mask.close()


BODY = make_body()


def request(port, method, path, token=None, data=b""):
    """Send one HTTP request with the test bearer to the loopback service and return (status, body)."""
    headers = {"Authorization": "Bearer " + TOKEN}
    if method == "POST":
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Job-Token"] = token
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def capabilities(port):
    """GET /v1/capabilities and return the parsed JSON."""
    status, data = request(port, "GET", "/v1/capabilities")
    assert status == 200
    return json.loads(data)


def wait_ready(port, seconds=10):
    """Poll capabilities until the service reports ready, or fail after `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if capabilities(port)["ready"]:
            return
        time.sleep(.02)
    raise AssertionError("service never became ready")


@contextmanager
def running(implementation=engine, *, clock=time.monotonic, loader=None):
    """Run a WorkerService on a background thread for the duration of a `with` block, then stop it."""
    value = service.WorkerService(TOKEN, port=0, clock=clock)
    results, exits = [], []
    thread = threading.Thread(target=lambda: results.append(value.run(
        Path("/not-loaded"), loader=loader or loader_for(implementation), hard_exit=exits.append)),
        daemon=True)
    thread.start()
    try:
        yield value, value.address[1], results
    finally:
        value.request_stop()
        thread.join(15)
        assert not thread.is_alive()
        assert exits == []  # The hard-exit watchdog never fired.


def submit_local(store):
    """Submit the synthetic fixture directly to a store and wait for its engine thread."""
    image, mask = private_session.analytic_fixture()
    job = store.submit(image, mask)
    store._worker.join(3)
    assert not store._worker.is_alive()
    return job


def validated(clock, implementation=engine):
    """A ServiceSession whose self-test has already passed, with the given engine."""
    session = service.ServiceSession(clock=clock)
    session.install_engine(implementation)
    session.validate_smoke()
    return session


def json_lines(text):
    """Parse the service's stderr, one JSON object per line."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- bearer and bind -------------------------------------------------------

@pytest.mark.parametrize("text", [TOKEN, TOKEN + "\n"])
def test_bearer_accepts_exactly_one_hex64_line(text):
    """The bearer is accepted with or without one trailing newline."""
    assert service.read_bearer(io.StringIO(text)) == TOKEN


@pytest.mark.parametrize("text", [
    "", "\n", TOKEN.upper(), TOKEN[:-1], TOKEN + "0", " " + TOKEN, TOKEN + " ", TOKEN + "\r\n",
    TOKEN + "\n\n", TOKEN + "\n" + TOKEN, "g" * 64, "0x" + TOKEN[2:], "a" * 5000,
])
def test_bearer_rejects_anything_else_without_echoing_it(text):
    """Any other bearer text is refused with a fixed message that does not repeat it."""
    with pytest.raises(RuntimeError) as caught:
        service.read_bearer(io.StringIO(text))
    assert str(caught.value) == "Expected exactly one hex64 bearer on stdin"


def test_cli_rejects_bad_stdin_before_any_socket(monkeypatch, capsys):
    """A bad bearer on stdin stops the program before it opens a socket, and the bearer is never logged."""
    near_miss = TOKEN[:-1] + "G"
    monkeypatch.setattr(sys, "stdin", io.StringIO(near_miss + "\n"))
    monkeypatch.setattr(service, "ServiceServer", lambda *a: pytest.fail("must not bind"))
    assert service.main(["--snapshot", "/not-loaded"]) == 2
    captured = capsys.readouterr()
    assert near_miss not in captured.err and near_miss not in captured.out
    event = json_lines(captured.err)[-1]
    assert event["event"] == "stopped" and event["phase"] == "bearer_input"


@pytest.mark.parametrize("argv", [
    ["--bind", "0.0.0.0"], ["--bind", "10.0.0.5"], ["--bind", "::1"], ["--bind", "::"],
    ["--bind", "localhost"], ["--port", "0"], ["--port", "65536"], ["--port", "-1"],
])
def test_cli_refuses_non_loopback_or_bad_port_before_stdin_or_socket(argv, monkeypatch):
    """A non-loopback address or an invalid port is refused before stdin is read or a socket opened."""
    monkeypatch.setattr(service, "read_bearer", lambda *a: pytest.fail("must not read the bearer"))
    monkeypatch.setattr(service, "ServiceServer", lambda *a: pytest.fail("must not bind"))
    with pytest.raises(SystemExit) as caught:
        service.main(["--snapshot", "/not-loaded", *argv])
    assert caught.value.code == 2


@pytest.mark.parametrize("bind", ["0.0.0.0", "192.168.1.2", "::1", "localhost", 0])
def test_service_refuses_non_loopback_bind_before_creating_a_socket(bind):
    """WorkerService itself refuses non-loopback addresses before creating a socket."""
    def factory(*args):
        pytest.fail("must not bind")
    with pytest.raises(ValueError):
        service.WorkerService(TOKEN, bind=bind, port=0, server_factory=factory)


def test_remote_bind_requires_the_explicit_override_and_bearer_must_be_hex64():
    """Remote binding needs the explicit override, port 0 is test-only, and a non-hex64 bearer is refused."""
    with pytest.raises(ValueError):
        service.validate_bind("0.0.0.0", 8000)
    assert service.validate_bind("0.0.0.0", 8000, allow_remote_bind=True) == "0.0.0.0"
    assert service.validate_bind("127.0.0.1", 8000) == "127.0.0.1"
    with pytest.raises(ValueError):
        service.validate_bind("127.0.0.1", 0)  # Ephemeral ports are for tests only.
    for bearer in ("a" * 32, "A" * 64, TOKEN + "0", None):
        with pytest.raises(RuntimeError):
            service.WorkerService(bearer, port=0, server_factory=lambda *a: pytest.fail("bound"))


def test_bearer_comes_only_from_stdin_and_no_validation_bypass_exists():
    """The source has no flag or environment variable that could skip the self-test, and loading, self-test and ready happen in that order."""
    source = (WORKER_DIR / "facade_photoguard_service.py").read_text()
    for forbidden in ("--gpu-smoke-validated", "--engine-smoke-validated", "--max-jobs",
                      "--deadline", "FACADE_PHOTOGUARD_API_TOKEN", "environ.get", "getenv"):
        assert forbidden not in source
    tree = ast.parse(source)
    run = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    serve = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "load")
    smoke = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "validate_smoke")
    ready = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "log_event"
                 and node.args and getattr(node.args[0], "value", None) == "ready")
    assert serve.lineno < smoke.lineno < ready.lineno


# --- readiness ----------------------------------------------------------------

def test_listener_answers_not_ready_until_the_fresh_smoke_passes():
    """While the model loads, the listener answers but reports not ready and refuses jobs with 503."""
    gate, calls = threading.Event(), []
    def counted(image, mask, **hooks):
        calls.append(1)
        return engine(image, mask, **hooks)
    def slow_loader(snapshot, *, cancel):
        assert gate.wait(5)
        return counted, dict(FAKE_INFO)
    with running(loader=slow_loader) as (value, port, results):
        cap = capabilities(port)
        assert not cap["ready"] and not cap["gpu_smoke_validated"] and not cap["engine_smoke_validated"]
        assert request(port, "POST", "/v1/jobs", data=BODY)[0] == 503
        assert request(port, "GET", "/v1/capabilities")[0] == 200
        assert calls == []
        gate.set()
        wait_ready(port)
        assert calls == [1]  # Exactly one fresh smoke, before any user job.
        assert capabilities(port)["gpu_smoke_validated"] is True
    assert results == [0]


@pytest.mark.parametrize("mode", ["white", "bound", "short"])
def test_failed_fresh_smoke_never_opens_readiness_and_exits_nonzero(mode, capsys):
    """A self-test that changes a white pixel, exceeds the bound or stops early ends the service with exit code 2."""
    def invalid(image, mask, *, cancel, progress):
        for iteration in range(201 if mode != "short" else 3):
            progress(iteration, 200)
        result = image.copy()
        if mode == "white":
            result.putpixel((200, 200), (0, 0, 0))
        elif mode == "bound":
            result.putpixel((0, 0), (255, 255, 255))
        return result
    value = service.WorkerService(TOKEN, port=0)
    exits = []
    assert value.run(Path("/not-loaded"), loader=loader_for(invalid), hard_exit=exits.append) == 2
    assert exits == [] and not value.session.store.ready
    assert not value.session.store.capabilities()["gpu_smoke_validated"]
    stopped = json_lines(capsys.readouterr().err)[-1]
    assert stopped["event"] == "stopped" and stopped["status"] == "FAILED"
    assert stopped["phase"] == "fresh_synthetic_smoke"


def test_capabilities_match_the_website_contract_exactly():
    """Once ready, /v1/capabilities has exactly the fields and value types the website requires."""
    with running() as (value, port, results):
        wait_ready(port)
        cap = capabilities(port)
        for key, expected in WEBSITE_REQUIRED.items():
            assert cap[key] == expected and type(cap[key]) is type(expected), key
        assert cap == READY_CAPABILITIES
        assert value.session.store.capabilities() == READY_CAPABILITIES


# --- unlimited jobs and release after download ------------------------------

def test_twenty_five_sequential_jobs_without_delete_release_after_download(capsys):
    """Twenty-five jobs run back to back with no job cap or deadline; each slot frees 120 s after download, and nothing secret is logged."""
    now = [1000.]
    seen = []
    with running(clock=lambda: now[0]) as (value, port, results):
        wait_ready(port)
        store = value.session.store
        for number in range(25):
            status, data = request(port, "POST", "/v1/jobs", data=BODY)
            assert status == 202, number
            job = json.loads(data)
            seen.append(job)
            store._worker.join(3)
            path = "/v1/jobs/" + job["id"]
            status, data = request(port, "GET", path, job["access_token"])
            assert status == 200 and json.loads(data)["status"] == "completed"
            status, output = request(port, "GET", path + "/output", job["access_token"])
            assert status == 200
            with Image.open(io.BytesIO(output)) as delivered:
                assert delivered.size == (512, 512) and delivered.mode == "RGB" and not delivered.info
                assert delivered.getpixel((0, 0)) == (20, 40, 60)
            now[0] += 119
            assert request(port, "POST", "/v1/jobs", data=BODY)[0] == 409  # Still held.
            assert request(port, "GET", path + "/output", job["access_token"])[0] == 200
            now[0] += 1  # 120 s after the first download: released, no DELETE sent.
            if number == 12:
                now[0] += 90 * 86400  # Months pass; there is no session deadline.
                assert capabilities(port)["ready"]
        store.reap()
        assert store._job is None
        last = seen[-1]
        assert request(port, "GET", "/v1/jobs/" + last["id"], last["access_token"])[0] == 404
        assert request(port, "DELETE", "/v1/jobs/" + last["id"], last["access_token"])[0] == 404
        summary = value.session.summary()
        assert summary["admitted_jobs"] == 25 > private_session.MAX_SESSION_JOBS
        assert summary["checked_user_engine_calls"] == {"returned": 25, "raised": 0}
        assert summary["maximum_session_jobs"] is None
        assert capabilities(port) == READY_CAPABILITIES
    assert results == [0]
    err = capsys.readouterr().err
    assert TOKEN not in err and "iVBORw0KGgo" not in err
    for job in seen:
        assert job["id"] not in err and job["access_token"] not in err
    events = json_lines(err)
    assert [event["kind"] for event in events if event["event"] == "engine_call"] == ["smoke"] + ["job"] * 25
    assert events[-1]["event"] == "stopped" and events[-1]["status"] == "STOPPED"


def test_hard_expiry_still_bounds_undownloaded_and_late_downloaded_jobs():
    """The 1800 s expiry still frees a never-downloaded job, and a late download cannot extend it."""
    now = [0.]
    session = validated(lambda: now[0])
    store = session.store
    try:
        job = submit_local(store)
        now[0] = 1799
        with pytest.raises(worker.RequestError) as caught:
            submit_local(store)
        assert caught.value.code == "busy"
        now[0] = 1800  # Never downloaded: the unchanged 30-minute expiry releases it.
        second = submit_local(store)
        with pytest.raises(worker.RequestError):
            store.get(job["id"], job["access_token"])
        now[0] = 1800 + 1750  # Download 50 s before the hard expiry: not extended.
        assert store.get(second["id"], second["access_token"], output=True)
        assert store._job.expires == 3600
        now[0] = 3599
        store.reap()
        assert store.get(second["id"], second["access_token"])["status"] == "completed"
        now[0] = 3600
        store.reap()
        assert store._job is None
    finally:
        session.close()


def test_first_download_starts_the_window_and_early_reads_do_not():
    """Only the first successful output download starts the 120 s window; status reads, repeats and wrong tokens do not."""
    now = [0.]
    gate = threading.Event()
    def gated(image, mask, **hooks):
        assert gate.wait(3)
        return engine(image, mask, **hooks)
    session = validated(lambda: now[0], phased(gated))
    store = session.store
    try:
        image, mask = private_session.analytic_fixture()
        running_job = store.submit(image, mask)
        with pytest.raises(worker.RequestError) as caught:
            store.get(running_job["id"], running_job["access_token"], output=True)
        assert caught.value.code == "not_completed" and store._job.expires == 1800
        gate.set()
        store._worker.join(3)
        now[0] = 10
        store.get(running_job["id"], running_job["access_token"])  # Status reads do not release.
        assert store._job.expires == 1800
        store.get(running_job["id"], running_job["access_token"], output=True)
        assert store._job.expires == 130
        now[0] = 100
        store.get(running_job["id"], running_job["access_token"], output=True)
        assert store._job.expires == 130  # A second download does not extend it.
        with pytest.raises(worker.RequestError):
            store.get(running_job["id"], "0" * 64, output=True)  # Wrong token changes nothing.
        assert store._job.expires == 130
        now[0] = 130
        store.reap()
        assert store._job is None
    finally:
        gate.set()
        session.close()


def test_hard_expiry_still_cancels_an_active_job():
    """A job still running at 1800 s is cancelled, removed once its engine returns, and the slot reopens."""
    now = [0.]
    entered, release, saw_cancel = threading.Event(), threading.Event(), []
    def slow(image, mask, *, cancel, progress):
        if entered.is_set():
            return engine(image, mask, cancel=cancel, progress=progress)
        entered.set()
        assert release.wait(3)
        saw_cancel.append(cancel())
        raise RuntimeError("cancelled")
    session = validated(lambda: now[0], phased(slow))
    store = session.store
    try:
        image, mask = private_session.analytic_fixture()
        job = store.submit(image, mask)
        assert entered.wait(3)
        now[0] = 1800
        store.reap()
        assert store.get(job["id"], job["access_token"])["status"] == "cancelling"
        release.set()
        store._worker.join(3)
        assert saw_cancel == [True]
        assert store._job is None
        with pytest.raises(worker.RequestError):
            store.get(job["id"], job["access_token"])
        assert submit_local(store)["status"] == "queued"
    finally:
        release.set()
        session.close()


def test_per_job_output_validation_is_kept_and_a_failed_job_frees_the_slot():
    """A user job whose output changed a white pixel fails the checks and frees the slot at once."""
    now = [0.]
    def tamper(image, mask, *, cancel, progress):
        result = engine(image, mask, cancel=cancel, progress=progress)
        result.putpixel((256, 256), (1, 2, 3))  # Inside the white edit region.
        return result
    session = validated(lambda: now[0], phased(tamper))
    store = session.store
    try:
        job = submit_local(store)
        state = store.get(job["id"], job["access_token"])
        assert state["status"] == "failed" and state["cleanup_complete"]
        assert store._job.output is None
        submit_local(store)  # No DELETE and no clock advance needed after a failure.
        assert session.summary()["checked_user_engine_calls"] == {"returned": 0, "raised": 2}
    finally:
        session.close()


# --- shutdown -------------------------------------------------------------------

def test_stop_request_cancels_an_inflight_job_and_exits_cleanly():
    """A stop request cancels the running job, drops readiness and ends run() with exit code 0."""
    entered, saw_cancel = threading.Event(), []
    def slow(image, mask, *, cancel, progress):
        progress(0, 200)
        entered.set()
        while not cancel():
            time.sleep(.01)
        saw_cancel.append(True)
        raise RuntimeError("cancelled")
    with running(phased(slow)) as (value, port, results):
        wait_ready(port)
        assert request(port, "POST", "/v1/jobs", data=BODY)[0] == 202
        assert entered.wait(3)
        value.request_stop()
        deadline = time.monotonic() + 5
        while not results and time.monotonic() < deadline:
            time.sleep(.02)
        assert not value.session.store.ready
    assert results == [0] and saw_cancel == [True]


DRIVER = r"""
import sys
sys.path.insert(0, sys.argv[1])
import facade_photoguard_service as service

def engine(image, mask, *, cancel, progress):
    for iteration in range(201):
        if cancel():
            raise RuntimeError("cancelled")
        progress(iteration, 200)
    return image.copy()

service.load_cuda_engine = lambda snapshot, *, cancel=None: (engine, {"device": "fake"})
Base = service.ServiceServer
class Loopback(Base):
    def __init__(self, address, api):
        assert address == ("127.0.0.1", 8000)
        super().__init__(("127.0.0.1", 0), api)
        print(self.server_address[1], flush=True)
service.ServiceServer = Loopback
raise SystemExit(service.main(["--snapshot", "/not-loaded"]))
"""


def test_sigterm_stops_the_real_process_gracefully_and_logs_no_secrets():
    """A real child process serves one job, exits 0 on SIGTERM, logs the expected events and never logs a secret."""
    process = subprocess.Popen([sys.executable, "-c", DRIVER, str(WORKER_DIR)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        process.stdin.write(TOKEN + "\n")
        process.stdin.close()
        line = process.stdout.readline()
        assert line.strip().isdigit(), process.stderr.read()
        port = int(line)
        wait_ready(port, seconds=30)
        status, data = request(port, "POST", "/v1/jobs", data=BODY)
        assert status == 202
        job = json.loads(data)
        deadline = time.monotonic() + 10
        while True:
            status, data = request(port, "GET", "/v1/jobs/" + job["id"], job["access_token"])
            if json.loads(data)["status"] == "completed" or time.monotonic() > deadline:
                break
            time.sleep(.02)
        assert request(port, "GET", "/v1/jobs/" + job["id"] + "/output", job["access_token"])[0] == 200
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 0
        out, err = process.stdout.read(), process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
        process.stdout.close()
        process.stderr.close()
    for secret in (TOKEN, job["id"], job["access_token"], "iVBORw0KGgo"):
        assert secret not in out and secret not in err
    events = json_lines(err)
    names = [event["event"] for event in events]
    assert names[:3] == ["listening", "model_loaded", "engine_call"]
    assert "ready" in names and names[-1] == "stopped"
    assert events[0]["bind"] == "127.0.0.1" and events[0]["ready"] is False
    assert events[-1]["status"] == "STOPPED" and events[-1]["summary"]["admitted_jobs"] == 1
    with pytest.raises(OSError):
        request(port, "GET", "/v1/capabilities")  # The listener is closed.
