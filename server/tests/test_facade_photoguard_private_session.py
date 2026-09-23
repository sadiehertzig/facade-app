"""Session layer: fake engines, synthetic images, loopback HTTP only. No model or GPU."""
import base64
import errno
from io import BytesIO
import json
import threading

import pytest
from PIL import Image

import facade_photoguard_private_session as session_layer
import facade_photoguard_worker as worker


def engine(image, mask, *, cancel, progress):
    for iteration in range(201):
        if cancel():
            raise RuntimeError("cancelled")
        progress(iteration, 200)
    return image.copy()


def session(max_jobs=2, clock=None, deadline=None, implementation=engine):
    import time
    clock = clock or time.monotonic
    value = session_layer.PrivateSession(max_jobs=max_jobs, deadline=deadline or clock()+300, clock=clock)
    value.install_engine(implementation)
    return value


def png(image):
    result = BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


def submit(store):
    image, mask = session_layer.analytic_fixture()
    try:
        result = store.submit(image, mask)
    except BaseException:
        image.close(); mask.close()
        raise
    store._worker.join(3)
    assert not store._worker.is_alive()
    return result


def body():
    image, mask = session_layer.analytic_fixture()
    try:
        # An arbitrary owner-authored synthetic frame proves admission is no longer fixture-only.
        image.putpixel((0, 0), (20, 40, 60))
        return json.dumps({"image_png": base64.b64encode(png(image)).decode(),
                           "mask_png": base64.b64encode(png(mask)).decode(),
                           "recipe": session_layer.RECIPE}).encode()
    finally:
        image.close(); mask.close()


def request(port, method, path, bearer, token=None, data=b""):
    import http.client
    headers = {"Authorization": "Bearer " + bearer, "Content-Type": "application/json"}
    if token:
        headers["X-Job-Token"] = token
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_unvalidated_session_cannot_admit_and_smoke_is_once_only():
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        return engine(*args, **kwargs)
    value = session(implementation=fake)
    try:
        api = worker.WorkerAPI(value.store, "a"*64)
        assert api.handle("GET", "/v1/capabilities", {})[0] == 401
        assert api.handle("POST", "/v1/jobs", {"authorization": "Bearer " + "a"*64,
                                             "content-type": "application/json"}, body())[0] == 503
        assert calls == []
        value.validate_smoke()
        assert value.store.ready and value.store.capabilities()["gpu_smoke_validated"]
        with pytest.raises(RuntimeError):
            value.validate_smoke()
        assert calls == [1]
    finally:
        value.close()


@pytest.mark.parametrize("mode", ["short", "skip", "bad-total", "white", "bound"])
def test_invalid_fresh_smoke_never_opens_readiness(mode):
    def invalid(image, mask, *, cancel, progress):
        if mode == "short":
            progress(0, 200)
        elif mode == "skip":
            progress(0, 200); progress(2, 200)
        elif mode == "bad-total":
            progress(0, 199)
        else:
            for iteration in range(201):
                progress(iteration, 200)
        result = image.copy()
        if mode == "white":
            result.putpixel((200, 200), (0, 0, 0))
        elif mode == "bound":
            result.putpixel((0, 0), (255, 255, 255))
        return result
    value = session(implementation=invalid)
    try:
        with pytest.raises(RuntimeError):
            value.validate_smoke()
        assert not value.store.ready
        assert not value.store.capabilities()["gpu_smoke_validated"]
    finally:
        value.close()


def test_two_http_jobs_require_delete_have_isolated_tokens_and_finite_capacity():
    value = session()
    server = worker.WorkerServer(("127.0.0.1", 0), worker.WorkerAPI(value.store, "b"*64))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True).start()
    port = server.server_address[1]
    try:
        assert request(port, "GET", "/v1/capabilities", "wrong")[0] == 401
        status, data = request(port, "GET", "/v1/capabilities", "b"*64)
        assert status == 200 and not json.loads(data)["ready"]
        value.validate_smoke()
        for _ in range(2):
            status, data = request(port, "POST", "/v1/jobs", "b"*64, data=body())
            assert status == 202
            job = json.loads(data)
            value.store._worker.join(3)
            path = "/v1/jobs/" + job["id"]
            assert request(port, "GET", path, "b"*64, "0"*64)[0] == 404
            status, output = request(port, "GET", path+"/output", "b"*64, job["access_token"])
            assert status == 200
            with Image.open(BytesIO(output)) as delivered:
                assert delivered.mode == "RGB" and delivered.size == (512, 512) and not delivered.info
                assert delivered.getpixel((0, 0)) == (20, 40, 60)
            assert request(port, "POST", "/v1/jobs", "b"*64, data=body())[0] == 409
            status, data = request(port, "DELETE", path, "b"*64, job["access_token"])
            assert status == 200 and json.loads(data)["cleanup_complete"]
        assert value.store.drained()
        assert request(port, "POST", "/v1/jobs", "b"*64, data=body())[0] == 409
        summary = value.summary()
        assert summary["admitted_jobs"] == 2 and summary["checked_user_engine_calls"] == {"returned": 2, "raised": 0}
        serialized = json.dumps(summary)
        assert job["id"] not in serialized and job["access_token"] not in serialized
        assert "image_png" not in serialized and "mask_png" not in serialized
    finally:
        value.close()
        server.shutdown(); server.server_close()


def test_cancelled_engine_must_finish_and_join_before_next_admission():
    running, cancelled, release = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def slow(image, mask, *, cancel, progress):
        calls.append(1)
        if len(calls) == 2:
            running.set()
            assert cancelled.wait(2)
            assert cancel()
            assert release.wait(2)
        return engine(image, mask, cancel=cancel, progress=progress)
    value = session(implementation=slow)
    try:
        value.validate_smoke()
        image, mask = session_layer.analytic_fixture()
        job = value.store.submit(image, mask)
        assert running.wait(2)
        result = value.store.delete(job["id"], job["access_token"])
        assert result["status"] == "cancelling" and not result["cleanup_complete"]
        cancelled.set()
        with pytest.raises(worker.RequestError) as caught:
            submit(value.store)
        assert caught.value.code == "busy"
        release.set()
        previous = value.store._worker
        previous.join(2)
        assert not previous.is_alive()
        next_job = submit(value.store)
        assert value.store.get(next_job["id"], next_job["access_token"])["status"] == "completed"
        assert len(calls) == 3  # One smoke, cancelled job, then one replacement.
    finally:
        release.set(); cancelled.set()
        value.close()


def test_ttl_releases_retained_output_and_deadline_caps_next_job():
    now = [100.]
    value = session(clock=lambda: now[0], deadline=4000.)
    try:
        value.validate_smoke()
        first = submit(value.store)
        assert value.store._job.expires == 1900
        now[0] = 1900
        value.store.reap()
        with pytest.raises(worker.RequestError):
            value.store.get(first["id"], first["access_token"], output=True)
        second = submit(value.store)
        assert value.store._job.expires == 3700
        now[0] = 4000
        value.store.reap()
        assert not value.store.ready
        assert value.store._job is None
        with pytest.raises(worker.RequestError):
            value.store.get(second["id"], second["access_token"], output=True)
    finally:
        value.close()


def test_job_expiry_never_exceeds_session_deadline():
    now = [100.]
    value = session(clock=lambda: now[0], deadline=500.)
    try:
        value.validate_smoke(); submit(value.store)
        assert value.store._job.expires == 500
    finally:
        value.close()


@pytest.mark.parametrize("max_jobs,deadline", [(0, 200), (True, 200), (21, 200), (1, 100),
                                               (1, 5501), (1, float("inf")), (1, float("nan"))])
def test_invalid_session_limits_fail_closed(max_jobs, deadline):
    with pytest.raises(RuntimeError):
        session_layer.PrivateSession(max_jobs=max_jobs, deadline=deadline, clock=lambda: 100.)


def test_safe_diagnostics_exclude_exception_text():
    sensitive = "image_name_and_bearer_secret"
    try:
        raise OSError(errno.EADDRINUSE, sensitive)
    except OSError as error:
        detail = session_layer.sanitized_exception(error)
    assert detail["exception_chain"][0]["errno"] == errno.EADDRINUSE
    assert sensitive not in json.dumps(detail)
    assert detail["messages_omitted"] and detail["locals_omitted"]


def test_watchdog_terminates_after_deadline():
    import time
    exits = []
    value = session(deadline=time.monotonic()+.03)
    done, thread = session_layer.start_deadline_watchdog(value, grace_seconds=.01, hard_exit=exits.append)
    try:
        thread.join(1)
        assert not thread.is_alive() and exits == [124]
        assert value.stop.is_set()
    finally:
        done.set()
        value.close()


def test_successful_cleanup_disarms_watchdog():
    exits = []
    value = session()
    done, thread = session_layer.start_deadline_watchdog(value, grace_seconds=.01, hard_exit=exits.append)
    done.set()
    thread.join(1)
    value.close()
    assert not thread.is_alive() and exits == []


def test_delete_cannot_rewrite_checked_engine_outcome_counts():
    value = session()
    try:
        value.validate_smoke()
        job = submit(value.store)
        before = value.summary()["checked_user_engine_calls"]
        value.store.delete(job["id"], job["access_token"])
        assert value.store.get(job["id"], job["access_token"])["status"] == "cancelled"
        assert before == value.summary()["checked_user_engine_calls"] == {"returned": 1, "raised": 0}
    finally:
        value.close()
