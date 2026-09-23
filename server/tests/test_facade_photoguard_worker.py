"""Local fake-engine checks: no listening socket, model load, or provider call."""
import base64
from email.message import Message
import io
import json
import struct
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin

import facade_photoguard_worker as worker

API_TOKEN = "a" * 64
HEADERS = {"authorization": "Bearer " + API_TOKEN, "content-type": "application/json"}


def png(mode="RGB", color=42, size=(512, 512), metadata=None):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG", pnginfo=metadata)
    return buffer.getvalue()


def payload(**changes):
    value = {"image_png": base64.b64encode(png()).decode(),
             "mask_png": base64.b64encode(png("L")).decode(), "recipe": worker.RECIPE}
    value.update(changes)
    return json.dumps(value).encode()


def successful(image, mask, *, cancel, progress):
    assert image.mode == "RGB" and mask.mode == "L"
    assert not cancel()
    progress(0, 200)
    result = image.copy()
    result.info["patient_secret"] = "must be discarded"
    progress(200, 200)
    return result


def decode(response):
    return response[0], json.loads(response[2])


def wait_done(store, job):
    store._worker.join(timeout=3)
    assert not store._worker.is_alive()
    return store.get(job["id"], job["access_token"])


def submit(api):
    status, job = decode(api.handle("POST", "/v1/jobs", HEADERS, payload()))
    assert status == 202
    assert len(job["access_token"]) == 64
    assert job["total_iterations"] == 200 and job["iteration"] == 0
    return job


def job_headers(job, **changes):
    return dict(HEADERS, **{"x-job-token": job["access_token"]}, **changes)


def test_readiness_needs_loaded_engine_and_explicit_operator_validation():
    for engine, validated in [(None, False), (successful, False), (None, True)]:
        store = worker.JobStore(engine, gpu_smoke_validated=validated)
        api = worker.WorkerAPI(store, API_TOKEN)
        status, cap = decode(api.handle("GET", "/v1/capabilities", HEADERS))
        assert status == 200 and not cap["ready"]
        assert cap == {"ready": False, "recipe": worker.RECIPE, "image_side": 512,
                       "total_iterations": 200, "retention_seconds": 1800,
                       "experimental": True, "gpu_smoke_validated": validated,
                       "engine_smoke_validated": validated, "execution_device": "cuda:0"}
        assert decode(api.handle("POST", "/v1/jobs", HEADERS, payload())) == (503, {"error": "not_ready"})


@pytest.mark.parametrize("authorization", ["", "bearer " + API_TOKEN, "Bearer " + API_TOKEN + " ", "Bearer " + "b" * 64, "Bearer \u00e9"])
def test_strict_bearer_auth(authorization):
    api = worker.WorkerAPI(worker.JobStore(successful, gpu_smoke_validated=True), API_TOKEN)
    assert decode(api.handle("GET", "/v1/capabilities", {"authorization": authorization})) == (401, {"error": "unauthorized"})


def test_exact_job_capability_and_output_then_deletion():
    store = worker.JobStore(successful, gpu_smoke_validated=True)
    api = worker.WorkerAPI(store, API_TOKEN)
    job = submit(api)
    assert wait_done(store, job)["status"] == "completed"
    path = "/v1/jobs/" + job["id"]
    for headers in [HEADERS, dict(HEADERS, **{"x-job-token": "b" * 64}), dict(HEADERS, **{"x-job-token": job['access_token'].upper()})]:
        assert api.handle("GET", path, headers)[0] == 404
        assert api.handle("DELETE", path, headers)[0] == 404
        assert api.handle("GET", path + "/output", headers)[0] == 404
    status, public = decode(api.handle("GET", path, job_headers(job)))
    assert status == 200 and "access_token" not in public
    result = api.handle("GET", path + "/output", job_headers(job))
    assert result[0] == 200 and result[1]["Content-Type"] == "image/png"
    assert result[1]["Cache-Control"] == "no-store"
    with Image.open(io.BytesIO(result[2])) as image:
        assert image.size == (512, 512) and image.mode == "RGB"
        assert "patient_secret" not in image.info
    assert api.handle("POST", "/v1/jobs", HEADERS, payload())[0] == 409
    status, deleted = decode(api.handle("DELETE", path, job_headers(job)))
    assert status == 200 and deleted["status"] == "cancelled" and deleted["cleanup_complete"]
    assert deleted["iteration"] == 200 and deleted["total_iterations"] == 200
    assert store._job.output is None and store._job.image is None and store._job.mask is None
    assert api.handle("GET", path + "/output", job_headers(job))[0] == 409


@pytest.mark.parametrize("changes", [
    {"recipe": "other"}, {"extra": "private"}, {"image_png": "data:image/png;base64,abcd"},
    {"mask_png": "not base64!"}, {"image_png": None},
    {"image_png": base64.b64encode(png(size=(513, 512))).decode()},
])
def test_malformed_payload_rejected_without_job(changes):
    store = worker.JobStore(successful, gpu_smoke_validated=True)
    api = worker.WorkerAPI(store, API_TOKEN)
    assert decode(api.handle("POST", "/v1/jobs", HEADERS, payload(**changes))) == (400, {"error": "invalid_request"})
    assert store._job is None


def test_duplicate_json_huge_dimensions_and_body_limit():
    store = worker.JobStore(successful, gpu_smoke_validated=True)
    api = worker.WorkerAPI(store, API_TOKEN)
    duplicate = payload()[:-1] + b',"recipe":"photoguard-diffusion-l2-v1"}'
    assert api.handle("POST", "/v1/jobs", HEADERS, duplicate)[0] == 400
    raw = bytearray(png())
    raw[16:24] = struct.pack(">II", 2**31, 2**31)
    assert api.handle("POST", "/v1/jobs", HEADERS, payload(image_png=base64.b64encode(raw).decode()))[0] == 400
    assert api.handle("POST", "/v1/jobs", HEADERS, b"x" * (worker.MAX_REQUEST_BYTES + 1))[0] == 413
    assert store._job is None


def test_single_job_cancellation_remains_pollable_until_engine_releases():
    entered = threading.Event()
    release = threading.Event()
    def slow(image, mask, *, cancel, progress):
        progress(3, 200)
        entered.set()
        assert release.wait(3)
        assert cancel()
        return image.copy()
    store = worker.JobStore(slow, gpu_smoke_validated=True)
    api = worker.WorkerAPI(store, API_TOKEN)
    job = submit(api)
    assert entered.wait(2)
    path = "/v1/jobs/" + job["id"]
    try:
        assert api.handle("POST", "/v1/jobs", HEADERS, payload())[0] == 409
        status, response = decode(api.handle("DELETE", path, job_headers(job)))
        assert status == 200 and response["status"] == "cancelling" and not response["cleanup_complete"]
        status, response = decode(api.handle("GET", path, job_headers(job)))
        assert status == 200 and response["status"] == "cancelling"
        assert response["iteration"] == 3 and response["total_iterations"] == 200
        assert api.handle("POST", "/v1/jobs", HEADERS, payload())[0] == 409
    finally:
        release.set()
    assert wait_done(store, job)["status"] == "cancelled"
    assert store.get(job["id"], job["access_token"])["cleanup_complete"]
    assert store._job.image is store._job.mask is store._job.output is None


def test_ttl_expiry_cancels_active_job_then_removes_capability():
    clock = [0.0]
    entered, release = threading.Event(), threading.Event()
    def slow(image, mask, *, cancel, progress):
        entered.set()
        assert release.wait(3)
        assert cancel()
        return image.copy()
    store = worker.JobStore(slow, gpu_smoke_validated=True, clock=lambda: clock[0])
    api = worker.WorkerAPI(store, API_TOKEN)
    job = submit(api)
    assert entered.wait(2)
    try:
        clock[0] = 1800
        store.reap()
        assert store.get(job["id"], job["access_token"])["status"] == "cancelling"
        assert store.ready  # Capacity is checked separately from readiness.
    finally:
        release.set()
    store._worker.join(timeout=3)
    assert not store._worker.is_alive()
    assert store._job is None
    assert api.handle("GET", "/v1/jobs/" + job["id"], job_headers(job))[0] == 404


def test_ttl_removes_completed_output_and_no_input_or_exception_leaks(capsys):
    clock = [0.0]
    store = worker.JobStore(successful, gpu_smoke_validated=True, clock=lambda: clock[0])
    api = worker.WorkerAPI(store, API_TOKEN)
    job = submit(api)
    wait_done(store, job)
    clock[0] = 1800
    store.reap()
    assert store._job is None
    def broken(*args, **kwargs):
        raise RuntimeError("SECRET_INPUT_DO_NOT_LOG")
    store.engine = broken
    job = submit(api)
    response = wait_done(store, job)
    assert response["status"] == "failed" and response["cleanup_complete"]
    assert "SECRET" not in json.dumps(response)
    assert store._job.output is store._job.image is store._job.mask is None
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


def test_apng_rejected_and_metadata_stripped_on_decode():
    buffer = io.BytesIO()
    first, second = Image.new("RGB", (512, 512)), Image.new("RGB", (512, 512), "red")
    first.save(buffer, format="PNG", save_all=True, append_images=[second])
    with pytest.raises(ValueError):
        worker._decode_png(base64.b64encode(buffer.getvalue()).decode(), mask=False)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private", "not retained")
    image = worker._decode_png(base64.b64encode(png(metadata=metadata)).decode(), mask=False)
    assert not image.info
    image.close()


def test_header_duplicates_and_transfer_encoding_rejected():
    with pytest.raises(worker.RequestError):
        worker.WorkerAPI.headers([("Authorization", "a"), ("authorization", "b")])
    api = worker.WorkerAPI(worker.JobStore(), API_TOKEN)
    assert api.handle("GET", "/v1/capabilities", dict(HEADERS, **{"transfer-encoding": "chunked"}))[0] == 400


def test_http_adapter_rejects_large_unauthenticated_and_duplicate_headers_before_body_read():
    class NoRead(io.BytesIO):
        def read(self, *args):
            raise AssertionError("body must not be read")
    def dispatch(headers):
        handler = object.__new__(worker.WorkerHandler)
        handler.server = SimpleNamespace(api=worker.WorkerAPI(worker.JobStore(), API_TOKEN))
        handler.headers = Message()
        for key, value in headers:
            handler.headers.add_header(key, value)
        handler.command, handler.path, handler.rfile = "POST", "/v1/jobs", NoRead()
        responses = []
        handler._write = responses.append
        handler._handle()
        return responses[0][0]
    assert dispatch([("Content-Length", "999999999")]) == 401
    assert dispatch([("Authorization", "Bearer " + API_TOKEN), ("Content-Length", str(worker.MAX_REQUEST_BYTES + 1))]) == 413
    assert dispatch([("Authorization", "Bearer " + API_TOKEN), ("Content-Length", "1"), ("Content-Length", "2")]) == 400


def test_transparent_mask_and_photo_match_engine_compositing():
    transparent_white = base64.b64encode(png("RGBA", (255, 255, 255, 0))).decode()
    transparent_black = base64.b64encode(png("RGBA", (0, 0, 0, 0))).decode()
    mask = worker._decode_png(transparent_white, mask=True)
    photo = worker._decode_png(transparent_black, mask=False)
    assert mask.getextrema() == (0, 0)
    assert photo.getextrema() == ((255, 255),) * 3
    mask.close()
    photo.close()


def test_short_engine_result_cannot_claim_completed_recipe():
    def short(image, mask, *, cancel, progress):
        progress(1, 200)
        return image.copy()
    store = worker.JobStore(short, gpu_smoke_validated=True)
    api = worker.WorkerAPI(store, API_TOKEN)
    job = submit(api)
    state = wait_done(store, job)
    assert state["status"] == "failed" and state["iteration"] == 1
    assert state["cleanup_complete"] and store._job.output is None


def binary_mask_png():
    image = Image.new("L", (512, 512), 0)
    image.paste(255, (256, 0, 512, 512))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def test_encoder_recipe_is_explicit_and_other_recipe_refused():
    store = worker.JobStore(successful, recipe=worker.ENCODER_RECIPE, execution_device="mps",
                            engine_smoke_validated=True)
    api = worker.WorkerAPI(store, API_TOKEN)
    cap = decode(api.handle("GET", "/v1/capabilities", HEADERS))[1]
    assert cap["ready"] and cap["engine_smoke_validated"]
    assert cap["execution_device"] == "mps" and cap["recipe"] == worker.ENCODER_RECIPE
    assert not cap["gpu_smoke_validated"], "local MPS numerical validation is not CUDA validation"
    assert api.handle("POST", "/v1/jobs", HEADERS, payload(mask_png=binary_mask_png()))[0] == 400
    status, job = decode(api.handle("POST", "/v1/jobs", HEADERS,
        payload(recipe=worker.ENCODER_RECIPE, mask_png=binary_mask_png())))
    assert status == 202 and wait_done(store, job)["status"] == "completed"
    assert store.recipe == worker.ENCODER_RECIPE
    with pytest.raises(AttributeError):
        store.recipe = worker.DIFFUSION_RECIPE
    with pytest.raises(AttributeError):
        store.execution_device = "cuda:0"
    store.close()


@pytest.mark.parametrize("color", [0, 42, 128, 255])
def test_encoder_soft_or_single_region_masks_rejected_before_engine(color):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid encoder mask must never reach the engine")
    store = worker.JobStore(unexpected, recipe=worker.ENCODER_RECIPE, engine_smoke_validated=True)
    api = worker.WorkerAPI(store, API_TOKEN)
    assert api.handle("POST", "/v1/jobs", HEADERS,
        payload(recipe=worker.ENCODER_RECIPE, mask_png=base64.b64encode(png("L", color)).decode()))[0] == 400
    assert store._job is None


def test_encoder_readiness_does_not_inherit_other_recipe_or_device_validation():
    store = worker.JobStore(successful, recipe=worker.ENCODER_RECIPE, execution_device="mps")
    cap = store.capabilities()
    assert not cap["ready"] and not cap["engine_smoke_validated"] and not cap["gpu_smoke_validated"]
    with pytest.raises(ValueError, match="MPS validation"):
        worker.JobStore(successful, recipe=worker.ENCODER_RECIPE, execution_device="mps", gpu_smoke_validated=True)
    with pytest.raises(ValueError, match="no MPS"):
        worker.JobStore(successful, recipe=worker.DIFFUSION_RECIPE, execution_device="mps")
    with pytest.raises(ValueError, match="unsupported"):
        worker.JobStore(successful, recipe="guessed-recipe")


def test_worker_cli_defaults_encoder_but_diffusion_stays_explicit(monkeypatch):
    seen = []
    class FakeServer:
        def __init__(self, address, api):
            seen.append(api.store.capabilities())
        def serve_forever(self, **kwargs):
            pass
        def server_close(self):
            pass
    monkeypatch.setenv("FACADE_PHOTOGUARD_API_TOKEN", API_TOKEN)
    monkeypatch.setattr(worker, "WorkerServer", FakeServer)
    monkeypatch.setattr(worker.JobStore, "start_reaper", lambda self: None)
    worker.main([])
    worker.main(["--engine", "diffusion"])
    assert [row["recipe"] for row in seen] == [worker.ENCODER_RECIPE, worker.DIFFUSION_RECIPE]
    assert all(not row["ready"] and not row["gpu_smoke_validated"] for row in seen)
