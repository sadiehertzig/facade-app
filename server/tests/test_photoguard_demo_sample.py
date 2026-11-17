"""Tests for photoguard_demo/sample.py: the public, sample-only mode of the PhotoGuard demo.

CPU only, with the fake pipeline from test_photoguard_demo_core and a fake gradio
module that records the page. They check that the public page offers no way to
upload a photo or type a prompt, that a run can only ever be what join() checked
and counted, and that the daily limits, the line and the shared GPU lock hold.
"""
from __future__ import annotations

import ast
import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from photoguard_demo import app, core, quota, sample
from test_photoguard_demo_core import FakePipe, make_demo

SERVER = Path(__file__).resolve().parents[1]
DEMO_DIR = SERVER / "photoguard_demo"
REPO = SERVER.parent


# --- a fake gradio ------------------------------------------------------------------------


class _Error(Exception):
    def __init__(self, message, **kwargs):
        super().__init__(message)
        self.kwargs = kwargs


class _Listener:
    """One event listener and whatever is chained after it."""

    def __init__(self, fn, inputs, outputs, kwargs):
        self.fn, self.inputs, self.outputs, self.kwargs = fn, inputs, outputs, kwargs
        self.chained = {}

    def _chain(self, how, fn, inputs, outputs, **kwargs):
        self.chained[how] = _Listener(fn, inputs, outputs, kwargs)
        return self.chained[how]

    def then(self, *args, **kwargs):
        return self._chain("then", *args, **kwargs)

    def success(self, *args, **kwargs):
        return self._chain("success", *args, **kwargs)


class _Component:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.listeners = {}
        _Component.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _listen(self, name, fn, inputs, outputs, **kwargs):
        self.listeners[name] = _Listener(fn, inputs, outputs, kwargs)
        return self.listeners[name]

    def click(self, *args, **kwargs):
        return self._listen("click", *args, **kwargs)

    def change(self, *args, **kwargs):
        return self._listen("change", *args, **kwargs)

    def load(self, *args, **kwargs):
        return self._listen("load", *args, **kwargs)


_Component.created = []
KINDS = ("Blocks", "Row", "Column", "Accordion", "HTML", "Markdown", "Radio", "Image", "Number", "Button",
         "Gallery", "DownloadButton", "State")


def fake_gradio():
    fake = SimpleNamespace(**{kind: type(kind, (_Component,), {}) for kind in KINDS})
    fake.Error = _Error
    fake.Request = type("Request", (), {})
    fake.update = lambda **kwargs: kwargs
    return fake


def request(peer="127.0.0.1", forwarded="203.0.113.9"):
    """What Gradio hands a function with a `request: gr.Request` parameter, behind Caddy."""
    headers = {} if forwarded is None else {"x-forwarded-for": forwarded}
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


@pytest.fixture(scope="module")
def samples():
    return sample.load_samples()


class Page:
    """The public page built around a fake gradio, a fake pipeline and an in-memory quota."""

    def __init__(self, samples, tmp_path, *, per_visitor=3, per_day=150, max_waiting=sample.MAX_WAITING, demo=None):
        _Component.created.clear()
        self.gr = fake_gradio()
        self.demo, self.pipe = demo or make_demo(pgd_iters=2)
        self.line = sample.Line(quota.DailyQuota(None, per_visitor=per_visitor, per_day=per_day),
                                max_waiting=max_waiting)
        sample.build_sample_ui(self.gr, core, self.demo, self.line, samples, export_root=lambda: tmp_path,
                               settings=app.run_settings(core))
        self.components = list(_Component.created)
        self.button = self.only("Button")
        self.join = self.button.listeners["click"]
        self.run = self.join.chained["success"]

    def all(self, kind):
        return [c for c in self.components if type(c).__name__ == kind]

    def only(self, kind):
        (component,) = self.all(kind)
        return component

    def click(self, face="sample-1", prompt="wedding", seed=1234, who=None):
        """Press the button: join(), then run() with the token join() stored. Returns run()'s outputs."""
        token, status, *_ = self.join.fn(face, prompt, seed, who or request())
        return status, self.run.fn(token)


# --- the sample faces -------------------------------------------------------------------------


def test_sample_files_match_their_pins_and_the_masks_keep_face_and_hair(samples):
    """Both faces load with their pinned hashes; each mask keeps the head and leaves background and clothes to edit."""
    assert [s.spec.id for s in samples.values()] == ["sample-1", "sample-2"]
    for item in samples.values():
        for name, pinned in ((item.spec.image, item.spec.image_sha256), (item.spec.keep, item.spec.keep_sha256)):
            assert hashlib.sha256((sample.SAMPLES_DIR / name).read_bytes()).hexdigest() == pinned
        keep = np.asarray(item.keep) > 127
        h, w = keep.shape
        assert set(np.unique(np.asarray(item.keep))) == {0, 255}, "a hard mask"
        assert keep[int(h * 0.4), w // 2], "the middle of the face is kept"
        assert not keep[5, 5] and not keep[5, w - 6], "the top corners (background) may be edited"
        assert not keep[h - 5, w // 2], "the clothes at the bottom may be edited"
        assert 0.2 < keep.mean() < 0.6
        # Through the demo's own crop and inversion, the mask is usable (not all kept, not all edited).
        core.check_mask(core.resize_and_crop(Image.fromarray(255 - np.asarray(item.keep)).convert("RGB"),
                                             (core.SIDE, core.SIDE)))
        # The preview is the photo with only a thin line drawn on it.
        changed = np.any(np.asarray(item.preview) != np.asarray(item.image), axis=-1)
        assert 0 < changed.mean() < 0.05


def test_sample_faces_are_the_websites_fictional_examples():
    """The samples are byte-identical copies of the AI-generated faces the website already shows."""
    for name, website in (("sample-1.jpg", "pg-original.jpg"), ("sample-2.jpg", "swap-target.jpg")):
        assert (sample.SAMPLES_DIR / name).read_bytes() == (REPO / "web/public/example" / website).read_bytes()
    readme = (sample.SAMPLES_DIR / "README.md").read_text()
    assert "fictional" in readme and "AI-generated" in readme


def test_a_changed_sample_file_is_refused(tmp_path):
    """load_samples refuses any file whose bytes are not the pinned ones."""
    for spec in sample.SAMPLES:
        for name in (spec.image, spec.keep):
            (tmp_path / name).write_bytes((sample.SAMPLES_DIR / name).read_bytes())
    sample.load_samples(tmp_path)
    data = bytearray((tmp_path / "sample-2-keep.png").read_bytes())
    data[-20] ^= 1
    (tmp_path / "sample-2-keep.png").write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="pinned SHA-256"):
        sample.load_samples(tmp_path)


# --- the page offers no upload and no free text ----------------------------------------------------


def test_public_page_has_no_upload_and_no_free_text(samples, tmp_path):
    """Only radios with preset choices, a whole-number seed and display-only images: nothing to upload or type into."""
    page = Page(samples, tmp_path)
    kinds = {type(c).__name__ for c in page.components}
    assert not kinds & {"UploadButton", "ImageMask", "ImageEditor", "File", "Textbox", "Dropdown", "Slider"}
    radios = page.all("Radio")
    assert [r.kwargs["choices"] for r in radios] == [
        [("Sample face 1", "sample-1"), ("Sample face 2", "sample-2")],
        [(text, key) for key, text in sample.PROMPTS]]
    assert page.only("Number").kwargs["precision"] == 0
    for shown in page.all("Image") + page.all("Gallery"):
        assert shown.kwargs["interactive"] is False
    assert page.only("Image").kwargs["buttons"] == ["fullscreen"]
    # The run step accepts nothing from the browser but the server-side token.
    assert page.run.inputs == [page.only("State")]
    assert page.join.inputs == radios + [page.only("Number")]
    # Step 1 never waits for the GPU; step 2 queues for it and shows its place in line on the gallery.
    assert page.join.kwargs == {"api_visibility": "private", "concurrency_limit": None,
                                "concurrency_id": sample.JOIN_LANE, "show_progress": "hidden"}
    assert page.run.kwargs == {"api_visibility": "private", "concurrency_limit": "default",
                               "show_progress_on": [page.only("Gallery")]}
    assert page.run.chained["then"].fn.__name__ == "show_status"
    assert page.only("Blocks").kwargs["delete_cache"] == (600, 900)


def test_a_run_uses_only_the_preset_face_mask_prompt_and_settings(samples, tmp_path):
    """join() then run() edits the chosen sample with its own mask, the chosen preset prompt, 100 steps, guidance 7.5."""
    page = Page(samples, tmp_path)
    status, (gallery, summary, comparison, archive) = page.click("sample-2", "snow", 77)
    assert status.startswith("Your run is starting.") and "You have 2 of 3 runs left today." in status
    assert [caption for _, caption in gallery] == [
        "Protected image", "AI edit of the original", "AI edit of the protected image"]
    calls = page.pipe.calls
    assert len(calls) == 2 and all(c["prompt"] == "a person in a red winter coat in the snow" for c in calls)
    assert all(c["steps"] == core.NUM_INFERENCE_STEPS == 100 and c["guidance"] == 7.5 for c in calls)
    assert all(c["seed"] == 77 for c in calls)
    expected_input = np.asarray(core.crop_input(samples["sample-2"].image))
    assert np.array_equal(calls[0]["image"], expected_input)
    edit_region = np.asarray(core.resize_and_crop(
        Image.fromarray(255 - np.asarray(samples["sample-2"].keep)).convert("RGB"), (core.SIDE, core.SIDE)))
    assert np.array_equal(calls[0]["mask"], edit_region)
    assert "Sample face 2" in summary and "red winter coat" in summary and "Seed 77" in summary
    assert comparison["visible"] and archive["visible"]
    assert Path(comparison["value"]).parent.parent == tmp_path


@pytest.mark.parametrize("face, prompt, seed, message", [
    ("sample-3", "wedding", 1, sample.PICK_SAMPLE),
    ("../../etc/passwd", "wedding", 1, sample.PICK_SAMPLE),
    ("sample-1", "a person in handcuffs", 1, sample.PICK_PROMPT),
    ("sample-1", None, 1, sample.PICK_PROMPT),
    ("sample-1", "wedding", -1, "Seed must be a whole number"),
    ("sample-1", "wedding", 4.5, "Seed must be a whole number"),
])
def test_join_refuses_anything_but_the_presets_without_counting(samples, tmp_path, face, prompt, seed, message):
    """A made-up face, a typed prompt or a bad seed is refused before anything is counted or queued."""
    page = Page(samples, tmp_path)
    with pytest.raises(_Error, match=message):
        page.join.fn(face, prompt, seed, request())
    assert page.line.quota.site_used() == 0 and page.line.waiting() == 0


def test_run_without_a_counted_token_does_nothing(samples, tmp_path):
    """A script calling the queued step directly (no token, a made-up one, or one already used) gets nothing."""
    page = Page(samples, tmp_path)
    for token in (None, "made-up", 12345, {"prompt": "anything"}):
        with pytest.raises(_Error, match="Press Protect and compare"):
            page.run.fn(token)
    token = page.join.fn("sample-1", "beach", 1, request())[0]
    page.run.fn(token)
    with pytest.raises(_Error, match="Press Protect and compare"):
        page.run.fn(token)  # each token runs once
    assert len(page.pipe.calls) == 2


# --- limits, line and failures ----------------------------------------------------------------------


def test_three_runs_per_visitor_then_a_clear_refusal(samples, tmp_path):
    """The fourth run from one address is refused with the visitor message; another address still runs."""
    page = Page(samples, tmp_path)
    for left in (2, 1, 0):
        status, _ = page.click()
        assert f"You have {left} of 3 runs left today." in status
    with pytest.raises(_Error, match="used today's 3 runs") as refused:
        page.join.fn("sample-1", "wedding", 1, request())
    assert refused.value.kwargs == {"print_exception": False}
    page.click(who=request(forwarded="198.51.100.4"))
    assert page.line.quota.site_used() == 4


def test_forwarded_for_is_ignored_unless_it_comes_from_caddy(samples, tmp_path):
    """A visitor reaching the page without Caddy cannot pick a new address by sending X-Forwarded-For."""
    page = Page(samples, tmp_path, per_visitor=1)
    page.click(who=request(peer="198.51.100.4", forwarded="10.0.0.1"))
    with pytest.raises(_Error, match="used today's 1 runs"):
        page.join.fn("sample-1", "wedding", 1, request(peer="198.51.100.4", forwarded="10.0.0.2"))


def test_the_site_cap_refuses_everyone(samples, tmp_path):
    page = Page(samples, tmp_path, per_day=2)
    page.click(who=request(forwarded="10.0.0.1"))
    page.click(who=request(forwarded="10.0.0.2"))
    with pytest.raises(_Error, match="limit for everyone"):
        page.join.fn("sample-1", "wedding", 1, request(forwarded="10.0.0.3"))


def test_queue_position_and_wait_are_shown_and_a_full_line_is_refused_uncounted(samples, tmp_path):
    """Each joined run says how many are ahead and roughly how long; past MAX_WAITING, new runs are turned away."""
    page = Page(samples, tmp_path, max_waiting=3)
    statuses = [page.join.fn("sample-1", "wedding", 1, request(forwarded=f"10.0.0.{i}"))[1] for i in range(3)]
    assert statuses[0].startswith("Your run is starting. Results in about a minute.")
    assert statuses[1].startswith("Your run is in line, with 1 run ahead of you. Results in about 2 minutes.")
    assert statuses[2].startswith("Your run is in line, with 2 runs ahead of you. Results in about 3 minutes.")
    with pytest.raises(_Error, match="3 runs are already in line"):
        page.join.fn("sample-1", "wedding", 1, request(forwarded="10.0.0.9"))
    assert page.line.quota.site_used() == 3 and page.line.quota.left("10.0.0.9") == 3
    # The line under the button, as the page shows it on opening and after each run.
    show_status = page.run.chained["then"].fn
    assert show_status(request(forwarded="10.0.0.9")) == (
        "You have 3 of 3 runs left today. 3 runs are in line now; each takes about a minute.")


def test_a_server_failure_gives_the_run_back(samples, tmp_path):
    """If the GPU fails, the visitor gets the run back, the line is freed, and the message says so."""
    page = Page(samples, tmp_path, per_visitor=1)

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    page.demo.compare = broken
    token = page.join.fn("sample-1", "wedding", 1, request())[0]
    with pytest.raises(_Error, match=r"Processing failed \(RuntimeError\)\. This run was not counted"):
        page.run.fn(token)
    assert page.line.quota.left("203.0.113.9") == 1 and page.line.waiting() == 0


def test_an_abandoned_run_leaves_the_line_but_stays_counted():
    """A counted run that never starts (its visitor left) stops holding a place after the timeout."""
    clock = SimpleNamespace(now=0.0)
    line = sample.Line(quota.DailyQuota(None), timeout=600, clock=lambda: clock.now)
    joined = line.join("10.0.0.1", "sample-1", "a prompt", 1)
    assert joined.reservation and line.waiting() == 1
    clock.now = 601
    assert line.waiting() == 0 and line.start(joined.reservation.token) is None
    assert line.quota.left("10.0.0.1") == 2


def test_wait_text():
    assert [sample.wait_text(n) for n in (1, 2, 3, 10)] == [
        "about a minute", "about 2 minutes", "about 3 minutes", "about 9 minutes"]


# --- the shared GPU and the process ------------------------------------------------------------------


class _SlowPipe(FakePipe):
    """Records how many edits run at the same time."""

    def __init__(self):
        super().__init__()
        self.active = 0
        self.most = 0
        self._count = threading.Lock()

    def __call__(self, **kwargs):
        with self._count:
            self.active += 1
            self.most = max(self.most, self.active)
        time.sleep(0.05)
        with self._count:
            self.active -= 1
        return super().__call__(**kwargs)


def test_the_two_pages_take_turns_on_the_gpu():
    """compare() calls from the full demo and the sample mode at once never overlap on the GPU."""
    import torch
    pipe = _SlowPipe()
    demo = core.PhotoGuardDemo(pipe, device="cpu", dtype=torch.float32, pgd_iters=1)
    photo = Image.new("RGB", (64, 64), (120, 80, 60))
    keep = Image.new("L", (64, 64), 0)
    keep.paste(255, (0, 0, 32, 64))
    threads = [threading.Thread(target=demo.compare, args=(photo, keep, "p", seed, 10, 7.5)) for seed in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(pipe.calls) == 8 and pipe.most == 1


def test_a_bad_setting_never_waits_for_the_gpu(samples):
    """compare() checks its settings before taking the GPU lock."""
    demo, _ = make_demo()
    with demo.gpu_lock:  # the GPU is busy
        with pytest.raises(core.InputError):
            demo.compare(samples["sample-1"].image, samples["sample-1"].keep, "p", seed=-5)


def test_sample_mode_launch_is_loopback_only_and_takes_no_uploads():
    """sample.start() binds to the given host, never shares, queues one run at a time and caps files at 1 kB."""
    tree = ast.parse((DEMO_DIR / "sample.py").read_text())
    (launch,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "launch"]
    kwargs = {k.arg: ast.unparse(k.value) for k in launch.keywords}
    assert kwargs["server_name"] == "host" and kwargs["server_port"] == "port" and kwargs["share"] == "False"
    assert kwargs["max_file_size"] == "NO_UPLOADS" and sample.NO_UPLOADS == "1kb"
    assert kwargs["prevent_thread_lock"] == "True" and kwargs["run_history"] == "False"
    assert kwargs["mcp_server"] == "False" and kwargs["enable_monitoring"] == "False"
    (queue,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "queue"]
    assert {k.arg: ast.unparse(k.value) for k in queue.keywords} == {
        "default_concurrency_limit": "1", "max_size": "2 * MAX_WAITING", "api_open": "False"}
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all("gradio" not in ast.unparse(n) and "torch" not in ast.unparse(n) for n in top_level)


def test_app_starts_the_sample_mode_only_when_asked():
    """--sample-port turns the public page on; without it the full demo runs alone, as before."""
    args = app.parse_args(["--snapshot", "x"])
    assert args.sample_port is None and args.quota_file is None
    assert (args.runs_per_visitor, args.runs_per_day) == (3, 150)
    args = app.parse_args(["--snapshot", "x", "--sample-port", "7861", "--sample-root-path", "/photoguard-sample",
                           "--quota-file", "/var/lib/facade/sample-quota/counts.json"])
    assert (args.sample_port, args.sample_root_path) == (7861, "/photoguard-sample")
    for bad in (["--sample-port", "7860"], ["--runs-per-visitor", "0"], ["--sample-root-path", "photoguard"]):
        with pytest.raises(SystemExit):
            app.parse_args(["--snapshot", "x", *bad])
    source = (DEMO_DIR / "app.py").read_text()
    assert "if args.sample_port is not None:" in source and "sample.start(" in source


def test_the_safety_checker_stays_on_for_both_pages():
    """No unit or script on the server turns the Stable Diffusion safety checker off."""
    for path in [*(REPO / "deploy").rglob("*"), DEMO_DIR / "app.py", DEMO_DIR / "sample.py"]:
        if path.is_file():
            text = path.read_text(errors="ignore")
            assert "FACADE_SAFETY_CHECKER=0" not in text.replace(" ", ""), path
    assert core.safety_checker_enabled({})


def test_privacy_page_describes_the_sample_counts():
    """The privacy policy says what the sample mode stores (a daily count under a code, no address)."""
    page = (REPO / "deploy" / "public" / "privacy.html").read_text()
    assert "PhotoGuard sample page" in page and "cannot upload" in page
    assert "IP address itself" in page and "is not stored" in page and "midnight UTC" in page
    assert "password-protected research site" not in page, "the website has been public since 2026-10-25"
