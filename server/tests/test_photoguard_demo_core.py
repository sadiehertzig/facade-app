"""Tests for the PhotoGuard demo port: photoguard_demo/core.py and app.py.

CPU tests for the ported PhotoGuard demo: fake pipeline, no gradio, no model download.
FakePipe stands in for Stable Diffusion inpainting (its "edit" is seeded noise)
and a fake gradio module records how app.py builds the page. Two optional
tests compare the port with an upstream checkout named by PHOTOGUARD_UPSTREAM.
"""
from __future__ import annotations

import ast
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from photoguard_demo import app, core, io_utils  # app.py imports gradio only inside main()

SERVER = Path(__file__).resolve().parents[1]
DEMO_DIR = SERVER / "photoguard_demo"
ENCODER = SERVER / "worker" / "facade_photoguard_encoder.py"
# Optional extra checks run when these exist locally:
#   the pinned SD1.5-inpainting snapshot in the Hugging Face cache, and a checkout
#   of https://github.com/MadryLab/photoguard at the upstream commit, pointed to by
#   PHOTOGUARD_UPSTREAM (for example: git clone ... && git checkout 686bea75c786).
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting"
            / "snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb")
UPSTREAM = Path(os.environ.get("PHOTOGUARD_UPSTREAM", "/nonexistent-photoguard-checkout"))
GRAY = 2 * (128 / 255) - 1


@pytest.fixture(autouse=True, scope="module")
def few_cpu_threads():
    """Use two CPU threads for these tests, then restore the previous setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyVAE:
    """Differentiable stand-in for AutoencoderKL.encode: 8x pooling to a (1, 4, 64, 64) mean."""

    def __init__(self):
        self.calls = []

    def encode(self, x):
        detached = x.detach()
        self.calls.append(SimpleNamespace(
            requires_grad=x.requires_grad, grad_enabled=torch.is_grad_enabled(),
            gray=bool((detached - GRAY).abs().max() < 1e-6)))  # float32 2*128/255-1 is not exact
        pooled = F.avg_pool2d(x, 8)
        mean = torch.cat([pooled, pooled.pow(2).mean(dim=1, keepdim=True)], dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))


class FakePipe:
    """Records every inpainting call. Its 'edit' is noise drawn from the given generator."""

    def __init__(self, nsfw=False, safety_checker=True):
        self.vae = TinyVAE()
        self.safety_checker = object() if safety_checker else None
        self.nsfw = nsfw
        self.calls = []

    def __call__(self, *, prompt, image, mask_image, height, width, eta, guidance_scale, num_inference_steps,
                 generator, callback_on_step_end=None):
        noise = torch.rand((height, width, 3), generator=generator)
        self.calls.append(dict(prompt=prompt, image=np.asarray(image).copy(), mask=np.asarray(mask_image).copy(),
                               seed=generator.initial_seed(), noise=noise, eta=eta, guidance=guidance_scale,
                               steps=num_inference_steps, height=height, width=width))
        if callback_on_step_end is not None:
            for step in range(num_inference_steps):
                assert callback_on_step_end(self, step, 999 - step, {"latents": None}) == {"latents": None}
        edited = Image.fromarray((noise.numpy() * 255).astype(np.uint8), "RGB")
        return SimpleNamespace(images=[edited], nsfw_content_detected=[self.nsfw])


def photo(width=512, height=512, seed=0):
    """A random-noise RGB test photo (no real person), reproducible from `seed`."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB")


def keep_left_half(width=512, height=512):
    """Brush mask as the user paints it: white = painted = KEEP."""
    mask = np.zeros((height, width), np.uint8)
    mask[:, : width // 2] = 255
    return Image.fromarray(mask, "L")


def make_demo(pgd_iters=3, **pipe_kwargs):
    """A PhotoGuardDemo on the CPU around a FakePipe, with a short PGD run by default."""
    pipe = FakePipe(**pipe_kwargs)
    return core.PhotoGuardDemo(pipe, device="cpu", dtype=torch.float32, pgd_iters=pgd_iters), pipe


def test_upstream_recipe_constants():
    """The attack, edit and slider constants are upstream's numbers."""
    assert (core.PGD_EPS, core.PGD_STEP_SIZE, core.PGD_ITERS) == (0.12, 0.01, 200)
    assert (core.NUM_INFERENCE_STEPS, core.GUIDANCE_SCALE, core.DEFAULT_SEED, core.ETA) == (100, 7.5, 1234, 1)
    assert (core.STEPS_MIN, core.STEPS_MAX, core.GUIDANCE_MIN, core.GUIDANCE_MAX) == (10, 250, 0.1, 25.0)


def test_gray_target_matches_facade_encoder_binding():
    """The demo's gray target is the same RGB 128 image, with the same hash, as the server's encoder."""
    tree = ast.parse(ENCODER.read_text())
    bound = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", None) == "TARGET_RGB512_SHA256" for t in node.targets))
    assert core.TARGET_RGB512_SHA256 == bound
    target = core.get_target()
    assert target.size == (512, 512) and target.mode == "RGB"
    assert np.all(np.asarray(target) == 128)


def test_pgd_respects_linf_bound_clamp_and_mask():
    """PGD stays within 0.12 per pixel and within [-1, 1], and never touches pixels outside its mask."""
    vae = TinyVAE()
    X = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(1)) * 2 - 1
    perturb = torch.zeros(1, 1, 512, 512)
    perturb[..., :256] = 1
    X = X * perturb  # upstream attacks the masked image: the edit region is zero
    targets = vae.encode(torch.zeros(1, 3, 512, 512)).latent_dist.mean.detach()
    adv = core.pgd(X, targets=targets, model=vae.encode, criterion=torch.nn.MSELoss(), eps=0.12,
                   step_size=0.01, iters=200, clamp_min=-1, clamp_max=1, mask=perturb,
                   generator=torch.Generator().manual_seed(0)).detach()
    delta = adv - X
    assert delta.abs().max() <= 0.12 + 1e-6
    assert delta[..., :256].abs().max() > 0.1  # it moved, up to the bound
    assert torch.all(adv[..., 256:] == 0)  # nothing outside the mask
    assert adv.min() >= -1 and adv.max() <= 1


def test_pgd_random_start_is_the_seeded_cpu_stream():
    """PGD's seeded random start equals upstream's torch.manual_seed start."""
    X = torch.zeros(1, 3, 16, 16)
    identity = lambda x: SimpleNamespace(latent_dist=SimpleNamespace(mean=x))  # noqa: E731
    a = core.pgd(X, X, identity, None, eps=0.12, iters=0, generator=torch.Generator().manual_seed(7))
    torch.manual_seed(7)  # upstream: torch.manual_seed(seed), then torch.rand from the global CPU RNG
    b = core.pgd(X, X, identity, None, eps=0.12, iters=0)
    assert torch.equal(a, b)


def test_pgd_loop_keeps_upstream_statements_in_order():
    """pgd()'s loop has upstream's statements in upstream's order, apart from the added detach() and progress call."""
    if not (UPSTREAM / "demo/app.py").is_file():
        pytest.skip("set PHOTOGUARD_UPSTREAM to a MadryLab/photoguard checkout to run this check")

    def loop_body(tree):
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "pgd")
        loop = next(n for n in function.body if isinstance(n, ast.For))
        return [ast.dump(s) for s in loop.body if "pbar" not in ast.unparse(s) and "progress" not in ast.unparse(s)]

    upstream = loop_body(ast.parse((UPSTREAM / "demo/app.py").read_text()))
    ours = loop_body(ast.parse((DEMO_DIR / "core.py").read_text()))
    detach = ast.dump(ast.parse("X_adv = X_adv.detach()").body[0])
    assert [s for s in ours if s != detach] == upstream


def test_utils_are_verbatim_upstream():
    """The four helpers copied from upstream's demo/utils.py are identical to upstream."""
    if not (UPSTREAM / "demo/utils.py").is_file():
        pytest.skip("set PHOTOGUARD_UPSTREAM to a MadryLab/photoguard checkout to run this check")
    names = {"resize_and_crop", "recover_image", "preprocess", "prepare_mask_and_masked_image"}

    def functions(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}

    assert functions(DEMO_DIR / "core.py") == functions(UPSTREAM / "demo/utils.py")


def test_target_latent_is_computed_once():
    """The gray target's latent is computed once, without gradients, and reused by every run."""
    demo, pipe = make_demo()
    image, mask = photo(), keep_left_half()
    demo.compare(image, mask, "a prompt", 1, 10, 7.5)
    demo.compare(image, mask, "a prompt", 2, 10, 7.5)
    gray = [call for call in pipe.vae.calls if call.gray]
    assert len(gray) == 1 and not gray[0].grad_enabled and pipe.vae.calls[0].gray
    attack = [call for call in pipe.vae.calls if not call.gray]
    assert len(attack) == 2 * demo.pgd_iters and all(call.requires_grad for call in attack)
    expected = pipe.vae.encode(core.preprocess(core.get_target())).latent_dist.mean
    assert torch.equal(demo.targets, expected.detach())


def test_compare_returns_three_512_images_and_honors_upstream_mask_semantics():
    """compare() returns three 512 x 512 images, perturbs only the painted part, and both edits keep the original painted pixels."""
    demo, pipe = make_demo()
    stages = []
    original = photo()
    immunized, edit_original, edit_immunized, info = demo.compare(
        original, keep_left_half(), "  a person at a wedding ", None, 20, 5.0,
        progress=lambda stage, done, total: stages.append((stage, done, total)))

    for image in (immunized, edit_original, edit_immunized):
        assert isinstance(image, Image.Image) and image.size == (512, 512) and image.mode == "RGB"

    source = np.asarray(original).astype(np.int16)
    protected = np.asarray(immunized).astype(np.int16)
    # The painted (kept) left half is perturbed within the recipe bound; the unpainted edit region is untouched.
    assert np.abs(protected[:, :256] - source[:, :256]).max() <= 16
    assert np.abs(protected[:, :256] - source[:, :256]).mean() > 1
    assert np.array_equal(protected[:, 256:], source[:, 256:])

    # The editor gets the inverted mask: white = unpainted = region to regenerate.
    for call in pipe.calls:
        assert np.all(call["mask"][:, :256] == 0) and np.all(call["mask"][:, 256:] == 255)
        assert (call["eta"], call["guidance"], call["steps"], call["height"], call["width"]) == (1, 5.0, 20, 512, 512)
        assert call["prompt"] == "a person at a wedding"
    assert np.array_equal(pipe.calls[0]["image"], np.asarray(original))
    assert np.array_equal(pipe.calls[1]["image"], np.asarray(immunized))

    # recover_image: both edits keep the ORIGINAL pixels where the user painted.
    for edited in (edit_original, edit_immunized):
        assert np.array_equal(np.asarray(edited)[:, :256], np.asarray(original)[:, :256])
    assert np.array_equal(np.asarray(edit_original)[:, 256:], np.asarray(edit_immunized)[:, 256:])

    assert info["seed"] == core.DEFAULT_SEED and info["steps"] == 20 and info["guidance"] == 5.0
    assert all(info[key] >= 0 for key in ("immunize_seconds", "edit_original_seconds",
                                          "edit_immunized_seconds", "total_seconds"))
    assert not info["safety_flagged_original"] and not info["safety_flagged_immunized"]
    assert [s for s in stages if s[0] == "Immunizing"][-1] == ("Immunizing", 3, 3)
    assert ("Editing the original", 20, 20) in stages and ("Editing the immunized image", 20, 20) in stages


def test_both_edit_arms_use_identical_fresh_seeds():
    """Both edits start from identical noise for the same seed, and a repeat run gets the same noise again."""
    demo, pipe = make_demo()
    demo.compare(photo(), keep_left_half(), "p", 987654321, 10, 7.5)
    first, second = pipe.calls
    assert first["seed"] == second["seed"] == 987654321
    assert torch.equal(first["noise"], second["noise"])
    demo.compare(photo(), keep_left_half(), "p", 987654321, 10, 7.5)
    assert torch.equal(pipe.calls[2]["noise"], first["noise"])


def test_non_square_input_is_center_cropped_like_upstream():
    """A wide photo and its mask are center-cropped to the same 512 x 512 square."""
    demo, pipe = make_demo()
    wide = photo(width=768, height=512, seed=3)
    mask = np.zeros((512, 768), np.uint8)
    mask[:, 128:384] = 255  # after the center crop (x 128..640) this is the left half
    immunized, *_ = demo.compare(wide, Image.fromarray(mask, "L"), "p", 5, 10, 7.5)
    cropped = np.asarray(wide)[:, 128:640]
    assert np.array_equal(pipe.calls[0]["image"], cropped)
    assert np.array_equal(np.asarray(immunized)[:, 256:], cropped[:, 256:])
    assert np.all(pipe.calls[0]["mask"][:, :256] == 0) and np.all(pipe.calls[0]["mask"][:, 256:] == 255)


def test_soft_mask_seam_is_the_only_place_beyond_the_byte_bound():
    """Upstream blends a resampled (soft) mask; only that 1-2 px seam may exceed 16 levels."""
    demo, _ = make_demo(pgd_iters=20)
    image = photo(width=800, height=600, seed=4)
    keep = np.zeros((600, 800), np.uint8)
    keep[100:400, 250:550] = 255
    immunized = demo.compare(image, Image.fromarray(keep, "L"), "p", 1, 10, 7.5)[0]
    source = np.asarray(core.resize_and_crop(image, (512, 512))).astype(np.int16)
    delta = np.abs(np.asarray(immunized).astype(np.int16) - source).max(axis=-1)
    edit = np.asarray(core.resize_and_crop(Image.fromarray(255 - keep).convert("RGB"), (512, 512)))[..., 0]
    seam = (edit > 0) & (edit < 255)
    assert delta[edit == 0].max() <= 16 and delta[edit == 0].mean() > 1
    assert delta[edit == 255].max() == 0
    assert 0 < seam.sum() < 4 * 2 * 260 and delta[seam].max() > 16


def test_license_is_the_upstream_mit_text():
    """LICENSE is upstream's MIT text byte for byte."""
    import hashlib
    raw = (DEMO_DIR / "LICENSE").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "1279eae7ed1e1b45d1381ee0dc5d57f037b81798dda8b26b964768afa04dfcd4"
    assert b"Copyright (c) 2022 Madry Lab" in raw


def test_unusable_masks_and_settings_are_input_errors():
    """An empty or full mask, out-of-range steps, guidance or seed, or no photo, are user-facing input errors."""
    demo, _ = make_demo()
    blank = Image.new("L", (512, 512), 0)
    full = Image.new("L", (512, 512), 255)
    with pytest.raises(core.InputError, match="Paint over"):
        demo.compare(photo(), blank, "p", 1, 10, 7.5)
    with pytest.raises(core.InputError, match="unpainted"):
        demo.compare(photo(), full, "p", 1, 10, 7.5)
    with pytest.raises(core.InputError):
        demo.compare(photo(), keep_left_half(), "p", 1, 9, 7.5)
    with pytest.raises(core.InputError):
        demo.compare(photo(), keep_left_half(), "p", 1, 10, 30)
    with pytest.raises(core.InputError):
        demo.compare(photo(), keep_left_half(), "p", -1, 10, 7.5)
    with pytest.raises(core.InputError):
        demo.compare(None, keep_left_half(), "p", 1, 10, 7.5)
    assert core.parse_seed("") == core.parse_seed(None) == 1234 and core.parse_seed("42") == 42
    assert core.parse_seed(42.0) == 42


def test_safety_checker_env_switch():
    """The safety checker is on by default and off only with FACADE_SAFETY_CHECKER=0."""
    default = core.pipeline_kwargs({})
    assert default == {"dtype": torch.float16, "use_safetensors": False, "local_files_only": True}
    assert core.pipeline_kwargs({"FACADE_SAFETY_CHECKER": "1"}) == default
    off = core.pipeline_kwargs({"FACADE_SAFETY_CHECKER": "0"})
    assert off["safety_checker"] is None and off["requires_safety_checker"] is False
    assert core.safety_checker_enabled({}) and not core.safety_checker_enabled({"FACADE_SAFETY_CHECKER": "0"})


def test_load_pipeline_passes_env_switch_and_disables_progress(monkeypatch, tmp_path):
    """load_pipeline uses the local .bin snapshot, honours the safety switch, hides progress bars and freezes the weights."""
    (tmp_path / "model_index.json").write_text("{}")
    seen = {}

    class Module:
        def __init__(self):
            self.grad = True

        def requires_grad_(self, flag):
            self.grad = flag

    class FakeCls:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            seen.update(path=path, kwargs=kwargs)
            pipe = cls()
            pipe.vae, pipe.unet, pipe.text_encoder = Module(), Module(), Module()
            pipe.safety_checker = None if "safety_checker" in kwargs else Module()
            return pipe

        def to(self, device):
            seen["device"] = device
            return self

        def set_progress_bar_config(self, **kwargs):
            seen["progress"] = kwargs

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pipe = core.load_pipeline(tmp_path, env={"FACADE_SAFETY_CHECKER": "0"}, pipeline_cls=FakeCls)
    assert seen["path"] == str(tmp_path.resolve()) and seen["device"] == "cuda"
    assert seen["kwargs"]["safety_checker"] is None and seen["progress"] == {"disable": True}
    assert pipe.vae.grad is False and pipe.unet.grad is False
    core.load_pipeline(tmp_path, env={}, pipeline_cls=FakeCls)
    assert "safety_checker" not in seen["kwargs"] and seen["kwargs"]["dtype"] is torch.float16


def test_load_pipeline_requires_cuda(monkeypatch, tmp_path):
    """load_pipeline refuses to run without an NVIDIA GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs an NVIDIA GPU"):
        core.load_pipeline(tmp_path)
    with pytest.raises(RuntimeError, match="needs an NVIDIA GPU"):
        core.load_pipeline(tmp_path, device="cpu")


def test_editor_value_to_image_and_brush_mask():
    """Any brush stroke, even semi-transparent, marks a pixel as kept; no photo or no paint is an input error."""
    background = np.zeros((300, 400, 4), np.uint8)
    background[..., :3] = 77
    background[..., 3] = 255
    layer = np.zeros((300, 400, 4), np.uint8)
    layer[50:100, 60:120] = (255, 45, 85, 153)  # a 60 % opacity stroke
    image, mask = core.editor_to_image_and_mask({"background": background, "layers": [layer], "composite": None})
    assert image.mode == "RGB" and image.size == (400, 300) and np.all(np.asarray(image) == 77)
    painted = np.asarray(mask)
    assert mask.mode == "L" and painted[50:100, 60:120].min() == 255 and painted.sum() == 255 * 50 * 60
    pil_value = {"background": Image.fromarray(background, "RGBA"), "layers": [Image.fromarray(layer, "RGBA")]}
    assert np.array_equal(np.asarray(core.editor_to_image_and_mask(pil_value)[1]), painted)
    with pytest.raises(core.InputError, match="Upload"):
        core.editor_to_image_and_mask(None)
    with pytest.raises(core.InputError, match="Upload"):
        core.editor_to_image_and_mask({"background": None, "layers": [], "composite": None})
    with pytest.raises(core.InputError, match="Paint over"):
        core.editor_to_image_and_mask({"background": background, "layers": [], "composite": None})


def test_safety_flag_is_reported():
    """A safety-checker black-out, or the checker being off, is mentioned in the summary."""
    demo, _ = make_demo(nsfw=True)
    info = demo.compare(photo(), keep_left_half(), "p", 1, 10, 7.5)[3]
    assert info["safety_flagged_original"] and info["safety_flagged_immunized"]
    assert "blacked out" in core.format_summary(info)
    demo_off, _ = make_demo(safety_checker=False)
    info = demo_off.compare(photo(), keep_left_half(), "p", 1, 10, 7.5)[3]
    assert "FACADE_SAFETY_CHECKER=0" in core.format_summary(info)


def test_warmup_runs_one_step_everything():
    """The warm-up runs one PGD step and two one-step edits."""
    demo, pipe = make_demo(pgd_iters=200)
    result = demo.warmup()
    assert len(pipe.calls) == 2 and pipe.calls[0]["steps"] == 1
    assert sum(1 for call in pipe.vae.calls if not call.gray) == 1
    assert result[0].size == (512, 512)


# --- app.py, checked without importing gradio -------------------------------


def _calls(tree, name):
    """Every call in `tree` to a function or method named `name`."""
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and (getattr(n.func, "attr", None) == name or getattr(n.func, "id", None) == name)]


def test_core_never_imports_gradio():
    """core.py never imports gradio."""
    tree = ast.parse((DEMO_DIR / "core.py").read_text())
    imported = {alias.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "gradio" not in imported


def test_app_never_shares_and_binds_only_to_host():
    """app.py never makes a share link, binds only to --host, and queues one run at a time."""
    source = (DEMO_DIR / "app.py").read_text()
    tree = ast.parse(source)
    for keyword in (k for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords):
        if keyword.arg == "share":
            assert isinstance(keyword.value, ast.Constant) and keyword.value.value is False
    assert "share=True" not in source.replace(" ", "") and "0.0.0.0" not in source
    (launch,) = _calls(tree, "launch")
    kwargs = {k.arg: k.value for k in launch.keywords}
    assert ast.unparse(kwargs["server_name"]) == "args.host" and ast.unparse(kwargs["server_port"]) == "args.port"
    assert kwargs["share"].value is False and kwargs["run_history"].value is False
    (queue,) = _calls(tree, "queue")
    assert {k.arg: ast.literal_eval(k.value) for k in queue.keywords} == {
        "default_concurrency_limit": 1, "max_size": 8, "api_open": False}
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all("gradio" not in ast.unparse(n) for n in top_level)
    assert ast.unparse(kwargs["root_path"]) == "args.root_path"
    args = app.parse_args(["--snapshot", "/srv/facade/models/sd15-inpaint-8a4288"])
    assert (args.host, args.port, args.root_path) == ("127.0.0.1", 7860, "")


@pytest.mark.parametrize("value", ["/photoguard", "/a/b-c_d.e"])
def test_root_path_accepts_a_url_prefix(value):
    """--root-path accepts a URL prefix such as /photoguard."""
    assert app.parse_args(["--snapshot", "x", "--root-path", value]).root_path == value


@pytest.mark.parametrize("value", ["photoguard", "/photoguard/", "/", "//x", "/a b", "/a?b", "https://x/y"])
def test_root_path_rejects_anything_else(value, capsys):
    """--root-path refuses anything that is not a clean prefix without a trailing slash."""
    with pytest.raises(SystemExit):
        app.parse_args(["--snapshot", "x", "--root-path", value])


class _Block:
    created = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        _Block.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def click(self, fn, inputs, outputs, **kwargs):
        self.clicked = _Event(fn, inputs, outputs, kwargs)
        return self.clicked

    def upload(self, fn, inputs, outputs, **kwargs):
        self.uploaded = _Event(fn, inputs, outputs, kwargs)
        return self.uploaded


class _Event(tuple):
    """(fn, inputs, outputs, kwargs) of one listener, plus the listener chained after it with .then()."""

    def __new__(cls, fn, inputs, outputs, kwargs):
        return super().__new__(cls, (fn, inputs, outputs, kwargs))

    def then(self, fn, inputs, outputs, **kwargs):
        self.next = _Event(fn, inputs, outputs, kwargs)
        return self.next


class _GradioError(Exception):
    def __init__(self, message, **kwargs):
        super().__init__(message)


def _fake_gradio():
    """A stand-in for the gradio module that records every component and listener app.py creates."""
    kinds = ("Blocks", "Row", "Column", "Accordion", "HTML", "Markdown", "ImageMask", "Textbox", "Number",
             "Slider", "Button", "Gallery", "UploadButton", "DownloadButton")
    fake = SimpleNamespace(**{kind: type(kind, (_Block,), {}) for kind in kinds})
    fake.Brush = lambda **kwargs: kwargs
    fake.Progress = lambda: (lambda *args, **kwargs: None)
    fake.Error = _GradioError
    fake.warnings = []
    fake.Warning = fake.warnings.append
    fake.update = lambda **kwargs: kwargs
    return fake


def _built_ui(demo, gradio=None):
    """Build the page with the fake gradio and return its components by kind."""
    _Block.created.clear()
    app.build_ui(gradio or _fake_gradio(), core, demo)
    return {type(block).__name__: block for block in _Block.created}


HIDDEN = {"value": None, "visible": False}
CLEARED = ([], "", HIDDEN, HIDDEN)  # results, summary and both download buttons


def _jpeg(image, orientation):
    """Encode `image` as a JPEG carrying the given EXIF orientation tag."""
    import io
    exif = Image.Exif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, exif=exif.tobytes())
    return buffer.getvalue()


def test_ui_callback_with_fake_gradio_and_fake_pipeline(monkeypatch, tmp_path):
    """The page's components and events are wired as intended, and a run returns the gallery, summary and both downloads."""
    import io
    import json
    import zipfile
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    demo, pipe = make_demo()
    by_kind = _built_ui(demo)
    editor = by_kind["ImageMask"].kwargs
    assert editor["type"] == "pil" and editor["image_mode"] == "RGBA" and editor["fixed_canvas"] is False
    assert editor["format"] == "png" and editor["brush"]["color_mode"] == "fixed"
    # Photos reach the editor only through Choose photo (ROUTE_PHOTOS_JS), so it needs no upload listener.
    assert editor["sources"] == ("upload",) and editor["elem_id"] == app.EDITOR_ID
    assert not hasattr(by_kind["ImageMask"], "uploaded")
    gallery = by_kind["Gallery"].kwargs
    assert gallery["format"] == "png" and gallery["columns"] == 3
    assert by_kind["Blocks"].kwargs["delete_cache"] == (600, 900)
    assert by_kind["Blocks"].kwargs["analytics_enabled"] is False
    # The click first clears the previous results in the browser (no Python, no queue), then runs.
    clear, clear_inputs, outputs, clear_kwargs = by_kind["Button"].clicked
    assert clear is None and clear_inputs is None
    assert clear_kwargs == {"js": app.CLEAR_RESULTS_JS, "queue": False, "api_visibility": "private"}
    run, inputs, run_outputs, click_kwargs = by_kind["Button"].clicked.next
    assert len(inputs) == 5 and run_outputs == outputs and len(outputs) == 4
    assert click_kwargs == {"api_visibility": "private", "concurrency_limit": "default"}
    assert outputs[:2] == [by_kind["Gallery"], by_kind["Markdown"]]
    downloads = [block for block in _Block.created if type(block).__name__ == "DownloadButton"]
    assert outputs[2:] == downloads and all(block.kwargs["visible"] is False for block in downloads)

    background = np.dstack([np.asarray(photo(640, 480)), np.full((480, 640), 255, np.uint8)])
    layer = np.zeros((480, 640, 4), np.uint8)
    layer[:, :320, 3] = 255
    value = {"background": background, "layers": [layer], "composite": background}
    gallery_items, summary, comparison, archive = run(value, "a prompt", 1234, 10, 7.5)
    assert [caption for _, caption in gallery_items] == [
        "Immunized image", "Edit of the original", "Edit of the immunized image"]
    assert all(image.size == (512, 512) for image, _ in gallery_items)
    assert "Seed 1234" in summary and len(pipe.calls) == 2

    # Downloads: written inside Gradio's cache directory, so delete_cache removes them.
    assert comparison["visible"] is True and archive["visible"] is True
    comparison_path, archive_path = Path(comparison["value"]), Path(archive["value"])
    assert comparison_path.parent == archive_path.parent and comparison_path.parent.parent == tmp_path.resolve()
    assert comparison_path.name.endswith("-seed1234-comparison.png") and archive_path.suffix == ".zip"
    with zipfile.ZipFile(archive_path) as bundle:
        assert sorted(bundle.namelist()) == sorted(["original.png", "immunized.png", "edit_original.png",
                                                    "edit_immunized.png", "comparison.png", "settings.json"])
        original = Image.open(io.BytesIO(bundle.read("original.png")))
        assert np.array_equal(np.asarray(original), np.asarray(core.crop_input(photo(640, 480))))
        assert np.array_equal(np.asarray(Image.open(io.BytesIO(bundle.read("immunized.png")))),
                              np.asarray(gallery_items[0][0]))
        settings = json.loads(bundle.read("settings.json"))
    assert settings["prompt"] == "a prompt" and settings["seed"] == 1234 and settings["steps"] == 10
    assert settings["model_revision"] == core.MODEL_REVISION and settings["pgd_eps"] == core.PGD_EPS

    with pytest.raises(_GradioError, match="Paint over"):
        run({"background": background, "layers": [], "composite": None}, "p", 1, 10, 7.5)


def test_ui_run_normalizes_the_editor_photo_and_keeps_the_mask_aligned(monkeypatch, tmp_path):
    """A background larger than MAX_SIDE is scaled down; the painted layer is scaled with it."""
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(io_utils, "MAX_SIDE", 320)
    monkeypatch.setattr(io_utils.normalize_editor_value, "__defaults__", (320,))
    demo, pipe = make_demo()
    run = _built_ui(demo)["Button"].clicked.next[0]
    background = np.asarray(photo(640, 480))
    layer = np.zeros((480, 640, 4), np.uint8)
    layer[:, 80:320, 3] = 255  # after the center crop of the 4:3 frame: the left half of the square
    run({"background": background, "layers": [layer], "composite": None}, "p", 1, 10, 7.5)
    mask = pipe.calls[0]["mask"]  # 240 px upscaled to 512, so the edges are soft (upstream resize_and_crop)
    assert np.all(mask[:, 8:248] == 0) and np.all(mask[:, 264:504] == 255)


def test_ui_photo_upload_is_upright_capped_and_sets_a_finger_sized_brush(monkeypatch):
    """Choose photo turns a sideways phone photo upright, caps it at 1024 px, sizes the brush, and handles bad files with a toast."""
    demo, _ = make_demo()
    gradio = _fake_gradio()
    by_kind = _built_ui(demo, gradio)
    editor = by_kind["ImageMask"]
    results = [by_kind["Gallery"], by_kind["Markdown"],
               *[block for block in _Block.created if type(block).__name__ == "DownloadButton"]]
    load_photo, inputs, outputs, kwargs = by_kind["UploadButton"].uploaded
    assert kwargs == {"trigger_mode": "multiple", "api_visibility": "private"} and outputs == [editor, *results]
    choose = by_kind["UploadButton"].kwargs
    assert choose["type"] == "binary" and choose["file_types"] == ["image", ".heic", ".heif"]

    stored = np.zeros((3024, 4032, 3), np.uint8)
    stored[:, :2016] = (200, 0, 0)  # the stored left half is red; EXIF orientation 6 turns it to the top
    update, *cleared = load_photo(_jpeg(Image.fromarray(stored), orientation=6))
    background = update["value"]["background"]
    assert background.size == (768, 1024) and background.mode == "RGB"
    assert background.getpixel((384, 10))[0] > 150 and background.getpixel((384, 1016))[0] < 50
    assert update["value"]["layers"] == [] and update["brush"]["default_size"] == 43
    assert tuple(cleared) == CLEARED  # a new photo clears the previous photo's results and downloads
    assert gradio.warnings == []

    # A bad file keeps the current photo and results (every output unchanged) and says why in a toast,
    # and like every photo it takes at least MIN_EVENT_SECONDS (a Gradio frontend race; see app.py).
    started = time.monotonic()
    assert load_photo(b"not an image") == ({},) * 5
    assert time.monotonic() - started >= app.MIN_EVENT_SECONDS
    assert load_photo(None) == ({},) * 5
    assert len(gradio.warnings) == 2 and "not an image" in gradio.warnings[0]
    assert gradio.warnings[1] == core.NEED_IMAGE



def test_browser_scripts_match_the_ui():
    """ROUTE_PHOTOS_JS and CLEAR_RESULTS_JS name the same elements and outputs that build_ui creates."""
    route = app.ROUTE_PHOTOS_JS
    assert f'getElementById("{app.EDITOR_ID}")' in route
    assert "input[type=file][accept*=\".heic\"]" in route and ".heic" in io_utils.UPLOAD_FILE_TYPES
    for event in ("dragenter", "dragover", "drop", "change", "paste"):
        assert f'"{event}"' in route, event
    assert route.count(", true);") == 4  # capture phase, ahead of the editor's own handlers
    clear = app.CLEAR_RESULTS_JS
    assert 'return [[], "", hidden, hidden];' in clear and "visible: false" in clear
    # Both are passed to Gradio, and the per-step progress bar stays off (see run()).
    tree = ast.parse((DEMO_DIR / "app.py").read_text())
    (launch,) = _calls(tree, "launch")
    assert {k.arg: ast.unparse(k.value) for k in launch.keywords}["js"] == "ROUTE_PHOTOS_JS"
    assert _calls(tree, "Progress") == []


def test_temp_dir_falls_back_when_configured_dir_is_unusable(monkeypatch, tmp_path):
    """An unusable GRADIO_TEMP_DIR falls back to the next folder, and analytics and network access are switched off."""
    preferred, fallback = tmp_path / "missing-parent" / "gradio", tmp_path / "fallback"
    monkeypatch.setattr(app, "TEMP_DIR_FALLBACKS", (str(fallback),))
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(preferred))
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "True")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    assert app.configure_environment() == str(fallback)
    assert fallback.is_dir() and not preferred.exists()
    assert os.environ["GRADIO_TEMP_DIR"] == str(fallback)
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False" and os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "direct"))
    assert app.configure_environment() == str(tmp_path / "direct")


@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="local pinned snapshot not present")
def test_real_snapshot_configs_load_without_weights():
    """With the pinned snapshot present, its configs load and it really has no .safetensors files."""
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
    config = StableDiffusionInpaintPipeline.load_config(str(SNAPSHOT))
    assert config["_class_name"] == "StableDiffusionInpaintPipeline"
    assert config["safety_checker"] == ["stable_diffusion", "StableDiffusionSafetyChecker"]
    scheduler = DDIMScheduler.from_pretrained(str(SNAPSHOT), subfolder="scheduler", local_files_only=True)
    assert isinstance(scheduler, DDIMScheduler)  # upstream's eta=1 is a DDIM parameter
    assert not any(SNAPSHOT.rglob("*.safetensors"))  # hence use_safetensors=False
