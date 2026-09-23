"""Analytic CPU fixtures only: no real weights, GPU, provider, or source photos."""
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

import facade_photoguard_diffusion as pg


class AnalyticVAE:
    config = SimpleNamespace(scaling_factor=0.18215)

    def __init__(self, recorder):
        self.recorder = recorder

    def encode(self, image):
        pooled = F.avg_pool2d(image, 8)
        mean = torch.cat([pooled, pooled.mean(dim=1, keepdim=True)], dim=1)
        def sample(generator):
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            self.recorder["posterior_noise"].append(noise.detach().clone())
            return mean + noise * 0.03
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=sample))

    def decode(self, latent):
        self.recorder["decode"] += 1
        return SimpleNamespace(sample=F.interpolate(latent[:, :3], scale_factor=8, mode="nearest"))


class AnalyticUNet:
    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, image, timestep, encoder_hidden_states):
        assert image.shape[0:2] == (2, 9)
        assert encoder_hidden_states.shape[0] == 2
        self.recorder["unet"] += 1
        self.recorder["initial_or_later_latent"].append(image[:, :4].detach().clone())
        return SimpleNamespace(sample=0.1 * image[:, :4] + 0.2 * image[:, 5:9] + 0.01 * image[:, 4:5])


class AnalyticScheduler:
    init_noise_sigma = 1.0

    def __init__(self, config=None):
        self.config = {} if config is None else config

    @classmethod
    def from_config(cls, config):
        return cls(dict(config))

    def set_timesteps(self, count, device):
        self.timesteps = torch.arange(count - 1, -1, -1, device=device)

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, noise_prediction, timestep, latent, eta, generator):
        fresh = torch.randn(latent.shape, dtype=latent.dtype, device=latent.device, generator=generator)
        return SimpleNamespace(prev_sample=0.8 * latent + 0.2 * noise_prediction + eta * 0.005 * fresh)


def components(dtype=torch.float32):
    recorder = {"unet": 0, "decode": 0, "posterior_noise": [], "initial_or_later_latent": []}
    def tokenize(prompts, **kwargs):
        assert prompts == ["", ""]
        return SimpleNamespace(input_ids=torch.zeros(2, 77, dtype=torch.long))
    value = pg.LoadedComponents(AnalyticVAE(recorder), AnalyticUNet(recorder), tokenize,
        lambda tokens: (torch.zeros(2, 77, 2, dtype=dtype),), AnalyticScheduler(),
        torch.device("cpu"), dtype)
    return value, recorder


def tensors(dtype=torch.float32):
    image = torch.linspace(-0.7, 0.7, 3 * 8 * 8, dtype=dtype).reshape(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, dtype=dtype)
    mask[:, :, :, :3] = 1
    return image, mask


def recipe(**kwargs):
    return replace(pg.DEFAULT_RECIPE, side=8, iterations=2, gradient_repetitions=2,
                   **kwargs)


def test_complete_diffusion_gradient_matches_independent_finite_difference():
    comp, seen = components(torch.float64)
    image, mask = tensors(torch.float64)
    configured = recipe()
    embeddings = pg._prompt_embeddings(comp, "")
    x = (image * (1 - mask)).requires_grad_(True)
    def objective(candidate):
        generator = torch.Generator().manual_seed(123)
        scheduler = AnalyticScheduler()
        output = pg._forward(comp, scheduler, candidate, mask, embeddings, configured, generator, None)
        return output.square().sum().sqrt()
    gradient = torch.autograd.grad(objective(x), x)[0]
    direction = torch.linspace(-1, 1, x.numel(), dtype=x.dtype).reshape(x.shape)
    direction /= direction.norm()
    expected = (gradient * direction).sum().item()
    h = 1e-4
    observed = ((objective(x.detach() + h * direction) - objective(x.detach() - h * direction)) / (2*h)).item()
    assert abs(expected) > 1e-7
    assert observed == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert seen["unet"] == 3 * 4 and seen["decode"] == 3


def test_optimizer_projects_masked_context_and_reports_completed_iterations():
    comp, seen = components()
    image, mask = tensors()
    configured = recipe(epsilon=0.12, step_size=0.1)
    updates = []
    with torch.no_grad():
        result = pg._attack_tensors(image, mask, comp, recipe=configured,
                                   progress=lambda current, total: updates.append((current, total)))
    assert updates == [(0, 2), (1, 2), (2, 2)]
    assert torch.equal(result * mask, image * mask)
    assert torch.linalg.vector_norm(result - image) <= configured.epsilon + 1e-6
    assert result.min() >= -1 and result.max() <= 1
    assert not torch.equal(result, image)
    assert seen["unet"] == 2 * 2 * 4 and seen["decode"] == 2 * 2
    assert not hasattr(comp.scheduler, "timesteps"), "shared scheduler must remain untouched"


def test_one_small_step_reduces_same_noise_analytic_target_loss():
    comp, _ = components()
    image, mask = tensors()
    configured = replace(recipe(epsilon=0.1, step_size=0.05), iterations=1, gradient_repetitions=1)
    embeddings = pg._prompt_embeddings(comp, "")
    def loss(candidate):
        output = pg._forward(comp, AnalyticScheduler(), candidate * (1-mask), mask,
                             embeddings, configured, torch.Generator().manual_seed(configured.seed), None)
        return output.float().norm().item()
    before = loss(image)
    after_image = pg._attack_tensors(image, mask, comp, recipe=configured)
    assert loss(after_image) < before


def test_noise_advances_within_job_and_repeats_only_for_same_new_job_seed():
    image, mask = tensors()
    a, seen_a = components()
    b, seen_b = components()
    first = pg._attack_tensors(image, mask, a, recipe=recipe())
    second = pg._attack_tensors(image, mask, b, recipe=recipe())
    assert torch.equal(first, second)
    assert not torch.equal(seen_a["posterior_noise"][0], seen_a["posterior_noise"][1])
    assert not torch.equal(seen_a["initial_or_later_latent"][0], seen_a["initial_or_later_latent"][4])
    assert all(torch.equal(x, y) for x, y in zip(seen_a["posterior_noise"], seen_b["posterior_noise"]))


def test_cancel_before_work_returns_no_partial_result():
    comp, seen = components()
    image, mask = tensors()
    progress = []
    with pytest.raises(pg.AttackCancelled):
        pg._attack_tensors(image, mask, comp, recipe=recipe(), cancel=lambda: True,
                           progress=lambda *v: progress.append(v))
    assert seen["unet"] == 0 and progress == []


def test_cancel_between_denoising_steps_stops_current_iteration():
    comp, seen = components()
    image, mask = tensors()
    progress = []
    with pytest.raises(pg.AttackCancelled):
        pg._attack_tensors(image, mask, comp, recipe=recipe(), cancel=lambda: seen["unet"] >= 2,
                           progress=lambda *v: progress.append(v))
    assert seen["unet"] == 2 and progress == [(0, 2)]


def test_cancel_from_progress_stops_before_next_iteration():
    comp, seen = components()
    image, mask = tensors()
    progress = []
    with pytest.raises(pg.AttackCancelled):
        pg._attack_tensors(image, mask, comp, recipe=recipe(), cancel=lambda: (1, 2) in progress,
                           progress=lambda *v: progress.append(v))
    assert progress == [(0, 2), (1, 2)] and seen["unet"] == 8


def test_nonfinite_decoder_rejected_before_progress_or_output():
    comp, _ = components()
    image, mask = tensors()
    comp.vae.decode = lambda z: SimpleNamespace(sample=z[:, :3] * float("nan"))
    with pytest.raises(FloatingPointError, match="objective"):
        pg._attack_tensors(image, mask, comp, recipe=recipe())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 0, 17])
def test_invalid_l2_recipe_refuses_before_model_calls(bad):
    comp, seen = components()
    image, mask = tensors()
    with pytest.raises(ValueError):
        pg._attack_tensors(image, mask, comp, recipe=recipe(epsilon=bad))
    assert seen["unet"] == 0


@pytest.mark.parametrize("kind", ["all_repaint", "all_context", "fractional", "nan"])
def test_invalid_mask_is_rejected(kind):
    comp, _ = components()
    image, mask = tensors()
    if kind == "all_repaint": mask.fill_(1)
    if kind == "all_context": mask.zero_()
    if kind == "fractional": mask[:, :, 0, 0] = 0.5
    if kind == "nan": mask[:, :, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="mask"):
        pg._attack_tensors(image, mask, comp, recipe=recipe())


def test_local_snapshot_rejects_changed_bytes_and_optional_unbound_files(tmp_path, monkeypatch):
    path = tmp_path / "vae/config.json"
    path.parent.mkdir()
    path.write_bytes(b"frozen analytic config")
    monkeypatch.setattr(pg, "SOURCE_FILES", {"vae/config.json": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert pg.verify_local_snapshot(tmp_path) == tmp_path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        pg.verify_local_snapshot(tmp_path)
    path.write_bytes(b"frozen analytic config")
    (path.parent / "optional.json").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        pg.verify_local_snapshot(tmp_path)


def test_cpu_real_load_refuses_without_snapshot_access(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "version", lambda name: pg.REQUIRED_VERSIONS[name])
    monkeypatch.setattr(pg, "verify_local_snapshot", lambda *a: pytest.fail("snapshot must not be read"))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        pg.load_local_components(tmp_path, device="cpu")


def test_public_api_rejects_cpu_injection_and_inference_mode():
    image = Image.new("RGB", (512, 512), "gray")
    mask = Image.new("L", (512, 512), 0)
    mask.paste(255, (0, 0, 256, 512))
    comp, _ = components()
    with pytest.raises(ValueError, match="verified local CUDA"):
        pg.protect(image, mask, components=comp)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        pg.protect(image, mask, components=comp)


def png(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_png_bounds_transparency_and_metadata_stripping():
    photograph = Image.new("RGBA", (512, 512), (50, 60, 70, 0))
    result = pg.decode_png(png(photograph))
    assert result.getpixel((0, 0)) == (255, 255, 255) and result.info == {}
    assert pg.decode_png(png(photograph), mask=True).getpixel((0, 0)) == 0
    with pytest.raises(ValueError, match="cropped"):
        pg.decode_png(png(Image.new("RGB", (16, 16))))
    with pytest.raises(ValueError, match="16 MiB"):
        pg.decode_png(b"x" * (pg.MAX_PNG_BYTES + 1))
    result.info["comment"] = "must not persist"
    recovered = pg.decode_png(pg.encode_png(result))
    assert recovered.info == {}


def test_quantization_never_exceeds_global_l2_or_changes_repaint_pixels():
    original = Image.new("RGB", (512, 512), (127, 127, 127))
    mask = np.zeros((512, 512), np.float32)
    mask[:, :256] = 1
    candidate = torch.full((1, 3, 512, 512), 127 / 127.5 - 1 + 0.004)
    naive = np.rint((candidate.numpy()[0].transpose(1, 2, 0) + 1) * 127.5)
    assert np.linalg.norm((naive[:, 256:] - 127) / 127.5) > 2.6
    result = np.asarray(pg._quantized_output(original, candidate, mask, epsilon=2.6))
    assert np.array_equal(result[:, :256], np.asarray(original)[:, :256])
    assert np.linalg.norm((result.astype(np.float64) - 127) / 127.5) <= 2.6
    assert result.dtype == np.uint8
