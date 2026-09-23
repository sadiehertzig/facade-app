#!/usr/bin/env python3
"""Experimental PhotoGuard diffusion worker; no GPU smoke or efficacy validation.

Port of MadryLab/photoguard notebooks/demo_complex_attack_inpainting.ipynb
(MIT code, see licenses/photoguard-MIT.txt; the model weights are separately
under CreativeML OpenRAIL-M, see licenses/CreativeML-OpenRAIL-M.txt). The default recipe
uses the complete differentiable VAE/UNet/DDIM/decoder path with four denoising
steps, not the encoder-only phone prototype. It is a 0.40.0 port, not a claim of
bitwise reproduction of the upstream 0.10.2 CUDA notebook. Optimizer arithmetic
and the image-space loss use FP32; model computation uses FP16. The public
entry point only accepts the frozen recipe and CUDA-loaded local components.

This library performs no provisioning, downloads, disk image writes, logging of
images, or scientific evaluation. A server that uses it must pass its own GPU
smoke test before it reports ready. The server does not run this engine today: the
worker service uses the encoder attack (facade_photoguard_encoder.py). The small
PNG helpers and the cancellation type below are shared with the encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Callable

ENGINE_ID = "photoguard_diffusion_inpainting_sd15_experimental_v1"
IMPLEMENTATION_STATUS = "experimental_unvalidated"
GPU_SMOKE_VALIDATED = False
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
MINIMUM_GPU_MEMORY_BYTES = 40_000_000_000
MAX_PNG_BYTES = 16 * 1024 * 1024
REQUIRED_VERSIONS = {"torch": "2.8.0", "diffusers": "0.40.0", "transformers": "5.16.1"}
SOURCE_FILES = {
    "vae/config.json": "4cd0ce4803567837175fbb8412689fe77d069c1ef8171beb83979db309a8bad5",
    "vae/diffusion_pytorch_model.bin": "1b134cded8eb78b184aefb8805b6b572f36fa77b255c483665dda931fa0130c5",
    "unet/config.json": "669f6581087fac5b16c44e42c58b359a35eadf2af5691d7945480bd7ceac8e6d",
    "unet/diffusion_pytorch_model.bin": "af4e22eecaca7a1c5dd849f51924effdccc1f765141e898c323b775cc80a43b3",
    "text_encoder/config.json": "845df614cb9327ae7bbea027316246fae917827407da6df13572e41b5f93b4cc",
    "text_encoder/pytorch_model.bin": "770a47a9ffdcfda0b05506a7888ed714d06131d60267e6cf52765d61cf59fd67",
    "tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer/special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
    "tokenizer/tokenizer_config.json": "00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882",
    "tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    "scheduler/scheduler_config.json": "8b457848b74deeeab2a967b75048dd2cb6f393cfc17751eca1b5ec655cff0183",
}
Cancel = Callable[[], bool]
Progress = Callable[[int, int], None]


class AttackCancelled(RuntimeError):
    """Cooperative cancellation; no partially protected image is returned."""


@dataclass(frozen=True)
class DiffusionRecipe:
    side: int = 512
    iterations: int = 200
    gradient_repetitions: int = 10
    denoising_steps: int = 4
    epsilon: float = 16.0
    step_size: float = 1.0
    guidance_scale: float = 7.5
    eta: float = 1.0
    seed: int = 786349
    prompt: str = ""

    def validate(self) -> None:
        for name, maximum in (("side", 512), ("iterations", 200),
                              ("gradient_repetitions", 10), ("denoising_steps", 4)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid bounded {name}")
        if self.side % 8 or self.side < 8:
            raise ValueError("side must be a multiple of eight")
        for name in ("epsilon", "step_size", "guidance_scale", "eta"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"nonfinite {name}")
        if not (0 < self.epsilon <= 16 and 0 < self.step_size <= min(1, self.epsilon)
                and 1 <= self.guidance_scale <= 7.5 and 0 <= self.eta <= 1):
            raise ValueError("invalid bounded optimization parameters")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 or self.prompt != "":
            raise ValueError("the frozen recipe requires an integer seed and empty prompt")


DEFAULT_RECIPE = DiffusionRecipe()


@dataclass
class LoadedComponents:
    vae: Any
    unet: Any
    tokenizer: Any
    text_encoder: Any
    scheduler: Any
    device: Any
    dtype: Any
    source_digest: str = ""


def _check_cancel(cancel: Cancel | None) -> None:
    if cancel is not None and cancel():
        raise AttackCancelled("PhotoGuard operation cancelled")


def source_digest() -> str:
    payload = json.dumps(SOURCE_FILES, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_local_snapshot(snapshot: Path, cancel: Cancel | None = None) -> Path:
    """Verify exact component bytes; cache symlinks are supported, not trusted."""
    root = Path(snapshot).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("an existing local SD1.5 inpainting snapshot is required; downloads are disabled")
    # Tokenizer loaders can consult optional files. Reject unbound extras in
    # every consumed component directory so no optional variant changes loading.
    expected = set(SOURCE_FILES)
    actual = {str(path.relative_to(root)) for component in ("vae", "unet", "text_encoder", "tokenizer", "scheduler")
              for path in (root / component).rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("local component inventory differs from the frozen snapshot")
    for relative, expected_hash in SOURCE_FILES.items():
        _check_cancel(cancel)
        digest = hashlib.sha256()
        with (root / relative).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                _check_cancel(cancel)
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"local snapshot checksum mismatch: {relative}")
    return root


def load_local_components(snapshot: Path, *, device: str = "cuda:0",
                          cancel: Cancel | None = None) -> LoadedComponents:
    """Explicit operator action: load only verified local weights onto CUDA.

    Upstream reports >=40 GB GPU memory for its full differentiable demo. This
    capacity check is a prerequisite, not a measured memory/speed guarantee.
    Cancellation cannot interrupt a synchronous weight load or CUDA kernel.
    """
    _check_cancel(cancel)
    for distribution, expected in REQUIRED_VERSIONS.items():
        if version(distribution).split("+")[0] != expected:
            raise RuntimeError(f"{distribution}=={expected} is required by this experimental port")
    import torch

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full diffusion PhotoGuard requires CUDA; CPU/phone fallback is unavailable")
    if torch.cuda.get_device_properties(selected).total_memory < MINIMUM_GPU_MEMORY_BYTES:
        raise RuntimeError("upstream full diffusion PhotoGuard requires a GPU with at least 40 GB memory")
    root = verify_local_snapshot(snapshot, cancel)
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    def load(model_type, component):
        _check_cancel(cancel)
        model = model_type.from_pretrained(str(root / component), local_files_only=True,
                                           use_safetensors=False, torch_dtype=torch.float16)
        model.eval().requires_grad_(False)
        model = model.to(selected)
        _check_cancel(cancel)
        return model

    vae = load(AutoencoderKL, "vae")
    unet = load(UNet2DConditionModel, "unet")
    text_encoder = load(CLIPTextModel, "text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
    scheduler = DDIMScheduler.from_config(json.loads((root / "scheduler/scheduler_config.json").read_text()))
    if (unet.config.in_channels != 9 or vae.config.latent_channels != 4
            or float(vae.config.scaling_factor) != 0.18215 or tokenizer.model_max_length != 77):
        raise ValueError("loaded architecture differs from the frozen inpainting recipe")
    _check_cancel(cancel)
    return LoadedComponents(vae, unet, tokenizer, text_encoder, scheduler,
                            selected, torch.float16, source_digest())


def _clean_pil(image, *, mask: bool = False, side: int = 512):
    from PIL import Image, ImageOps

    if not isinstance(image, Image.Image) or image.size != (side, side):
        raise ValueError(f"image and mask must already be cropped to {side}x{side}")
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("animated images are not supported")
    image.load()
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    # Browser mask canvases can have transparent backgrounds: transparent mask
    # pixels mean retained context. Transparent photograph pixels become white.
    base = Image.new("RGBA", rgba.size, (0, 0, 0, 255) if mask else (255, 255, 255, 255))
    value = Image.alpha_composite(base, rgba).convert("L" if mask else "RGB")
    value.info.clear()
    return value


def decode_png(data: bytes, *, mask: bool = False):
    """Decode one bounded PNG in memory, stripping metadata and transparency."""
    from PIL import Image

    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_PNG_BYTES:
        raise ValueError("PNG input must contain 1..16 MiB")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("only PNG input is supported")
            return _clean_pil(image, mask=mask)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("invalid or oversized PNG input") from error


def encode_png(image) -> bytes:
    clean = _clean_pil(image)
    buffer = BytesIO()
    clean.save(buffer, format="PNG")
    data = buffer.getvalue()
    if len(data) > MAX_PNG_BYTES:
        raise ValueError("encoded output exceeds the PNG bound")
    return data


def _validate_tensors(image, mask, recipe: DiffusionRecipe) -> None:
    import torch

    recipe.validate()
    if image.shape != (1, 3, recipe.side, recipe.side) or mask.shape != (1, 1, recipe.side, recipe.side):
        raise ValueError("image/mask tensor shape mismatch")
    if not image.is_floating_point() or not torch.isfinite(image).all() or image.min() < -1 or image.max() > 1:
        raise ValueError("image tensor must be finite normalized RGB in[-1,1]")
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("mask tensor must be binary")
    if not (mask == 0).any() or not (mask == 1).any():
        raise ValueError("mask must contain both a replacement region and retained context")


def _project_l2(candidate, original, mask, epsilon: float):
    import torch

    delta = (candidate.float() - original.float()) * (1 - mask.float())
    norm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
    scale = torch.minimum(torch.ones_like(norm), epsilon / norm.clamp_min(1e-12))
    return (original.float() + delta * scale).clamp(-1, 1)


def _prompt_embeddings(components: LoadedComponents, prompt: str):
    import torch

    # Both are empty for the frozen recipe; retain the upstream two-branch CFG
    # layout rather than silently selecting a different memory optimization.
    tokens = components.tokenizer(["", prompt], padding="max_length", max_length=77,
                                  truncation=True, return_tensors="pt")
    with torch.no_grad():
        return components.text_encoder(tokens.input_ids.to(components.device))[0].detach()


def _forward(components: LoadedComponents, scheduler, candidate, mask, embeddings,
             recipe: DiffusionRecipe, generator, cancel: Cancel | None):
    import torch
    import torch.nn.functional as functional

    _check_cancel(cancel)
    shape = (1, 4, recipe.side // 8, recipe.side // 8)
    # Preserve the original RNG ordering: initial noise, posterior sample, then
    # one fresh DDIM variance draw per denoising timestep when eta is nonzero.
    latents = torch.randn(shape, device=components.device, dtype=components.dtype, generator=generator)
    latent_mask = functional.interpolate(mask.to(components.dtype), size=shape[-2:], mode="nearest")
    posterior = components.vae.encode(candidate.to(components.dtype)).latent_dist
    condition = components.vae.config.scaling_factor * posterior.sample(generator=generator)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(recipe.denoising_steps, device=components.device)
    for timestep in scheduler.timesteps:
        _check_cancel(cancel)
        expanded = scheduler.scale_model_input(torch.cat([latents, latents]), timestep)
        model_input = torch.cat([expanded, torch.cat([latent_mask, latent_mask]),
                                 torch.cat([condition, condition])], dim=1)
        uncond, conditioned = components.unet(model_input, timestep, encoder_hidden_states=embeddings).sample.chunk(2)
        guided = uncond + recipe.guidance_scale * (conditioned - uncond)
        latents = scheduler.step(guided, timestep, latents, eta=recipe.eta, generator=generator).prev_sample
    _check_cancel(cancel)
    return components.vae.decode(latents / components.vae.config.scaling_factor).sample


def _attack_tensors(image, mask, components: LoadedComponents, *, recipe: DiffusionRecipe = DEFAULT_RECIPE,
                    cancel: Cancel | None = None, progress: Progress | None = None):
    """Differentiable core; injectable CPU analytic components are for tests only.

    Each repetition releases its autograd graph. FP32 optimizer state enforces
    the continuous L2 radius; this deliberate precision choice is recorded in
    the module contract and has not been GPU-validated against upstream.
    """
    import torch

    _validate_tensors(image, mask, recipe)
    _check_cancel(cancel)
    original = image.detach().to(device=components.device, dtype=torch.float32)
    mask = mask.detach().to(device=components.device, dtype=torch.float32)
    masked_original = original * (1 - mask)
    candidate = masked_original.clone()
    scheduler = type(components.scheduler).from_config(components.scheduler.config)
    generator = torch.Generator(device=components.device).manual_seed(recipe.seed)
    embeddings = _prompt_embeddings(components, recipe.prompt)
    if progress is not None:
        progress(0, recipe.iterations)
    # Input autograd must work even when an embedding preparation or caller
    # used no_grad. inference_mode is explicitly rejected by the public entry.
    with torch.enable_grad():
        for iteration in range(recipe.iterations):
            _check_cancel(cancel)
            average = torch.zeros_like(candidate)
            for _ in range(recipe.gradient_repetitions):
                _check_cancel(cancel)
                current = candidate.detach().requires_grad_(True)
                decoded = _forward(components, scheduler, current, mask, embeddings, recipe, generator, cancel)
                loss = torch.linalg.vector_norm(decoded.float())
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite diffusion objective")
                gradient = torch.autograd.grad(loss, current, create_graph=False)[0].detach() * (1 - mask)
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite diffusion input gradient")
                average.add_(gradient / recipe.gradient_repetitions)
                del decoded, loss, gradient, current
            norm = torch.linalg.vector_norm(average)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient norm")
            direction = average / (norm + 1e-10)
            with torch.no_grad():
                candidate = _project_l2(candidate - recipe.step_size * direction,
                                        masked_original, mask, recipe.epsilon).detach()
            _check_cancel(cancel)
            if progress is not None:
                progress(iteration + 1, recipe.iterations)
    _check_cancel(cancel)
    return (original * mask + candidate * (1 - mask)).detach().cpu()


def _quantized_output(original, candidate, mask, epsilon: float):
    """Keep the normalized L2 bound after PNG's unavoidable 8-bit rounding."""
    import numpy as np
    from PIL import Image

    pixels = np.asarray(original, dtype=np.uint8)
    proposed = candidate.detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.float64)
    if proposed.shape != pixels.shape or not np.isfinite(proposed).all() or not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("invalid output tensor or bound")
    normalized = pixels.astype(np.float64) / 127.5 - 1
    delta = (proposed - normalized) * (1 - mask[..., None])

    def quantize(scale):
        result = np.rint(np.clip((normalized + scale * delta + 1) * 127.5, 0, 255)).astype(np.uint8)
        result[mask.astype(bool)] = pixels[mask.astype(bool)]
        distance = np.linalg.norm((result.astype(np.float64) - pixels.astype(np.float64)) / 127.5)
        return result, distance

    result, distance = quantize(1)
    if distance > epsilon:
        lower, upper = 0.0, 1.0
        result, _ = quantize(0)
        for _ in range(32):
            midpoint = (lower + upper) / 2
            proposal, distance = quantize(midpoint)
            if distance <= epsilon:
                lower, result = midpoint, proposal
            else:
                upper = midpoint
    return Image.fromarray(result)


def protect(image, mask, *, snapshot: Path | None = None, components: LoadedComponents | None = None,
            cancel: Cancel | None = None, progress: Progress | None = None):
    """Return a fresh 512px protected PNG-compatible PIL image, experimentally.

    White mask pixels identify where AI may replace content; their original
    pixels remain byte-identical in this output. Perturbations affect dark
    retained-context pixels only. Epsilon 16 is a global normalized L2 radius,
    not a 16/255 per-pixel budget. Callers must serialize access to components.
    """
    import numpy as np
    import torch

    _check_cancel(cancel)
    if torch.is_inference_mode_enabled():
        raise RuntimeError("PhotoGuard requires autograd; inference_mode is unsupported")
    clean = _clean_pil(image)
    clean_mask = _clean_pil(mask, mask=True)
    mask_array = (np.asarray(clean_mask, dtype=np.uint8) >= 128).astype(np.float32)
    image_tensor = torch.from_numpy(np.asarray(clean, dtype=np.float32).transpose(2, 0, 1).copy())[None] / 127.5 - 1
    mask_tensor = torch.from_numpy(mask_array.copy())[None, None]
    _validate_tensors(image_tensor, mask_tensor, DEFAULT_RECIPE)
    if components is None:
        if snapshot is None:
            raise ValueError("explicit local snapshot or validated CUDA components are required")
        components = load_local_components(snapshot, cancel=cancel)
    elif snapshot is not None:
        raise ValueError("choose a snapshot or loaded components, not both")
    if torch.device(components.device).type != "cuda" or components.source_digest != source_digest():
        raise ValueError("public protection requires verified local CUDA components")
    candidate = _attack_tensors(image_tensor, mask_tensor, components, cancel=cancel, progress=progress)
    _check_cancel(cancel)
    result = _quantized_output(clean, candidate, mask_array, DEFAULT_RECIPE.epsilon)
    _check_cancel(cancel)
    return result
