#!/usr/bin/env python3
"""PhotoGuard's full diffusion attack, plus PNG helpers the encoder engine shares.

Port of MadryLab/photoguard notebooks/demo_complex_attack_inpainting.ipynb
(MIT code, see licenses/photoguard-MIT.txt; the model weights are separately
under CreativeML OpenRAIL-M, see licenses/CreativeML-OpenRAIL-M.txt). The default recipe
uses the complete differentiable VAE/UNet/DDIM/decoder path with four denoising
steps, not the encoder-only version that runs on the phone. It is written for
diffusers 0.40.0, so it does not reproduce demo_complex_attack_inpainting.ipynb
(which used diffusers 0.10.2 on CUDA) bit for bit. Optimizer arithmetic and the image-space
loss use FP32; model computation uses FP16. The public entry point only accepts
the default recipe and components loaded from local files onto CUDA.

This library performs no provisioning, downloads, disk image writes, logging of
images, or measurement of how well it protects. A server that uses it must pass
its own GPU smoke test before it reports ready. The server does not run this
engine today: the worker service uses the encoder attack
(facade_photoguard_encoder.py). The small PNG helpers (`decode_png`,
`encode_png`, `_clean_pil`) and the cancellation pieces (`AttackCancelled`,
`_check_cancel`) below are shared with the encoder.

How the diffusion attack differs from the encoder attack. The encoder attack
only looks at the first stage of Stable Diffusion (the VAE encoder). This one
runs the WHOLE inpainting edit, differentiably: encode the photo, run four DDIM
denoising steps of the UNet, decode back to pixels. It then asks "which way
should each protected pixel move so the edited picture comes out as close to
all zeros (flat mid-gray) as possible?" and steps that way. Every one of the
200 iterations averages the gradient over 10 noisy runs of that edit, so one
job is 200 x 10 x 4 = 8,000 UNet passes. That is why upstream asks for a 40 GB
GPU, and why the server uses the much cheaper encoder attack instead.

Two more differences: the perturbation budget is an L2 ball (the total size of
the change over the whole image is at most 16 on the [-1, 1] scale) rather
than a per-pixel limit, and each step moves along the normalized gradient
rather than along its sign.

Mask polarity matches the encoder: WHITE (255, or 1 as a tensor) marks where
an editor may repaint, and those pixels are returned untouched; BLACK (0) is
the context that gets perturbed. The protection is experimental and nothing
here measures it.
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

# Names for this engine. GPU_SMOKE_VALIDATED stays False: this engine has not
# been run end to end on the server's GPU.
ENGINE_ID = "photoguard_diffusion_inpainting_sd15_experimental_v1"
IMPLEMENTATION_STATUS = "experimental_unvalidated"
GPU_SMOKE_VALIDATED = False
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
# Upstream reports that the full differentiable attack needs a GPU with at
# least 40 GB of memory; loading refuses a smaller GPU instead of failing later.
MINIMUM_GPU_MEMORY_BYTES = 40_000_000_000
# Upper bound for a PNG going through decode_png or encode_png. A 512 x 512 RGB
# PNG is under 1 MB, so 16 MiB only stops absurd inputs. (The /v1 API has its
# own decoder and a tighter 4 MiB limit.)
MAX_PNG_BYTES = 16 * 1024 * 1024
# The exact library versions this port was written against; anything else is
# refused because a different version could change the arithmetic.
REQUIRED_VERSIONS = {"torch": "2.8.0", "diffusers": "0.40.0", "transformers": "5.16.1"}
# Every file this attack loads, pinned by SHA-256. The safety checker and the
# feature extractor are not listed because this attack never runs them.
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
# Callback types. `cancel()` returns True when the caller wants the attack to
# stop; `progress(done, total)` is called after every iteration.
Cancel = Callable[[], bool]
Progress = Callable[[int, int], None]


class AttackCancelled(RuntimeError):
    """Cooperative cancellation; no partially protected image is returned.

    "Cooperative" means the attack checks `cancel()` between steps and raises
    this exception; nothing interrupts a GPU call that is already running.
    The encoder engine imports and raises this same exception.
    """


@dataclass(frozen=True)
class DiffusionRecipe:
    """The attack's numbers, all taken from the `super_l2` call in demo_complex_attack_inpainting.ipynb.

    `frozen=True` makes an instance immutable, so a recipe cannot change after
    `validate()` accepted it. Upstream's attack uses the empty prompt, and
    `validate()` insists on that.
    """
    # image is side x side pixels
    side: int = 512
    # optimizer steps
    iterations: int = 200
    # noisy edits averaged per step (upstream grad_reps=10)
    gradient_repetitions: int = 10
    # DDIM steps inside each differentiable edit
    denoising_steps: int = 4
    # L2 radius of the whole change, on the [-1, 1] scale
    epsilon: float = 16.0
    # L2 length of each step (no decay, as upstream)
    step_size: float = 1.0
    # classifier-free guidance strength
    guidance_scale: float = 7.5
    # DDIM randomness: 1 adds fresh noise at every denoising step
    eta: float = 1.0
    # upstream's SEED, used for every random draw in a job
    seed: int = 786349
    # the text the simulated editor is given
    prompt: str = ""

    def validate(self) -> None:
        """Refuse any value outside the range the defaults sit in, before any model call."""
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
    """The loaded Stable Diffusion parts the attack runs through.

    `source_digest` records which pinned files they were built from; `protect()`
    refuses components whose digest does not match this module's.
    """
    # AutoencoderKL: encodes the photo and decodes the edit
    vae: Any
    # UNet2DConditionModel: the denoiser (9 input channels for inpainting)
    unet: Any
    # CLIPTokenizer for the prompt
    tokenizer: Any
    # CLIPTextModel: turns the tokens into embeddings
    text_encoder: Any
    # DDIMScheduler; each job works on its own copy
    scheduler: Any
    # torch.device, always CUDA for the public entry point
    device: Any
    # torch.float16
    dtype: Any
    source_digest: str = ""


def _check_cancel(cancel: Cancel | None) -> None:
    """Raise AttackCancelled if the caller's `cancel()` says so; do nothing without one."""
    if cancel is not None and cancel():
        raise AttackCancelled("PhotoGuard operation cancelled")


def source_digest() -> str:
    """A fingerprint of the pinned file hashes, used to match components to this module."""
    payload = json.dumps(SOURCE_FILES, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_local_snapshot(snapshot: Path, cancel: Cancel | None = None) -> Path:
    """Verify exact component bytes; cache symlinks are supported, not trusted.

    A Hugging Face cache stores files as symlinks into a blob folder. They are
    followed, but what counts is the hash of the bytes at the end, so a link
    pointing somewhere unexpected still fails. Files are read in 1 MiB blocks,
    checking `cancel` between blocks, so a cancel does not wait for the whole
    multi-gigabyte UNet to be hashed.
    """
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
    check only refuses a smaller GPU; passing it does not promise that a job
    fits in memory or how long it takes. Cancellation cannot interrupt a
    synchronous weight load or CUDA kernel.

    Steps: check library versions, check for a CUDA GPU with enough memory,
    verify every file's hash, load each model in FP16 with gradients off (the
    attack needs gradients with respect to the IMAGE, not the weights), and
    check that the loaded models have the inpainting architecture.
    """
    _check_cancel(cancel)
    # importlib.metadata.version() may append a local tag such as "+cu128";
    # only the part before "+" has to match.
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
        """Load one model from its local folder: no network, .bin weights, FP16, eval mode."""
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
    # The inpainting UNet takes 9 channels: 4 noisy latent + 1 mask + 4 latent
    # of the masked photo. 0.18215 is SD 1.5's latent scaling factor and 77 is
    # CLIP's prompt length in tokens.
    if (unet.config.in_channels != 9 or vae.config.latent_channels != 4
            or float(vae.config.scaling_factor) != 0.18215 or tokenizer.model_max_length != 77):
        raise ValueError("loaded architecture differs from the frozen inpainting recipe")
    _check_cancel(cancel)
    return LoadedComponents(vae, unet, tokenizer, text_encoder, scheduler,
                            selected, torch.float16, source_digest())


def _clean_pil(image, *, mask: bool = False, side: int = 512):
    """Return a fresh, metadata-free copy of a 512 x 512 image: "L" for a mask, "RGB" for a photo.

    Refuses the wrong size and animated images, applies the EXIF rotation, and
    flattens transparency (see the comment below for which colour). Both
    engines run every input through this, and encode_png runs every output
    through it.
    """
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
    """Decode one bounded PNG in memory, stripping metadata and transparency.

    Used by the encoder's command-line tool. Pillow's DecompressionBombError
    (a tiny file that claims a gigantic size) is turned into a plain ValueError
    like every other bad input.
    """
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
    """Serialize a 512 x 512 image as an RGB PNG with no metadata, in memory."""
    clean = _clean_pil(image)
    buffer = BytesIO()
    clean.save(buffer, format="PNG")
    data = buffer.getvalue()
    if len(data) > MAX_PNG_BYTES:
        raise ValueError("encoded output exceeds the PNG bound")
    return data


def _validate_tensors(image, mask, recipe: DiffusionRecipe) -> None:
    """Refuse inputs the attack math does not expect, before any model runs.

    The image must be (1, 3, side, side), finite and within [-1, 1]; the mask
    (1, 1, side, side) with only 0 and 1 and both present (an all-repaint or
    all-context mask leaves nothing to protect or nothing to edit).
    """
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
    """Pull the candidate back inside the L2 ball of radius epsilon around the original.

    The change is first limited to the context (mask 0). If its total length
    is above epsilon it is scaled down to exactly epsilon, keeping its
    direction; a shorter change is left as it is. Then pixels are clamped to
    [-1, 1]. Upstream does the same with torch.renorm.
    """
    import torch

    # Only the context may change; the editable region's change is zeroed.
    delta = (candidate.float() - original.float()) * (1 - mask.float())
    norm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
    # min(1, epsilon / norm): 1 inside the ball, a shrink factor outside it.
    # clamp_min avoids dividing by zero when nothing has changed yet.
    scale = torch.minimum(torch.ones_like(norm), epsilon / norm.clamp_min(1e-12))
    return (original.float() + delta * scale).clamp(-1, 1)


def _prompt_embeddings(components: LoadedComponents, prompt: str):
    """CLIP embeddings for ["", prompt]: the unconditional branch first, then the prompt.

    Computed once per job with gradients off; the attack never changes them.
    """
    import torch

    # Both are empty for the default recipe. The two-branch layout of
    # classifier-free guidance (CFG) is kept as upstream has it, instead of
    # quietly switching to a cheaper single-branch shortcut.
    tokens = components.tokenizer(["", prompt], padding="max_length", max_length=77,
                                  truncation=True, return_tensors="pt")
    with torch.no_grad():
        return components.text_encoder(tokens.input_ids.to(components.device))[0].detach()


def _forward(components: LoadedComponents, scheduler, candidate, mask, embeddings,
             recipe: DiffusionRecipe, generator, cancel: Cancel | None):
    """One differentiable inpainting edit of `candidate`; returns the decoded pixels.

    This mirrors upstream's `attack_forward`: random starting latents, the VAE
    latent of the (perturbed) masked photo as the condition, `denoising_steps`
    DDIM steps of the UNet with classifier-free guidance, and the VAE decoder.
    Unlike the normal pipeline, gradients are kept the whole way through, so
    the result can be differentiated with respect to `candidate`.
    """
    import torch
    import torch.nn.functional as functional

    _check_cancel(cancel)
    # 512 px / 8 = 64: Stable Diffusion's latent is 4 channels of 64 x 64.
    shape = (1, 4, recipe.side // 8, recipe.side // 8)
    # Preserve the original RNG ordering: initial noise, posterior sample, then
    # one fresh DDIM variance draw per denoising timestep when eta is nonzero.
    latents = torch.randn(shape, device=components.device, dtype=components.dtype, generator=generator)
    # The mask shrunk to latent size (64 x 64); "nearest" keeps it 0/1.
    latent_mask = functional.interpolate(mask.to(components.dtype), size=shape[-2:], mode="nearest")
    # The editor's view of the photo. This is where the perturbation enters,
    # and where the gradient flows back to.
    posterior = components.vae.encode(candidate.to(components.dtype)).latent_dist
    condition = components.vae.config.scaling_factor * posterior.sample(generator=generator)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(recipe.denoising_steps, device=components.device)
    for timestep in scheduler.timesteps:
        _check_cancel(cancel)
        # Batch of two: row 0 is the unconditional branch, row 1 the prompt.
        expanded = scheduler.scale_model_input(torch.cat([latents, latents]), timestep)
        # The 9 UNet input channels: noisy latent, mask, masked-photo latent.
        model_input = torch.cat([expanded, torch.cat([latent_mask, latent_mask]),
                                 torch.cat([condition, condition])], dim=1)
        uncond, conditioned = components.unet(model_input, timestep, encoder_hidden_states=embeddings).sample.chunk(2)
        # Classifier-free guidance: push the prediction away from "no prompt".
        guided = uncond + recipe.guidance_scale * (conditioned - uncond)
        latents = scheduler.step(guided, timestep, latents, eta=recipe.eta, generator=generator).prev_sample
    _check_cancel(cancel)
    # Undo the latent scaling and decode back to (1, 3, 512, 512) pixels.
    return components.vae.decode(latents / components.vae.config.scaling_factor).sample


def _attack_tensors(image, mask, components: LoadedComponents, *, recipe: DiffusionRecipe = DEFAULT_RECIPE,
                    cancel: Cancel | None = None, progress: Progress | None = None):
    """Differentiable core; injectable CPU analytic components are for tests only.

    This follows upstream's `super_l2`. For each of the 200 iterations:
    run the differentiable edit `gradient_repetitions` times (each run draws
    different noise), take the gradient of the loss with respect to the
    image each time, zero it outside the context, and average. Then step
    `step_size` along the normalized average, downhill, and project back into
    the epsilon L2 ball. The loss is the L2 norm of the edited picture, i.e.
    its distance from an all-zero (mid-gray) target, as in demo_complex_attack_inpainting.ipynb.

    Each repetition releases its autograd graph. The candidate and the
    optimizer arithmetic stay in FP32 so the L2 radius is enforced exactly,
    while the models run in FP16. That choice is this port's own and has not
    been checked against upstream on a GPU. Returns the result on the CPU,
    with the editable region set back to the original values.
    """
    import torch

    _validate_tensors(image, mask, recipe)
    _check_cancel(cancel)
    original = image.detach().to(device=components.device, dtype=torch.float32)
    mask = mask.detach().to(device=components.device, dtype=torch.float32)
    # The editable region is zeroed (0 is mid-gray on the [-1, 1] scale), which
    # is what an inpainting editor is given as its "masked image".
    masked_original = original * (1 - mask)
    candidate = masked_original.clone()
    # A private scheduler copy: set_timesteps changes a scheduler's state, and
    # the shared one in `components` must stay untouched.
    scheduler = type(components.scheduler).from_config(components.scheduler.config)
    # One generator per job, seeded with the recipe's seed: the same job gives
    # the same output, the noise still differs between repetitions inside the
    # job, and torch's global random state is never touched.
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
                # Distance of the edited picture from all zeros (upstream's target).
                loss = torch.linalg.vector_norm(decoded.float())
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite diffusion objective")
                # The gradient is kept only on the context (mask 0), as upstream.
                gradient = torch.autograd.grad(loss, current, create_graph=False)[0].detach() * (1 - mask)
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite diffusion input gradient")
                average.add_(gradient / recipe.gradient_repetitions)
                # Drop this repetition's tensors now so its graph is freed
                # before the next repetition builds another one.
                del decoded, loss, gradient, current
            norm = torch.linalg.vector_norm(average)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient norm")
            # Unit-length direction; 1e-10 (upstream's value) avoids dividing by zero.
            direction = average / (norm + 1e-10)
            with torch.no_grad():
                # Step downhill by step_size, then project back into the ball.
                candidate = _project_l2(candidate - recipe.step_size * direction,
                                        masked_original, mask, recipe.epsilon).detach()
            _check_cancel(cancel)
            if progress is not None:
                progress(iteration + 1, recipe.iterations)
    _check_cancel(cancel)
    # Original pixels in the editable region, attacked pixels in the context.
    return (original * mask + candidate * (1 - mask)).detach().cpu()


def _quantized_output(original, candidate, mask, epsilon: float):
    """Keep the normalized L2 bound after PNG's unavoidable 8-bit rounding.

    Rounding every pixel to a whole 0..255 value can make the total change
    slightly larger than epsilon. If it does, a binary search (32 halvings)
    finds the largest fraction of the change that still fits after rounding.
    The editable region (mask 1) always keeps its original bytes.
    """
    import numpy as np
    from PIL import Image

    pixels = np.asarray(original, dtype=np.uint8)
    proposed = candidate.detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.float64)
    if proposed.shape != pixels.shape or not np.isfinite(proposed).all() or not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("invalid output tensor or bound")
    normalized = pixels.astype(np.float64) / 127.5 - 1
    delta = (proposed - normalized) * (1 - mask[..., None])

    def quantize(scale):
        """Apply `scale` times the change, round to bytes, restore the editable region, measure L2."""
        result = np.rint(np.clip((normalized + scale * delta + 1) * 127.5, 0, 255)).astype(np.uint8)
        result[mask.astype(bool)] = pixels[mask.astype(bool)]
        distance = np.linalg.norm((result.astype(np.float64) - pixels.astype(np.float64)) / 127.5)
        return result, distance

    result, distance = quantize(1)
    if distance > epsilon:
        lower, upper = 0.0, 1.0
        # Scale 0 is the unchanged photo, which always fits; start from it so
        # the result is never over the bound even if the search finds nothing.
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

    The public entry point, with the same shape as the encoder's `protect()`:
    Pillow images in, a Pillow image out. Pass either `snapshot` (load now) or
    `components` (already loaded), never both. Only real CUDA components with
    this module's `source_digest` are accepted, so the CPU test doubles cannot
    be used here.
    """
    import numpy as np
    import torch

    _check_cancel(cancel)
    # inference_mode turns autograd off completely, and the attack needs it.
    if torch.is_inference_mode_enabled():
        raise RuntimeError("PhotoGuard requires autograd; inference_mode is unsupported")
    clean = _clean_pil(image)
    clean_mask = _clean_pil(mask, mask=True)
    # Mask to 0/1 floats: bright pixels (>= 128) are the editable region.
    mask_array = (np.asarray(clean_mask, dtype=np.uint8) >= 128).astype(np.float32)
    # HWC uint8 -> 1 x C x H x W float on the [-1, 1] scale (x/127.5 - 1).
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
