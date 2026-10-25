#!/usr/bin/env python3
"""The engine: PhotoGuard's encoder attack on a 512 x 512 photo, as run by the server.

Adapted from MadryLab/photoguard demo/app.py at commit
686bea75c786cb46c88fc396a0cd0ee3d7d28c2e (MIT; see licenses/photoguard-MIT.txt).
This is the masked-context encoder attack, not the full diffusion attack
(facade_photoguard_diffusion.py) and not the smaller version that runs on the
phone. On an NVIDIA GPU it runs in FP16 under autocast, the same arithmetic path
as upstream; on a Mac (MPS) it runs in FP32, which is a different numerical
path and is only used for local development. Importing this module or calling
`protect()` never downloads, logs or writes an image.

This is the engine the worker service runs on the server. Run this file directly to
protect one 512x512 PNG on your own GPU or Mac (see --help).

The idea in one paragraph. Stable Diffusion edits a photo by first compressing
it with an encoder (the VAE) into a small "latent" grid, editing that, and
decoding it back. PhotoGuard's encoder attack adds a tiny perturbation to the
photo (at most 0.12 on a [-1, 1] pixel scale, so at most 16 of 255 levels)
chosen so that the VAE's latent for the perturbed photo lands as close as
possible to the latent of a flat gray image. An editor that starts from that
latent has lost the picture, so its edits come out wrong. The perturbation is
found by projected gradient descent (PGD): 200 small steps, each one moving
every pixel a fixed amount in the direction that lowers the latent distance,
then clipping back inside the allowed range.

Mask polarity, used everywhere in this file: the mask is WHITE (255) where an
editor may repaint and those pixels are left exactly as they were; it is BLACK
(0) on the part that gets protected (the "context"). The protection is
experimental: nothing in this file measures how well it works.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Any, Callable

from facade_photoguard_diffusion import AttackCancelled, _check_cancel, _clean_pil, decode_png, encode_png

# ENGINE_ID appears in the service's "model_loaded" log line; the service
# refuses to start unless RECIPE_ID equals the recipe name /v1/capabilities
# reports. "linf" in the recipe id is the L-infinity bound: every pixel may
# move at most epsilon.
ENGINE_ID = "photoguard_encoder_sd15_experimental_v1"
RECIPE_ID = "photoguard-encoder-linf-v1"
IMPLEMENTATION_STATUS = "experimental_unvalidated"
ENGINE_SMOKE_VALIDATED = False
GPU_SMOKE_VALIDATED = False
UPSTREAM_COMMIT = "686bea75c786cb46c88fc396a0cd0ee3d7d28c2e"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
# The exact library versions this port was written against. A different torch
# or diffusers could change the arithmetic, so loading refuses anything else.
REQUIRED_VERSIONS = {"torch": "2.8.0", "diffusers": "0.40.0"}
# Only the VAE is needed for the encoder attack: its config and its weights,
# pinned by SHA-256 so a swapped or corrupted file is refused before it loads.
SOURCE_FILES = {
    "vae/config.json": "4cd0ce4803567837175fbb8412689fe77d069c1ef8171beb83979db309a8bad5",
    "vae/diffusion_pytorch_model.bin": "1b134cded8eb78b184aefb8805b6b572f36fa77b255c483665dda931fa0130c5",
}
# The exact upstream PNG was independently inspected before this port. Its RGB
# conversion and 512 resize are uniformly 128; generate those identical pixels.
TARGET_SOURCE_URL = "https://www.rtings.com/images/test-materials/2015/204_Gray_Uniformity.png"
TARGET_SOURCE_SHA256 = "1034946e419c95d023e44069545462aab36daf3bcab3d7d62f01099f9432ecbb"
TARGET_RGB512_SHA256 = "fd92de53403dba9b673df2cba7ee3de7c7c4ca5ef4b8e1f7f2e8c0d72ed1e193"
QUANTIZED_MAX_DELTA = 16  # ceil(0.12 * 127.5), includes upstream uint8 truncation.
# Callback types. `cancel()` returns True when the caller wants the attack to
# stop; `progress(done, total)` is called after every iteration.
Cancel = Callable[[], bool]
Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class EncoderRecipe:
    """The attack's numbers. The defaults are upstream's and are what the server runs.

    `frozen=True` makes an instance immutable, so a recipe cannot be changed
    after `validate()` has accepted it.
    """
    # image is side x side pixels; the VAE shrinks it by 8 per axis
    side: int = 512
    # PGD steps
    iterations: int = 200
    # max change per pixel on the [-1, 1] scale (0.12 * 127.5 = 15.3 levels)
    epsilon: float = 0.12
    # how far each PGD step moves a pixel, decaying toward 1/100 of this
    step_size: float = 0.01
    # seed of the random starting perturbation, so a job repeats exactly
    seed: int = 1234

    def validate(self):
        """Refuse any value outside the ranges the defaults sit in."""
        if type(self.side) is not int or self.side < 8 or self.side > 512 or self.side % 8:
            raise ValueError("side must be a bounded multiple of eight")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 200:
            raise ValueError("invalid iteration count")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= .12:
            raise ValueError("invalid L-infinity epsilon")
        if not math.isfinite(self.step_size) or not 0 < self.step_size <= .01:
            raise ValueError("invalid step size")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("invalid seed")


DEFAULT_RECIPE = EncoderRecipe()


@dataclass
class LoadedComponents:
    """What `protect()` needs at run time: the loaded VAE and the gray target's latent.

    `source_digest` records which weights, target and recipe the components
    were built from; `protect()` refuses components whose digest does not match
    this module's, so an engine from a different configuration cannot slip in.
    """
    # the diffusers AutoencoderKL (only .encode is used)
    vae: Any
    # latent mean of the gray target, shape (1, 4, 64, 64)
    target_mean: Any
    # torch.device, "cuda:0" on the server
    device: Any
    # float16 on CUDA, float32 on MPS
    dtype: Any
    source_digest: str = ""


def source_digest() -> str:
    """A fingerprint of everything that defines this engine: weight hashes, target, recipe."""
    value = {"files": SOURCE_FILES, "target_rgb512_sha256": TARGET_RGB512_SHA256,
             "upstream_commit": UPSTREAM_COMMIT, "recipe": DEFAULT_RECIPE.__dict__}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_local_snapshot(snapshot: Path, cancel: Cancel | None = None) -> Path:
    """Check that snapshot/vae/ holds exactly the two pinned files with the pinned hashes.

    The inventory check matters as much as the hashes: any extra file in vae/
    (a stray .safetensors copy, a second config) is refused, so the folder the
    loader reads holds exactly the two pinned files and nothing else. Files are
    hashed in 1 MiB blocks so a cancel can land between blocks of the 335 MB
    weight file.
    """
    root = Path(snapshot).expanduser().resolve()
    expected = set(SOURCE_FILES)
    actual = {str(path.relative_to(root)) for path in (root / "vae").rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("local VAE inventory differs from the frozen snapshot")
    for relative, digest in SOURCE_FILES.items():
        h = hashlib.sha256()
        _check_cancel(cancel)
        with (root / relative).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                _check_cancel(cancel)
                h.update(block)
        if h.hexdigest() != digest:
            raise ValueError("local VAE checksum mismatch: " + relative)
    return root


def _target_tensor(*, side=512):
    """The gray target as a (1, 3, side, side) tensor on the [-1, 1] scale.

    Upstream downloaded a gray test image from rtings.com; every pixel of it
    is RGB(128, 128, 128) after its resize, so the same pixels are generated
    here and, at 512 px, checked against the recorded hash.
    """
    import numpy as np
    import torch
    pixels = np.full((side, side, 3), 128, dtype=np.uint8)
    if side == 512 and hashlib.sha256(pixels.tobytes()).hexdigest() != TARGET_RGB512_SHA256:
        raise RuntimeError("upstream target pixel binding mismatch")
    # Keep upstream preprocess ordering: /255 then 2*x-1, not zero or /127.5.
    # 128/255*2-1 = 0.0039..., not exactly 0, and the tests pin that value.
    values = pixels.astype(np.float32) / 255.0
    return 2.0 * torch.from_numpy(values[None].transpose(0, 3, 1, 2).copy()) - 1.0


def _autocast(components):
    """FP16 autocast on CUDA (upstream's path); a no-op context anywhere else."""
    import torch
    return torch.autocast("cuda", dtype=torch.float16) if torch.device(components.device).type == "cuda" else nullcontext()


def load_local_components(snapshot: Path, *, device: str = "cuda:0", cancel: Cancel | None = None) -> LoadedComponents:
    """Load the exact local VAE only. MPS uses FP32; CUDA uses FP16/autocast.

    Steps: check the library versions, pick the device and precision, verify
    the files on disk, load the VAE in eval mode with gradients off (PGD needs
    gradients with respect to the INPUT image, not the weights), then encode
    the gray target once and keep its latent mean for every later job.

    Unlike the full diffusion engine there is no 40 GB memory gate: the VAE
    alone fits easily. A synchronous weight load or GPU kernel cannot be
    interrupted by `cancel`; it is only checked between the steps above.
    """
    _check_cancel(cancel)
    # importlib.metadata.version() may append a local tag such as "+cu128";
    # only the part before "+" has to match.
    for package, expected in REQUIRED_VERSIONS.items():
        if version(package).split("+")[0] != expected:
            raise RuntimeError(package + "==" + expected + " is required")
    import torch
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        dtype = torch.float16
    elif selected.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        dtype = torch.float32
    else:
        raise ValueError("real encoder loading requires explicit CUDA or MPS; CPU fixtures are test-only")
    root = verify_local_snapshot(snapshot, cancel)
    from diffusers import AutoencoderKL
    # local_files_only: never touch the network. use_safetensors=False: the
    # pinned snapshot ships .bin weights and the inventory check above has
    # already refused any other file.
    vae = AutoencoderKL.from_pretrained(str(root / "vae"), local_files_only=True,
        use_safetensors=False, torch_dtype=dtype, low_cpu_mem_usage=False).eval().requires_grad_(False).to(selected)
    _check_cancel(cancel)
    if vae.config.latent_channels != 4:
        raise ValueError("wrong VAE architecture")
    components = LoadedComponents(vae, None, selected, dtype, source_digest())
    # The target latent never changes, so it is computed once here instead of
    # on every job (upstream recomputed it per request).
    with torch.no_grad(), _autocast(components):
        target = vae.encode(_target_tensor().to(device=selected, dtype=dtype)).latent_dist.mean.detach()
    # 512 px / 8 = 64: the VAE's latent is 4 channels of 64 x 64.
    if target.shape != (1, 4, 64, 64) or not torch.isfinite(target).all():
        raise FloatingPointError("invalid target latent mean")
    components.target_mean = target
    _check_cancel(cancel)
    return components


def _validate_tensors(image, mask, components, recipe):
    """Refuse anything the attack math does not expect, before the VAE sees it.

    The image must be a (1, 3, side, side) float tensor within [-1, 1]; the mask
    a (1, 1, side, side) tensor holding only 0 and 1 with both values present
    (an all-protect or all-edit mask would make the job meaningless).
    """
    import torch
    recipe.validate()
    if image.shape != (1, 3, recipe.side, recipe.side) or mask.shape != (1, 1, recipe.side, recipe.side):
        raise ValueError("wrong image/mask shape")
    if not image.is_floating_point() or not torch.isfinite(image).all() or image.min() < -1 or image.max() > 1:
        raise ValueError("image must be finite RGB in [-1,1]")
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all() or not (mask == 0).any() or not (mask == 1).any():
        raise ValueError("mask must contain binary edit and context regions")
    target = components.target_mean
    if target.shape != (1, 4, recipe.side//8, recipe.side//8) or not torch.isfinite(target).all():
        raise FloatingPointError("invalid target latent mean")


def _attack_tensors(image, mask, components, *, recipe=DEFAULT_RECIPE, cancel=None, progress=None):
    """The PGD loop itself, on tensors. Returns the perturbed image on the CPU.

    This follows upstream's `pgd()` statement for statement, so a test can run
    an independent transcription of upstream next to it and compare every
    intermediate tensor. Two upstream details are kept on purpose: the random
    start is FP32 uniform noise drawn on the CPU (then moved to the device), and
    the first iterate is neither masked nor clamped; masking happens after
    every update. The loss is the L2 distance between the VAE latent mean of
    the current image and the latent mean of the gray target. The step size
    decays linearly, the result is projected back into the epsilon box around
    the masked original, then clamped to [-1, 1]. `components.vae` may be a
    tiny CPU stand-in in the tests.
    """
    import torch
    _validate_tensors(image, mask, components, recipe)
    _check_cancel(cancel)
    original = image.detach().to(device=components.device, dtype=components.dtype)
    edit_mask = mask.detach().to(device=components.device, dtype=components.dtype)
    # mask is 1 where the editor may repaint; context is 1 where we protect.
    context = 1 - edit_mask
    # The attack works on the photo with the editable region zeroed (0 is
    # mid-gray on the [-1, 1] scale), which is what an inpainting editor is
    # given as its "masked image".
    masked = original * context
    # A private generator seeded per job: the same job gives the same output,
    # and torch's global random state is never touched.
    rng = torch.Generator(device="cpu").manual_seed(recipe.seed)
    # Uniform noise in [-epsilon, +epsilon] as the starting perturbation.
    noise = (torch.rand(tuple(masked.shape), generator=rng, dtype=torch.float32) * (2 * recipe.epsilon) - recipe.epsilon).to(components.device)
    candidate = masked.clone().detach() + noise
    target = components.target_mean.detach()
    if progress:
        progress(0, recipe.iterations)
    with torch.enable_grad(), _autocast(components):
        for iteration in range(recipe.iterations):
            _check_cancel(cancel)
            # Fresh leaf tensor each step: the previous step's graph is freed
            # (upstream kept all 200 graphs alive, which cost memory but
            # changed no values).
            current = candidate.detach().requires_grad_(True)
            mean = components.vae.encode(current).latent_dist.mean
            if mean.shape != target.shape:
                raise ValueError("encoder target/output shape mismatch")
            # Objective: how far the current latent is from the gray latent.
            loss = (mean - target).norm()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite encoder objective")
            # d(loss)/d(pixels): which way to nudge every pixel to shrink the loss.
            gradient = torch.autograd.grad(loss, current, create_graph=False)[0]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("nonfinite encoder input gradient")
            _check_cancel(cancel)
            # Linear decay from step_size down to step_size/100 over the run:
            # big moves early, fine adjustments late.
            step = recipe.step_size - (recipe.step_size - recipe.step_size/100) / recipe.iterations * iteration
            with torch.no_grad():
                # Signed gradient descent: every pixel moves exactly `step`
                # in the direction that lowers the loss (the L-infinity PGD step).
                candidate = current - gradient.detach().sign() * step
                # Projection: stay within epsilon of the masked original...
                candidate = torch.minimum(torch.maximum(candidate, masked - recipe.epsilon), masked + recipe.epsilon)
                # ...and within valid pixel range, then re-zero the editable region.
                candidate = candidate.clamp(-1, 1) * context
            del current, mean, loss, gradient
            if not torch.isfinite(candidate).all():
                raise FloatingPointError("nonfinite encoder iterate")
            if progress:
                progress(iteration + 1, recipe.iterations)
    _check_cancel(cancel)
    return candidate.detach().cpu()


def _output_image(original, candidate, mask):
    """Match upstream ToPILImage truncation; preserve white-region bytes exactly.

    Converts the [-1, 1] tensor back to 8-bit pixels the way torchvision's
    ToPILImage does (scale to [0, 255] and truncate, not round), copies the
    original pixels back over the white (editable) region, and refuses the
    result if any byte moved by more than QUANTIZED_MAX_DELTA (16).
    """
    import numpy as np
    import torch
    from PIL import Image
    pixels = np.asarray(original, dtype=np.uint8)
    edited = np.asarray(mask, dtype=np.uint8) >= 128
    if candidate.shape != (1, 3, *pixels.shape[:2]) or not torch.isfinite(candidate).all():
        raise ValueError("invalid protected output tensor")
    # CUDA reference candidate is FP32 due to CPU-FP32 random-start addition.
    normalized = (candidate.detach().cpu().float()[0] / 2 + .5).clamp(0, 1)
    result = (normalized * 255).to(torch.uint8).permute(1, 2, 0).numpy().copy()
    result[edited] = pixels[edited]
    delta = np.abs(result.astype(np.int16) - pixels.astype(np.int16))
    if delta.max() > QUANTIZED_MAX_DELTA:
        raise ValueError("exported PNG exceeds the encoder recipe byte bound")
    return Image.fromarray(result)


def protect(image, mask, *, snapshot=None, components=None, device="cuda:0", cancel=None, progress=None):
    """Return a 512x512 RGB image using the fixed 200-step masked-context encoder recipe.

    White mask means the region an editor may replace. Its pixels are unchanged
    in the protected output; only dark retained context receives perturbations.
    Only binary 0/255 masks with both regions are supported; soft edges are rejected.
    This function does not assert effectiveness, quality or deployment readiness.

    This is the public entry point the worker calls once per job. It takes
    Pillow images in and gives a Pillow image out; the tensor work happens in
    `_attack_tensors`. Pass either `snapshot` (load the VAE now) or
    `components` (already loaded, the server's case), never both.
    """
    import numpy as np
    import torch
    _check_cancel(cancel)
    # torch.inference_mode() disables autograd entirely, and PGD needs the
    # gradient with respect to the image, so refuse to run inside it.
    if torch.is_inference_mode_enabled():
        raise RuntimeError("encoder protection requires autograd, not inference_mode")
    clean = clean_mask = output = None
    try:
        clean = _clean_pil(image)
        clean_mask = _clean_pil(mask, mask=True)
        # Only pure black and pure white, and both must appear. A soft-edged
        # mask would make "protected" a matter of degree, which this recipe
        # does not define.
        values = np.asarray(clean_mask, dtype=np.uint8)
        if not np.all((values == 0) | (values == 255)) or not np.any(values == 0) or not np.any(values == 255):
            raise ValueError("encoder mask must be binary 0/255 with both edit and context regions")
        # HWC uint8 -> 1 x C x H x W float on the [-1, 1] scale (x/127.5 - 1).
        image_tensor = torch.from_numpy(np.asarray(clean, dtype=np.float32).transpose(2, 0, 1).copy())[None] / 127.5 - 1
        mask_tensor = torch.from_numpy((np.asarray(clean_mask, dtype=np.uint8) >= 128).astype(np.float32))[None, None]
        if components is None:
            if snapshot is None:
                raise ValueError("explicit local snapshot or verified loaded components required")
            components = load_local_components(Path(snapshot), device=device, cancel=cancel)
        elif snapshot is not None:
            raise ValueError("choose snapshot or loaded components")
        # The components must have come from load_local_components with this
        # module's exact configuration: right device, right precision, and the
        # same source_digest. A CPU test double is refused here on purpose.
        expected_dtype = torch.float16 if torch.device(components.device).type == "cuda" else torch.float32
        if (torch.device(components.device).type not in {"cuda", "mps"} or components.dtype != expected_dtype
                or components.source_digest != source_digest()):
            raise ValueError("public protection requires verified CUDA-FP16 or MPS-FP32 components")
        candidate = _attack_tensors(image_tensor, mask_tensor, components, cancel=cancel, progress=progress)
        _check_cancel(cancel)
        output = _output_image(clean, candidate, clean_mask)
        _check_cancel(cancel)
        # Hand the output to the caller and forget it here, so the finally
        # block below does not close the very image being returned.
        returned, output = output, None
        return returned
    finally:
        for value in (clean, clean_mask, output):
            if value is not None:
                value.close()


def main(argv=None):
    """Protect one 512x512 PNG on a local CUDA GPU or Apple MPS; never downloads anything.

        python facade_photoguard_encoder.py --snapshot DIR --input photo.png \\
            --mask mask.png --output protected.png [--device mps]

    The mask is white where an editor may repaint (left unchanged) and black on
    the context that gets protected. The snapshot directory needs the pinned
    vae/ files (see server/models/fetch_models.py).

    Progress is printed every 20 iterations and a one-line JSON summary at the
    end. The output file must not exist already; it is never overwritten.
    """
    import argparse
    import time
    import numpy as np
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True, help="local SD1.5-inpainting snapshot directory")
    parser.add_argument("--input", type=Path, required=True, help="512x512 PNG photo")
    parser.add_argument("--mask", type=Path, required=True, help="512x512 PNG mask (white = editable region)")
    parser.add_argument("--output", type=Path, required=True, help="where to write the protected PNG (must not exist)")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (FP16) or mps (FP32)")
    parser.add_argument("--deadline-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    # "Cooperative" means the deadline is only checked between steps; a single
    # GPU kernel that hangs is not interrupted.
    if not math.isfinite(args.deadline_seconds) or not 0 < args.deadline_seconds <= 1800:
        parser.error("deadline must be 1..1800 seconds; cancellation is cooperative")
    if args.output.exists():
        parser.error(f"{args.output} already exists")
    started = time.monotonic()
    image = mask = result = None
    try:
        image = decode_png(args.input.read_bytes())
        mask = decode_png(args.mask.read_bytes(), mask=True)

        def cancelled():
            return time.monotonic() - started >= args.deadline_seconds

        def progress(current, total):
            if current % 20 == 0 or current == total:
                print(f"iteration {current}/{total} ({time.monotonic() - started:.1f} s)", flush=True)

        components = load_local_components(args.snapshot, device=args.device, cancel=cancelled)
        result = protect(image, mask, components=components, cancel=cancelled, progress=progress)
        raw = encode_png(result)
        # "xb": create the file, fail if it appeared since the check above.
        with args.output.open("xb") as handle:
            handle.write(raw)
        delta = np.asarray(result).astype(np.int16) - np.asarray(image).astype(np.int16)
        print(json.dumps({"output": str(args.output), "recipe": RECIPE_ID, "device": str(components.device),
                          "seconds": round(time.monotonic() - started, 1),
                          "max_absolute_byte_delta": int(np.abs(delta).max())}))
    finally:
        for value in (image, mask, result):
            if value is not None:
                value.close()


if __name__ == "__main__":
    main()
