"""PhotoGuard encoder-attack demo core, ported from the official MadryLab demo.

Upstream: https://github.com/MadryLab/photoguard at commit
686bea75c786cb46c88fc396a0cd0ee3d7d28c2e, files demo/app.py and demo/utils.py.
MIT License, Copyright (c) 2022 Madry Lab (see LICENSE beside this file).
Paper: Salman, Khaddaj, Leclerc, Ilyas, Madry, "Raising the Cost of Malicious
AI-Powered Image Editing" (arXiv:2302.06588).

This module never imports gradio. The pipeline is injected, so tests run on CPU
with a fake pipeline; the real service loads it with `load_pipeline()`, which
requires CUDA.

Mask semantics are upstream's, unchanged:

* The user paints the region to KEEP, for example a face. That painted-white
  image is the "brush mask" passed to `compare()`.
* Upstream inverts it (`ImageOps.invert`). In the inverted `mask_image`, white
  marks the region Stable Diffusion may replace (the edit region) and black
  marks the kept region.
* PGD perturbs only the kept (painted) region: `pgd(..., mask=1-mask)`. The
  edit region of the immunized image is the original pixels
  (`recover_image(..., background=True)`).
* The inpainting editor regenerates the white (unpainted) region. For both
  arms, its output is pasted over the ORIGINAL image outside that region
  (`recover_image(edited, init_image, mask_image)`).

What one click of "Immunize and compare" does, in `PhotoGuardDemo.compare()`:

1. Center-crop the photo to 512 x 512 and turn the brush mask into upstream's
   inverted `mask_image` (white = the part the editor may change).
2. Immunize: 200 steps of PGD against the pipeline's own VAE encoder (`pgd()`),
   pushing the latent of the kept region toward the latent of a flat gray
   image. Every pixel moves by at most 0.12 on the [-1, 1] scale.
3. Edit the ORIGINAL crop with Stable Diffusion inpainting and the user's
   prompt and seed.
4. Edit the IMMUNIZED image with the same prompt, the same seed and the same
   starting noise.
5. Return the immunized image and both edits, so they can be shown side by
   side. If the immunization worked, the second edit looks clearly worse.

The two edits differ only in their input image, which is what makes the
comparison fair. This is a demonstration: nothing here measures how well
the protection works.

Upstream-copied code is marked below. Tests compare `pgd()`'s loop and the
four `utils.py` helpers with upstream statement by statement, so their
explanations sit in comments above them, not inside them.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage, ToTensor

# app.py configures logging; this module only writes through this logger.
LOG = logging.getLogger("photoguard_demo")

UPSTREAM_REPOSITORY = "https://github.com/MadryLab/photoguard"
UPSTREAM_COMMIT = "686bea75c786cb46c88fc396a0cd0ee3d7d28c2e"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"

## Good params for editing that we used all over the paper --> decent quality and speed (upstream)
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 100
DEFAULT_SEED = 1234
# DDIM's eta: 1 adds fresh random noise at every denoising step, as upstream.
ETA = 1
# Every image the models see is SIDE x SIDE pixels.
SIDE = 512
# Upstream immunize_fn() arguments to pgd().
PGD_EPS = 0.12
PGD_STEP_SIZE = 0.01
PGD_ITERS = 200
# Upstream UI slider bounds.
STEPS_MIN, STEPS_MAX = 10, 250
GUIDANCE_MIN, GUIDANCE_MAX = 0.1, 25.0
# The largest seed accepted: the largest signed 64-bit integer.
SEED_MAX = 2**63 - 1

# Upstream downloaded rtings.com 204_Gray_Uniformity.png and resized it to
# 512x512 RGB. FACADE inspected those bytes (server/worker/facade_photoguard_encoder.py):
# the result is uniformly RGB(128,128,128). Generate it locally instead.
TARGET_RGB = (128, 128, 128)
TARGET_RGB512_SHA256 = "fd92de53403dba9b673df2cba7ee3de7c7c4ca5ef4b8e1f7f2e8c0d72ed1e193"

SAFETY_ENV = "FACADE_SAFETY_CHECKER"
# Messages the UI shows the user word for word.
CUDA_REQUIRED = (
    "The PhotoGuard demo needs an NVIDIA GPU, but torch.cuda.is_available() is False. "
    "Check `nvidia-smi` and that torch is the +cu128 build."
)
NEED_IMAGE = "Upload an image first."
NEED_PAINT = ("Paint over the part of the image you want to keep, such as a face. "
              "The unpainted part is what the editor will change.")
NEED_UNPAINTED = "Leave some of the image unpainted: that is the part the editor changes."

# Upstream's converters. ToTensor maps 0..255 pixels to 0..1 floats in
# (channels, height, width) order; ToPILImage goes back, truncating to whole
# 0..255 values rather than rounding.
totensor = ToTensor()
topil = ToPILImage()
# progress(stage_name, done, total), for example ("Immunizing", 37, 200).
Progress = Callable[[str, int, int], None]


class InputError(ValueError):
    """A problem with what the user submitted. Safe to show verbatim in the UI."""


# --- upstream demo/utils.py, verbatim --------------------------------------
#
# These four functions are copied character for character from upstream, and a
# test checks that they still are, so all explanation is in these comments.
#
# resize_and_crop(img, size): scale the image so it covers `size`, then cut off
# whatever sticks out (ImageOps.fit). With "center", equal amounts come off
# both sides. A 1024 x 768 photo becomes its middle 768 x 768 square, shrunk
# to 512 x 512.


def resize_and_crop(img, size, crop_type="center"):
    '''Resize and crop the image to the given size.'''
    if crop_type == "top":
        center = (0, 0)
    elif crop_type == "center":
        center = (0.5, 0.5)
    else:
        raise ValueError

    resize = list(size)
    if size[0] is None:
        resize[0] = img.size[0]
    if size[1] is None:
        resize[1] = img.size[1]
    return ImageOps.fit(img, resize, centering=center)


# recover_image(image, init_image, mask, background=False): mix two images
# pixel by pixel using `mask` (white = 1 = the editor's region):
#   background=False: editor's region from `image`, the rest from `init_image`.
#     Used after each edit, so only the edit region can show the editor's work
#     and the kept part is always the untouched original.
#   background=True: the reverse. Used after immunizing, so the edit region
#     stays the original pixels and only the kept part carries the perturbation.
# Only the first channel of the mask is used (`[0]`); the mask is gray anyway.


def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)[0]
    init_image = totensor(init_image)

    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)


# preprocess(image): a Pillow image -> a (1, 3, H, W) tensor on the [-1, 1]
# scale that the VAE expects. Width and height are first rounded down to a
# multiple of 32; at 512 x 512 nothing changes. Used here for the gray target.
#
# prepare_mask_and_masked_image(image, mask): returns
#   mask: (1, 1, H, W), 1 where mask_image is white (>= 0.5), else 0;
#   masked_image: the photo on the [-1, 1] scale with the white (editor's)
#     region set to 0, which is mid-gray. This is what the attack perturbs.


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def prepare_mask_and_masked_image(image, mask):

    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


# --- upstream demo/app.py attack --------------------------------------------
#
# How pgd() works (projected gradient descent with an L-infinity budget):
#
# * Start from X plus uniform random noise in [-eps, +eps].
# * Each iteration: run the VAE encoder on the current image, measure the L2
#   distance between its latent mean and the gray target's latent (`loss`),
#   and ask autograd for d(loss)/d(pixels).
# * Move every pixel by exactly `actual_step_size` against the SIGN of its
#   gradient. The step shrinks linearly from step_size (0.01) toward
#   step_size/100 over the run: big moves first, fine adjustments last.
# * Project: keep every pixel within eps (0.12) of X, then within [-1, 1].
# * Multiply by `mask` (here 1 - edit mask, i.e. the kept region), so the
#   editor's region stays 0. Note that the very first noisy start is not
#   masked; the mask is only applied after each update, as upstream does.
#
# `criterion` is accepted but never used, in upstream as here; the loss is
# always the latent L2 distance. A test compares this loop's statements, in
# order, with upstream's (ignoring the progress call and the added detach()).


def pgd(X, targets, model, criterion, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None,
        generator=None, progress=None):
    """Upstream pgd(). FACADE changes (values are unchanged):

    * the random start is drawn from `generator` (a CPU generator; upstream used
      the CPU global RNG right after torch.manual_seed(seed), which is the same
      stream) and moved to X's device instead of a hard-coded `.cuda()`;
    * `X_adv.detach()` at the top of each iteration drops the previous
      iteration's autograd graph, which upstream kept alive for all 200 steps;
    * the tqdm bar and per-step `loss.item()` became an optional progress callback.

    Returns the perturbed tensor, still on the [-1, 1] scale and on X's device.
    """
    X_adv = X.clone().detach() + (torch.rand(*X.shape, generator=generator)*2*eps-eps).to(X.device)
    for i in range(iters):
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        X_adv = X_adv.detach()
        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean - targets).norm()

        grad, = torch.autograd.grad(loss, [X_adv])

        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        if mask is not None:
            X_adv.data *= mask

        if progress is not None:
            progress(i + 1, iters)

    return X_adv


def get_target():
    """Upstream's gray target image, generated locally (no rtings.com download)."""
    target_image = Image.new("RGB", (SIDE, SIDE), TARGET_RGB)
    if hashlib.sha256(target_image.tobytes()).hexdigest() != TARGET_RGB512_SHA256:
        raise RuntimeError("gray target pixels do not match the bound upstream target")
    return target_image


def _autocast(device):
    """FP16 autocast on CUDA, as upstream's `torch.autocast('cuda')`; a no-op on the CPU tests."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def compute_target_latent(vae, device, dtype):
    """Upstream computed this inside every immunize_fn() call; it is constant, so once.

    The VAE latent mean of the gray target, shape (1, 4, 64, 64). Gradients
    are off because nothing is optimized here.
    """
    with torch.no_grad(), _autocast(device):
        targets = vae.encode(preprocess(get_target()).to(device=device, dtype=dtype)).latent_dist.mean
    targets = targets.detach()
    if targets.shape != (1, 4, SIDE // 8, SIDE // 8) or not torch.isfinite(targets).all():
        raise FloatingPointError("invalid target latent")
    return targets


def immunize_fn(init_image, mask_image, *, vae, targets, device, dtype, iters=PGD_ITERS,
                generator=None, progress=None):
    """Upstream immunize_fn() with the model, target and device passed in.

    `mask_image` is upstream's inverted mask (white = the editor's region).
    Returns the immunized 512 x 512 RGB Pillow image: perturbed where the user
    painted, the original pixels everywhere else. `generator` seeds the random
    start, so the same seed gives the same immunized image.
    """
    # enable_grad: PGD needs gradients even if the caller turned them off.
    with _autocast(device), torch.enable_grad():
        mask, X = prepare_mask_and_masked_image(init_image, mask_image)
        X = X.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        adv_X = pgd(X,
                    targets=targets,
                    model=vae.encode,
                    criterion=torch.nn.MSELoss(),
                    clamp_min=-1,
                    clamp_max=1,
                    eps=PGD_EPS,
                    step_size=PGD_STEP_SIZE,
                    iters=iters,
                    mask=1-mask,
                    generator=generator,
                    progress=progress,
                    )

        # Back from [-1, 1] to [0, 1], then to 8-bit pixels.
        adv_X = (adv_X / 2 + 0.5).clamp(0, 1)

        adv_image = topil(adv_X[0]).convert("RGB")
        adv_image = recover_image(adv_image, init_image, mask_image, background=True)
        return adv_image


# --- input handling ----------------------------------------------------------


def parse_seed(seed) -> int:
    """Upstream: an empty seed means DEFAULT_SEED.

    The UI's seed box is free text. "42" and "42.0" are accepted; "4.2", a
    negative number or anything too big for torch's generator is refused
    with a message the user can act on.
    """
    if seed is None or (isinstance(seed, str) and not seed.strip()):
        return DEFAULT_SEED
    try:
        value = float(seed)
    except (TypeError, ValueError):
        raise InputError("Seed must be a whole number.") from None
    if not value.is_integer() or not 0 <= value <= SEED_MAX:
        raise InputError(f"Seed must be a whole number from 0 to {SEED_MAX}.")
    return int(value)


def parse_steps(steps) -> int:
    """Denoising steps: a whole number within upstream's slider range (10 to 250)."""
    try:
        value = float(steps)
    except (TypeError, ValueError):
        raise InputError("Steps must be a number.") from None
    if not value.is_integer() or not STEPS_MIN <= value <= STEPS_MAX:
        raise InputError(f"Steps must be a whole number from {STEPS_MIN} to {STEPS_MAX}.")
    return int(value)


def parse_guidance(guidance) -> float:
    """Guidance scale: a number within upstream's slider range (0.1 to 25)."""
    try:
        value = float(guidance)
    except (TypeError, ValueError):
        raise InputError("Guidance must be a number.") from None
    if not GUIDANCE_MIN <= value <= GUIDANCE_MAX:
        raise InputError(f"Guidance must be from {GUIDANCE_MIN} to {GUIDANCE_MAX}.")
    return value


def crop_input(image) -> Image.Image:
    """The 512 x 512 RGB center crop that compare() immunizes and edits (upstream resize_and_crop)."""
    return resize_and_crop(image.convert("RGB"), (SIDE, SIDE))


def check_mask(mask_image):
    """Reject masks that make the comparison meaningless (all kept or all edited).

    `mask_image` is the inverted mask, so `edit` is True where the editor may
    change the photo. All True means the user painted nothing; all False
    means they painted everything.
    """
    edit = np.asarray(mask_image.convert("L")) >= 128  # the 0.5 threshold of prepare_mask_and_masked_image
    if edit.all():
        raise InputError(NEED_PAINT)
    if not edit.any():
        raise InputError(NEED_UNPAINTED)


def _as_pil(value) -> Image.Image:
    """Accept what Gradio may hand over (a Pillow image, a numpy array or a file path) as a Pillow image."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        array = value
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
    if isinstance(value, (str, Path)):
        # Copy into memory and close the file at once, so no file handle is
        # left open on Gradio's temporary upload.
        with Image.open(value) as opened:
            opened.load()
            return opened.copy()
    raise InputError(NEED_IMAGE)


def editor_to_image_and_mask(value) -> tuple[Image.Image, Image.Image]:
    """Convert a Gradio 6 ImageEditor/ImageMask value to (RGB image, brush mask).

    The value is {"background", "layers", "composite"}. The background is the
    photo at its own size (the app does not use a fixed canvas). The brush
    strokes live in the layers' alpha channel, and any painted pixel counts as
    KEEP. The returned brush mask is white where painted, the same meaning as
    upstream Gradio 3's `image['mask']`. The composite has the strokes burnt
    in, so it is never used as the image.
    """
    if not isinstance(value, Mapping) or value.get("background") is None:
        raise InputError(NEED_IMAGE)
    background = _as_pil(value["background"])
    image = background.convert("RGB")
    # numpy arrays are (height, width) while Pillow sizes are (width, height).
    painted = np.zeros((image.size[1], image.size[0]), dtype=bool)
    for layer in value.get("layers") or []:
        if layer is None:
            continue
        layer = _as_pil(layer).convert("RGBA")
        if layer.size != image.size:
            layer = layer.resize(image.size, Image.NEAREST)
        # Channel 3 is alpha: any brush stroke, whatever its colour, is opaque there.
        painted |= np.asarray(layer)[..., 3] > 0
    if not painted.any():
        raise InputError(NEED_PAINT)
    return image, Image.fromarray(painted.astype(np.uint8) * 255, "L")


# --- pipeline ----------------------------------------------------------------


def safety_checker_enabled(env: Mapping[str, str] | None = None) -> bool:
    """On unless FACADE_SAFETY_CHECKER is 0 (or false/no/off)."""
    env = os.environ if env is None else env
    return str(env.get(SAFETY_ENV, "1")).strip().lower() not in {"0", "false", "no", "off"}


def pipeline_kwargs(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """from_pretrained() arguments for the local, pinned, .bin-only snapshot.

    local_files_only: never reach the network. use_safetensors=False: the
    pinned snapshot has .bin weights, and those are the files whose hashes
    fetch_models.py checked. With the safety checker off, diffusers also has
    to be told not to warn that it is missing.
    """
    kwargs: dict[str, Any] = {"dtype": torch.float16, "use_safetensors": False, "local_files_only": True}
    if not safety_checker_enabled(env):
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False
    return kwargs


def load_pipeline(snapshot, *, device="cuda", env: Mapping[str, str] | None = None, pipeline_cls=None):
    """Load StableDiffusionInpaintPipeline from a local snapshot directory onto CUDA.

    fetch_models.py writes a `.verified` marker only after every file's
    SHA-256 matched; a snapshot without it still loads (handy for a hand-made
    snapshot on a test machine) but is logged as unchecked. On the server the
    demo's systemd unit waits for the model download service to finish first.
    `pipeline_cls` lets tests pass a fake pipeline class.
    """
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(CUDA_REQUIRED)
    root = Path(snapshot).expanduser().resolve()
    if not (root / "model_index.json").is_file():
        raise FileNotFoundError(f"no model_index.json in snapshot directory {root}")
    if not (root / ".verified").exists():
        LOG.warning("snapshot %s has no .verified marker; checksums were not confirmed", root)
    if pipeline_cls is None:
        from diffusers import StableDiffusionInpaintPipeline as pipeline_cls
    kwargs = pipeline_kwargs(env)
    LOG.info("loading %s@%s from %s (safety checker %s)", MODEL_ID, MODEL_REVISION[:7], root,
             "off" if "safety_checker" in kwargs else "on")
    pipe = pipeline_cls.from_pretrained(str(root), **kwargs)
    pipe = pipe.to(device)
    # diffusers' own tqdm bar would write to the journal on every step.
    pipe.set_progress_bar_config(disable=True)
    for name in ("vae", "unet", "text_encoder", "safety_checker"):
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False)  # PGD needs input gradients only
    return pipe


class PhotoGuardDemo:
    """Holds the loaded pipeline and the target latent, which is computed once.

    One instance serves every user of the demo. The Gradio queue in app.py
    runs one request at a time, so calls never overlap on the GPU.
    """

    def __init__(self, pipe, *, device="cuda", dtype=torch.float16, pgd_iters: int = PGD_ITERS):
        self.pipe = pipe
        self.device = torch.device(device)
        self.dtype = dtype
        self.pgd_iters = pgd_iters
        self.safety_checker = getattr(pipe, "safety_checker", None) is not None
        self.targets = compute_target_latent(pipe.vae, self.device, self.dtype)

    @classmethod
    def from_snapshot(cls, snapshot, *, device="cuda", env: Mapping[str, str] | None = None):
        """Load the real pipeline from `snapshot` and wrap it; this is what app.py calls."""
        return cls(load_pipeline(snapshot, device=device, env=env), device=device, dtype=torch.float16)

    def _sync(self):
        """Wait for queued GPU work to finish, so the timings in `info` are real.

        CUDA calls return before the GPU is done; without this, the time of one
        stage would be counted in the next.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _edit(self, image, mask_image, prompt, seed, steps, guidance, stage, progress):
        """Run one Stable Diffusion inpainting edit; return (image, flagged by the safety checker)."""
        # A fresh generator per arm: both arms start from identical noise.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        extra = {}
        if progress is not None:
            def on_step_end(pipe, step, timestep, callback_kwargs):
                """diffusers calls this after each denoising step; it must return callback_kwargs."""
                progress(stage, step + 1, steps)
                return callback_kwargs
            extra["callback_on_step_end"] = on_step_end
        # height and width come from size[0] and size[1] exactly as in
        # upstream's run(). Pillow's size is (width, height), so the two are
        # swapped, which is harmless because the crop is always square.
        output = self.pipe(prompt=prompt,
                           image=image,
                           mask_image=mask_image,
                           height=image.size[0],
                           width=image.size[1],
                           eta=ETA,
                           guidance_scale=guidance,
                           num_inference_steps=steps,
                           generator=generator,
                           **extra,
                           )
        # None when the safety checker is off; otherwise one flag per image.
        flags = getattr(output, "nsfw_content_detected", None)
        return output.images[0], bool(flags and flags[0])

    def _run(self, image, mask, prompt, seed, steps, guidance, progress, pgd_iters):
        """The steps of compare() on already-checked settings; see the module docstring.

        `pgd_iters` is a parameter so that warmup() can run a single step.
        """
        init_image = crop_input(image)
        mask = mask.convert("L")
        # NEAREST keeps the mask hard black and white when it is resized.
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        # Upstream's inversion: painted (keep) becomes black, unpainted white.
        mask_image = ImageOps.invert(mask.convert("RGB"))
        # The same center crop as the photo, so the two stay lined up.
        mask_image = resize_and_crop(mask_image, init_image.size)
        check_mask(mask_image)

        started = time.perf_counter()
        report = None if progress is None else (lambda done, total: progress("Immunizing", done, total))
        # PGD's random start uses a private CPU generator seeded with the
        # user's seed: the same stream upstream got from torch.manual_seed(seed),
        # without touching the global random state.
        immunized_image = immunize_fn(init_image, mask_image, vae=self.pipe.vae, targets=self.targets,
                                      device=self.device, dtype=self.dtype, iters=pgd_iters,
                                      generator=torch.Generator(device="cpu").manual_seed(seed),
                                      progress=report)
        self._sync()
        immunized_at = time.perf_counter()

        edit_original, flagged_original = self._edit(init_image, mask_image, prompt, seed, steps, guidance,
                                                     "Editing the original", progress)
        edit_original = recover_image(edit_original, init_image, mask_image)
        self._sync()
        original_at = time.perf_counter()

        edit_immunized, flagged_immunized = self._edit(immunized_image, mask_image, prompt, seed, steps, guidance,
                                                       "Editing the immunized image", progress)
        edit_immunized = recover_image(edit_immunized, init_image, mask_image)
        self._sync()
        finished = time.perf_counter()

        # Settings and timings only; no image data. export.py writes this
        # into settings.json and onto the comparison picture.
        info = {
            "prompt": prompt,
            "immunize_seconds": immunized_at - started,
            "edit_original_seconds": original_at - immunized_at,
            "edit_immunized_seconds": finished - original_at,
            "total_seconds": finished - started,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "pgd_iters": pgd_iters,
            "safety_checker": self.safety_checker,
            "safety_flagged_original": flagged_original,
            "safety_flagged_immunized": flagged_immunized,
        }
        return immunized_image, edit_original, edit_immunized, info

    def compare(self, image, mask, prompt, seed=DEFAULT_SEED, steps=NUM_INFERENCE_STEPS,
                guidance=GUIDANCE_SCALE, progress: Progress | None = None):
        """Immunize `image`, then edit the original and the immunized image with the same seed.

        `mask` is the brush mask: white where the user painted, which is the region
        to KEEP (upstream semantics; see the module docstring). Returns
        (immunized_image, edit_of_original, edit_of_immunized, info). All three
        images are 512x512 RGB. `info` holds the timings in seconds, the settings
        used, and whether the safety checker blacked out either edit.
        """
        if image is None:
            raise InputError(NEED_IMAGE)
        if mask is None:
            raise InputError(NEED_PAINT)
        prompt = "" if prompt is None else str(prompt).strip()
        return self._run(image, mask, prompt, parse_seed(seed), parse_steps(steps), parse_guidance(guidance),
                         progress, self.pgd_iters)

    def warmup(self):
        """One PGD step and a one-step edit on a synthetic image, to exercise every GPU path.

        Run once at start-up, before the server listens, so a broken GPU stack
        fails then and not on the first visitor. The image is a red/green
        gradient (no photo of anyone) and the "kept" region is its left half.
        """
        ramp = np.linspace(0, 255, SIDE, dtype=np.float32)
        pixels = np.stack([np.tile(ramp, (SIDE, 1)), np.tile(ramp[:, None], (1, SIDE)),
                           np.full((SIDE, SIDE), 128, np.float32)], axis=-1).astype(np.uint8)
        keep = np.zeros((SIDE, SIDE), np.uint8)
        keep[:, : SIDE // 2] = 255
        result = self._run(Image.fromarray(pixels), Image.fromarray(keep, "L"), "a photo", DEFAULT_SEED,
                           1, GUIDANCE_SCALE, None, 1)
        LOG.info("warm-up done in %.1f s", result[3]["total_seconds"])
        return result


def is_cuda_oom(error: BaseException) -> bool:
    """True if `error` is the GPU running out of memory, which app.py reports separately."""
    return isinstance(error, torch.cuda.OutOfMemoryError)


def release_cuda_memory():
    """Give cached but unused GPU memory back after an out-of-memory error."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_summary(info: Mapping[str, Any]) -> str:
    """Markdown summary of one compare() run for the UI.

    Two lines of settings and timings, then a note if the safety checker
    blacked out an edit or is switched off.
    """
    lines = [
        f"Seed {info['seed']} · {info['steps']} steps · guidance {info['guidance']:g} · "
        f"PGD {info['pgd_iters']} iterations, ε {PGD_EPS} (on the [-1, 1] scale)",
        f"Immunize {info['immunize_seconds']:.1f} s · edit original {info['edit_original_seconds']:.1f} s · "
        f"edit immunized {info['edit_immunized_seconds']:.1f} s · total {info['total_seconds']:.1f} s",
    ]
    flagged = [name for name, key in (("edit of the original", "safety_flagged_original"),
                                      ("edit of the immunized image", "safety_flagged_immunized")) if info.get(key)]
    if flagged:
        lines.append("The safety checker blacked out the " + " and the ".join(flagged)
                     + ". Try another seed or prompt.")
    if not info.get("safety_checker", True):
        lines.append("The safety checker is off (FACADE_SAFETY_CHECKER=0).")
    return "\n\n".join(lines)
