"""Loading and normalizing uploaded photos for the PhotoGuard demo.

No gradio and no torch here, so the tests run anywhere Pillow does.

Phone photos arrive as JPEG or HEIC, are often stored sideways with an EXIF
orientation tag, and are much larger than the demo needs. Every upload goes
through `normalize_image()` before the editor or the 512 x 512 crop sees it:

* the EXIF orientation is applied (`ImageOps.exif_transpose`), so the photo is upright;
* transparency is flattened onto white and the result is RGB;
* an embedded color profile (for example an iPhone's Display P3) is converted to sRGB;
* the long side is capped at `MAX_SIDE` pixels. The 512 x 512 center crop in
  `core.py` is unchanged. Gradio's image editor sends the whole photo back to the
  server as PNGs several times (on load, on every run, and again when the photo
  changes), and that traffic grows with the square of the side. At 1024 px it
  stays at a few MB, and a 4:3 photo keeps a 768 px short side, 1.5x the crop;
* all metadata (EXIF including GPS, XMP, the color profile) is dropped.

HEIC/HEIF (the iPhone default) is decoded with pi-heif, the decode-only build of
pillow-heif, when it is installed; pillow-heif itself also works.

Used by app.py: `open_upload` for the Choose photo button, `editor_value` and
`brush_radius` to show the photo in the editor, and `normalize_editor_value`
when a run starts. Every error a visitor can cause becomes an `UploadError`
with a plain message.
"""
from __future__ import annotations

import functools
import importlib
import io
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Longest side, in pixels, of every photo after upload (see the docstring).
MAX_SIDE = 1024
# For gr.UploadButton: "image" becomes accept="image/*", which on iOS and Android offers
# Photo Library, Take Photo and Choose File. The suffixes let desktop browsers pick HEIC files.
UPLOAD_FILE_TYPES = ["image", ".heic", ".heif"]
# Tried in this order; the first one installed is used.
HEIF_MODULES = ("pi_heif", "pillow_heif")
WHITE = (255, 255, 255)
# Messages the UI shows the visitor word for word.

NOT_AN_IMAGE = "That file is not an image this demo can read. Use a JPEG, PNG, WebP or HEIC photo."
NO_HEIF = "HEIC photos are not supported on this server. Export the photo as JPEG and upload that."
TOO_LARGE = "That image has too many pixels for this demo. Use a smaller photo."


class UploadError(ValueError):
    """A problem with an uploaded file. Safe to show verbatim in the UI."""


@functools.cache
def register_heif_opener() -> str | None:
    """Teach Pillow to open HEIC/HEIF files. Returns the module used, or None if neither is installed.

    functools.cache runs the body only once per process; later calls just
    return the first answer.
    """
    for name in HEIF_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        module.register_heif_opener()
        return name
    return None


def _is_heif(data: bytes) -> bool:
    """ISO base media file with a HEIF brand: 'ftyp' at offset 4, then the major brand.

    A HEIC file starts with a box whose 4-byte type "ftyp" sits at bytes 4-8,
    followed by a 4-byte "brand" naming the format. Checking these 12 bytes
    tells a HEIC apart from a file that is simply not an image, so the visitor
    gets the right message when no HEIC decoder is installed.
    """
    return data[4:8] == b"ftyp" and data[8:12] in {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx",
                                                    b"mif1", b"msf1", b"heif"}


def open_upload(source: bytes | str | Path) -> Image.Image:
    """Decode an uploaded file (bytes or a path) and return it normalized; see `normalize_image`.

    The whole file is decoded in memory. DecompressionBombError is Pillow's
    refusal to decode an image with an enormous pixel count (a small file can
    claim to be gigapixels); it gets its own message.
    """
    data = bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    if register_heif_opener() is None and _is_heif(data[:12]):
        raise UploadError(NO_HEIF)
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            decoded = opened.copy()  # keeps .info (EXIF, ICC) for normalize_image
    except Image.DecompressionBombError:
        raise UploadError(TOO_LARGE) from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise UploadError(NOT_AN_IMAGE) from None
    return normalize_image(decoded)


def _to_srgb(image: Image.Image, icc: bytes | None) -> Image.Image:
    """Convert colours from the photo's embedded profile to standard sRGB.

    iPhones save photos in Display P3, a wider colour space. Without this,
    the model and the downloads would show them slightly washed out, because
    everything downstream assumes sRGB. A photo with no profile is already
    treated as sRGB and is returned as it is.
    """
    if not icc:
        return image
    try:
        from PIL import ImageCms
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return ImageCms.profileToProfile(image, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
    except Exception:  # no littlecms, or a broken profile: keep the pixels as they are
        return image


def _as_image(value: Any) -> Image.Image:
    """Accept what Gradio may send as the editor's background (image, array, path or bytes) as a Pillow image."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        array = value if value.dtype == np.uint8 else np.clip(value, 0, 255).astype(np.uint8)
        return Image.fromarray(array)
    if isinstance(value, (str, Path, bytes, bytearray)):
        return open_upload(value)
    raise UploadError(NOT_AN_IMAGE)


def normalize_image(image: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Upright, RGB, sRGB, at most `max_side` px on the long side, and without metadata."""
    # Read the colour profile first: the conversions below create new images
    # that may not carry it along.
    icc = image.info.get("icc_profile")
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # malformed EXIF: keep the pixels as stored
        pass
    # Any image with transparency: paste it onto white using its alpha channel.
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, WHITE)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        image = flat
    else:
        image = image.convert("RGB")
    width, height = image.size
    if max(width, height) > max_side:
        # Keep the aspect ratio; LANCZOS is Pillow's highest-quality downscale.
        scale = max_side / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(size, Image.LANCZOS)
    image = _to_srgb(image, icc)
    clean = Image.frombytes("RGB", image.size, image.tobytes())  # a new image: no EXIF, GPS, XMP or ICC
    return clean


def editor_value(image: Image.Image) -> dict[str, Any]:
    """A fresh ImageMask value: the normalized photo as the background and nothing painted yet."""
    return {"background": image, "layers": [], "composite": None}


def normalize_editor_value(value: Any, max_side: int = MAX_SIDE) -> Any:
    """Normalize the background of an ImageMask value; the painted layers pass through unchanged.

    The layers were painted over the photo as the browser showed it (upright), so
    they match the upright background; core.editor_to_image_and_mask resizes them
    to the background if it was downscaled here.
    """
    if not isinstance(value, Mapping) or value.get("background") is None:
        return value
    return {**value, "background": normalize_image(_as_image(value["background"]), max_side)}


def brush_radius(image: Image.Image) -> int:
    """A finger-sized brush: about 1/24 of the long side, within the editor's 1-100 slider."""
    return int(min(100, max(8, round(max(image.size) / 24))))
