"""Downloads for one PhotoGuard demo run: a labeled comparison PNG and a ZIP of everything.

No gradio and no torch here. The comparison is one PNG for a slide:

    title
    Original | Immunized | AI edit of original | AI edit of immunized     (labels)
    [512 x 512] [512 x 512] [512 x 512] [512 x 512]                       (panels)
    Prompt: "..."
    Seed · steps · guidance · run time · date (UTC) · credit

The ZIP holds original.png, immunized.png, edit_original.png, edit_immunized.png,
comparison.png and settings.json. Everything is PNG (lossless). Text uses Pillow's
built-in font (`ImageFont.load_default(size=...)`, Pillow 10.1 or later), so no
font file is needed. That font covers ASCII and a little punctuation only, so text
on the image is folded to what it can draw ("café — ✓" becomes "cafe - ?");
settings.json keeps the exact prompt.
"""
from __future__ import annotations

import functools
import io
import json
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

TITLE = "PhotoGuard: immunized photo vs. AI inpainting edit"
CREDIT = "PhotoGuard encoder method (MadryLab) · FACADE"
LABELS = ("Original", "Immunized", "AI edit of original", "AI edit of immunized")
ARCHIVE_NAMES = ("original.png", "immunized.png", "edit_original.png", "edit_immunized.png")

PANEL = 512
PAD = 32
GAP = 16
TITLE_SIZE = 44
LABEL_SIZE = 30
CAPTION_SIZE = 24
BACKGROUND = (255, 255, 255)
INK = (17, 17, 17)
MUTED = (68, 68, 68)
FRAME = (208, 208, 208)

NO_GLYPH = "\uffff"  # a noncharacter: every font draws it as its missing-glyph box
# Missing characters that NFKD does not reduce to ones the font has. iOS turns "--"
# into an em dash, so dashes are the common case.
FOLDS = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2212": "-", "\u2022": "-", "\u2032": "'", "\u2033": '"', "\u00d7": "x", "\u20ac": "EUR",
    "\u00df": "ss", "\u1e9e": "SS", "\u00e6": "ae", "\u00c6": "AE", "\u0153": "oe", "\u0152": "OE",
    "\u00f8": "o", "\u00d8": "O", "\u0142": "l", "\u0141": "L", "\u0111": "d", "\u0110": "D",
    "\u00f0": "d", "\u00d0": "D", "\u00fe": "th", "\u00de": "Th", "\u0131": "i",
})
# Never drawn: combining accents (after folding), zero-width and control characters.
INVISIBLE = {"Mn", "Me", "Cf", "Cc"}


@dataclass(frozen=True)
class ExportFiles:
    comparison: Path
    archive: Path


@functools.lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _ink(font: ImageFont.FreeTypeFont, text: str) -> bytes:
    side = 2 * int(font.size)
    canvas = Image.new("L", (side, side))
    ImageDraw.Draw(canvas).text((0, 0), text, font=font, fill=255)
    return canvas.tobytes()


@functools.lru_cache(maxsize=4096)
def _has_glyph(font, char: str) -> bool:
    """Whether `font` draws `char` as itself rather than as its missing-glyph box."""
    if not isinstance(font, ImageFont.FreeTypeFont):
        return ord(char) < 256  # Pillow's fallback bitmap font (no FreeType) encodes Latin-1
    return _ink(font, char) != _ink(font, NO_GLYPH)


def _fold(char: str, font) -> str:
    """`char` if the font has it, else the nearest text it can draw (é -> e, — -> -), else "?"."""
    if char.isspace():
        return " "
    if unicodedata.category(char) in INVISIBLE:
        return ""
    if _has_glyph(font, char):
        return char
    folded = "".join(c for c in unicodedata.normalize("NFKD", char.translate(FOLDS))
                     if unicodedata.category(c) not in INVISIBLE)
    return folded if all(c == " " or _has_glyph(font, c) for c in folded) else "?"


def _drawable(text: str, font) -> str:
    """`text` using only characters `font` can draw, so the image never shows empty boxes."""
    return "".join(_fold(char, font) for char in unicodedata.normalize("NFC", text))


def _fit(text: str, font, width: int) -> str:
    """`text`, cut with an ellipsis if it is wider than `width` pixels."""
    # Every drawn character is at least 1 px wide, so no more than `width` of them fit. Cutting
    # first keeps a huge prompt from costing a glyph check per character.
    cut = len(text) > width
    text = _drawable(text[:width], font)
    if not cut and font.getlength(text) <= width:
        return text
    ellipsis = _drawable("…", font)
    low, high = 0, len(text)
    while low < high:  # longest prefix that fits with the ellipsis
        middle = (low + high + 1) // 2
        if font.getlength(text[:middle].rstrip() + ellipsis) <= width:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + ellipsis


def _line_height(font) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def stamp(created: datetime) -> str:
    """Short UTC timestamp for file names, e.g. 20260923-140512Z."""
    return created.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def file_stem(created: datetime, seed: int) -> str:
    return f"photoguard-{stamp(created)}-seed{seed}"


def caption_lines(prompt: str, info: Mapping[str, Any], created: datetime) -> list[str]:
    """The text under the panels: the prompt, then the settings, run time, date and credit."""
    prompt = " ".join(str(prompt or "").split())
    when = created.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f'Prompt: "{prompt}"' if prompt else "Prompt: (none)",
             f"Seed {info['seed']} · {info['steps']} steps · guidance {info['guidance']:g} · "
             f"run time {info['total_seconds']:.1f} s · {when} · {CREDIT}"]
    flagged = [name for name, key in (("AI edit of original", "safety_flagged_original"),
                                      ("AI edit of immunized", "safety_flagged_immunized")) if info.get(key)]
    if flagged:
        lines.append("Blacked out by the Stable Diffusion safety checker: " + " and ".join(flagged) + ".")
    return lines


def comparison_image(images: Sequence[Image.Image], *, prompt: str, info: Mapping[str, Any],
                     created: datetime, labels: Sequence[str] = LABELS, title: str = TITLE) -> Image.Image:
    """The four panels side by side with a title, a label over each panel and a caption underneath."""
    if len(images) != len(labels):
        raise ValueError(f"expected {len(labels)} images, got {len(images)}")
    title_font, label_font, caption_font = _font(TITLE_SIZE), _font(LABEL_SIZE), _font(CAPTION_SIZE)
    captions = caption_lines(prompt, info, created)
    width = 2 * PAD + len(images) * PANEL + (len(images) - 1) * GAP
    inner = width - 2 * PAD
    title_h, label_h, caption_h = (_line_height(f) for f in (title_font, label_font, caption_font))
    panels_top = PAD + title_h + 20 + label_h + 10
    height = panels_top + PANEL + 20 + len(captions) * (caption_h + 6) + PAD

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, PAD), _fit(title, title_font, inner), font=title_font, fill=INK)
    for index, (image, label) in enumerate(zip(images, labels)):
        left = PAD + index * (PANEL + GAP)
        panel = image.convert("RGB")
        if panel.size != (PANEL, PANEL):
            panel = ImageOps.pad(panel, (PANEL, PANEL), color=BACKGROUND)
        canvas.paste(panel, (left, panels_top))
        draw.rectangle((left - 1, panels_top - 1, left + PANEL, panels_top + PANEL), outline=FRAME)
        text = _fit(label, label_font, PANEL)
        draw.text((left + (PANEL - label_font.getlength(text)) / 2, PAD + title_h + 20), text,
                  font=label_font, fill=INK)
    y = panels_top + PANEL + 20
    for line in captions:
        draw.text((PAD, y), _fit(line, caption_font, inner), font=caption_font, fill=MUTED)
        y += caption_h + 6
    return canvas


def settings_record(prompt: str, info: Mapping[str, Any], created: datetime,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """settings.json: the exact prompt, seed and settings, to cite this comparison or rerun it.

    The painted mask is not saved, so a rerun needs the region painted again.
    """
    record: dict[str, Any] = {"created_utc": created.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "prompt": prompt}
    record.update({key: value for key, value in info.items() if key != "prompt"})
    record.update(extra or {})
    record["files"] = {
        "original.png": "512 x 512 center crop of the uploaded photo (the input to both arms)",
        "immunized.png": "the original after PhotoGuard's encoder attack",
        "edit_original.png": "Stable Diffusion inpainting edit of the original",
        "edit_immunized.png": "the same edit (same seed) of the immunized image",
        "comparison.png": "all four side by side, with the settings",
    }
    record["credit"] = CREDIT
    return record


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_entry(name: str, created: datetime) -> zipfile.ZipInfo:
    entry = zipfile.ZipInfo(name, date_time=created.astimezone(timezone.utc).timetuple()[:6])
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.external_attr = 0o644 << 16
    return entry


def write_exports(directory: Path, *, original: Image.Image, immunized: Image.Image, edit_original: Image.Image,
                  edit_immunized: Image.Image, prompt: str, info: Mapping[str, Any], created: datetime,
                  extra_settings: Mapping[str, Any] | None = None) -> ExportFiles:
    """Write <stem>-comparison.png and <stem>.zip into `directory` and return their paths.

    Only these two files touch the disk; the ZIP's members are built in memory.
    """
    directory = Path(directory)
    images = (original, immunized, edit_original, edit_immunized)
    comparison = png_bytes(comparison_image(images, prompt=prompt, info=info, created=created))
    stem = file_stem(created, info["seed"])
    comparison_path = directory / f"{stem}-comparison.png"
    comparison_path.write_bytes(comparison)
    archive_path = directory / f"{stem}.zip"
    settings = json.dumps(settings_record(prompt, info, created, extra_settings), indent=2, ensure_ascii=False)
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, image in zip(ARCHIVE_NAMES, images):
            archive.writestr(_zip_entry(name, created), png_bytes(image.convert("RGB")))
        archive.writestr(_zip_entry("comparison.png", created), comparison)
        archive.writestr(_zip_entry("settings.json", created), settings.encode("utf-8") + b"\n")
    return ExportFiles(comparison=comparison_path, archive=archive_path)
