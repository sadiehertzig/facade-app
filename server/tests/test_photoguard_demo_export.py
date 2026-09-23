"""Photo upload normalization (io_utils) and slide downloads (export) of the PhotoGuard demo.

Pillow and numpy only: no gradio, torch, GPU or network. The HEIC decode test runs
where pi-heif (what the server installs) or pillow-heif is installed and is skipped
otherwise.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
import pytest

from photoguard_demo import export, io_utils

RED, BLUE, GREEN = (220, 20, 20), (20, 20, 220), (20, 200, 20)
# Made on macOS with `sips -s format heic` from a 128 x 64 JPEG (left half RED, right half BLUE)
# with EXIF orientation 6. Like an iPhone photo it stores the pixels sideways, with an irot box
# and an EXIF orientation, so a decoder that turned it upright twice would fail the test.
HEIC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "photoguard-orientation6.heic"
CREATED = datetime(2026, 9, 23, 14, 5, 12, tzinfo=timezone.utc)
INFO = {"prompt": "a person at a wedding", "seed": 1234, "steps": 100, "guidance": 7.5, "pgd_iters": 200,
        "immunize_seconds": 20.25, "edit_original_seconds": 19.5, "edit_immunized_seconds": 19.75,
        "total_seconds": 59.5, "safety_checker": True, "safety_flagged_original": False,
        "safety_flagged_immunized": False}


def halves(width=64, height=32, left=RED, right=BLUE, mode="RGB"):
    """Left half one color, right half another: easy to tell which way it was turned."""
    image = Image.new(mode, (width, height), right)
    image.paste(Image.new(mode, (width // 2, height), left), (0, 0))
    return image


def jpeg_bytes(image, orientation=None, **save):
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
        exif.get_ifd(0x8825)[2] = (40.0, 26.0, 0.0)  # a GPS latitude, which must not survive
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, exif=exif.tobytes(), **save)
    return buffer.getvalue()


def close(pixel, color, tolerance=40):
    return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(pixel, color))


def panel(color, size=512):
    array = np.zeros((size, size, 3), np.uint8)
    array[...] = color
    array[: size // 2, : size // 2] = (255, 255, 255)
    return Image.fromarray(array)


# --- io_utils: uploads -----------------------------------------------------------


def test_exif_orientation_6_is_turned_upright_and_metadata_is_dropped():
    stored = halves(64, 32)  # stored landscape; orientation 6 means "rotate 90 degrees clockwise to view"
    upright = io_utils.open_upload(jpeg_bytes(stored, orientation=6))
    assert upright.size == (32, 64) and upright.mode == "RGB"
    # Turned clockwise, the stored left (red) half becomes the top.
    assert close(upright.getpixel((16, 8)), RED) and close(upright.getpixel((16, 56)), BLUE)
    assert "exif" not in upright.info and "icc_profile" not in upright.info
    assert dict(upright.getexif()) == {}


# EXIF orientation -> the transpose that shows the photo upright (TIFF/EXIF 2.3, tag 0x0112).
UPRIGHT = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
           5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
           8: Image.Transpose.ROTATE_90}


def test_every_orientation_is_turned_upright(tmp_path):
    stored = halves(48, 24)
    for orientation in range(1, 9):
        data = jpeg_bytes(stored, orientation=orientation)
        path = tmp_path / f"o{orientation}.jpg"
        path.write_bytes(data)
        decoded = Image.open(io.BytesIO(data)).convert("RGB")
        expected = decoded.transpose(UPRIGHT[orientation]) if orientation in UPRIGHT else decoded
        for source in (data, path, str(path)):
            assert np.array_equal(np.asarray(io_utils.open_upload(source)), np.asarray(expected)), orientation


def test_large_photos_are_capped_at_max_side_and_small_ones_are_untouched():
    assert io_utils.MAX_SIDE == 1024  # the editor re-uploads the photo as PNGs, so keep it small
    big = io_utils.normalize_image(halves(4032, 3024))
    assert big.size == (1024, 768)
    tall = io_utils.normalize_image(halves(3000, 4000))
    assert tall.size == (768, 1024)
    assert min(big.size) >= 1.5 * 512  # a 4:3 photo still has headroom over the 512 x 512 crop
    small = halves(640, 480)
    assert np.array_equal(np.asarray(io_utils.normalize_image(small)), np.asarray(small))
    assert io_utils.normalize_image(halves(100, 50), max_side=40).size == (40, 20)


def test_transparency_is_flattened_onto_white_and_modes_become_rgb():
    rgba = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    rgba.paste(Image.new("RGBA", (4, 8), (*RED, 255)), (0, 0))
    flat = io_utils.normalize_image(rgba)
    assert flat.mode == "RGB" and flat.getpixel((1, 1)) == RED and flat.getpixel((6, 1)) == (255, 255, 255)
    for mode in ("1", "L", "LA", "P", "CMYK", "I;16"):
        assert io_utils.normalize_image(Image.new(mode, (4, 4))).mode == "RGB"


def test_srgb_profile_is_applied_then_dropped():
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image = halves(16, 8)
    image.info["icc_profile"] = icc
    result = io_utils.normalize_image(image)
    assert "icc_profile" not in result.info
    assert np.abs(np.asarray(result).astype(int) - np.asarray(halves(16, 8)).astype(int)).max() <= 2


def test_unreadable_uploads_are_upload_errors():
    with pytest.raises(io_utils.UploadError, match="not an image"):
        io_utils.open_upload(b"definitely not an image")
    with pytest.raises(io_utils.UploadError, match="not an image"):
        io_utils.open_upload(jpeg_bytes(halves())[:200])  # truncated
    with pytest.raises(io_utils.UploadError):
        io_utils.normalize_editor_value({"background": object(), "layers": []})


def test_heif_brand_sniffing():
    assert io_utils._is_heif(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00")
    assert io_utils._is_heif(b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00")
    assert not io_utils._is_heif(jpeg_bytes(halves())[:12])


def test_heic_without_a_decoder_gets_a_clear_message(monkeypatch):
    monkeypatch.setattr(io_utils, "register_heif_opener", lambda: None)
    with pytest.raises(io_utils.UploadError, match="HEIC"):
        io_utils.open_upload(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)


def test_heic_decodes_upright():
    data = HEIC_FIXTURE.read_bytes()
    assert io_utils._is_heif(data[:12]) and b"irot" in data and b"Exif" in data
    if io_utils.register_heif_opener() is None:
        pytest.skip("neither pi-heif nor pillow-heif is installed")
    for source in (data, HEIC_FIXTURE):
        upright = io_utils.open_upload(source)
        assert upright.size == (64, 128) and upright.mode == "RGB"
        # Turned clockwise, the stored left (red) half becomes the top.
        assert close(upright.getpixel((32, 16)), RED) and close(upright.getpixel((32, 112)), BLUE)
        assert upright.info == {} and dict(upright.getexif()) == {}


def test_editor_values():
    photo = halves(64, 32)
    fresh = io_utils.editor_value(photo)
    assert fresh == {"background": photo, "layers": [], "composite": None}
    layer = np.zeros((32, 64, 4), np.uint8)
    rgba = np.dstack([np.asarray(halves(64, 32)), np.full((32, 64), 255, np.uint8)])
    normalized = io_utils.normalize_editor_value({"background": rgba, "layers": [layer], "composite": rgba},
                                                 max_side=32)
    assert normalized["background"].size == (32, 16) and normalized["background"].mode == "RGB"
    assert normalized["layers"] == [layer] and normalized["composite"] is rgba
    assert io_utils.normalize_editor_value(None) is None
    assert io_utils.normalize_editor_value({"background": None, "layers": []}) == {"background": None, "layers": []}


def test_brush_radius_scales_with_the_photo_within_the_slider_range():
    assert io_utils.brush_radius(Image.new("RGB", (1024, 768))) == 43
    assert io_utils.brush_radius(Image.new("RGB", (480, 640))) == 27
    assert io_utils.brush_radius(Image.new("RGB", (64, 64))) == 8
    assert io_utils.brush_radius(Image.new("RGB", (9000, 100))) == 100


# --- export: comparison PNG and ZIP ------------------------------------------------


def four_panels():
    return [panel(color) for color in (RED, BLUE, GREEN, (128, 128, 128))]


def test_comparison_places_the_four_panels_with_labels_and_caption():
    images = four_panels()
    sheet = export.comparison_image(images, prompt=INFO["prompt"], info=INFO, created=CREATED)
    width = 2 * export.PAD + 4 * export.PANEL + 3 * export.GAP
    assert sheet.mode == "RGB" and sheet.width == width and sheet.height > export.PANEL + 150
    pixels = np.asarray(sheet)
    dark = pixels.max(axis=-1) < 120
    # Find each panel exactly where the layout puts it; the pixels are copied unchanged.
    top = np.where((pixels == RED).all(axis=-1).any(axis=1))[0].min()  # panel 0's right half is red to the top
    for index, image in enumerate(images):
        left = export.PAD + index * (export.PANEL + export.GAP)
        assert np.array_equal(pixels[top:top + 512, left:left + 512], np.asarray(image)), index
        assert dark[export.PAD:top, left:left + 512].any(), f"no label over panel {index}"
    assert dark[export.PAD:export.PAD + 40, : width // 2].any(), "no title"
    assert dark[top + 512 + 10:, :].sum() > 500, "no caption"


def test_caption_lines_hold_prompt_settings_date_and_credit():
    first, second = export.caption_lines(INFO["prompt"], INFO, CREATED)
    assert first == 'Prompt: "a person at a wedding"'
    for text in ("Seed 1234", "100 steps", "guidance 7.5", "run time 59.5 s", "2026-09-23 14:05 UTC",
                 "PhotoGuard encoder method (MadryLab) · FACADE"):
        assert text in second, text
    flagged = dict(INFO, safety_flagged_immunized=True)
    assert "AI edit of immunized" in export.caption_lines("", flagged, CREATED)[-1]
    assert export.caption_lines("  ", INFO, CREATED)[0] == "Prompt: (none)"


def test_long_text_is_cut_to_fit():
    font = export._font(export.CAPTION_SIZE)
    text = export._fit("word " * 400, font, 300)
    assert text.endswith("…") and font.getlength(text) <= 300
    assert export._fit("short", font, 300) == "short"
    sheet = export.comparison_image(four_panels(), prompt="very long prompt " * 100, info=INFO, created=CREATED)
    assert sheet.width == 2 * export.PAD + 4 * export.PANEL + 3 * export.GAP


def test_text_the_font_lacks_is_folded_not_drawn_as_boxes():
    font = export._font(export.CAPTION_SIZE)
    # iOS turns "--" into an em dash; the built-in font has neither it nor accented letters.
    assert export._drawable("a café scene — at night", font) == "a cafe scene - at night"
    assert export._drawable("café — ✓", font) == "cafe - ?"
    assert export._drawable("Straße – Œuvre 5×3 €", font) == "Strasse - OEuvre 5x3 EUR"
    assert export._drawable("e\u0301 x\u200dy\u00a0z\x00", font) == "e xy z"  # combining, zero-width, NBSP
    assert export._drawable("日本", font) == "??"
    for kept in ("“quoted” ‘x’", "Seed 1 · 2 steps", "…", "plain ASCII 0-9 ~!@#$%^&*()"):
        assert export._drawable(kept, font) == kept  # the font has these
    for text in ("a café — ✓", "Straße – Œuvre", "日本 👍"):
        assert all(export._has_glyph(font, char) or char == " " for char in export._drawable(text, font))
    # The whole sheet is drawn from the folded text, so it is identical to the ASCII version.
    fancy = export.comparison_image(four_panels(), prompt="a café scene — at night ✓", info=INFO, created=CREATED)
    plain = export.comparison_image(four_panels(), prompt="a cafe scene - at night ?", info=INFO, created=CREATED)
    assert np.array_equal(np.asarray(fancy), np.asarray(plain))


def test_settings_json_keeps_the_exact_prompt(tmp_path):
    prompt = "a café scene — at night ✓ 日本"
    files = export.write_exports(tmp_path, original=panel(RED), immunized=panel(BLUE), edit_original=panel(GREEN),
                                 edit_immunized=panel(RED), prompt=prompt, info=dict(INFO, prompt=prompt),
                                 created=CREATED)
    with zipfile.ZipFile(files.archive) as archive:
        assert json.loads(archive.read("settings.json"))["prompt"] == prompt


def test_non_square_panels_are_padded_not_stretched():
    images = four_panels()
    images[1] = Image.new("RGB", (256, 128), BLUE)
    sheet = export.comparison_image(images, prompt="p", info=INFO, created=CREATED)
    assert sheet.width == 2 * export.PAD + 4 * export.PANEL + 3 * export.GAP
    with pytest.raises(ValueError):
        export.comparison_image(images[:3], prompt="p", info=INFO, created=CREATED)


def test_write_exports_names_and_zip_contents(tmp_path):
    images = four_panels()
    files = export.write_exports(tmp_path, original=images[0], immunized=images[1], edit_original=images[2],
                                 edit_immunized=images[3], prompt=INFO["prompt"], info=INFO, created=CREATED,
                                 extra_settings={"model": "m", "pgd_eps": 0.12})
    assert files.comparison == tmp_path / "photoguard-20260923-140512Z-seed1234-comparison.png"
    assert files.archive == tmp_path / "photoguard-20260923-140512Z-seed1234.zip"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([files.comparison.name, files.archive.name])

    with zipfile.ZipFile(files.archive) as archive:
        assert archive.testzip() is None
        assert sorted(archive.namelist()) == sorted(["original.png", "immunized.png", "edit_original.png",
                                                     "edit_immunized.png", "comparison.png", "settings.json"])
        for name, image in zip(export.ARCHIVE_NAMES, images):
            with Image.open(io.BytesIO(archive.read(name))) as stored:
                assert stored.format == "PNG" and stored.size == (512, 512)
                assert np.array_equal(np.asarray(stored.convert("RGB")), np.asarray(image)), name
        assert archive.read("comparison.png") == files.comparison.read_bytes()
        settings = json.loads(archive.read("settings.json"))
        assert archive.getinfo("original.png").date_time == (2026, 9, 23, 14, 5, 12)
    with Image.open(files.comparison) as comparison:
        assert comparison.format == "PNG" and comparison.width == 2 * export.PAD + 4 * 512 + 3 * export.GAP
    assert settings["prompt"] == "a person at a wedding" and settings["seed"] == 1234
    assert settings["created_utc"] == "2026-09-23T14:05:12Z" and settings["total_seconds"] == 59.5
    assert settings["model"] == "m" and settings["pgd_eps"] == 0.12
    assert settings["credit"] == "PhotoGuard encoder method (MadryLab) · FACADE"
    assert set(settings["files"]) == {"original.png", "immunized.png", "edit_original.png", "edit_immunized.png",
                                      "comparison.png"}
