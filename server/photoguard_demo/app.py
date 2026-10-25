#!/usr/bin/env python3
"""FACADE private PhotoGuard demo: a Gradio 6 port of the official MadryLab demo UI.

Thin wrapper around core.py, which holds all model and attack code. The app
binds only to --host (127.0.0.1 behind Caddy). It never uses share links and
sends no analytics. See README.md for provenance and the changes from upstream.

On the server, Caddy serves it at https://<host>/photoguard/, so the service passes
--root-path /photoguard. Run it locally without --root-path.
"""
# The docstring above doubles as the --help text, so the longer explanation is
# kept in these comments.
#
# Where this fits: deploy/systemd/facade-demo.service runs this file as
# facade-demo on the GPU server. Caddy forwards https://<host>/photoguard/ to
# 127.0.0.1:7860, where Gradio listens. Gradio serves the page and calls the
# Python functions in build_ui() when the visitor clicks something.
#
# Start-up, in main(): set environment variables that turn off analytics and
# network access BEFORE gradio or diffusers are imported, load the model from
# the local snapshot, run a tiny warm-up on the GPU, then build the page and
# start listening. All model and attack code is in core.py; this file is only
# the page.
#
# Two kinds of browser-side JavaScript are passed to Gradio as strings:
# ROUTE_PHOTOS_JS (every photo goes through the Choose photo button) and
# CLEAR_RESULTS_JS (old results vanish the moment a new run starts). Both are
# explained where they are defined.
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("photoguard_demo.app")
# Where Gradio keeps uploads and results if GRADIO_TEMP_DIR is not usable. The
# first is the GPU machine's fast local NVMe disk, if it has one.
TEMP_DIR_FALLBACKS = ("/opt/dlami/nvme/facade-gradio", "/var/tmp/facade-gradio")
# Room for an original full-resolution phone photo, which is far larger than
# the 512 x 512 PNGs the /v1 worker accepts.
MAX_FILE_SIZE = "64mb"  # per upload; matches the /photoguard route's request_body in deploy/Caddyfile
# A bright pink at 60% opacity, so the photo still shows through the strokes.
BRUSH_COLOR = ("#ff2d55", 0.6)
BRUSH_DEFAULT = 48  # radius in image pixels until a photo sets its own (io_utils.brush_radius)
# Gradio 6.28's frontend sometimes misses the end of an event that finishes within a few
# milliseconds of starting (an unreadable file, a tiny photo): the warning never shows and the
# spinner stays until the next photo. Choose photo therefore always takes at least this long.
MIN_EVENT_SECONDS = 0.3

# Page text. These strings are shown to visitors exactly as written.
HEADER = """
<h1 class="pg-title" style="font-weight: 800; margin: 6px 0 4px;">PhotoGuard: FACADE private demo</h1>
<p style="margin: 0;">A port of the <b>official PhotoGuard demo</b> from
<a href="https://github.com/MadryLab/photoguard" target="_blank" rel="noopener">MadryLab/photoguard</a>
(MIT License, &copy; 2022 Madry Lab). It is hosted privately by FACADE and is not affiliated with or endorsed by the authors.</p>
"""

DESCRIPTION = """
**Raising the Cost of Malicious AI-Powered Image Editing.** Hadi Salman, Alaa Khaddaj, Guillaume Leclerc,
Andrew Ilyas and Aleksander Madry, MIT. [Paper](https://arxiv.org/abs/2302.06588) ·
[Blog post](https://gradientscience.org/photoguard/) · [GitHub](https://github.com/MadryLab/photoguard)

This page runs the paper's **encoder attack** (L∞ ε = 0.12 on the [-1, 1] scale, 200 PGD steps, gray target)
on a 512 × 512 crop. It then edits the original and the immunized image with Stable Diffusion 1.5 inpainting
(DDIM, η = 1), using the same seed for both. Immunization should make the second edit look unrealistic.
See Section 5 of the paper for the intended uses and limitations. Immunization is not a guarantee of protection.

*Private research demo. Only upload photos of yourself, or of people who have agreed to it. Photos are processed on the team's private server and are not kept.*
"""

HOW_TO = """
1. Tap **Choose photo** to pick one from your photo library or take a new one. On a computer it opens a file
   chooser; you can also drop a photo onto the editor or paste one. JPEG, PNG, WebP and HEIC (the iPhone
   default) all work. Photos are turned upright and scaled down to at most 1024 pixels on the long side.
2. **Paint over the parts to KEEP**, e.g. face and background, and leave what the AI may change unpainted.
   The painted part is what gets immunized. The brush menu in the editor's toolbar sets the brush size.
3. Write a prompt that describes the edit, for example *a person at a wedding*.
4. Tap **Immunize and compare**. A run takes about a minute. You get three images: the immunized image (which
   should look essentially identical to the original), the edit of the original, and the edit of the immunized
   image. Everything is done on a 512 × 512 center crop. Change the seed to get different edits.
5. **Save for slides:** *Download comparison (PNG)* is one labeled image with the original crop, the
   immunized image and both edits, plus the prompt and settings. *Download all (ZIP)* adds each image as its
   own PNG and a settings.json. On an iPhone, downloads go to the Files app (Downloads).
"""

NOTICE = """
Stable Diffusion's use restrictions apply
([CreativeML OpenRAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)).
Photos are processed on the team's private server and are not kept: uploads and results are deleted
automatically within about 30 minutes.
"""

PAINT_TIP = """
**Paint over the parts to KEEP**, e.g. face and background, and leave what the AI may change unpainted.
"""

# Every photo goes through Choose photo, so the server decodes it (HEIC too), turns it upright and
# scales it down before the editor shows it. A file dropped on the editor, picked with the editor's
# own upload button or pasted is handed to that button. Otherwise the browser alone would decode it:
# most browsers cannot draw HEIC, and the editor would silently swap the photo for a blank canvas.
# The selectors match EDITOR_ID and the ".heic" in io_utils.UPLOAD_FILE_TYPES.
EDITOR_ID = "pg-editor"
ROUTE_PHOTOS_JS = """
() => {
  const inEditor = (node) => node instanceof Node && !!document.getElementById("pg-editor")?.contains(node);
  const stop = (event) => { event.preventDefault(); event.stopImmediatePropagation(); };
  const choose = (files) => {
    const input = document.querySelector('input[type=file][accept*=".heic"]');
    if (!input || !files || !files.length) return false;
    const transfer = new DataTransfer();
    transfer.items.add(files[0]);
    input.files = transfer.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  };
  for (const type of ["dragenter", "dragover"]) {
    document.addEventListener(type, (event) => { if (inEditor(event.target)) stop(event); }, true);
  }
  document.addEventListener("drop", (event) => {
    if (inEditor(event.target)) { stop(event); choose(event.dataTransfer && event.dataTransfer.files); }
  }, true);
  document.addEventListener("change", (event) => {
    const input = event.target;
    if (input instanceof HTMLInputElement && input.type === "file" && inEditor(input)) {
      stop(event);
      choose(input.files);
      input.value = "";
    }
  }, true);
  document.addEventListener("paste", (event) => {
    if (event.target instanceof Element && event.target.closest("input, textarea")) return;
    const images = [...(event.clipboardData?.files || [])].filter((file) => file.type.startsWith("image/"));
    if (images.length && choose(images)) stop(event);
  }, true);
}
"""

# The browser-side twin of build_ui's cleared(): empty gallery and summary, hidden downloads.
CLEAR_RESULTS_JS = """
() => {
  const hidden = { __type__: "update", value: null, visible: false };
  return [[], "", hidden, hidden];
}
"""

# Midnight Harbor colours, shared with the FACADE app and website. The variable block overrides
# Soft's light theme only: Gradio puts .dark on <body> in dark mode, and it then does not apply.
# Theme variables that Soft derives from others (e.g. --block-title-background-fill) are resolved
# at :root, so each one used on this page is set here directly. The two rules after the block
# apply in both modes: underline spacing and an undimmed notice.
CSS = """
body:not(.dark), body:not(.dark) .gradio-container {
  /* Gradio's own page colours behind the app, seen when the page bounces. The -dark pair is
     what a device in dark mode uses even after ?__theme=light, so it is set here too. */
  --bg: #F3F5F9; --bg-dark: #F3F5F9; --col: #111F34; --col-dark: #111F34;
  --body-background-fill: #F3F5F9; --body-text-color: #111F34; --body-text-color-subdued: #515D6E;
  --color-accent: #244B92; --color-accent-soft: #D6E8FC;
  --button-primary-background-fill: #244B92; --button-primary-background-fill-hover: #050845;
  --button-primary-text-color: #FFFFFF; --button-primary-text-color-hover: #FFFFFF;
  --button-secondary-background-fill: #D6E8FC; --button-secondary-background-fill-hover: #E7EDF5;
  --button-secondary-text-color: #244B92; --button-secondary-text-color-hover: #244B92;
  --block-label-background-fill: #D6E8FC; --block-label-text-color: #244B92;
  --block-title-background-fill: #D6E8FC; --block-title-text-color: #244B92;
  --block-info-text-color: #515D6E; --accordion-text-color: #111F34; --border-color-accent: #D6E8FC;
  --link-text-color: #244B92; --link-text-color-hover: #050845;
  --link-text-color-visited: #244B92; --link-text-color-active: #244B92;
  --slider-color: #244B92; --checkbox-background-color-selected: #244B92; --loader-color: #244B92;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.gradio-container a { text-decoration: underline; text-underline-offset: 2px; }
#pg-results { min-height: 280px; }
.pg-note { font-size: 0.9em; }
@media (max-width: 640px) {
  .pg-title { font-size: 1.6rem !important; }
  /* Phones: one result per row, full width, instead of three thumbnails. */
  #pg-results .grid-wrap { height: auto !important; max-height: none !important; }
  #pg-results .grid-container { --grid-cols: 1 !important; }
  /* Keep the two-line gallery label clear of the gallery's own download icon. */
  #pg-results label { padding-right: 2.5rem; }
}
"""


def parse_args(argv=None):
    """The command-line options; see deploy/systemd/facade-demo.service for the values the server uses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="local stable-diffusion-inpainting snapshot directory (HF layout)")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root-path", default="", type=root_path,
                        help="public URL prefix when served behind a proxy, e.g. /photoguard (default: none)")
    parser.add_argument("--no-warmup", action="store_true", help="skip the one-step GPU warm-up at startup")
    return parser.parse_args(argv)


def root_path(value: str) -> str:
    """'' or a path like /photoguard: one leading slash, no trailing slash, URL-safe characters."""
    if value and not re.fullmatch(r"(/[A-Za-z0-9._~-]+)+", value):
        raise argparse.ArgumentTypeError("use '' or a path like /photoguard (no trailing slash)")
    return value


def configure_environment():
    """Must run before gradio, diffusers or huggingface_hub are imported.

    Those libraries read these variables once, when they are imported.
    Analytics and Hugging Face telemetry are switched off, the Hugging Face
    hub is told to stay offline, and a private (mode 0700) temporary folder is
    chosen for Gradio. Returns the folder; exits if none is writable.
    """
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    candidates = [os.environ.get("GRADIO_TEMP_DIR"), *TEMP_DIR_FALLBACKS]
    # dict.fromkeys drops duplicates while keeping the order; the first folder
    # that can be created (or already exists) and is writable wins.
    for candidate in dict.fromkeys(c for c in candidates if c):
        path = Path(candidate)
        try:
            path.mkdir(mode=0o700, exist_ok=True)  # the parent must already exist
        except OSError:
            continue
        if os.access(path, os.W_OK | os.X_OK):
            os.environ["GRADIO_TEMP_DIR"] = str(path)
            return str(path)
    raise SystemExit("no writable GRADIO_TEMP_DIR among " + ", ".join(dict.fromkeys(c for c in candidates if c)))


def export_root() -> Path:
    """Gradio's cache directory (its own default when GRADIO_TEMP_DIR is unset).

    Files returned from inside it are served in place and registered as Gradio
    temporary files, so delete_cache removes them like any upload.
    """
    root = Path(os.environ.get("GRADIO_TEMP_DIR") or Path(tempfile.gettempdir()) / "gradio").resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def run_settings(core) -> dict:
    """The fixed recipe, recorded in each download's settings.json next to the per-run values.

    With these numbers in the download, a slide made from it says exactly how
    the images were produced.
    """
    return {
        "method": "PhotoGuard encoder attack (Salman et al., arXiv:2302.06588)",
        "pgd_eps": core.PGD_EPS,
        "pgd_step_size": core.PGD_STEP_SIZE,
        "pgd_scale": "[-1, 1]",
        "pgd_target": "uniform gray RGB(128, 128, 128)",
        "image_size": core.SIDE,
        "eta": core.ETA,
        "model": core.MODEL_ID,
        "model_revision": core.MODEL_REVISION,
        "upstream": core.UPSTREAM_REPOSITORY,
        "upstream_commit": core.UPSTREAM_COMMIT,
    }


def build_ui(gr, core, demo):
    """Build the Gradio page and wire its two events: Choose photo and Immunize and compare.

    `gr`, `core` and `demo` are passed in rather than imported, so the tests
    can build the page around fakes. The functions inside run on the server
    each time a visitor triggers an event.
    """
    from photoguard_demo import export, io_utils

    def brush(size=BRUSH_DEFAULT):
        """One fixed-colour brush of the given radius; visitors only choose its size."""
        return gr.Brush(default_size=size, colors=[BRUSH_COLOR], color_mode="fixed")

    def cleared():
        """No results and no downloads, so nothing from an earlier photo or run stays on screen."""
        hidden = gr.update(value=None, visible=False)
        return [], "", hidden, hidden

    def show_photo(image):
        """The editor starts on the new photo with a brush sized for it; old results are cleared."""
        editor_update = gr.update(value=io_utils.editor_value(image), brush=brush(io_utils.brush_radius(image)))
        return (editor_update, *cleared())

    def unchanged(error=None):
        """Keep the photo and results as they are and say why in a toast, not an overlay over the editor.

        Call it inside the `except` block, so an unexpected error is logged with its traceback.
        """
        if error is None:
            message = core.NEED_IMAGE
        elif isinstance(error, io_utils.UploadError):
            message = str(error)
        else:
            LOG.exception("could not read an uploaded photo")
            message = f"Could not read that photo ({type(error).__name__}). Try a JPEG."
        gr.Warning(message)
        # gr.update() with no arguments means "leave this component alone", for
        # the editor and all four result components.
        return (gr.update(),) * 5

    def load_photo(data):
        """Choose photo: decode (HEIC too), turn upright, cap the size, and start the editor on it."""
        started = time.monotonic()
        try:
            if not data:
                return unchanged()
            try:
                image = io_utils.open_upload(data)
            except Exception as error:
                return unchanged(error)
            return show_photo(image)
        finally:
            # Pad a very fast event to MIN_EVENT_SECONDS (see its comment).
            time.sleep(max(0.0, MIN_EVENT_SECONDS - (time.monotonic() - started)))

    def downloads(image, immunized, edit_original, edit_immunized, info):
        """Write the comparison PNG and the ZIP for one run; return updates for the two buttons.

        They go in a fresh folder inside Gradio's temporary directory, so
        delete_cache removes them along with the uploads. If writing fails the
        buttons stay hidden; the results on screen are still shown.
        """
        created = datetime.now(timezone.utc)
        try:
            directory = Path(tempfile.mkdtemp(prefix="photoguard-", dir=export_root()))
            files = export.write_exports(
                directory, original=core.crop_input(image), immunized=immunized, edit_original=edit_original,
                edit_immunized=edit_immunized, prompt=info.get("prompt", ""), info=info, created=created,
                extra_settings=run_settings(core))
        except Exception:
            LOG.exception("could not write the downloads")
            hidden = gr.update(value=None, visible=False)
            return hidden, hidden
        return (gr.update(value=str(files.comparison), visible=True),
                gr.update(value=str(files.archive), visible=True))

    def run(editor_value, prompt, seed, steps, guidance):
        """Immunize and edit. The click has already cleared the previous results (CLEAR_RESULTS_JS).

        There is deliberately no gr.Progress. On every status message, Gradio 6.28's
        frontend re-reads each component that has been an input or output of any
        event, and the image editor answers by uploading the whole photo again
        (background, composite and layers). A per-step progress bar sends hundreds of
        messages per run, which re-uploaded the photo hundreds of times. The status
        timer (elapsed and estimated seconds) still shows while a run works.
        """
        # Three kinds of failure, three messages: a problem with the input is
        # shown as is; running out of GPU memory gets its own advice; anything
        # else shows only the exception's type, and the details go to the log.
        try:
            image, mask = core.editor_to_image_and_mask(io_utils.normalize_editor_value(editor_value))
            immunized, edit_original, edit_immunized, info = demo.compare(
                image, mask, prompt, seed, steps, guidance)
        except (core.InputError, io_utils.UploadError) as error:
            raise gr.Error(str(error), print_exception=False) from None
        except Exception as error:
            if core.is_cuda_oom(error):
                core.release_cuda_memory()
                raise gr.Error("The GPU ran out of memory, possibly because the worker is busy. "
                               "Try again in a minute.") from None
            LOG.exception("compare failed")
            raise gr.Error(f"Processing failed ({type(error).__name__}). Check the facade-demo logs.") from None
        gallery = [(immunized, "Immunized image"),
                   (edit_original, "Edit of the original"),
                   (edit_immunized, "Edit of the immunized image")]
        comparison, archive = downloads(image, immunized, edit_original, edit_immunized, info)
        return gallery, core.format_summary(info), comparison, archive

    # delete_cache=(600, 900): every 600 s, delete temporary files older than
    # 900 s. So an upload or result lives 15 to 25 minutes at most.
    with gr.Blocks(title="PhotoGuard · FACADE private demo", analytics_enabled=False,
                   delete_cache=(600, 900)) as ui:
        gr.HTML(HEADER)
        gr.Markdown(DESCRIPTION)
        with gr.Accordion("How to use", open=False):
            gr.Markdown(HOW_TO)
        with gr.Row():
            # min_width makes the two columns stack on a phone instead of squeezing side by side.
            with gr.Column(min_width=320):
                # type="binary": the server gets the file's raw bytes and decodes
                # them itself (io_utils), HEIC included.
                choose = gr.UploadButton("Choose photo (library or camera)", type="binary",
                                         file_types=io_utils.UPLOAD_FILE_TYPES, variant="secondary", size="lg")
                gr.Markdown(NOTICE, elem_classes=["pg-note"])
                gr.Markdown(PAINT_TIP)
                editor = gr.ImageMask(
                    label="Photo",
                    elem_id=EDITOR_ID,
                    type="pil",
                    image_mode="RGBA",  # layers must keep alpha: that is where the strokes are
                    # "upload" only for its drop zone and button; ROUTE_PHOTOS_JS hands those files
                    # (and pasted images) to Choose photo. No editor.upload listener is needed.
                    sources=("upload",),
                    # no crop or rotate tools: the server does that
                    transforms=(),
                    # a single paint layer, so "painted" is unambiguous
                    layers=False,
                    brush=brush(),
                    # Keep the canvas at the photo's own size. A fixed canvas letterboxes the
                    # photo with opaque fill, and those bars would end up in the 512 crop.
                    fixed_canvas=False,
                    format="png",  # the photo the server sends back to the editor stays lossless
                )
                prompt = gr.Textbox(label="Prompt", placeholder="a photo of a person at a wedding")
                # precision=0 makes the box hold whole numbers; core.parse_seed checks again.
                seed = gr.Number(label="Seed (change it to get different edits)", value=core.DEFAULT_SEED,
                                 precision=0, minimum=0)
                with gr.Accordion("Advanced options", open=False):
                    guidance = gr.Slider(label="Guidance scale", minimum=core.GUIDANCE_MIN,
                                         maximum=core.GUIDANCE_MAX, value=core.GUIDANCE_SCALE, step=0.1)
                    steps = gr.Slider(label="Number of inference steps", minimum=core.STEPS_MIN,
                                      maximum=core.STEPS_MAX, value=core.NUM_INFERENCE_STEPS, step=5)
                button = gr.Button("Immunize and compare", variant="primary")
            with gr.Column(min_width=320):
                results = gr.Gallery(label="Immunized · edit of original · edit of immunized", show_label=True,
                                     elem_id="pg-results", columns=3, height="auto", object_fit="contain",
                                     format="png", preview=False)
                summary = gr.Markdown()
                with gr.Row():
                    comparison = gr.DownloadButton("Download comparison (PNG)", visible=False, variant="primary")
                    archive = gr.DownloadButton("Download all (ZIP)", visible=False)
        # The four result components, in the order cleared() and run() return them.
        outputs = [results, summary, comparison, archive]
        # trigger_mode="multiple": should Gradio still miss the end of an event (MIN_EVENT_SECONDS),
        # the default "once" would keep Choose photo dead until the page is reloaded.
        choose.upload(load_photo, [choose], [editor, *outputs], trigger_mode="multiple", api_visibility="private")
        # Clear the previous results in the browser first, so a run that fails leaves no stale
        # results or dead download links. It runs no Python, so it sends no status messages
        # (each of which would make the editor upload the photo again; see run()).
        button.click(None, None, outputs, js=CLEAR_RESULTS_JS, queue=False, api_visibility="private").then(
            run, [editor, prompt, seed, steps, guidance], outputs,
            api_visibility="private", concurrency_limit="default")
    return ui


def main(argv=None):
    """Start the demo: environment, model, warm-up, then the web page (blocks until stopped)."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    temp_dir = configure_environment()
    LOG.info("Gradio temporary files in %s", temp_dir)

    # Run as a script, sys.path[0] is this directory, so `core`/`app` would shadow any
    # library module of the same name. Import through the package instead.
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != here]
    sys.path.append(str(here.parent))
    from photoguard_demo import core, io_utils

    # Teach Pillow to open HEIC, the iPhone's default photo format.
    heif = io_utils.register_heif_opener()
    LOG.info("HEIC/HEIF uploads: %s", heif or "not supported (install pi-heif)")
    demo = core.PhotoGuardDemo.from_snapshot(args.snapshot)
    if not args.no_warmup:
        demo.warmup()

    import gradio as gr

    ui = build_ui(gr, core, demo)
    # One run at a time on the GPU, at most 8 visitors waiting, and no public
    # API endpoint for scripts to call.
    ui.queue(default_concurrency_limit=1, max_size=8, api_open=False)
    # Everything Gradio could add beyond the page itself is switched off: no
    # share link, no footer, no run history, no monitoring or MCP endpoint, no
    # server-side rendering and no installable web app.
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        root_path=args.root_path,
        share=False,
        inbrowser=False,
        show_error=True,
        footer_links=[],
        run_history=False,
        enable_monitoring=False,
        mcp_server=False,
        ssr_mode=False,
        pwa=False,
        # Choose photo uploads the original phone photo as is; the editor then re-uploads the
        # (at most 1024 px) photo as PNGs. Caddy's request_body for the /photoguard route is
        # the same 64MB (deploy/Caddyfile).
        max_file_size=MAX_FILE_SIZE,
        theme=gr.themes.Soft(),
        css=CSS,
        js=ROUTE_PHOTOS_JS,
    )


if __name__ == "__main__":
    main()
