"""FACADE public sample mode: PhotoGuard on the team's sample faces, no password.

The full demo (app.py, https://<host>/photoguard/) accepts any photo and any
prompt, so it stays behind the team password: an open upload box plus
free-text Stable Diffusion inpainting would be a free tool for AI-editing
anyone's photo, which is the misuse FACADE exists to fight. This page is the
public alternative, at https://<host>/photoguard-sample/:

* Only the team's own sample faces (AI-generated pictures of fictional people,
  in samples/) can be used, each with a fixed mask of the part to keep (face
  and hair). There is no upload component and no text box for a prompt: the
  visitor picks a face, one of the PROMPTS and a seed (a whole number).
* The code path is the full demo's: the same PhotoGuardDemo object, model,
  settings (100 steps, guidance 7.5) and Stable Diffusion safety checker.
* Each visitor gets quota.RUNS_PER_VISITOR_PER_DAY runs per day and the page
  quota.RUNS_PER_DAY in total, counted in a file so a restart does not reset
  them (quota.py explains what is stored).
* The GPU runs one job at a time (about 47 s each on the server's L4). The page
  says where a run is in line and roughly how long it will take, and turns new
  runs away, uncounted, once MAX_WAITING are already in line.

How a run works (two events, see build_sample_ui):

1. "Protect and compare" first calls join(), in a lane of Gradio's queue of
   its own with no limit, so it answers at once instead of waiting for the GPU.
   (It is queued at all because Gradio 6.28 shows no error message for an event
   outside the queue.) join() checks the choices, the line and the visitor's
   allowance, counts the run, and returns a random token that Gradio keeps on
   the server (gr.State; the browser never sees it). A refusal is shown as a
   message and nothing is counted.
2. Only if join() succeeded, run() waits in Gradio's queue for the GPU. It
   accepts nothing from the browser but the token, so it can only ever run
   what join() checked. If the server fails, the run is given back.

Both pages run in one process (one copy of the model) and so share Gradio's
temporary folder. Gradio serves a file from it only by its full path, which
contains a hash under a secret seed, so this page cannot list or guess the
full demo's uploads; it can only fetch the results it was itself given.

Like app.py, this module never imports gradio at the top: gradio (`gr`),
core and the loaded demo are passed in, so the tests use fakes.
"""
# No `from __future__ import annotations` here: Gradio finds the `request: gr.Request`
# parameters by evaluating their type hints, and `gr` is a local name it could not resolve.
import hashlib
import logging
import math
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from PIL import Image, ImageChops, ImageFilter

from photoguard_demo import quota as quota_module

LOG = logging.getLogger("photoguard_demo.sample")

SAMPLES_DIR = Path(__file__).resolve().with_name("samples")


@dataclass(frozen=True)
class SampleSpec:
    """One sample face: its photo and keep mask (white = face and hair, kept and protected), pinned by SHA-256."""
    id: str
    label: str
    image: str
    image_sha256: str
    keep: str
    keep_sha256: str


# The team's sample faces (see samples/README.md for where they come from).
SAMPLES = (
    SampleSpec("sample-1", "Sample face 1", "sample-1.jpg",
               "e057d2e541b6b37d98d6810e017531841829222a32e7ac37695c0c758e3e509d",
               "sample-1-keep.png", "10e8be69a3d9e55c34b030642f46863f98eeedd8cfda54cd047393f0e4f2aa0a"),
    SampleSpec("sample-2", "Sample face 2", "sample-2.jpg",
               "be4c51c58bbde0ad27bd6187a8de5afd44ccd786ae6f0fd7db31a58b0b6d5194",
               "sample-2-keep.png", "ed0eebef6ab2b521fb52dffccd9e5bc0d1785fdf2f0ffc9f02547ba9dab0127b"),
)

# The only edits the public page offers: (id, prompt). The first is the full demo's
# example prompt, the one used for the website's example. Each changes the background
# and clothing, which is what the masks leave to the editor.
PROMPTS = (
    ("wedding", "a photo of a person at a wedding"),
    ("beach", "a person on a beach at sunset"),
    ("snow", "a person in a red winter coat in the snow"),
    ("office", "a person in a business suit in an office"),
    ("city", "a person on a busy city street at night"),
)
PROMPT_TEXT = dict(PROMPTS)

# Seconds one run takes on the server's L4 (about 47 s measured), rounded up, for wait estimates.
RUN_SECONDS = 50
# Runs that may be in line (waiting or running) before new ones are turned away: about 8 minutes.
MAX_WAITING = 10
# A counted run that has not finished this long after it was counted (its visitor
# left, say) stops holding a place in line. It stays counted.
LINE_TIMEOUT_SECONDS = 20 * 60
# How often a background timer starts the new day's counts after midnight UTC.
ROLL_OVER_SECONDS = 300
# Gradio's per-file upload limit for this page. It has no upload component, Caddy
# refuses its upload endpoint, and this keeps anything that slips past tiny.
NO_UPLOADS = "1kb"
# The queue lane join() runs in, apart from the GPU's (see the module docstring).
JOIN_LANE = "sample-join"
# Outline drawn around the kept region in the preview: the full demo's brush colour.
OUTLINE_RGB = (255, 45, 85)

# Messages shown to visitors word for word.
BUSY = ("The server is busy: {waiting} runs are already in line. Try again in a few minutes. "
        "This try was not counted.")
VISITOR_LIMIT = ("You have used today's {limit} runs. They come back at midnight UTC. "
                 "You can also protect your own photos on your iPhone with the FACADE app.")
SITE_LIMIT = "This demo has reached today's limit for everyone. Try again after midnight UTC."
PICK_SAMPLE = "Pick one of the sample faces."
PICK_PROMPT = "Pick one of the edits."
NOT_STARTED = "Press Protect and compare to start a run."
OUT_OF_MEMORY = ("The GPU ran out of memory, possibly because it is busy. "
                 "This run was not counted; try again in a minute.")
FAILED = "Processing failed ({kind}). This run was not counted; try again."


# --- the sample faces ----------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """A loaded sample face: the photo, the brush mask compare() expects, and a preview."""
    spec: SampleSpec
    image: Image.Image  # RGB
    keep: Image.Image  # L, white = painted = KEEP (upstream's brush mask)
    preview: Image.Image  # the photo with the kept region outlined


def outline(image: Image.Image, keep: Image.Image, color=OUTLINE_RGB) -> Image.Image:
    """The photo with a line drawn along the edge of the kept region."""
    size = max(3, (min(image.size) // 128) | 1)  # an odd filter size: 5 px for a 512-640 px photo
    edge = ImageChops.subtract(keep.filter(ImageFilter.MaxFilter(size)), keep.filter(ImageFilter.MinFilter(size)))
    framed = image.copy()
    framed.paste(Image.new("RGB", image.size, color), mask=edge)
    return framed


def load_samples(directory: Path = SAMPLES_DIR, specs=SAMPLES) -> dict[str, Sample]:
    """Read every sample face and its mask, refusing any file whose SHA-256 is not the pinned one."""
    samples = {}
    for spec in specs:
        images = []
        for name, expected in ((spec.image, spec.image_sha256), (spec.keep, spec.keep_sha256)):
            data = (directory / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"{name} does not match its pinned SHA-256")
            with Image.open(directory / name) as opened:
                opened.load()
                images.append(opened.copy())
        image, keep = images[0].convert("RGB"), images[1].convert("L")
        if image.size != keep.size:
            raise RuntimeError(f"{spec.keep} is not the size of {spec.image}")
        samples[spec.id] = Sample(spec, image, keep, outline(image, keep))
    return samples


# --- the line of counted runs ----------------------------------------------------------


@dataclass
class Reservation:
    """One counted run: what join() checked, and the quota ticket to give back if the server fails."""
    token: str
    ticket: quota_module.Ticket
    sample_id: str
    prompt: str
    seed: int
    created: float
    started: bool = False


@dataclass(frozen=True)
class Joined:
    """join()'s answer: a reservation, or the reason it was refused ("busy", "visitor" or "site")."""
    reservation: Reservation | None
    reason: str | None
    ahead: int  # runs in line in front of this one
    left: int  # runs the visitor has left today, after this one


@dataclass
class Line:
    """The runs that have been counted and have not finished yet, in the order they were counted.

    Thread-safe. `quota` counts the runs per visitor and per day; the line
    itself lives in memory only (a restart empties Gradio's queue anyway).
    """
    quota: quota_module.DailyQuota
    max_waiting: int = MAX_WAITING
    timeout: float = LINE_TIMEOUT_SECONDS
    clock: Callable[[], float] = time.monotonic
    _active: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _sweep_locked(self) -> None:
        now = self.clock()
        for token in [t for t, r in self._active.items() if now - r.created >= self.timeout]:
            del self._active[token]

    def waiting(self) -> int:
        """Runs in line now, the running one included."""
        with self._lock:
            self._sweep_locked()
            return len(self._active)

    def join(self, address: str, sample_id: str, prompt: str, seed: int) -> Joined:
        """Count a run for this visitor and put it in line, or refuse without counting it."""
        with self._lock:
            self._sweep_locked()
            ahead = len(self._active)
            if ahead >= self.max_waiting:
                return Joined(None, "busy", ahead, self.quota.left(address))
            decision = self.quota.reserve(address)
            if not decision.ok:
                return Joined(None, decision.reason, ahead, decision.left)
            reservation = Reservation(secrets.token_urlsafe(16), decision.ticket, sample_id, prompt, seed,
                                      self.clock())
            self._active[reservation.token] = reservation
            return Joined(reservation, None, ahead, decision.left)

    def start(self, token) -> Reservation | None:
        """The reservation for `token` if it is in line and has not started; marks it started."""
        with self._lock:
            self._sweep_locked()
            reservation = self._active.get(token) if isinstance(token, str) else None
            if reservation is None or reservation.started:
                return None
            reservation.started = True
            return reservation

    def finish(self, reservation: Reservation, *, failed: bool) -> None:
        """Take a run out of the line; if the server failed, give the run back to the visitor."""
        with self._lock:
            self._active.pop(reservation.token, None)
        if failed:
            self.quota.release(reservation.ticket)


def wait_text(runs: int) -> str:
    """'about a minute' or 'about N minutes' for `runs` runs of RUN_SECONDS each."""
    seconds = runs * RUN_SECONDS
    return "about a minute" if seconds <= 90 else f"about {math.ceil(seconds / 60)} minutes"


def runs_left_text(left: int, limit: int) -> str:
    return f"You have {left} of {limit} runs left today."


def line_text(waiting: int) -> str:
    if waiting == 0:
        return "The server is free right now."
    return f"{waiting} {'run is' if waiting == 1 else 'runs are'} in line now; each takes about a minute."


def joined_text(joined: Joined, limit: int) -> str:
    """What the page says right after a run is counted: its place in line and the expected wait."""
    if joined.ahead == 0:
        where = f"Your run is starting. Results in {wait_text(1)}."
    else:
        ahead = f"{joined.ahead} {'run' if joined.ahead == 1 else 'runs'} ahead of you"
        where = f"Your run is in line, with {ahead}. Results in {wait_text(joined.ahead + 1)}."
    return f"{where} {runs_left_text(joined.left, limit)}"


# --- the page ---------------------------------------------------------------------------

HEADER = """
<h1 class="pg-title" style="font-weight: 800; margin: 6px 0 4px;">PhotoGuard: try it on our sample faces</h1>
<p style="margin: 0;">A public, sample-only version of FACADE's hosted copy of the <b>official PhotoGuard demo</b> from
<a href="https://github.com/MadryLab/photoguard" target="_blank" rel="noopener">MadryLab/photoguard</a>
(MIT License, &copy; 2022 Madry Lab). It is not affiliated with or endorsed by the authors.</p>
"""

DESCRIPTION = """
PhotoGuard adds small changes to a photo so that AI editors like Stable Diffusion work badly on it.
Pick one of our sample faces and an AI edit. The server protects the photo (the encoder method: 200 steps, every
color value changes by a small, bounded amount), then runs the same Stable Diffusion 1.5 edit, with the same seed, on
the original and on the protected copy. If the protection works, the second edit looks clearly worse.
Results vary with the photo, the edit and the seed, and protection is never guaranteed.

Both faces are **AI-generated pictures of fictional people**. To keep this page from being used to edit real
people's photos, it accepts no uploads and no typed prompts.
"""

PREVIEW_NOTE = """
PhotoGuard protects and keeps the part inside the pink line (face and hair). The AI may repaint everything else.
"""

# Added to app.CSS: the Midnight Harbor blue for the selected choice in each list
# (Soft's own is purple), in light mode only, like the rest of app.CSS.
CSS = """
body:not(.dark), body:not(.dark) .gradio-container {
  --checkbox-label-background-fill-selected: #244B92; --checkbox-label-text-color-selected: #FFFFFF;
  --checkbox-border-color-selected: #244B92; --checkbox-label-border-color-selected: #244B92;
}
"""

NOTICE = """
The server runs one job at a time, and a run takes about a minute. Each visitor can start
{per_visitor} runs a day (they come back at midnight UTC). A run counts when it starts; if the server fails, it is
given back. Results are deleted from the server within about 30 minutes. Stable Diffusion's use restrictions apply
([CreativeML OpenRAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)).
"""


def build_sample_ui(gr, core, demo, line: Line, samples: Mapping[str, Sample], *,
                    export_root: Callable[[], Path], settings: Mapping):
    """Build the public page and wire its events. `gr`, `core` and `demo` are passed in (fakes in the tests)."""
    from photoguard_demo import export

    first = next(iter(samples.values()))
    limit = line.quota.per_visitor

    def address(request) -> str:
        """The visitor's address: X-Forwarded-For from the local Caddy only (see quota.client_address)."""
        client = getattr(request, "client", None)
        headers = getattr(request, "headers", None) or {}
        return quota_module.client_address(getattr(client, "host", None), headers.get("x-forwarded-for"))

    def show_status(request: gr.Request):
        """The line under the button when the page opens and after each run: runs left and the line now."""
        return f"{runs_left_text(line.quota.left(address(request)), limit)} {line_text(line.waiting())}"

    def show_preview(sample_id):
        sample = samples.get(sample_id)
        return gr.update() if sample is None else sample.preview

    def cleared():
        hidden = gr.update(value=None, visible=False)
        return [], "", hidden, hidden

    def join(sample_id, prompt_id, seed, request: gr.Request):
        """Step 1, at once: check the choices, count the run and put it in line, or refuse."""
        if sample_id not in samples:
            raise gr.Error(PICK_SAMPLE, print_exception=False)
        if prompt_id not in PROMPT_TEXT:
            raise gr.Error(PICK_PROMPT, print_exception=False)
        try:
            seed = core.parse_seed(seed)
        except core.InputError as error:
            raise gr.Error(str(error), print_exception=False) from None
        joined = line.join(address(request), sample_id, PROMPT_TEXT[prompt_id], seed)
        if joined.reservation is None:
            message = {"busy": BUSY.format(waiting=joined.ahead), "visitor": VISITOR_LIMIT.format(limit=limit),
                       "site": SITE_LIMIT}[joined.reason]
            raise gr.Error(message, print_exception=False)
        return (joined.reservation.token, joined_text(joined, limit), *cleared())

    def downloads(sample, immunized, edit_original, edit_immunized, info):
        """The full demo's two downloads, in a fresh folder that delete_cache later removes."""
        created = datetime.now(timezone.utc)
        try:
            directory = Path(tempfile.mkdtemp(prefix="photoguard-sample-", dir=export_root()))
            files = export.write_exports(
                directory, original=core.crop_input(sample.image), immunized=immunized,
                edit_original=edit_original, edit_immunized=edit_immunized, prompt=info.get("prompt", ""),
                info=info, created=created, extra_settings={**settings, "sample": sample.spec.id})
        except Exception:
            LOG.exception("could not write the downloads")
            hidden = gr.update(value=None, visible=False)
            return hidden, hidden
        return (gr.update(value=str(files.comparison), visible=True),
                gr.update(value=str(files.archive), visible=True))

    def run(token):
        """Step 2, in the queue: run exactly what join() counted. The browser can send nothing else."""
        reservation = line.start(token)
        if reservation is None:
            raise gr.Error(NOT_STARTED, print_exception=False)
        sample = samples[reservation.sample_id]
        failed = True
        try:
            immunized, edit_original, edit_immunized, info = demo.compare(
                sample.image, sample.keep, reservation.prompt, reservation.seed,
                core.NUM_INFERENCE_STEPS, core.GUIDANCE_SCALE)
            failed = False
        except Exception as error:
            if core.is_cuda_oom(error):
                core.release_cuda_memory()
                raise gr.Error(OUT_OF_MEMORY, print_exception=False) from None
            LOG.exception("sample run failed")
            raise gr.Error(FAILED.format(kind=type(error).__name__), print_exception=False) from None
        finally:
            line.finish(reservation, failed=failed)
        gallery = [(immunized, "Protected image"),
                   (edit_original, "AI edit of the original"),
                   (edit_immunized, "AI edit of the protected image")]
        summary = f"{sample.spec.label} · edit: “{reservation.prompt}”\n\n{core.format_summary(info)}"
        return (gallery, summary, *downloads(sample, immunized, edit_original, edit_immunized, info))

    # delete_cache=(600, 900): as in the full demo, results live 15 to 25 minutes at most.
    with gr.Blocks(title="PhotoGuard · FACADE sample demo", analytics_enabled=False,
                   delete_cache=(600, 900)) as ui:
        gr.HTML(HEADER)
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column(min_width=320):
                face = gr.Radio(choices=[(s.spec.label, s.spec.id) for s in samples.values()],
                                  value=first.spec.id, label="1. Pick a sample face")
                gr.Markdown(PREVIEW_NOTE, elem_classes=["pg-note"])
                # interactive=False: a picture to look at, never an upload box. No "share" button
                # (it posts to Hugging Face); fullscreen only.
                preview = gr.Image(value=first.preview, label="Kept and protected: inside the pink line",
                                   interactive=False, format="png", buttons=["fullscreen"])
                prompt = gr.Radio(choices=[(text, key) for key, text in PROMPTS], value=PROMPTS[0][0],
                                  label="2. Pick an AI edit")
                # precision=0 makes the box hold whole numbers; core.parse_seed checks again.
                seed = gr.Number(label="3. Seed (change it to get different edits)", value=core.DEFAULT_SEED,
                                 precision=0, minimum=0)
                button = gr.Button("Protect and compare", variant="primary")
                status = gr.Markdown(elem_id="pg-status")
                gr.Markdown(NOTICE.format(per_visitor=limit), elem_classes=["pg-note"])
            with gr.Column(min_width=320):
                results = gr.Gallery(label="Protected · edit of original · edit of protected", show_label=True,
                                     elem_id="pg-results", columns=3, height="auto", object_fit="contain",
                                     format="png", preview=False, interactive=False)
                summary = gr.Markdown()
                with gr.Row():
                    comparison = gr.DownloadButton("Download comparison (PNG)", visible=False, variant="primary")
                    archive = gr.DownloadButton("Download all (ZIP)", visible=False)
        token = gr.State(None)
        outputs = [results, summary, comparison, archive]
        ui.load(show_status, None, [status], queue=False, api_visibility="private")
        face.change(show_preview, [face], [preview], queue=False, api_visibility="private")
        # Step 1 answers at once (its own queue lane, no limit); step 2 runs only if step 1
        # succeeded, and shows Gradio's own queue position and timer on the gallery while it
        # waits for the GPU.
        joined = button.click(join, [face, prompt, seed], [token, status, *outputs], api_visibility="private",
                              concurrency_limit=None, concurrency_id=JOIN_LANE, show_progress="hidden")
        joined.success(run, [token], outputs, api_visibility="private", concurrency_limit="default",
                       show_progress_on=[results]).then(
            show_status, None, [status], queue=False, api_visibility="private")
    return ui


def start(gr, core, demo, *, host: str, port: int, root_path: str, quota_file: Path | None,
          per_visitor: int, per_day: int, css: str, export_root: Callable[[], Path], settings: Mapping):
    """Build and serve the public page on host:port in background threads; returns the page and its line."""
    samples = load_samples()
    line = Line(quota_module.DailyQuota(quota_file, per_visitor=per_visitor, per_day=per_day))
    LOG.info("public sample mode on %s:%s%s: %d faces, %d edits, %d runs per visitor, %d per day, counts in %s",
             host, port, root_path or "/", len(samples), len(PROMPTS), per_visitor, per_day,
             quota_file or "memory only")

    def roll_over():
        while True:
            time.sleep(ROLL_OVER_SECONDS)
            try:
                line.quota.roll_over()
            except Exception:
                LOG.exception("could not start the new day's counts")

    threading.Thread(target=roll_over, name="sample-roll-over", daemon=True).start()
    ui = build_sample_ui(gr, core, demo, line, samples, export_root=export_root, settings=settings)
    # One run at a time, like the full demo; core.PhotoGuardDemo.gpu_lock makes the two pages take
    # turns on the GPU. Gradio's own queue limit is never reached: join() turns runs away first.
    ui.queue(default_concurrency_limit=1, max_size=2 * MAX_WAITING, api_open=False)
    # The same switched-off extras as the full demo (app.py), plus: no uploads, and it returns
    # at once (prevent_thread_lock) so the full demo can start next in the same process.
    # show_error=True: the page shows the gr.Error messages above (a limit reached, a busy
    # server); run() turns every other failure into one of those messages.
    ui.launch(
        server_name=host,
        server_port=port,
        root_path=root_path,
        share=False,
        inbrowser=False,
        show_error=True,
        quiet=True,
        prevent_thread_lock=True,
        footer_links=[],
        run_history=False,
        enable_monitoring=False,
        mcp_server=False,
        ssr_mode=False,
        pwa=False,
        max_file_size=NO_UPLOADS,
        theme=gr.themes.Soft(),
        css=css + CSS,
    )
    return ui, line
