# PhotoGuard demo (FACADE private port)

This directory ports the official PhotoGuard demo to the FACADE GPU server. The
demo immunizes an image with the paper's **encoder attack**. It then shows
Stable Diffusion 1.5 inpainting edits of the original and of the immunized
image, made with the same seed.

- `core.py`: pipeline loading, upstream preprocessing, the PGD attack and the
  edit comparison. It never imports gradio. Its one entry point is
  `PhotoGuardDemo.compare(image, mask, prompt, seed, steps, guidance)`, which
  returns `(immunized, edit_of_original, edit_of_immunized, info)`.
- `io_utils.py`: turns an uploaded photo into what the editor gets: HEIC/HEIF
  decoded (pi-heif), EXIF orientation applied, RGB, sRGB, at most 1024 px on
  the long side, all metadata (including GPS) dropped. No gradio or torch.
- `export.py`: the downloads for one run: a labeled comparison PNG (original
  crop | immunized | edit of original | edit of immunized, with the prompt,
  settings, date and credit) and a ZIP of the four PNGs, that comparison and a
  `settings.json`. No gradio or torch. Pillow's built-in font draws ASCII and
  a little punctuation only, so text on the image is folded to what it can
  draw ("café — ✓" becomes "cafe - ?"); `settings.json` keeps the exact prompt.
- `app.py`: a thin Gradio 6.28 UI. Run it as
  `python app.py --snapshot /srv/facade/models/sd15-inpaint-8a4288 --host 127.0.0.1 --port 7860`.
  On the server, Caddy serves it at `https://<host>/photoguard/`, so
  `deploy/systemd/facade-demo.service` adds `--root-path /photoguard`.
- Tests: `server/tests/test_photoguard_demo_core.py` and
  `server/tests/test_photoguard_demo_export.py`. They run on CPU with a fake
  pipeline and need neither gradio nor model weights. The HEIC decode test reads
  `server/tests/fixtures/photoguard-orientation6.heic` (595 bytes, encoded on
  macOS like an iPhone photo: pixels stored sideways plus a rotation) and runs
  wherever pi-heif or pillow-heif is installed, as on the server.

## Provenance

- Upstream: [MadryLab/photoguard](https://github.com/MadryLab/photoguard) at
  commit `686bea75c786cb46c88fc396a0cd0ee3d7d28c2e`, files `demo/app.py` and
  `demo/utils.py`. `LICENSE` is the upstream MIT text, byte for byte
  (SHA-256 `1279eae7…dfcd4`), Copyright (c) 2022 Madry Lab. The same text is in
  `licenses/photoguard-MIT.txt` at the repository root.
- Paper: Salman, Khaddaj, Leclerc, Ilyas, Madry, *Raising the Cost of Malicious
  AI-Powered Image Editing*, [arXiv:2302.06588](https://arxiv.org/abs/2302.06588).
- Model: `stable-diffusion-v1-5/stable-diffusion-inpainting` at revision
  `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`, loaded from a local snapshot only
  (`local_files_only=True`, `.bin` weights, fp16 on CUDA). The model is under
  the CreativeML OpenRAIL-M license (`licenses/CreativeML-OpenRAIL-M.txt`),
  which is separate from this code's MIT license. The weights are downloaded on
  the server by `server/models/fetch_models.py`; they are not in this repository.

## Behaviour kept from upstream

- Attack: `pgd()` with ε 0.12 on the [-1, 1] scale, step 0.01 decaying
  linearly toward 0.0001, 200 iterations, and the L2 distance between VAE latent
  means as the loss, under CUDA fp16 autocast. A test checks that its loop
  statements match upstream's in order.
- The four `utils.py` helpers are copied verbatim, which a test also checks.
- Masks: the user paints the region to **keep**, such as a face. Upstream
  inverts that brush mask, so in `mask_image` white is the region the editor
  regenerates. PGD perturbs only the painted region, and the edit region of the
  immunized image stays the original pixels. After each edit,
  `recover_image` pastes the **original** pixels back over the painted region,
  in both arms.
- Edits: 100 DDIM steps by default, guidance 7.5, η 1, seed 1234, and the
  snapshot's DDIM scheduler. Inputs are center-cropped and resized to 512 × 512.
- Seam: upstream's soft-mask blend mixes gray into the 1–2 px anti-aliased
  mask edge. Pixel changes stay within 16/255 everywhere else but can reach
  about 31/255 on that seam.

## Changes from upstream

1. **Target:** upstream downloaded rtings.com's gray-uniformity PNG. That image
   is uniformly RGB(128,128,128) at 512 × 512, so the port generates it locally
   (hash-bound to `facade_photoguard_encoder.TARGET_RGB512_SHA256`). Its latent is
   computed once at startup instead of on every request.
2. **Seeding:** upstream reseeded the global RNG with `torch.manual_seed(seed)`
   on each submit. Here PGD's random start uses a CPU generator seeded with
   `seed`, which gives the same stream as upstream's first CPU draw (a test
   checks this). Each edit arm gets its own freshly seeded generator, so both
   arms start from identical noise, and the global RNG is never touched.
3. **PGD housekeeping:** the random start moves to the input's device instead of
   calling `.cuda()`. Each iteration calls `X_adv.detach()`, which frees the
   autograd graph upstream kept alive across all 200 steps without changing any
   values. The tqdm bar became an optional progress callback, which the UI
   leaves off (see 12).
4. **One button:** upstream needed two submits, one plain and one with
   "Immunize" ticked. Here one click returns the immunized image and both edits.
5. **Safety checker on** (upstream disabled it). Set `FACADE_SAFETY_CHECKER=0`
   to turn it off. A flagged edit comes back black, and the UI says so.
6. **Gradio 3 → 6:** `ImageMask` → Gradio 6 `ImageMask` (an `ImageEditor`).
   Brush strokes are read from the layer alpha, and any painted pixel means
   "keep". The canvas stays at the photo's size (`fixed_canvas=False`), so no
   letterbox bars get into the crop. Other changes: a 3-column PNG gallery,
   `queue(default_concurrency_limit=1, max_size=8)`, `delete_cache=(600, 900)`,
   and theme and css passed to `launch()`.
7. **Private hosting:** binds only to `--host`, with no `share=True`, no
   analytics, no run history, no monitoring or MCP endpoint, and no footer.
   There are no example photos (upstream's showed real people), and webcam input
   is off. Empty or all-painted masks, and settings outside upstream's slider
   ranges, are rejected with a clear message.
8. **Startup warm-up:** one PGD step and a 1-step edit run before the server
   listens, so a broken GPU stack fails at start, not on the first user.
   `--no-warmup` skips it.
9. **Phone photos:** a **Choose photo (library or camera)** button
   (`accept="image/*, .heic, .heif"`, which on iOS and Android offers Photo
   Library, Take Photo and Choose File). Every photo goes through it and then
   through `io_utils`: HEIC decoded on the server, EXIF orientation applied,
   RGB, sRGB, at most 1024 px on the long side, metadata dropped. A photo
   dropped on the editor, picked with the editor's own upload button, or
   pasted is handed to that button by a small script (`ROUTE_PHOTOS_JS`,
   passed to `launch(js=...)`). Without it the browser alone decoded such a
   file, and since browsers other than Safari cannot draw HEIC, the editor
   silently replaced the photo with a blank canvas. The 512 × 512 center crop
   is unchanged. The brush radius is set to about 1/24 of the photo's long
   side so it is finger-sized on a phone; the editor's brush menu changes it.
   A file that cannot be read shows a warning and leaves the current photo and
   results in place. On narrow screens the two columns stack and the results
   show one per row.
10. **Downloads for slides:** after each run, *Download comparison (PNG)* and
    *Download all (ZIP)* (see `export.py`). Files are named
    `photoguard-<UTC timestamp>-seed<seed>`. They are written inside Gradio's
    temporary directory, so `delete_cache` removes them with the uploads. A new
    photo, and every click on *Immunize and compare*, first clears the previous
    results and hides both buttons, so a failed run leaves no stale results or
    dead links behind.
11. **Notice:** the page says that Stable Diffusion's use restrictions
    (CreativeML OpenRAIL-M) apply and that uploads and results are deleted
    automatically (Gradio's `delete_cache=(600, 900)` removes them 15 to 25
    minutes after they are written).
12. **Upload traffic:** on every status message of any event, Gradio 6.28's
    frontend re-reads each component that has been an input or output, and
    the image editor answers by uploading the whole photo again (background,
    composite and layers, about 3 MB at 1024 px). The UI is built to send few
    status messages: no per-step progress bar (a run shows Gradio's elapsed and
    estimated seconds instead; each progress update re-uploaded the photo, and
    a real run sends about 400 of them), results are cleared in the browser
    (`CLEAR_RESULTS_JS`), and the editor has no upload listener. Measured in
    headless Chromium with a fake pipeline and a 12 MP JPEG: choosing a photo
    now uploads it once plus one 1 MB PNG (was 17 MB), and a run uploads about
    20 MB however long it takes (a 20-second run with a progress update every
    0.1 s uploaded 627 MB before). The 1024 px cap keeps each upload small.
13. **Choose photo takes at least 0.3 s** (`MIN_EVENT_SECONDS`), and a new
    choice is never blocked by an earlier one (`trigger_mode="multiple"`).
    Gradio 6.28's frontend sometimes misses the end of an event that finishes
    within a few milliseconds of starting (an unreadable file, a tiny photo):
    the warning never showed, a spinner stayed, and Choose photo stopped
    responding until the page was reloaded.

This is a demonstration port. It makes no claim about protection efficacy.
Only upload photos of yourself, or of people who have agreed to it.
