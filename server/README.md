# server/

The Python code that runs on the FACADE GPU server. `deploy/` installs and runs
it; `deploy/README.md` explains the server itself.

| Path | What it is |
|---|---|
| `worker/facade_photoguard_service.py` | **the worker program** (`facade-worker`): the `/v1` API the website calls, on 127.0.0.1:8000 |
| `worker/facade_photoguard_worker.py` | the `/v1` HTTP API: bearer check, PNG checks, one job at a time, per-job tokens |
| `worker/facade_photoguard_private_session.py` | the job session layer: the start-up self-test and the output checks every job gets |
| `worker/facade_photoguard_encoder.py` | **the engine**: PhotoGuard's encoder attack on the Stable Diffusion VAE (200 PGD steps) |
| `worker/facade_photoguard_diffusion.py` | PhotoGuard's full diffusion attack (not used by the server today) and PNG helpers the encoder shares |
| `photoguard_demo/` | **the demo program** (`facade-demo`): a Gradio port of the official PhotoGuard demo with phone uploads (HEIC too) and downloads for slides, see its README |
| `models/fetch_models.py`, `models/MODEL_SHA256SUMS` | downloads the pinned Stable Diffusion 1.5 inpainting weights and checks every sha256 |
| `health.py` | the check `facade-health` runs every minute |
| `requirements-gpu.lock.txt` | exact versions of the Python packages on the server (torch and gradio are pinned in `deploy/bootstrap.sh`) |
| `tests/` | pytest tests; no GPU, model weights, gradio or network needed |

The worker modules import each other by file name, so run them from
`server/worker/` as scripts (`python server/worker/facade_photoguard_service.py --help`).

## Running the tests

With Python 3.12 and `torch`, `torchvision`, `numpy`, `pillow`, `diffusers`,
`transformers` and `pytest` installed (the versions in
`requirements-gpu.lock.txt` plus torch 2.8.0 and torchvision 0.23.0):

```bash
python -m pytest -q server/tests
```

A few checks run only when extra things exist on your computer, and are skipped
otherwise:

- the pinned model snapshot in your Hugging Face cache
  (`~/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting`)
- a checkout of [MadryLab/photoguard](https://github.com/MadryLab/photoguard) at
  commit `686bea75c786`, named by `PHOTOGUARD_UPSTREAM=/path/to/photoguard`, to
  check that the ported code still matches upstream line for line
- `shellcheck`, for `deploy/bootstrap.sh`

## Trying the engine on your own computer

On a Mac with Apple silicon (`--device mps`) or an NVIDIA GPU:

```bash
python server/models/fetch_models.py --dest ~/facade-models/sd15-inpaint-8a4288    # about 5.5 GB
python server/worker/facade_photoguard_encoder.py --snapshot ~/facade-models/sd15-inpaint-8a4288 \
    --input photo.png --mask mask.png --output protected.png --device mps
```

Both images must be 512 x 512 PNGs. The mask is white where an editor may
repaint (left untouched) and black on the part that gets protected. Use only
synthetic or fictional images. This measures nothing about how well the
protection works; it only runs it.
