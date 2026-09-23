# Third-party notices

FACADE includes or builds on the work below. Full license texts are in
[`licenses/`](licenses/) or next to the code they cover.

## Code included in this repository

| What | Where in FACADE | License | Text |
|---|---|---|---|
| [PhotoGuard](https://github.com/MadryLab/photoguard) by Madry Lab, commit `686bea75c786cb46c88fc396a0cd0ee3d7d28c2e`: the demo app and its image helpers (`demo/app.py`, `demo/utils.py`), and the encoder and diffusion attacks (`demo/app.py`, `notebooks/demo_complex_attack_inpainting.ipynb`) | `server/photoguard_demo/`, `server/worker/facade_photoguard_encoder.py`, `server/worker/facade_photoguard_diffusion.py` | MIT, Copyright (c) 2022 Madry Lab | [`licenses/photoguard-MIT.txt`](licenses/photoguard-MIT.txt), also `server/photoguard_demo/LICENSE` |
| [shadcn/ui](https://ui.shadcn.com) components | `web/components/ui/` | MIT | `web/components/ui/LICENSE.md` |
| Resampling coefficient and rounding conventions of [Pillow](https://python-pillow.org) 12.3.0 `src/libImaging/Resample.c` | `app/ios/FACADE/Native/NativePixels.c` | MIT-CMU (HPND) | [`licenses/Pillow-HPND.txt`](licenses/Pillow-HPND.txt) |
| The float summation order of [PyTorch](https://github.com/pytorch/pytorch) `SumKernel.cpp` at commit `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7` | `app/ios/FACADE/Native/NativeNetwork.cpp` | BSD-3-Clause | [`licenses/PyTorch-BSD-3-Clause.txt`](licenses/PyTorch-BSD-3-Clause.txt) |
| The ArcFace 112-pixel five-point face alignment template (constants only) from [InsightFace](https://github.com/deepinsight/insightface) | `app/ios/FACADE/App/NativeVision.swift`, `app/ios/FACADE/App/FACADEFaceOffRenderer.swift` | MIT, Copyright (c) 2022 Jiankang Deng and Jia Guo | [`licenses/InsightFace-MIT.txt`](licenses/InsightFace-MIT.txt) |

PhotoGuard paper: Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas
and Aleksander Madry, *Raising the Cost of Malicious AI-Powered Image Editing*,
[arXiv:2302.06588](https://arxiv.org/abs/2302.06588). FACADE's PhotoGuard demo
is a port hosted by FACADE; it is not affiliated with or endorsed by the
authors.

The InsightFace MIT license covers its code only. InsightFace's pretrained
models are for non-commercial research only; none are in this repository.

## Libraries built into the iOS app (downloaded at build time, not in this repository)

`app/ios/scripts/build_native_deps.sh` downloads these source releases, checks
their sha256, and builds static libraries that are linked into the app. Anyone
who distributes the app must ship these notices with it.

| What | License | Text |
|---|---|---|
| [OpenCV](https://opencv.org) 5.0.0 | Apache-2.0 | [`licenses/OpenCV-Apache-2.0.txt`](licenses/OpenCV-Apache-2.0.txt), [`licenses/OpenCV-COPYRIGHT.txt`](licenses/OpenCV-COPYRIGHT.txt) |
| [KleidiCV](https://gitlab.arm.com/kleidi/kleidicv) 26.03 (OpenCV's Arm acceleration layer) | Apache-2.0 | [`licenses/KleidiCV-Apache-2.0.txt`](licenses/KleidiCV-Apache-2.0.txt) |
| [SLEEF](https://sleef.org) at commit `5a1d179df9cf652951b59010a2d2075372d67f68` | Boost Software License 1.0 | [`licenses/SLEEF-BSL-1.0.txt`](licenses/SLEEF-BSL-1.0.txt) |
| [libjpeg-turbo](https://libjpeg-turbo.org) 3.1.4.1 | IJG License, BSD-3-Clause and zlib License | [`licenses/libjpeg-turbo-LICENSE.md`](licenses/libjpeg-turbo-LICENSE.md), [`licenses/libjpeg-turbo-README.ijg`](licenses/libjpeg-turbo-README.ijg) |
| [zlib](https://zlib.net), as bundled in OpenCV's `3rdparty/` | zlib License | [`licenses/zlib.txt`](licenses/zlib.txt) |

This software is based in part on the work of the Independent JPEG Group.

## Model weights downloaded at run time (not in this repository)

| What | Used by | License | Text |
|---|---|---|---|
| [`stable-diffusion-v1-5/stable-diffusion-inpainting`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) at revision `8a4288a76071f7280aedbdb3253bdb9e9d5d84bb` (Stable Diffusion 1.5 inpainting: VAE, UNet, CLIP text encoder, tokenizer, scheduler and safety checker) | the worker (VAE only) and the PhotoGuard demo (full pipeline) | CreativeML OpenRAIL-M, Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors | [`licenses/CreativeML-OpenRAIL-M.txt`](licenses/CreativeML-OpenRAIL-M.txt) |

The server downloads these weights from Hugging Face with
`server/models/fetch_models.py` and checks each file against
`server/models/MODEL_SHA256SUMS`. FACADE does not redistribute them or any
model derived from them. The license's use-based restrictions (paragraph 5 and
Attachment A) apply to everyone who uses the hosted demo or worker, and FACADE
must require its users to follow them. The website's upload consent and its
link to the demo both ask users to agree to them.

## Software installed on the server (not in this repository)

`deploy/bootstrap.sh` installs these on the server from their official
sources. They are not redistributed here.

| What | License |
|---|---|
| [PyTorch](https://pytorch.org) 2.8.0 and torchvision 0.23.0 | BSD-3-Clause |
| [Diffusers](https://github.com/huggingface/diffusers) 0.40.0, [Transformers](https://github.com/huggingface/transformers) 5.16.1, [huggingface_hub](https://github.com/huggingface/huggingface_hub) 1.29.0 | Apache-2.0 |
| [Gradio](https://github.com/gradio-app/gradio) 6.28.0 | Apache-2.0 |
| [NumPy](https://numpy.org) 2.1.2 | BSD-3-Clause |
| [Pillow](https://python-pillow.org) 12.3.0 | MIT-CMU (HPND) |
| [pi-heif](https://github.com/bigcat88/pillow_heif) 1.4.0 (the decode-only build of pillow-heif; its wheel bundles [libheif](https://github.com/strukturag/libheif) and [libde265](https://github.com/strukturag/libde265)) | BSD-3-Clause; the bundled libheif and libde265 are LGPL-3.0 |
| the other Python packages pinned in `server/requirements-gpu.lock.txt` | each under its own license |
| [Node.js](https://nodejs.org) 22 LTS | MIT and bundled third-party licenses |
| the website's npm dependencies, from `web/package-lock.json` | each under its own license |
| [Caddy](https://caddyserver.com) | Apache-2.0 |
