# Models (not in git)

The FACADE app runs its models on the phone, but **no model files are stored in
git**. Some are derived from models whose licences allow research use only, and
all of them are large. The project owner keeps the verified copies and shares
them with teammates through a private release of a separate repository,
`sadiehertzig/facade-models` (see below).
Please do not commit them, upload them anywhere, or send them on.

The app builds and runs without any of these files. Each feature that needs a
missing model tells you so in the app ("FACADE models are not installed",
"Mini PhotoGuard · unavailable in this build", or a Face-Off error), instead of
doing anything else.

## Download them

If you have been invited to the private `sadiehertzig/facade-models`
repository, get them from its
[`models-v1` release](https://github.com/sadiehertzig/facade-models/releases/tag/models-v1)
(1.15 GB). From this repository's root:

```bash
gh release download models-v1 -R sadiehertzig/facade-models -p FACADE-Models-v1.zip -D ~/Downloads
unzip -oq ~/Downloads/FACADE-Models-v1.zip -d app/ios/Models
(cd app/ios/Models && shasum -a 256 -c --quiet MODEL_SHA256SUMS && echo "models OK")
```

This puts every folder in the table below in place, plus `MODEL_SHA256SUMS`
(git ignores both). If the check prints a `FAILED` line, download again.

## Where the files go

Put each folder directly inside this `Models/` folder, with exactly the name
shown. During every build, the Xcode build phase "Copy private models"
(`scripts/copy_models.sh`) copies whichever of them exist into the app. A folder
can also be a symbolic link to a copy stored somewhere else on your Mac.

| Folder in `Models/` | Needed for | Contents |
| --- | --- | --- |
| `NativeModels/` | FACADE protection | `inswapper/graph.json` + `weights.f32`, `simswap/graph.json` + `weights.f32`, `detector.mlpackage`, `parser.mlpackage`, `emap.f32` |
| `FACADEModelHandoff/` | FACADE protection | `manifest.json` plus its `Models/`, `Fixtures/`, `Docs/` and `Licenses/` folders, unchanged |
| `FaceOffModels/` | the face-swap comparison | `renderer.mlmodelc`, `recognizer-float32.mlmodelc`, `identity-map-float32.bin` |
| `FaceOffAssets/` or `FaceOffTarget/` | the face-swap comparison | the fictional target face: `FaceOffAssets/synthetic_blonde_woman_faceoff_target.png` or `FaceOffTarget/target.png` |
| `MiniPhotoGuardPayload/` | Mini PhotoGuard | `manifest.json` and its four parity fixtures |
| `photoguard_encoder_gradient_256_fp32.mlpackage` | Mini PhotoGuard | the Core ML encoder named in that manifest |
| `SampleFace/` | "Try a fictional sample" | `sample.png`: any photo you have the right to use, for example a fictional face |

The engines check what they load. FACADE's model handoff manifest, the
Face-Off identity map and target face, and the Mini PhotoGuard manifest are
pinned to exact SHA-256 values in the Swift code, so a different or re-exported
copy is refused rather than silently used.

## What these models are

- **FACADE** (`NativeModels/`, `FACADEModelHandoff/`) uses face detection,
  face parsing and face-recognition models, some derived from InsightFace and
  SimSwap models. Those licences restrict use to non-commercial research.
- **Face-Off** (`FaceOffModels/`) is an on-device face swapper derived from
  InSwapper. It only exists to show what a face-swap tool does to a protected
  photo. Its similarity scores come from the swapper's own recognizer; they are
  not proof that protection works.
- **Mini PhotoGuard** is a 256 × 256 encoder exported from the Stable Diffusion
  1.5 inpainting VAE (CreativeML OpenRAIL-M licence, with its use restrictions).

None of these results are a guarantee of protection. Treat them as the
experimental measurements they are.
