# Sample faces for the public PhotoGuard page

The public sample mode (`../sample.py`, served at `/photoguard-sample/`) runs
PhotoGuard only on these two pictures. Both are **AI-generated pictures of
fictional people**, the same files the website already shows:

| File | Same bytes as | Shown on the website as |
| --- | --- | --- |
| `sample-1.jpg` (512 × 512) | `web/public/example/pg-original.jpg` | the PhotoGuard example, and the face-swap example's "Your photo" |
| `sample-2.jpg` (640 × 640) | `web/public/example/swap-target.jpg` | the face-swap example's "Someone else" |

Each `*-keep.png` is the matching mask, the same size as its photo: white is the
part PhotoGuard protects and keeps (face, hair and neck), black is what the AI
edit may repaint (background and clothes). That is what a visitor would paint in
the full demo. The masks were made by separating the plain studio background and
the sweater by colour, then checked by eye. The page draws the white region's
outline on the photo, so visitors see exactly what is kept.

`sample.py` pins every file's SHA-256 and refuses to start the page if one
changes. To change a sample, replace the files and update `SAMPLES` there. Use
only pictures of fictional people, never photos of real people.
