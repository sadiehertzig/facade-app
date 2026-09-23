# FACADE

FACADE helps people protect their photos from AI misuse, such as face-swap
tools and AI edits that change clothing or backgrounds. It adds small, carefully
chosen changes to a photo so that these tools work less well on the protected
copy, and it points people to official help services if their images have
already been misused.

FACADE has three parts:

- **An iPhone app** that protects a photo on the phone itself. Nothing is
  uploaded.
- **A website** where you crop a photo, mark the area an editor might replace,
  and (with your consent) send it to the team's GPU server to apply
  [PhotoGuard](https://github.com/MadryLab/photoguard).
- **A GPU server** that runs the PhotoGuard engine for the website, plus a
  hosted copy of the original PhotoGuard demo.

Protection is experimental and never guaranteed. Please keep everything we
write, in the app, on the website and in this repository, honest about that.

## What is where

| Folder | What it is | Start here |
| --- | --- | --- |
| [`app/ios/`](app/ios/) | The FACADE iPhone app (SwiftUI, with C/C++ for the protection engine) | [`app/ios/README.md`](app/ios/README.md) |
| [`web/`](web/) | The website (Next.js) | [`web/README.md`](web/README.md) |
| [`server/`](server/) | Python code for the GPU server: the PhotoGuard worker (`/v1` API), the PhotoGuard demo, model download, health check, tests | [`server/README.md`](server/README.md) |
| [`deploy/`](deploy/) | How the server is set up and updated: `bootstrap.sh`, the Caddy and systemd files, and `facade_server.py` | [`deploy/README.md`](deploy/README.md) |
| [`licenses/`](licenses/), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Licenses of the code and models FACADE builds on | |

How the pieces fit together on the server:

```
browser ──► https://<server>/             website (web/)            ──► worker on 127.0.0.1:8000
        ──► https://<server>/photoguard/  PhotoGuard demo (server/photoguard_demo/)
        ──► https://<server>/v1/...       worker API (server/worker/), bearer token only
```

The website and the demo ask for the team password. The iPhone app protects
photos on the phone and does not send them to the server; it can only open the
website in a browser.

## Running each part

You only need the tools for the part you work on.

**Website** (Node.js 22 or newer). No GPU needed: a pretend worker stands in
for the server.

```bash
cd web
npm ci
cp .env.example .env.local     # then put a made-up token in it, see web/README.md
npm run fake-worker            # terminal 1
npm run dev                    # terminal 2, then open http://localhost:3000
```

**Server code** (Python 3.12 with PyTorch; see `server/README.md` for the
exact versions). The tests need no GPU, model weights or network:

```bash
python -m pytest -q server/tests
```

**iPhone app** (a Mac with Xcode 26 or newer, and CMake):

```bash
app/ios/scripts/build_native_deps.sh     # once: builds the C/C++ libraries
app/ios/scripts/run_host_tests.sh        # quick tests on your Mac
open app/ios/FACADE.xcodeproj            # then pick the FACADE scheme and press Run
```

The app builds and runs without its models, but the protection buttons stay off
until the models are installed. The models are not in this repository; see
[`app/ios/Models/README.md`](app/ios/Models/README.md).

**The server itself** is updated only by the project owner, from `main`, with
`deploy/facade_server.py`. You do not need AWS access to work on any part of
FACADE.

## Working together

1. Get the latest `main`: `git switch main && git pull`.
2. Make a branch for your change, named after what it does:
   `git switch -c web/clearer-consent-text`.
3. Make small commits with messages that say what changed and why.
4. Run the checks for the part you touched:
   - website: `npm test`, `npm run lint` and `npm run build` in `web/`
   - server: `python -m pytest -q server/tests`
   - iPhone app: `app/ios/scripts/run_host_tests.sh` and a simulator build
5. Push the branch (`git push -u origin <branch>`) and open a pull request
   against `main`. Say what you changed, how you tested it, and add a screenshot
   for anything visible.
6. Someone else on the team reviews it. When it is approved, it is merged into
   `main`. Please do not push directly to `main`.

A few rules keep the repository safe to share:

- **Never commit secrets**: tokens, passwords, `.env.local`, your
  `Local.xcconfig` (it holds your Apple team ID), certificates or provisioning
  profiles.
- **Never commit model weights or other large binary files.** Models are
  shared privately; see `app/ios/Models/README.md`.
- **Use only synthetic or fictional images** when you test, never photos of
  real people, and do not commit test images.
- **Changing the protection engine changes the protection.** The numerical code
  in `app/ios/FACADE/Native/`, the `app/ios/FACADE/App/Native*.swift` files and
  `server/worker/facade_photoguard_encoder.py` needs a teammate who knows it to
  review the change.

The `.gitignore` files already keep most of these out; run `git status` before
you commit and check that only the files you meant to change are listed.

## Credits and licenses

FACADE's own code, and the FACADE app icon made by the team, are released under
the [MIT License](LICENSE).

The website's online protection and the hosted demo are built on
[PhotoGuard](https://github.com/MadryLab/photoguard) by Hadi Salman, Alaa
Khaddaj, Guillaume Leclerc, Andrew Ilyas and Aleksander Madry (MIT). FACADE is
not affiliated with its authors. The server uses Stable Diffusion 1.5 weights,
which come with use restrictions (CreativeML OpenRAIL-M) that everyone using the
website or demo must follow. Every third-party component and its license is
listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
