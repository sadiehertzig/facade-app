# FACADE website

A [Next.js](https://nextjs.org) app. It has three tabs:

- **Prepare a photo**: choose a photo, crop it to 512 × 512 in the browser,
  mark the area an editor might replace, and (with the user's consent) send the
  crop to the team's GPU server to apply PhotoGuard. It also links to the full
  PhotoGuard demo at `/photoguard/`.
- **Get help**: support services for people whose images were misused.
- **Our approach**: what FACADE does and what we have measured.

## Folder map

| Path | What it is |
| --- | --- |
| `app/page.tsx` | The page and its three tabs |
| `app/layout.tsx`, `app/globals.css` | Page shell and all styles |
| `components/protection-workspace.tsx` | Photo crop, mask brush, consent, job progress and download |
| `components/research-metrics.tsx`, `lib/face-swap-results.ts` | The face-swap results table and its numbers |
| `components/ui/` | Buttons, tabs and other building blocks from [shadcn/ui](https://ui.shadcn.com) (MIT, see `LICENSE.md` there) |
| `app/api/processing/*/route.ts` | Server routes the page calls: capabilities, jobs, output |
| `lib/processor.ts` | Server-side gateway to the PhotoGuard worker; all the safety checks live here |
| `lib/protection-mask.ts`, `lib/protection-recipe.ts` | Mask clean-up and the recipe name, shared by page and server |
| `scripts/fake-worker.mjs` | A pretend worker for local development (no GPU, no real protection) |
| `tests/` | Tests for the gateway and routes (`npm test`) |

## How the pieces talk

```
browser ──► Next.js routes (/api/processing/*) ──► PhotoGuard worker (/v1/*)
            lib/processor.ts adds the bearer        server/worker/ in this repo
```

The browser never talks to the worker and never sees its token. The gateway:

- only reports "available" when the worker declares exactly the expected recipe
  (`photoguard-encoder-linf-v1`, 512 px, 200 steps, 30-minute retention,
  experimental, GPU smoke test passed);
- accepts only JSON `{recipe, image_png, mask_png}` up to 4 MB, with both
  images as base64 PNGs;
- only accepts uploads and deletes sent from the site's own pages (Origin check);
- keeps each job's id and access token in an `HttpOnly`, `SameSite=Strict`
  cookie, so page scripts cannot read them;
- never passes the worker's error text to the browser and gives up on the
  worker after 15 seconds.

## Run it on your computer

You need Node.js 22 or newer.

```bash
cd web
npm ci                          # install exactly what package-lock.json lists
cp .env.example .env.local      # local settings; .env.local is gitignored
```

Put a made-up token into `.env.local`, for example the output of
`openssl rand -hex 32`:

```
FACADE_PROCESSOR_URL=http://127.0.0.1:8000
FACADE_PROCESSOR_TOKEN=<your made-up token>
FACADE_PUBLIC_ORIGIN=http://localhost:3000
```

Then, in two terminals:

```bash
npm run fake-worker   # terminal 1: pretend GPU server on 127.0.0.1:8000
npm run dev           # terminal 2: the website on http://localhost:3000
```

The fake worker "processes" a crop in about 10 seconds and gives it back
unchanged. It is only for working on the website. Like the real worker it holds
one job at a time, so a second upload gets "busy" until you clear the first job
(or two minutes after you download its result).

Without a worker (leave `FACADE_PROCESSOR_URL` empty) the site still runs; it
just says online processing is unavailable. The `/photoguard/` link only works
on the real server, where Caddy sends it to the separate PhotoGuard demo.

To try the real GPU server instead, set `FACADE_PROCESSOR_URL` to
`https://<server host>` and `FACADE_PROCESSOR_TOKEN` to the worker's bearer
token. Ask whoever runs the server; never commit or share that token.

## Checks

```bash
npm test            # gateway and route tests (vitest); no network needed
npm run lint        # ESLint
npm run typecheck   # TypeScript
npm run build       # production build (also type-checks)
```

Please run `npm test` and `npm run build` before opening a pull request.

## Settings

| Variable | Meaning |
| --- | --- |
| `FACADE_PROCESSOR_URL` | Worker base URL. Must be `https://`, except `http://` is allowed for `127.0.0.1` and `localhost`. |
| `FACADE_PROCESSOR_TOKEN` | Worker bearer token, 32–256 printable characters, no spaces. |
| `FACADE_PUBLIC_ORIGIN` | The address people open, e.g. `https://facade.example.org`. Used for the Origin check and the cookie's `Secure` flag, because behind Caddy the app only sees `http://127.0.0.1:3000`. **Required in production**; without it uploads are refused. In `npm run dev` it defaults to the request's own origin. |

These are server-only. Never rename them to `NEXT_PUBLIC_*`, which would send
them to every browser.

## How it is deployed

On the server (see `deploy/` at the top of this repo), bootstrap installs
Node 22 LTS, then, as the `facade` user in `/opt/facade/src/web`:

```bash
npm ci && npm run build
```

The `facade-web` service runs `next start -H 127.0.0.1 -p 3000` with its
settings in `/etc/facade/web.env` (`FACADE_PROCESSOR_URL=http://127.0.0.1:8000`,
the worker bearer, `FACADE_PUBLIC_ORIGIN=https://<host>`, `NODE_ENV=production`).
Caddy serves HTTPS, puts the site behind basic auth, sends `/photoguard*` to the
PhotoGuard demo and `/v1/*` to the worker, and everything else to this app.

## Credits

PhotoGuard is research by Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew
Ilyas and Aleksander Madry ([MadryLab/photoguard](https://github.com/MadryLab/photoguard),
MIT). FACADE is not affiliated with its authors or with the support services
the site links to. UI components are from shadcn/ui (MIT).
