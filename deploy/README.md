# Deploying the FACADE server

FACADE runs on one always-on AWS GPU server (a g6.xlarge with an NVIDIA L4, in
us-east-1). Caddy serves everything on one hostname with automatic HTTPS:

| URL | What | Runs on the server as | Protected by |
|---|---|---|---|
| `https://<host>/` | the FACADE website (`web/`) | `facade-web`, Next.js on 127.0.0.1:3000 | public; PhotoGuard runs rate-limited per visitor (`web/lib/quota.ts`) |
| `https://<host>/photoguard/` | the PhotoGuard demo (`server/photoguard_demo/`) | `facade-demo`, Gradio on 127.0.0.1:7860 | basic auth (user `owner`) |
| `https://<host>/v1/...` | the PhotoGuard worker API (`server/worker/`) | `facade-worker` on 127.0.0.1:8000 | the worker's own bearer token |

The website calls the worker directly on 127.0.0.1:8000. `/v1` is also reachable
from outside, but only with the bearer token.

`<host>` is **facadephotoshield.com**. `www.facadephotoshield.com` and the
server's original address redirect there; TestFlight builds 1.0 (3) and (4)
still link to the original address. The domain's DNS is in Route 53 (two A
records, apex and `www`, pointing at the server's fixed Elastic IP address).

## Who does what

- **Teammates:** work on a branch and open a pull request. Run the tests first
  (see the top-level README). You do not need AWS access.
- **The owner** (anyone holding the `facade` AWS access key): reviews and
  merges to `main`, then deploys `main` with `deploy/facade_server.py`.

Creating, resizing, stopping or deleting the server is not done from this
repository. `facade_server.py` only finds the running server, copies code to
it, restarts services, and reads its status and logs.

## One-time setup for the person who deploys

1. Install the AWS CLI v2 and the Session Manager plugin (macOS:
   `brew install awscli` and `brew install --cask session-manager-plugin`).
2. `aws configure --profile facade` with the access key of the IAM user
   `facade`, region `us-east-1`.
3. Check it: `aws sts get-caller-identity --profile facade` must print an ARN
   ending in `:user/facade`. The tool refuses anything else (root, roles, other
   accounts).

There is no SSH port and no SSH key to keep: the tool reaches the server through
AWS Systems Manager, pushing a throwaway key that is valid for 60 seconds.

## Commands

Run them from a checkout of this repository with any Python 3.9 or newer.
The tool uses only the standard library.

```bash
python3 deploy/facade_server.py status                       # state, URL, worker readiness
python3 deploy/facade_server.py deploy                       # copy this checkout, run bootstrap.sh, wait until ready
python3 deploy/facade_server.py logs --service web           # worker | demo | web | caddy | models | health | bootstrap | all
python3 deploy/facade_server.py restart --service demo       # worker | demo | web | caddy | all
python3 deploy/facade_server.py ssh                          # a root-capable shell on the server
python3 deploy/facade_server.py ssh --forward 7860           # then open http://localhost:7860/photoguard/
```

`status` also checks that `/` answers 200 (the website is public) and that
`/photoguard/` answers 401 without a password, which means Caddy is up and
basic auth is still in front of the demo.

### What `deploy` does

1. Bundles the files git tracks (`git ls-files`), refusing uncommitted changes
   unless you pass `--allow-dirty`. Untracked files such as `web/.env.local`
   never leave your computer.
2. Checks it is running as the `facade` user and finds the one running
   instance tagged `Project=FACADE`.
3. Mirrors the bundle to `/opt/facade/src` with `rsync --delete` (keeping the
   server-built `web/node_modules/` and `web/.next/`), then checks every file's
   sha256 on the server. If rsync fails it sends a tar stream instead and then
   deletes every file the bundle does not list, the same way.
4. Runs `sudo bash /opt/facade/src/deploy/bootstrap.sh` as a transient systemd
   unit and follows its log. A laptop that sleeps does not stop it; run
   `deploy` again to follow it again.
5. Waits until `https://<host>/v1/capabilities` reports ready.

`bootstrap.sh` is safe to run again and again. On each run it:

- keeps the Python environment in `/opt/facade/venv`, rebuilding it only when
  `server/requirements-gpu.lock.txt` or the torch/gradio pins change
- keeps Node.js 22 LTS in `/opt/facade/node`, from the official nodejs.org
  tarball, with the version and sha256 pinned in the script
- rebuilds the website (`npm ci` and `npm run build` in `web/`, as the `facade`
  user) only when files in `web/` changed
- reads the secrets from SSM and writes them to root/facade-only files
- validates the Caddyfile before replacing the live one
- restarts the services and waits for the loopback health check

A deploy with no Python or website changes takes a few minutes: the worker
reloads the model and reruns its 200-step GPU self-test before it reports
ready. The first boot of a new server takes 20 to 30 minutes: packages, about
8 GB of wheels and about 5.5 GB of model weights.

## Secrets

They live only in SSM Parameter Store (SecureString) and in root/facade-only
files on the server. None of them are in this repository, and the tool never
prints them.

| SSM parameter | Used for |
|---|---|
| `/facade/prod/processor-token` | the worker's bearer token (64 hex characters); also given to the website |
| `/facade/prod/basic-auth-hash` | bcrypt hash of the `owner` password for the website and the demo |
| `/facade/prod/host` | the public hostname Caddy gets a certificate for |

The owner changes them directly with the AWS CLI; the new value reaches the
server on the next `deploy`.

```bash
# New website/demo password (prompts twice; only the bcrypt hash is stored)
htpasswd -nBC 10 owner | cut -d: -f2 | aws ssm put-parameter --profile facade --region us-east-1 \
  --name /facade/prod/basic-auth-hash --type SecureString --value file:///dev/stdin --overwrite

# New worker token, at once if it ever leaks
openssl rand -hex 32 | tr -d '\n' | aws ssm put-parameter --profile facade --region us-east-1 \
  --name /facade/prod/processor-token --type SecureString --value file:///dev/stdin --overwrite
python3 deploy/facade_server.py deploy
```

## On the server

| Path | What |
|---|---|
| `/opt/facade/src` | a copy of this repository, root-owned and read-only for the services (`web/` belongs to `facade` only while the website builds; `next start` may write only `web/.next/cache`) |
| `/opt/facade/venv` | Python 3.12 with torch, diffusers and gradio |
| `/opt/facade/node` | Node.js 22 LTS |
| `/srv/facade/models/sd15-inpaint-8a4288` | the Stable Diffusion 1.5 inpainting weights, downloaded and sha256-checked by `facade-models` |
| `/etc/facade/` | `worker.bearer`, `web.env`, `host.env` |
| `/var/log/facade-bootstrap.log` | the log of every bootstrap run |

The units are in `deploy/systemd/`: `facade-models` (one-time weight
download), `facade-worker`, `facade-demo`, `facade-web`, `facade-health` (a
check every 60 s; `logs --service health`), plus a drop-in that gives Caddy the
hostname. Every file in `deploy/` starts with a comment that explains it: the
Caddyfile explains each route, and the units explain each setting, including
every security restriction, line by line.

- **Uploads.** Caddy allows 12 MB per request on `/v1` and the website and
  64 MB on the demo, whose image editor re-uploads full-size photos.
  `MAX_FILE_SIZE` in `server/photoguard_demo/app.py` must match the Caddyfile.
- **Reboots.** The worker and demo run as the unprivileged `facade` user. A
  root step before each start (`nvidia-modprobe -u -c=0`) creates
  `/dev/nvidia-uvm`; without it CUDA fails after a reboot.
- **Security updates** stay on but never reboot the server or restart the
  FACADE services. Restart at a planned time with `restart --service all`.

## When something is wrong

1. `status`: is the instance running, and is the worker ready?
2. `logs --service bootstrap` if a deploy failed, otherwise `logs --service <name>`.
3. Fix it on a branch, merge, and `deploy` again.

## Security notes

- Only Caddy listens publicly (ports 80 and 443). All three services bind to
  127.0.0.1, and Caddy keeps no access log.
- The services run as the unprivileged `facade` user and cannot change the code
  in `/opt/facade` (`ReadOnlyPaths=` in their units).
- The worker never writes photos, masks, outputs or tokens to disk or to its
  logs. Its log lines are lifecycle events only.
- Use only synthetic or fictional images, never photos of real people.
