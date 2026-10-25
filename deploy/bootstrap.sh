#!/usr/bin/env bash
# FACADE server bootstrap: idempotent, safe to re-run, run as root.
#
#   sudo bash /opt/facade/src/deploy/bootstrap.sh
#
# Expects a copy of this repository at /opt/facade/src (same relative paths;
# deploy/facade_server.py puts it there). It then:
#   1. installs python3.12-venv and Caddy (official apt repo, key fingerprint
#      pinned), apt-mark holds the installed kernel/NVIDIA packages, and
#      installs Node.js 22 LTS from the official nodejs.org tarball (exact
#      version and sha256 pinned below)
#   2. creates the no-login system user `facade`
#   3. builds /opt/facade/venv when missing or when the dependency recipe
#      (lock hash + torch/gradio pins) changed
#   4. fetches /facade/prod/{processor-token,basic-auth-hash,host} from SSM with
#      the instance role and writes them to root/facade-only files, including
#      /etc/facade/web.env for the website
#   5. builds the website (npm ci + npm run build in web/, as `facade`) when its
#      sources or the Node version changed
#   6. installs the systemd units, Caddyfile and caddy drop-in, caps journald
#      at 1G, disables core dumps
#   7. enables and (re)starts facade-models, facade-worker, facade-demo,
#      facade-web, facade-health.timer and caddy, then waits for the loopback
#      health check
#
# Secrets are never printed: no xtrace, values only pass through bash variables
# and files. Everything is logged to /var/log/facade-bootstrap.log.
#
# HOW IT IS RUN. deploy/facade_server.py copies the repository to the server
# and starts this script as a temporary systemd unit (facade-bootstrap), so it
# keeps running even if the laptop that started it goes to sleep.
#
# SAFE TO RUN AGAIN. Every step checks first and only does work that is
# needed: packages already installed are kept, the Python environment and the
# website are rebuilt only when their inputs changed (a sha256 "stamp" file
# remembers the last build), and the model weights are downloaded only once.
#
# A PATTERN USED THROUGHOUT. Secret files are written to a temporary file in
# the same folder, given their owner and permissions, and only then renamed
# into place with `mv -f`. A rename is atomic, so a service never sees a
# half-written file or one with the wrong permissions.
#
# The shell settings below: -e stops at the first failing command, -u treats
# an unset variable as an error, and pipefail makes a pipeline fail when any
# part of it fails, not just the last.
set -euo pipefail
# New files are readable by everyone unless a step says otherwise (secret
# files are tightened explicitly). PATH is fixed so the script does not depend
# on whoever started it. No apt prompts, and the AWS CLI must not open a pager.
umask 022
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin
export DEBIAN_FRONTEND=noninteractive
export AWS_PAGER=""

# ---- Fixed paths and pinned versions ------------------------------------------
#   SRC              the copy of this repository that facade_server.py pushed
#   VENV             the Python environment for the worker and the demo
#   SNAPSHOT         the sha256-checked Stable Diffusion 1.5 inpainting weights
#   SSM_PREFIX       where the secrets live in AWS Systems Manager Parameter Store
#   TORCH_*/GRADIO   exact versions, so every rebuild installs the same thing
#   CADDY_KEY_FPR    the fingerprint of the Caddy package signing key: the
#                    downloaded key is accepted only if it matches exactly
#   APT              apt-get that waits up to 15 minutes for another apt run
#                    (e.g. automatic security updates) instead of failing
LOG=/var/log/facade-bootstrap.log
SRC=/opt/facade/src
DEPLOY="$SRC/deploy"
SERVER="$SRC/server"
WEB="$SRC/web"
VENV=/opt/facade/venv
SYSTEM_PYTHON=/usr/bin/python3.12
MODELS_ROOT=/srv/facade/models
SNAPSHOT="$MODELS_ROOT/sd15-inpaint-8a4288"
REGION=us-east-1
SSM_PREFIX=/facade/prod
TORCH_INDEX=https://download.pytorch.org/whl/cu128
TORCH_PINS=(torch==2.8.0 torchvision==0.23.0)
GRADIO_PIN="gradio==6.28.0"
CADDY_KEY_URL=https://dl.cloudsmith.io/public/caddy/stable/gpg.key
CADDY_KEY_FPR=65760C51EDEA2017CEA2CA15155B6D79CA56EA34
CADDY_KEYRING=/usr/share/keyrings/caddy-stable-archive-keyring.gpg
CADDY_LIST=/etc/apt/sources.list.d/caddy-stable.list
# Node.js 22 LTS ("Jod"), linux-x64. To upgrade, take both values from
# https://nodejs.org/dist/<version>/SHASUMS256.txt
NODE_VERSION=v22.23.2
NODE_SHA256=d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307
NODE_TARBALL="node-$NODE_VERSION-linux-x64.tar.xz"
NODE_URL="https://nodejs.org/dist/$NODE_VERSION/$NODE_TARBALL"
NODE_DIR=/opt/facade/node
NPM_CACHE=/opt/facade/.npm
WEB_STAMP=/opt/facade/.web-build-sha256
APT=(apt-get -y -q -o DPkg::Lock::Timeout=900)
HEALTH_WAIT_SECONDS=${FACADE_HEALTH_WAIT_SECONDS:-1200}

# Only root can install packages and write /etc. Then copy everything the
# script prints, from here on, into the log file as well as the terminal.
if [[ ${EUID} -ne 0 ]]; then
  echo "bootstrap.sh must run as root (sudo)." >&2
  exit 1
fi
touch "$LOG"
chmod 0600 "$LOG"
exec > >(tee -a "$LOG") 2>&1

# log: one timestamped line. die: log an error and stop. The EXIT trap adds a
# final "FAILED" line pointing at the full log whenever the script stops early.
# apt_update retries `apt-get update` for up to about 2.5 minutes.
log() { printf '[facade-bootstrap %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }
trap 'rc=$?; if [[ $rc -ne 0 ]]; then log "FAILED (exit $rc); full log: $LOG"; fi' EXIT
apt_update() {  # the apt lists lock is not covered by DPkg::Lock::Timeout on every apt
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    "${APT[@]}" update && return 0
    log "apt-get update failed (attempt $attempt); retrying in 15 s"
    sleep 15
  done
  return 1
}

log "start (log: $LOG)"

# --- 0. code present --------------------------------------------------------
# Refuse to start unless every file later steps need is present.
for required in \
  "$DEPLOY/Caddyfile" "$DEPLOY/systemd/facade-models.service" \
  "$DEPLOY/systemd/facade-worker.service" "$DEPLOY/systemd/facade-demo.service" \
  "$DEPLOY/systemd/facade-web.service" "$DEPLOY/systemd/facade-health.service" \
  "$DEPLOY/systemd/facade-health.timer" "$DEPLOY/systemd/caddy-facade.conf" \
  "$SERVER/models/MODEL_SHA256SUMS" "$SERVER/models/fetch_models.py" \
  "$SERVER/requirements-gpu.lock.txt" "$SERVER/health.py" \
  "$SERVER/worker/facade_photoguard_service.py" "$SERVER/worker/facade_photoguard_worker.py" \
  "$SERVER/worker/facade_photoguard_encoder.py" "$SERVER/photoguard_demo/app.py" \
  "$WEB/package.json" "$WEB/package-lock.json"; do
  [[ -f "$required" ]] || die "missing $required - copy the repository to $SRC first (deploy/facade_server.py deploy)"
done
# Code is root-owned and read-only for the services (they run as `facade`, and
# u+rwX,go+rX,go-w means only root may change a file). web/ is handled in step 5:
# it belongs to `facade` only while npm ci and next build run, so it is left out
# here to avoid flipping its build output twice.
chown root:root "$SRC"
chmod 0755 "$SRC"
find "$SRC" -mindepth 1 -maxdepth 1 ! -name web -exec chown -R root:root {} + \
  -exec chmod -R u+rwX,go+rX,go-w {} +

# --- 1. packages ------------------------------------------------------------
# The server starts from Amazon's Deep Learning AMI (DLAMI) for Ubuntu 24.04,
# which already has Python 3.12, the NVIDIA driver and the AWS CLI.
[[ -x "$SYSTEM_PYTHON" ]] || die "$SYSTEM_PYTHON not found (expected on the Ubuntu 24.04 DLAMI)"
log "apt: update + python3.12-venv and helpers"
apt_update || die "apt-get update keeps failing"
"${APT[@]}" install python3.12-venv ca-certificates curl gnupg xz-utils

# Caddy comes from its official apt repository. Its signing key is downloaded,
# but only trusted when it is exactly one primary key with the pinned
# fingerprint, so a tampered download cannot slip in a different key. gpg runs
# with a throwaway GNUPGHOME so nothing is added to root's own keyring.
if [[ ! -s "$CADDY_KEYRING" || ! -s "$CADDY_LIST" ]] || ! command -v caddy >/dev/null 2>&1; then
  log "apt: adding the official Caddy stable repo (key fingerprint pinned)"
  keytmp=$(mktemp -d)
  curl -fsSL --retry 5 --retry-delay 3 "$CADDY_KEY_URL" -o "$keytmp/caddy.asc"
  keyinfo=$(GNUPGHOME="$keytmp" gpg --batch --show-keys --with-colons "$keytmp/caddy.asc") \
    || die "cannot parse the Caddy repo key"
  primaries=$(awk -F: '$1=="pub"{n++} END{print n+0}' <<<"$keyinfo")
  fpr=$(awk -F: '$1=="pub"{p=1;next} p&&$1=="fpr"{print $10; exit}' <<<"$keyinfo")
  [[ "$primaries" -eq 1 && "$fpr" == "$CADDY_KEY_FPR" ]] \
    || die "Caddy repo key fingerprint mismatch (got '${fpr}', ${primaries} primary keys)"
  GNUPGHOME="$keytmp" gpg --batch --yes --dearmor -o "$CADDY_KEYRING" "$keytmp/caddy.asc"
  chmod 0644 "$CADDY_KEYRING"
  printf 'deb [signed-by=%s] https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main\n' \
    "$CADDY_KEYRING" > "$CADDY_LIST"
  chmod 0644 "$CADDY_LIST"
  rm -rf "$keytmp"
  apt_update || die "apt-get update failed after adding the Caddy repo"
fi
command -v caddy >/dev/null 2>&1 || "${APT[@]}" install caddy
log "caddy: $(caddy version 2>/dev/null | head -n1)"

# Freeze the kernel and NVIDIA driver packages ("apt-mark hold"): an automatic
# update to either could leave the GPU unusable until a reboot. Other security
# updates still install, but never reboot the server on their own.
log "apt: holding installed kernel and NVIDIA packages; no automatic reboot"
# shellcheck disable=SC2016  # dpkg-query/awk fields, not shell expansions
mapfile -t HOLD < <(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' \
  | awk '$1=="ii"{print $2}' \
  | grep -E '^(linux-(image|headers|modules|aws|tools|objects|signatures)|linux-.*-aws$)|nvidia|^cuda-(drivers|compat)|^libxnvctrl' || true)
if [[ ${#HOLD[@]} -gt 0 ]]; then
  apt-mark hold "${HOLD[@]}" >/dev/null
  log "held ${#HOLD[@]} packages"
fi
cat > /etc/apt/apt.conf.d/52facade-no-auto-reboot <<'EOF'
// FACADE server: security updates stay on, but never reboot unattended.
Unattended-Upgrade::Automatic-Reboot "false";
EOF
# needrestart (Ubuntu 24.04) would otherwise restart the services after an
# unattended python/libc/openssl update: a surprise model reload and 200-step
# smoke. Updates now only log the pending restart; restart at a planned time.
install -d -m 0755 /etc/needrestart/conf.d
cat > /etc/needrestart/conf.d/60-facade.conf <<'NRCONF'
# FACADE server: never auto-restart the FACADE services or Caddy after upgrades.
$nrconf{override_rc}{qr(^facade-)} = 0;
$nrconf{override_rc}{qr(^caddy)} = 0;
NRCONF

# Report the GPU in the log, and warn about anything the services will need.
if command -v nvidia-smi >/dev/null 2>&1; then
  log "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo unavailable)"
else
  log "WARNING: nvidia-smi not found; the services need the DLAMI NVIDIA driver"
fi
# facade-worker/facade-demo run it as root (ExecStartPre=-+) to create
# /dev/nvidia-uvm before dropping privileges under NoNewPrivileges.
if [[ ! -x /usr/bin/nvidia-modprobe ]]; then
  log "WARNING: /usr/bin/nvidia-modprobe not found; after a reboot CUDA may fail until /dev/nvidia-uvm exists"
fi

# Node.js: the exact pinned release, checked against its sha256 before unpacking.
# It is unpacked into a temporary folder and only then swapped into
# /opt/facade/node, so a failed download never leaves a half-installed Node.
if [[ -x "$NODE_DIR/bin/node" && "$("$NODE_DIR/bin/node" --version 2>/dev/null || true)" == "$NODE_VERSION" ]]; then
  log "node: $NODE_VERSION already installed"
else
  [[ "$(uname -m)" == "x86_64" ]] || die "the pinned Node.js tarball is linux-x64; this machine is $(uname -m)"
  log "node: installing $NODE_VERSION from $NODE_URL (sha256 pinned)"
  install -d -m 0755 /opt/facade
  nodetmp=$(mktemp -d /opt/facade/.node-install.XXXXXX)
  curl -fsSL --retry 5 --retry-delay 3 "$NODE_URL" -o "$nodetmp/$NODE_TARBALL"
  printf '%s  %s\n' "$NODE_SHA256" "$nodetmp/$NODE_TARBALL" | sha256sum --check --strict --quiet \
    || die "sha256 mismatch for $NODE_TARBALL; not installing it"
  install -d -m 0755 "$nodetmp/node"
  tar -xJf "$nodetmp/$NODE_TARBALL" -C "$nodetmp/node" --strip-components=1 --no-same-owner
  chown -R root:root "$nodetmp/node"
  chmod -R u+rwX,go+rX,go-w "$nodetmp/node"
  rm -rf "$NODE_DIR"
  mv "$nodetmp/node" "$NODE_DIR"
  rm -rf "$nodetmp"
  log "node: $("$NODE_DIR/bin/node" --version) installed at $NODE_DIR"
fi

# --- 2. user and directories -------------------------------------------------
# The `facade` system user runs every service. It cannot log in (nologin
# shell). The video and render groups give it access to the GPU devices.
# /etc/facade (root-owned, readable by the facade group) holds the secrets.
if ! id -u facade >/dev/null 2>&1; then
  log "creating system user facade"
  useradd --system --user-group --home-dir /var/lib/facade --create-home --shell /usr/sbin/nologin facade
fi
for group in video render; do
  if getent group "$group" >/dev/null; then usermod -aG "$group" facade; fi
done
install -d -m 0700 -o facade -g facade /var/lib/facade
install -d -m 0755 -o root -g root /opt/facade "$MODELS_ROOT"
install -d -m 0750 -o root -g facade /etc/facade
install -d -m 0700 -o facade -g facade "$NPM_CACHE"

# --- 3. venv ----------------------------------------------------------------
# The environment is rebuilt only when its "recipe" changes. The recipe hash
# covers the lock file (every package at an exact version), the Python version,
# and the torch and gradio pins; it is stored in a stamp file inside the venv.
# Install order: torch from PyTorch's CUDA 12.8 index, then the lock file with
# no extra dependencies, then gradio held to the lock file's versions.
# Afterwards a short Python check confirms the key versions import correctly,
# and `pip freeze` records exactly what was installed.
LOCK="$SERVER/requirements-gpu.lock.txt"
RECIPE_HASH=$({ sha256sum "$LOCK" | cut -d' ' -f1; "$SYSTEM_PYTHON" -V
  echo "$TORCH_INDEX|${TORCH_PINS[*]}|$GRADIO_PIN"; } | sha256sum | cut -d' ' -f1)
STAMP="$VENV/.facade-recipe-sha256"
if [[ -x "$VENV/bin/python" && -f "$STAMP" && "$(cat "$STAMP")" == "$RECIPE_HASH" ]]; then
  log "venv: up to date ($RECIPE_HASH)"
else
  log "venv: building $VENV (recipe $RECIPE_HASH); several GB of wheels, ~5-10 min"
  systemctl stop facade-worker.service facade-demo.service 2>/dev/null || true
  rm -rf "$VENV"
  "$SYSTEM_PYTHON" -m venv "$VENV"
  PIP=("$VENV/bin/python" -m pip --disable-pip-version-check --no-input install --no-cache-dir)
  "${PIP[@]}" --index-url "$TORCH_INDEX" "${TORCH_PINS[@]}"
  "${PIP[@]}" --no-deps -r "$LOCK"
  "${PIP[@]}" "$GRADIO_PIN" -c "$LOCK"
  "$VENV/bin/python" -m pip check || log "WARNING: pip check reported conflicts (see above)"
  "$VENV/bin/python" -m pip freeze --all > /opt/facade/requirements-aws.lock.txt
  "$VENV/bin/python" - <<'EOF'
from importlib.metadata import version
import torch
assert version("torch").split("+")[0] == "2.8.0", version("torch")
assert version("diffusers") == "0.40.0" and version("transformers") == "5.16.1"
import diffusers, transformers, gradio  # noqa: F401  import smoke
print("venv ok: torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "gradio", gradio.__version__)
EOF
  echo "$RECIPE_HASH" > "$STAMP"
  log "venv: built; frozen to /opt/facade/requirements-aws.lock.txt"
fi

# --- 4. secrets and host from SSM ---------------------------------------------
command -v aws >/dev/null 2>&1 || die "aws CLI not found (expected on the DLAMI)"
# Secrets come from SSM Parameter Store, read with the server's own AWS role
# (no keys are stored on the server). Each value is checked against its exact
# expected shape before it is written anywhere, and never printed: the log
# lines name the parameter and the destination file, not the value. SSM's
# error text goes to a temporary file, so it can be shown without the value.
SSM_ERR=$(mktemp)
ssm_get() {  # value goes to stdout only; callers capture it and never echo it
  local name=$1 attempt
  for attempt in 1 2 3 4 5 6 7 8; do
    if aws ssm get-parameter --region "$REGION" --name "$name" --with-decryption \
         --query Parameter.Value --output text 2>"$SSM_ERR"; then
      return 0
    fi
    sleep $((attempt * 3))
  done
  return 1
}
ssm_die() { die "cannot read SSM $1 with the instance role: $(tail -n 2 "$SSM_ERR" | tr '\n' ' ')"; }

# The worker's bearer token: 64 lowercase hex characters. Owner facade, mode
# 0400 (only `facade` can read it, nobody can write it).
log "ssm: fetching $SSM_PREFIX/processor-token -> /etc/facade/worker.bearer"
token=$(ssm_get "$SSM_PREFIX/processor-token") || ssm_die "$SSM_PREFIX/processor-token"
[[ "$token" =~ ^[0-9a-f]{64}$ ]] || die "SSM $SSM_PREFIX/processor-token is not 64 lowercase hex"
tmp=$(mktemp /etc/facade/.worker.bearer.XXXXXX)
printf '%s\n' "$token" > "$tmp"
chown facade:facade "$tmp"
chmod 0400 "$tmp"
mv -f "$tmp" /etc/facade/worker.bearer

# The website/demo password, as a bcrypt hash. It becomes a small Caddy
# snippet that the Caddyfile imports; owner root, group caddy, mode 0640, so
# only root can change it and only Caddy can read it.
log "ssm: fetching $SSM_PREFIX/basic-auth-hash -> /etc/caddy/facade-auth.caddy"
hash=$(ssm_get "$SSM_PREFIX/basic-auth-hash") || ssm_die "$SSM_PREFIX/basic-auth-hash"
[[ "$hash" =~ ^\$2[aby]\$[0-9]{2}\$[./A-Za-z0-9]{53}$ ]] || die "SSM $SSM_PREFIX/basic-auth-hash is not a bcrypt hash"
install -d -m 0755 /etc/caddy
tmp=$(mktemp /etc/caddy/.facade-auth.XXXXXX)
printf 'basic_auth {\n\towner %s\n}\n' "$hash" > "$tmp"
unset hash
chown root:caddy "$tmp"
chmod 0640 "$tmp"
mv -f "$tmp" /etc/caddy/facade-auth.caddy

# The public hostname (facadephotoshield.com). Lower-cased and checked to be a
# plain DNS name; it is not secret, so the file is world-readable.
log "ssm: fetching $SSM_PREFIX/host -> /etc/facade/host.env"
host=$(ssm_get "$SSM_PREFIX/host") || ssm_die "$SSM_PREFIX/host"
rm -f "$SSM_ERR"
host=${host,,}
[[ "$host" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$ ]] \
  || die "SSM $SSM_PREFIX/host is not a DNS hostname: '$host'"
tmp=$(mktemp /etc/facade/.host.env.XXXXXX)
printf 'FACADE_HOST=%s\n' "$host" > "$tmp"
chmod 0644 "$tmp"
mv -f "$tmp" /etc/facade/host.env
log "host: $host"

# The website's settings: the worker on loopback, the bearer token, and its own
# public address (for the Origin check and the Secure cookie). The token is
# cleared from this script's memory right after.
log "writing /etc/facade/web.env (website -> loopback worker; values not shown)"
tmp=$(mktemp /etc/facade/.web.env.XXXXXX)
printf 'FACADE_PROCESSOR_URL=http://127.0.0.1:8000\nFACADE_PROCESSOR_TOKEN=%s\nFACADE_PUBLIC_ORIGIN=https://%s\nNODE_ENV=production\n' "$token" "$host" > "$tmp"
unset token
chown facade:facade "$tmp"
chmod 0400 "$tmp"
mv -f "$tmp" /etc/facade/web.env

# --- 5. website build ------------------------------------------------------------
# npm ci and next build write inside web/, so web/ is handed to `facade` for the
# build only. Afterwards the whole tree is root-owned again, so the services
# (which all run as `facade` and parse untrusted uploads) cannot rewrite the
# website's code; `next start` may write only web/.next/cache. The build is
# skipped when the web sources (not node_modules/ or .next/) and the Node
# version are unchanged since the last one.
# WEB_HASH: one sha256 over every web/ source file (sorted, so the order is
# always the same) plus the Node version.
WEB_HASH=$({ cd "$SRC" && find web -type f \
    ! -path 'web/node_modules/*' ! -path 'web/.next/*' ! -name 'next-env.d.ts' ! -name '*.tsbuildinfo' -print0 \
    | LC_ALL=C sort -z | xargs -0 -r sha256sum; echo "$NODE_VERSION"; } | sha256sum | cut -d' ' -f1)
if [[ -f "$WEB/.next/BUILD_ID" && -d "$WEB/node_modules" && "$(cat "$WEB_STAMP" 2>/dev/null || true)" == "$WEB_HASH" ]]; then
  log "web: build up to date ($WEB_HASH)"
else
  log "web: npm ci + npm run build in $WEB as facade (a few minutes)"
  systemctl stop facade-web.service 2>/dev/null || true
  rm -f "$WEB_STAMP"
  chown -R facade:facade "$WEB"
  chmod -R u+rwX,go+rX,go-w "$WEB"
  # A clean environment with no secrets at build time: the website reads its
  # settings from /etc/facade/web.env when facade-web starts.
  # shellcheck disable=SC2016  # $1 is expanded by the inner bash
  runuser -u facade -- env -i HOME=/var/lib/facade PATH="$NODE_DIR/bin:/usr/bin:/bin" \
    npm_config_cache="$NPM_CACHE" NEXT_TELEMETRY_DISABLED=1 \
    bash -c 'cd "$1" && npm ci --no-audit --no-fund && npm run build' web-build "$WEB" \
    || die "web build failed (see the npm output above)"
  echo "$WEB_HASH" > "$WEB_STAMP"
  log "web: built ($WEB_HASH)"
fi
# Back to root-owned and read-only, except the cache folder `next start` uses.
chown -R root:root "$WEB"
chmod -R u+rwX,go+rX,go-w "$WEB"
install -d -m 0700 -o facade -g facade "$WEB/.next/cache"
chown -R facade:facade "$WEB/.next/cache"

# --- 6. units, Caddy, journald, core dumps ----------------------------------------
# Copy the unit files from deploy/systemd/ into place, plus the drop-in that
# hands Caddy the hostname (FACADE_HOST).
log "installing systemd units, caddy drop-in and Caddyfile"
for unit in facade-models.service facade-worker.service facade-demo.service \
            facade-web.service facade-health.service facade-health.timer; do
  install -m 0644 "$DEPLOY/systemd/$unit" "/etc/systemd/system/$unit"
done
install -d -m 0755 /etc/systemd/system/caddy.service.d
install -m 0644 "$DEPLOY/systemd/caddy-facade.conf" /etc/systemd/system/caddy.service.d/facade.conf
# Validate a staged copy: only a config Caddy accepts replaces /etc/caddy/Caddyfile,
# so a rejected file can never break the next caddy restart or reboot.
staged=$(mktemp /etc/caddy/.Caddyfile.XXXXXX)
install -m 0644 "$DEPLOY/Caddyfile" "$staged"
validate_out=$(mktemp)
if ! FACADE_HOST="$host" caddy validate --config "$staged" --adapter caddyfile >"$validate_out" 2>&1; then
  grep -v 'basic_auth\|\$2[aby]\$' "$validate_out" | tail -n 5 || true
  rm -f "$validate_out" "$staged"
  die "caddy validate rejected $DEPLOY/Caddyfile; /etc/caddy/Caddyfile left unchanged"
fi
rm -f "$validate_out"
mv -f "$staged" /etc/caddy/Caddyfile

# Cap the system journal at 1 GB so logs can never fill the disk; restart
# journald (below) only if the setting changed.
install -d -m 0755 /etc/systemd/journald.conf.d
JOURNALD_CONF=/etc/systemd/journald.conf.d/60-facade.conf
JOURNALD_WANT=$'[Journal]\nSystemMaxUse=1G'
if [[ "$(cat "$JOURNALD_CONF" 2>/dev/null || true)" != "$JOURNALD_WANT" ]]; then
  printf '%s\n' "$JOURNALD_WANT" > "$JOURNALD_CONF"
  RESTART_JOURNALD=1
fi

# Switch off core dumps everywhere. A core dump is a copy of a crashed
# program's memory written to disk, and a worker's memory can hold a visitor's
# photo or the bearer token. Each line closes a different route: systemd's
# coredump store, the default limit for all units, login sessions, Ubuntu's
# crash reporter (apport), and the kernel itself (core_pattern sends dumps
# nowhere; suid_dumpable=0 covers privileged programs).
install -d -m 0755 /etc/systemd/coredump.conf.d /etc/systemd/system.conf.d /etc/security/limits.d
printf '[Coredump]\nStorage=none\nProcessSizeMax=0\n' > /etc/systemd/coredump.conf.d/60-facade.conf
printf '[Manager]\nDefaultLimitCORE=0\n' > /etc/systemd/system.conf.d/60-facade-nocore.conf
printf '* hard core 0\n* soft core 0\n' > /etc/security/limits.d/60-facade-nocore.conf
systemctl disable --now apport.service >/dev/null 2>&1 || true
printf 'kernel.core_pattern=|/bin/false\nfs.suid_dumpable=0\n' > /etc/sysctl.d/60-facade-nocore.conf
sysctl -q -p /etc/sysctl.d/60-facade-nocore.conf

# Make systemd re-read the unit files just installed.
systemctl daemon-reload
if [[ -n "${RESTART_JOURNALD:-}" ]]; then systemctl restart systemd-journald; fi

# --- 7. start everything ------------------------------------------------------
# Enable everything to start at boot. The model download runs first (it is a
# no-op once the weights are verified), then the services are restarted so
# they pick up the new code, settings and secrets.
systemctl enable facade-models.service facade-worker.service facade-demo.service \
  facade-web.service facade-health.timer caddy.service >/dev/null
if [[ -f "$SNAPSHOT/.verified" ]]; then
  log "models: $SNAPSHOT already verified"
else
  log "models: fetching + verifying ~5.5 GB (journalctl -u facade-models -f)"
fi
systemctl start facade-models.service || die "facade-models failed: journalctl -u facade-models"
[[ -f "$SNAPSHOT/.verified" ]] || die "model snapshot not verified after facade-models"

log "starting facade-worker, facade-demo, facade-web, caddy, facade-health.timer"
systemctl restart facade-worker.service facade-demo.service facade-web.service
systemctl restart caddy.service
systemctl restart facade-health.timer

# Wait until server/health.py, run as `facade`, sees the website, the worker
# and the demo answering on loopback. The worker is ready only after its
# 200-step GPU self-test, so this takes a few minutes. Exit code 2 means
# "installed, but not healthy yet" (the services keep starting).
log "waiting up to ${HEALTH_WAIT_SECONDS}s for the loopback health check (website, worker smoke, demo load)"
if runuser -u facade -- /usr/bin/python3 "$SERVER/health.py" --no-public --wait "$HEALTH_WAIT_SECONDS"; then
  log "loopback healthy. Public check: journalctl -u facade-health -n 3 (TLS needs DNS for $host -> this server)"
else
  log "loopback NOT healthy yet: journalctl -u facade-worker -u facade-demo -u facade-web -n 100"
  exit 2
fi
for unit in facade-worker facade-demo facade-web caddy facade-health.timer; do
  log "$unit: $(systemctl is-active "$unit" || true)"
done
log "done"
