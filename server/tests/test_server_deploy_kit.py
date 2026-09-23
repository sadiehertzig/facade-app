"""Static checks of the server deploy kit; no network, AWS, systemd, GPU or model bytes.

Covers deploy/ (systemd units, Caddyfile routing, bootstrap.sh syntax and
secret hygiene) and the server-side helpers it runs: server/models/ (the model
checksum list vs the hash-bound loaders, fetch_models.py), server/health.py and
the pinned GPU lock. fetch_models.py and health.py run with injected fakes on
tiny temporary files.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys

import pytest

import facade_photoguard_diffusion as diffusion
import facade_photoguard_encoder as encoder

SERVER = Path(__file__).resolve().parents[1]
REPO = SERVER.parent
DEPLOY = REPO / "deploy"
UNITS = DEPLOY / "systemd"
MODELS = SERVER / "models"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch_models = _load_module("facade_fetch_models", MODELS / "fetch_models.py")
health = _load_module("facade_health", SERVER / "health.py")

BOX_SRC = "/opt/facade/src/"
VENV_PY = "/opt/facade/venv/bin/python"
SNAPSHOT = "/srv/facade/models/sd15-inpaint-8a4288"
WORKER_EXEC = (f"{VENV_PY} /opt/facade/src/server/worker/facade_photoguard_service.py "
               f"--snapshot {SNAPSHOT} --bind 127.0.0.1 --port 8000")
DEMO_EXEC = (f"{VENV_PY} /opt/facade/src/server/photoguard_demo/app.py "
             f"--snapshot {SNAPSHOT} --host 127.0.0.1 --port 7860 --root-path /photoguard")
MODELS_EXEC = (f"{VENV_PY} /opt/facade/src/server/models/fetch_models.py "
               f"--sums /opt/facade/src/server/models/MODEL_SHA256SUMS --dest {SNAPSHOT}")
WEB_EXEC = "/opt/facade/node/bin/node node_modules/next/dist/bin/next start -H 127.0.0.1 -p 3000"
HEALTH_EXEC = "/usr/bin/python3 /opt/facade/src/server/health.py --bearer-file /etc/facade/worker.bearer"
EXTRA_MODEL_FILES = {
    "model_index.json": "636d700de8970a70f02b91d433ad949dc2ceb750f39ba0cf2cf39ef0b0b9d1e4",
    "feature_extractor/preprocessor_config.json": "2a1da83b5e1032aaeef397552ddb408dca0d8cd1dc58f61bf6abf38d6f33a0a2",
    "safety_checker/config.json": "d69bd41b02b5e8f4d3b148871ce03b1eaf108467ee3c505fe67b5b814ca3109e",
    "safety_checker/pytorch_model.bin": "193490b58ef62739077262e833bf091c66c29488058681ac25cf7df3d8190974",
}
LOCK_SHA256 = "e32bfd236901fed6b891638f5feefc4ad2fde8f92e81d64d78bdff5fb57e1867"
LOCAL_SNAPSHOT = Path.home() / (".cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting/"
                                "snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb")
UNIT_FILES = ["facade-models.service", "facade-worker.service", "facade-demo.service", "facade-web.service",
              "facade-health.service", "facade-health.timer", "caddy-facade.conf"]
KIT_FILES = [DEPLOY / "bootstrap.sh", DEPLOY / "Caddyfile", DEPLOY / "facade_server.py", DEPLOY / "README.md",
             MODELS / "fetch_models.py", MODELS / "MODEL_SHA256SUMS", SERVER / "health.py",
             SERVER / "requirements-gpu.lock.txt", *(UNITS / name for name in UNIT_FILES)]


def parse_unit(path: Path) -> dict[str, list[tuple[str, str]]]:
    sections: dict[str, list[tuple[str, str]]] = {}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        assert current is not None, f"{path.name}: key outside a section: {line}"
        assert not line.endswith("\\"), f"{path.name}: no line continuations expected"
        key, sep, value = line.partition("=")
        assert sep, f"{path.name}: malformed line {line}"
        sections[current].append((key.strip(), value.strip()))
    return sections


def values(unit, section, key):
    return [value for name, value in unit.get(section, []) if name == key]


def one(unit, section, key):
    found = values(unit, section, key)
    assert len(found) == 1, (section, key, found)
    return found[0]


def unit(name):
    return parse_unit(UNITS / name)


def environment(parsed):
    return {item.partition("=")[0]: item.partition("=")[2] for item in values(parsed, "Service", "Environment")}


def systemd_unescape(text):
    return text.replace("$$", "$").replace("%%", "%")


# --- kit inventory and unit files ------------------------------------------------

@pytest.mark.parametrize("path", KIT_FILES, ids=lambda path: str(path.relative_to(REPO)))
def test_kit_file_exists(path):
    assert path.is_file()


@pytest.mark.parametrize("name", UNIT_FILES)
def test_unit_files_parse(name):
    parsed = unit(name)
    assert parsed
    for section in parsed:
        assert section in {"Unit", "Service", "Install", "Timer"}


def test_exec_start_matches_contract():
    assert one(unit("facade-worker.service"), "Service", "ExecStart") == WORKER_EXEC
    assert one(unit("facade-demo.service"), "Service", "ExecStart") == DEMO_EXEC
    assert one(unit("facade-models.service"), "Service", "ExecStart") == MODELS_EXEC
    assert one(unit("facade-web.service"), "Service", "ExecStart") == WEB_EXEC
    assert one(unit("facade-health.service"), "Service", "ExecStart") == HEALTH_EXEC


def _box_paths():
    found = set()
    for path in UNITS.iterdir():
        for token in re.findall(r"/opt/facade/src/[A-Za-z0-9_./-]+", path.read_text()):
            found.add(token)
    return sorted(found)


def test_units_reference_files_in_this_repository():
    paths = _box_paths()
    assert paths, "units should run code from /opt/facade/src"
    for box_path in paths:
        relative = box_path[len(BOX_SRC):]
        assert (REPO / relative).exists(), box_path


@pytest.mark.parametrize("script, flags", [
    ("server/worker/facade_photoguard_service.py", ("--snapshot", "--bind", "--port")),
    ("server/photoguard_demo/app.py", ("--snapshot", "--host", "--port", "--root-path")),
    ("server/models/fetch_models.py", ("--sums", "--dest")),
    ("server/health.py", ("--bearer-file",)),
])
def test_service_scripts_accept_the_flags_their_units_pass(script, flags):
    assert any(path.startswith(BOX_SRC + script) for path in _box_paths())
    text = (REPO / script).read_text()
    for flag in flags:
        assert f'"{flag}"' in text or f"'{flag}'" in text, (script, flag)


def test_worker_unit_contract():
    parsed = unit("facade-worker.service")
    assert one(parsed, "Service", "StandardInput") == "file:/etc/facade/worker.bearer"
    assert "facade-models.service" in " ".join(values(parsed, "Unit", "Requires"))
    assert "facade-models.service" in " ".join(values(parsed, "Unit", "After"))
    assert one(parsed, "Service", "Restart") == "always"
    assert one(parsed, "Service", "User") == "facade"
    assert one(parsed, "Unit", "StartLimitIntervalSec") == "0"
    env = environment(parsed)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    exec_start = one(parsed, "Service", "ExecStart")
    assert "--max-jobs" not in exec_start and "--deadline-epoch" not in exec_start
    assert one(parsed, "Service", "LimitCORE") == "0"
    assert one(parsed, "Install", "WantedBy") == "multi-user.target"
    assert_creates_nvidia_uvm_as_root_first(parsed)


def assert_creates_nvidia_uvm_as_root_first(parsed):
    """NoNewPrivileges blocks CUDA's setuid nvidia-modprobe, so after a reboot or
    stop/start /dev/nvidia-uvm must be created by a privileged (+), non-fatal (-)
    pre-start step, before any other pre-start step."""
    assert one(parsed, "Service", "NoNewPrivileges") == "yes"
    pres = values(parsed, "Service", "ExecStartPre")
    assert pres and pres[0] == "-+/usr/bin/nvidia-modprobe -u -c=0", pres
    assert "nvidia-persistenced.service" in " ".join(values(parsed, "Unit", "After"))


def test_demo_unit_contract():
    parsed = unit("facade-demo.service")
    assert "facade-models.service" in " ".join(values(parsed, "Unit", "Requires"))
    assert "facade-models.service" in " ".join(values(parsed, "Unit", "After"))
    assert one(parsed, "Service", "Restart") == "always"
    assert one(parsed, "Service", "User") == "facade"
    env = environment(parsed)
    assert env["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["GRADIO_TEMP_DIR"] == "/var/tmp/facade-gradio"  # fallback default
    assert one(parsed, "Service", "EnvironmentFile") == "-/run/facade-demo.env"
    assert_creates_nvidia_uvm_as_root_first(parsed)
    (pre,) = [line for line in values(parsed, "Service", "ExecStartPre") if line.startswith("+/bin/sh -c ")]
    script = shlex.split(systemd_unescape(pre[1:]))[2]
    assert "mountpoint -q /opt/dlami/nvme" in script
    assert "d=/opt/dlami/nvme/facade-gradio" in script and "d=/var/tmp/facade-gradio" in script
    assert 'echo "GRADIO_TEMP_DIR=$d" > /run/facade-demo.env' in script
    shell = shutil.which("bash") or shutil.which("sh")
    subprocess.run([shell, "-n", "-c", script], check=True, stdin=subprocess.DEVNULL)
    assert "PrivateTmp" not in {key for key, _ in parsed["Service"]}, "private /var/tmp would hide the fallback dir"
    assert one(parsed, "Install", "WantedBy") == "multi-user.target"


def test_web_unit_contract():
    parsed = unit("facade-web.service")
    assert one(parsed, "Service", "User") == "facade"
    assert one(parsed, "Service", "WorkingDirectory") == "/opt/facade/src/web"
    assert one(parsed, "Service", "EnvironmentFile") == "/etc/facade/web.env"
    assert one(parsed, "Service", "Restart") == "always"
    assert one(parsed, "Unit", "StartLimitIntervalSec") == "0"
    env = environment(parsed)
    assert env["NEXT_TELEMETRY_DISABLED"] == "1"
    assert env["PATH"].startswith("/opt/facade/node/bin:")
    assert one(parsed, "Service", "NoNewPrivileges") == "yes"
    assert one(parsed, "Service", "LimitCORE") == "0"
    assert one(parsed, "Install", "WantedBy") == "multi-user.target"
    assert "127.0.0.1" in one(parsed, "Service", "ExecStart") and "0.0.0.0" not in (UNITS / "facade-web.service").read_text()


def test_web_scripts_exist_for_the_unit_and_bootstrap():
    package = json.loads((REPO / "web" / "package.json").read_text())
    assert "build" in package["scripts"]
    assert package["dependencies"].get("next"), "facade-web runs node_modules/next/dist/bin/next"


def test_models_unit_contract():
    parsed = unit("facade-models.service")
    assert one(parsed, "Service", "Type") == "oneshot"
    assert one(parsed, "Service", "RemainAfterExit") == "yes"
    assert one(parsed, "Unit", "ConditionPathExists") == f"!{SNAPSHOT}/.verified"
    assert "HF_HUB_OFFLINE" not in environment(parsed)
    assert one(parsed, "Service", "ReadWritePaths") == "/srv/facade/models"
    assert one(parsed, "Install", "WantedBy") == "multi-user.target"


def test_health_units_contract():
    service = unit("facade-health.service")
    assert one(service, "Service", "Type") == "oneshot"
    assert one(service, "Service", "User") == "facade"
    assert one(service, "Service", "ExecStart") == HEALTH_EXEC
    assert one(service, "Service", "EnvironmentFile") == "-/etc/facade/host.env"
    assert "Install" not in service  # started only by the timer
    timer = unit("facade-health.timer")
    assert one(timer, "Timer", "OnUnitActiveSec") in {"60", "60s", "1min"}
    assert one(timer, "Timer", "Unit") == "facade-health.service"
    assert one(timer, "Install", "WantedBy") == "timers.target"


def test_caddy_dropin_supplies_host():
    parsed = unit("caddy-facade.conf")
    assert one(parsed, "Service", "EnvironmentFile") == "/etc/facade/host.env"


# --- Caddyfile ---------------------------------------------------------------------

def _caddy_lines():
    return [line.split("#", 1)[0].rstrip() for line in (DEPLOY / "Caddyfile").read_text().splitlines()
            if line.split("#", 1)[0].strip()]


def _block(lines, opener):
    start = next(i for i, line in enumerate(lines) if line.strip() == opener)
    depth, body = 0, []
    for line in lines[start:]:
        depth += line.count("{") - line.count("}")
        body.append(line.strip())
        if depth == 0:
            return start, body
    raise AssertionError("unbalanced block " + opener)


def test_caddyfile_routes_v1_demo_and_website():
    lines = _caddy_lines()
    text = "\n".join(lines)
    assert text.count("{") == text.count("}")
    assert lines[0].strip() == "{$FACADE_HOST} {"
    v1_start, v1_body = _block(lines, "handle /v1/* {")
    demo_start, demo_body = _block(lines, "handle /photoguard/* {")
    web_start, web_body = _block(lines, "handle {")
    assert v1_start < demo_start < web_start, "specific routes before the catch-all"
    # /v1: the worker's bearer protects it; no basic auth in front.
    assert "reverse_proxy 127.0.0.1:8000 {" in v1_body
    assert not any("basic_auth" in line or "import" in line for line in v1_body)
    # /photoguard/*: basic auth, then the Gradio demo, path passed through unchanged.
    assert demo_body.index("import /etc/caddy/facade-auth.caddy") < demo_body.index("reverse_proxy 127.0.0.1:7860 {")
    assert "flush_interval -1" in demo_body
    assert "redir /photoguard /photoguard/ 308" in [line.strip() for line in lines]
    assert not any(line.strip().startswith(("handle_path", "uri strip_prefix")) for line in lines)
    # everything else: basic auth, then the website.
    assert web_body.index("import /etc/caddy/facade-auth.caddy") < web_body.index("reverse_proxy 127.0.0.1:3000")
    assert text.count("import /etc/caddy/facade-auth.caddy") == 2
    assert not any(re.match(r"\s*log\b", line) for line in lines), "no access log"
    assert "basic_auth" not in text, "the hash block is imported, never inlined"


def test_caddyfile_body_limits_are_per_route_and_match_the_demo():
    """/v1 and the website keep 12MB; the demo takes full-resolution editor PNGs (64MB), in step with app.py."""
    lines = _caddy_lines()
    _, v1_body = _block(lines, "handle /v1/* {")
    _, demo_body = _block(lines, "handle /photoguard/* {")
    _, web_body = _block(lines, "handle {")
    site_level = [line for line in lines if line.strip().startswith("request_body")]
    assert len(site_level) == 3, "one request_body per handle, none at site level"
    assert "max_size 12MB" in v1_body and "request_buffers 12MB" in v1_body
    assert "max_size 12MB" in web_body
    app_source = (SERVER / "photoguard_demo" / "app.py").read_text()
    match = re.search(r'^MAX_FILE_SIZE = "(\d+)mb"', app_source, re.M)
    assert match and f"max_size {match.group(1)}MB" in demo_body
    assert "max_file_size=MAX_FILE_SIZE" in app_source


def test_privacy_page_is_the_only_public_route_and_is_static():
    """Apple reviewers must reach /privacy without the password; it serves one static file, nothing else."""
    lines = _caddy_lines()
    privacy_start, privacy_body = _block(lines, "handle /privacy {")
    web_start, _ = _block(lines, "handle {")
    assert privacy_start < web_start, "public page before the catch-all"
    assert privacy_body == ["handle /privacy {", "root * /opt/facade/src/deploy/public",
                            "rewrite * /privacy.html", "file_server", "}"]
    assert not any("reverse_proxy" in line or "import" in line for line in privacy_body)
    assert "redir /privacy/ /privacy 308" in [line.strip() for line in lines]
    page = (DEPLOY / "public" / "privacy.html").read_text()
    assert "<title>FACADE Photo Shield: Privacy Policy</title>" in page
    assert "on your iPhone" in page and "never used to train" in page
    assert not re.search(r"<script|src=\"http", page), "self-contained: no scripts or remote assets"
    assert sorted(p.name for p in (DEPLOY / "public").iterdir()) == ["privacy.html"]


def test_demo_root_path_matches_the_caddy_route():
    demo = one(unit("facade-demo.service"), "Service", "ExecStart")
    root = demo.split("--root-path ", 1)[1].split()[0]
    assert f"handle {root}/* {{" in [line.strip() for line in _caddy_lines()]
    assert f"http://127.0.0.1:7860{root}/" in (SERVER / "health.py").read_text()


# --- model inventory -------------------------------------------------------------------

def test_model_sums_well_formed_and_complete():
    raw = (MODELS / "MODEL_SHA256SUMS").read_text()
    assert raw.endswith("\n") and "\r" not in raw
    sums = fetch_models.parse_sums(MODELS / "MODEL_SHA256SUMS")
    assert list(sums) == sorted(sums)
    for relative, digest in diffusion.SOURCE_FILES.items():
        assert sums[relative] == digest, relative
    for relative, digest in encoder.SOURCE_FILES.items():
        assert sums[relative] == digest, relative
    assert {k: v for k, v in sums.items() if k not in diffusion.SOURCE_FILES} == EXTRA_MODEL_FILES
    assert len(sums) == len(diffusion.SOURCE_FILES) + 4 == 15
    assert fetch_models.REVISION == diffusion.MODEL_REVISION == encoder.MODEL_REVISION
    assert fetch_models.REPO_ID == diffusion.MODEL_ID == encoder.MODEL_ID


def test_model_sums_match_local_snapshot_small_files():
    if not LOCAL_SNAPSHOT.is_dir():
        pytest.skip("pinned local snapshot not present")
    sums = fetch_models.parse_sums(MODELS / "MODEL_SHA256SUMS")
    for relative, digest in sums.items():
        path = LOCAL_SNAPSHOT / relative
        if path.stat().st_size <= 5_000_000:
            assert fetch_models.sha256_file(path) == digest, relative


def test_requirements_lock_is_the_pinned_gpu_lock():
    lock = SERVER / "requirements-gpu.lock.txt"
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == LOCK_SHA256
    pins = dict(line.split("==") for line in lock.read_text().split())
    assert pins["diffusers"] == "0.40.0" and pins["transformers"] == "5.16.1"
    assert "torch" not in pins and "gradio" not in pins  # bootstrap.sh pins those two
    # gradio 6.28.0's own dependencies, as resolved with `pip install gradio==6.28.0 -c lock`, so
    # nothing floats on the server; pi-heif decodes iPhone HEIC uploads in the demo.
    assert (pins["fastapi"], pins["starlette"], pins["uvicorn"]) == ("0.141.1", "1.7.0", "0.53.0")
    assert pins["gradio-client"] == "2.7.1" and pins["pi-heif"] == "1.4.0"
    assert list(pins) == sorted(pins) and all(name == name.lower() for name in pins)


# --- bootstrap.sh --------------------------------------------------------------------------

BOOTSTRAP = DEPLOY / "bootstrap.sh"


def test_bootstrap_bash_syntax():
    subprocess.run(["bash", "-n", str(BOOTSTRAP)], check=True, stdin=subprocess.DEVNULL)


def test_bootstrap_shellcheck():
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    subprocess.run(["shellcheck", "-s", "bash", str(BOOTSTRAP)], check=True, stdin=subprocess.DEVNULL)


def test_bootstrap_contract_steps():
    text = BOOTSTRAP.read_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert not re.search(r"set\s+-[a-z]*x|set -o xtrace|bash -x", text), "xtrace would print secrets"
    for needle in [
        "/var/log/facade-bootstrap.log", "python3.12-venv", "DPkg::Lock::Timeout", "apt-mark hold",
        "65760C51EDEA2017CEA2CA15155B6D79CA56EA34", "caddy-stable-archive-keyring.gpg",
        "useradd --system", "--shell /usr/sbin/nologin facade",
        "https://download.pytorch.org/whl/cu128", "torch==2.8.0 torchvision==0.23.0", "gradio==6.28.0",
        '--no-deps -r "$LOCK"', '-c "$LOCK"', "requirements-aws.lock.txt",
        "SSM_PREFIX=/facade/prod", '"$SSM_PREFIX/processor-token"', '"$SSM_PREFIX/basic-auth-hash"',
        '"$SSM_PREFIX/host"', "--region \"$REGION\"", "REGION=us-east-1", "--with-decryption",
        "/etc/facade/worker.bearer", "chmod 0400", "chown facade:facade",
        "/etc/caddy/facade-auth.caddy", "chown root:caddy", "chmod 0640", "/etc/facade/host.env",
        "SystemMaxUse=1G", "kernel.core_pattern=|/bin/false", "systemctl daemon-reload",
        "/etc/systemd/system/caddy.service.d", "/etc/caddy/Caddyfile", "caddy validate",
        'SRC=/opt/facade/src', 'DEPLOY="$SRC/deploy"', 'SERVER="$SRC/server"', 'WEB="$SRC/web"',
        "/etc/facade/web.env", "FACADE_PROCESSOR_URL=http://127.0.0.1:8000", "FACADE_PUBLIC_ORIGIN=https://%s",
        "NODE_ENV=production", "npm ci --no-audit --no-fund", "npm run build", "NPM_CACHE=/opt/facade/.npm",
        "runuser -u facade", "NEXT_TELEMETRY_DISABLED=1",
    ]:
        assert needle in text, needle
    enable = re.search(r"systemctl enable ([^\n]*\\\n[^\n]*)", text).group(1)
    for name in ["facade-models.service", "facade-worker.service", "facade-demo.service", "facade-web.service",
                 "facade-health.timer", "caddy.service"]:
        assert name in enable, name
    for name in UNIT_FILES:
        assert name in text, name
    for path in ["$DEPLOY/Caddyfile", "$SERVER/requirements-gpu.lock.txt", "$SERVER/health.py",
                 "$SERVER/models/MODEL_SHA256SUMS", "$SERVER/models/fetch_models.py"]:
        assert f'"{path}"' in text, path


def test_bootstrap_pins_node_22_lts_by_version_and_sha256():
    text = BOOTSTRAP.read_text()
    version = re.search(r"^NODE_VERSION=(v\d+\.\d+\.\d+)$", text, re.M).group(1)
    digest = re.search(r"^NODE_SHA256=([0-9a-f]{64})$", text, re.M).group(1)
    assert version.startswith("v22.") and len(digest) == 64
    assert 'NODE_TARBALL="node-$NODE_VERSION-linux-x64.tar.xz"' in text
    assert 'NODE_URL="https://nodejs.org/dist/$NODE_VERSION/$NODE_TARBALL"' in text
    check = text.index("sha256sum --check --strict")
    unpack = text.index('tar -xJf "$nodetmp/$NODE_TARBALL"')
    assert check < unpack, "verify the tarball before unpacking it"
    assert "NODE_DIR=/opt/facade/node" in text


def test_bootstrap_builds_the_website_as_facade_without_secrets():
    text = BOOTSTRAP.read_text()
    build = text[text.index("# --- 5. website build"):text.index("# --- 6. units")]
    assert "env -i" in build, "a clean environment: web.env's token never reaches the build"
    assert "web.env" not in build.split("runuser", 1)[1].split("||", 1)[0]
    assert "systemctl stop facade-web.service" in build
    # web/ belongs to facade only inside the rebuild branch; afterwards it is
    # root-owned again, apart from the cache `next start` writes.
    rebuild = build[build.index("\nelse\n"):build.index("\nfi\n")]
    assert 'chown -R facade:facade "$WEB"' in rebuild
    after = build[build.index("\nfi\n"):]
    assert 'chown -R root:root "$WEB"' in after
    assert after.index('chown -R root:root "$WEB"') < after.index('install -d -m 0700 -o facade -g facade "$WEB/.next/cache"')
    # web/ is kept out of the step-0 chown; step 5 sets its ownership.
    assert "! -name web -exec chown -R root:root" in text


def test_services_cannot_write_the_code():
    for name in ("facade-worker.service", "facade-demo.service", "facade-web.service"):
        assert one(unit(name), "Service", "ReadOnlyPaths") == "/opt/facade", name
    assert one(unit("facade-web.service"), "Service", "ReadWritePaths") == "-/opt/facade/src/web/.next/cache"


def test_bootstrap_validates_a_staged_caddyfile_before_replacing_the_live_one():
    text = BOOTSTRAP.read_text()
    validate = text.index('caddy validate --config "$staged"')
    replace = text.index('mv -f "$staged" /etc/caddy/Caddyfile')
    assert validate < replace
    assert not re.search(r'install[^\n]*"\$DEPLOY/Caddyfile" /etc/caddy/Caddyfile', text), "never install unvalidated"
    assert text.count("/etc/caddy/Caddyfile") >= 1


def test_bootstrap_keeps_needrestart_from_restarting_the_services():
    text = BOOTSTRAP.read_text()
    assert "/etc/needrestart/conf.d/60-facade.conf" in text
    assert "$nrconf{override_rc}{qr(^facade-)} = 0;" in text
    assert "$nrconf{override_rc}{qr(^caddy)} = 0;" in text
    assert "<<'NRCONF'" in text, "quoted heredoc: perl sigils must reach the file unexpanded"


def test_bootstrap_never_prints_secret_values():
    for number, line in enumerate(BOOTSTRAP.read_text().splitlines(), 1):
        if re.search(r"\$\{?(token|hash)\b", line):
            stripped = line.strip()
            assert not stripped.startswith(("log ", "die ", "echo ")), (number, line)
            if stripped.startswith("printf"):
                assert stripped.endswith('> "$tmp"'), (number, line)


# --- fetch_models.py ------------------------------------------------------------------------

def fake_sums():
    content = {relative: f"fake bytes for {relative}\n".encode() for relative in
               fetch_models.parse_sums(MODELS / "MODEL_SHA256SUMS")}
    return content, {relative: hashlib.sha256(data).hexdigest() for relative, data in content.items()}


def fake_downloader(content, calls, corrupt=()):
    def download(relative, staging, force):
        calls.append((relative, force))
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"corrupt" if (relative, len([c for c in calls if c[0] == relative])) in corrupt
                           else content[relative])
    return download


def test_fetch_publishes_read_only_verified_tree(tmp_path):
    content, sums = fake_sums()
    calls = []
    dest = tmp_path / "models" / "sd15-inpaint-8a4288"
    fetch_models.publish(sums, dest, attempts=[("fake", fake_downloader(content, calls))], sleep=lambda _: None)
    files = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert files == set(sums) | {".verified"}
    marker = json.loads((dest / ".verified").read_text())
    assert marker["revision"] == fetch_models.REVISION and marker["files"] == 15
    for path in [dest, *dest.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == (0o555 if path.is_dir() else 0o444), path
    assert sorted(parent.name for parent in dest.parent.iterdir()) == [dest.name]  # no staging left
    fetch_models.verify_tree(dest, sums)
    assert len(calls) == 15


def test_published_tree_is_accepted_by_both_hash_bound_loaders(tmp_path, monkeypatch):
    content, sums = fake_sums()
    dest = tmp_path / "sd15-inpaint-8a4288"
    fetch_models.publish(sums, dest, attempts=[("fake", fake_downloader(content, []))], sleep=lambda _: None)
    monkeypatch.setattr(diffusion, "SOURCE_FILES", {k: sums[k] for k in diffusion.SOURCE_FILES})
    monkeypatch.setattr(encoder, "SOURCE_FILES", {k: sums[k] for k in encoder.SOURCE_FILES})
    # Extras (model_index.json, safety_checker/, feature_extractor/, .verified) sit
    # outside the component directories the loaders inventory, so both accept them.
    assert diffusion.verify_local_snapshot(dest) == dest.resolve()
    assert encoder.verify_local_snapshot(dest) == dest.resolve()


def test_fetch_retries_a_corrupt_download_then_succeeds(tmp_path):
    content, sums = fake_sums()
    calls = []
    corrupt = {("unet/config.json", 1)}
    dest = tmp_path / "snap"
    fetch_models.publish(sums, dest, attempts=[("fake", fake_downloader(content, calls, corrupt))] * 2,
                         sleep=lambda _: None)
    assert [c for c in calls if c[0] == "unet/config.json"] == [("unet/config.json", False),
                                                                ("unet/config.json", True)]
    fetch_models.verify_tree(dest, sums)


def test_fetch_never_publishes_unverified_bytes(tmp_path):
    content, sums = fake_sums()
    corrupt = {("vae/config.json", 1), ("vae/config.json", 2)}
    dest = tmp_path / "snap"
    with pytest.raises(RuntimeError, match="vae/config.json"):
        fetch_models.publish(sums, dest, attempts=[("fake", fake_downloader(content, [], corrupt))] * 2,
                             sleep=lambda _: None)
    assert not dest.exists()


def test_fetch_resumes_from_verified_staging_and_skips_when_verified(tmp_path):
    content, sums = fake_sums()
    dest = tmp_path / "snap"
    staged = tmp_path / ".staging-snap" / "unet" / "config.json"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(content["unet/config.json"])
    calls = []
    fetch_models.publish(sums, dest, attempts=[("fake", fake_downloader(content, calls))], sleep=lambda _: None)
    assert "unet/config.json" not in {c[0] for c in calls} and len(calls) == 14
    again = []
    fetch_models.publish(sums, dest, attempts=[("fake", fake_downloader(content, again))], sleep=lambda _: None)
    assert again == []


def test_fetch_replaces_a_stale_unverified_tree(tmp_path):
    content, sums = fake_sums()
    dest = tmp_path / "snap"
    (dest / "vae").mkdir(parents=True)
    (dest / "vae" / "junk.bin").write_bytes(b"junk")
    fetch_models.publish(sums, dest, attempts=[("fake", fake_downloader(content, []))], sleep=lambda _: None)
    assert not (dest / "vae" / "junk.bin").exists()
    fetch_models.verify_tree(dest, sums)


@pytest.mark.parametrize("line", [
    "A" * 64 + "  vae/config.json",
    "a" * 64 + " vae/config.json",
    "a" * 64 + "  ../etc/passwd",
    "a" * 64 + "  /abs/path",
    "a" * 64 + "  .verified",
    "a" * 64 + "  .cache/x",
])
def test_parse_sums_rejects_bad_lines(tmp_path, line):
    path = tmp_path / "SUMS"
    path.write_text(line + "\n")
    with pytest.raises(ValueError):
        fetch_models.parse_sums(path)


def test_parse_sums_rejects_duplicates(tmp_path):
    path = tmp_path / "SUMS"
    path.write_text(("a" * 64 + "  vae/config.json\n") * 2)
    with pytest.raises(ValueError, match="twice"):
        fetch_models.parse_sums(path)


@pytest.mark.parametrize("path", [MODELS / "fetch_models.py", SERVER / "health.py", DEPLOY / "facade_server.py"],
                         ids=["fetch_models", "health", "facade_server"])
def test_kit_python_is_stdlib_only_at_import(path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            assert root == "__future__" or root in sys.stdlib_module_names, (path.name, name)


# --- health.py ------------------------------------------------------------------------------

BEARER = "0123456789abcdef" * 4
GOOD_CAPS = {"ready": True, "recipe": "photoguard-encoder-linf-v1", "image_side": 512, "total_iterations": 200,
             "retention_seconds": 1800, "experimental": True, "gpu_smoke_validated": True,
             "engine_smoke_validated": True, "execution_device": "cuda:0"}


def test_health_evaluates_the_contract_capabilities():
    assert health.evaluate_capabilities(200, json.dumps(GOOD_CAPS).encode())["ok"]
    smoke_pending = dict(GOOD_CAPS, ready=False, gpu_smoke_validated=False)
    result = health.evaluate_capabilities(200, json.dumps(smoke_pending).encode())
    assert not result["ok"] and result["mismatched"] == ["gpu_smoke_validated", "ready"]
    assert not health.evaluate_capabilities(200, json.dumps(dict(GOOD_CAPS, image_side=512.0)).encode())["ok"]
    assert not health.evaluate_capabilities(401, b"")["ok"]
    assert not health.evaluate_capabilities(200, b"not json")["ok"]
    assert health.evaluate_page(200)["ok"] and health.evaluate_page(308)["ok"] and not health.evaluate_page(502)["ok"]


def test_health_bearer_format(tmp_path):
    path = tmp_path / "bearer"
    path.write_text(BEARER + "\n")
    assert health.read_bearer(path) == BEARER
    for bad in [BEARER + "\n\n", BEARER.upper(), BEARER[:-1], " " + BEARER]:
        path.write_text(bad)
        with pytest.raises(ValueError):
            health.read_bearer(path)


def test_health_main_checks_loopback_and_public_without_leaking_bearer(tmp_path, monkeypatch, capsys):
    bearer_file = tmp_path / "bearer"
    bearer_file.write_text(BEARER + "\n")
    seen = []

    def fake_get(url, headers=None, timeout=10.0):
        seen.append((url, dict(headers or {})))
        if url in {"http://127.0.0.1:3000/", "http://127.0.0.1:7860/photoguard/"}:
            return 200, b"<html>"
        return 200, json.dumps(GOOD_CAPS).encode()

    monkeypatch.setattr(health, "http_get", fake_get)
    monkeypatch.setattr(health, "gpu_info", lambda: None)
    assert health.main(["--bearer-file", str(bearer_file), "--public-host", "facade.example.com"]) == 0
    out = capsys.readouterr().out
    assert BEARER not in out
    report = json.loads(out)
    assert report["facade_health"] == "ok" and set(report["checks"]) == {"web", "demo", "worker", "public"}
    urls = [url for url, _ in seen]
    assert urls == ["http://127.0.0.1:3000/", "http://127.0.0.1:7860/photoguard/",
                    "http://127.0.0.1:8000/v1/capabilities", "https://facade.example.com/v1/capabilities"]
    assert seen[0][1] == {} and seen[1][1] == {}  # the pages never receive the worker bearer
    assert all(headers == {"Authorization": "Bearer " + BEARER} for _, headers in seen[2:])


def test_health_main_fails_when_worker_is_not_ready(tmp_path, monkeypatch, capsys):
    bearer_file = tmp_path / "bearer"
    bearer_file.write_text(BEARER)

    def fake_get(url, headers=None, timeout=10.0):
        if url.endswith("/v1/capabilities"):
            return 200, json.dumps(dict(GOOD_CAPS, ready=False)).encode()
        raise ConnectionRefusedError()

    monkeypatch.setattr(health, "http_get", fake_get)
    monkeypatch.setattr(health, "gpu_info", lambda: None)
    assert health.main(["--bearer-file", str(bearer_file), "--no-public"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["demo"] == {"ok": False, "error": "ConnectionRefusedError"}
    assert report["checks"]["web"] == {"ok": False, "error": "ConnectionRefusedError"}
    assert report["checks"]["worker"]["mismatched"] == ["ready"]
    assert report["checks"]["public"] == {"ok": True, "skipped": True}


# --- license files --------------------------------------------------------------------------

def test_third_party_notices_point_at_license_files_that_exist():
    notices = (REPO / "THIRD_PARTY_NOTICES.md").read_text()
    linked = set(re.findall(r"\(licenses/([^)\s]+)\)", notices))
    present = {path.name for path in (REPO / "licenses").iterdir() if path.is_file()}
    assert linked and linked <= present, linked - present
    assert present <= linked, f"license files nobody cites: {present - linked}"
    assert "CreativeML Open RAIL-M" in (REPO / "licenses" / "CreativeML-OpenRAIL-M.txt").read_text()


def test_photoguard_license_copies_are_identical():
    assert (REPO / "licenses" / "photoguard-MIT.txt").read_bytes() == \
        (SERVER / "photoguard_demo" / "LICENSE").read_bytes()
