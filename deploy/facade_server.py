#!/usr/bin/env python3
"""Deploy to and look after the FACADE server (one AWS GPU instance).

    python3 deploy/facade_server.py status              # is it up? is the worker ready?
    python3 deploy/facade_server.py deploy              # push this checkout and run bootstrap.sh
    python3 deploy/facade_server.py logs --service web  # recent journal lines
    python3 deploy/facade_server.py restart --service demo
    python3 deploy/facade_server.py ssh [--forward 3000]

What it needs on your computer: the AWS CLI v2 with a profile named `facade`
that holds an access key of the IAM user `facade`, the AWS Session Manager
plugin, OpenSSH, curl and git (rsync is optional). Python 3.9+ standard
library only.

What it does NOT do: create, resize or delete AWS resources, or change
secrets. It only finds the one running instance tagged Project=FACADE and
talks to it through AWS Systems Manager (there is no open SSH port).

Secrets stay secret: the worker bearer is read from SSM into memory and handed
to curl on stdin, never put in argv or printed. deploy/README.md is the runbook.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# ---------------------------------------------------------------------------
# 1. The fixed AWS surface. The IAM policy of the `facade` user allows this.
# ---------------------------------------------------------------------------

PROFILE = "facade"
REGION = "us-east-1"
ACCOUNT_ID = "669707689345"
EXPECTED_ARN = f"arn:aws:iam::{ACCOUNT_ID}:user/facade"
PROJECT_KEY = "Project"
PROJECT_VALUE = "FACADE"
INSTANCE_NAME = "facade-box"
OS_USER = "ubuntu"
LIVE_STATES = ("pending", "running", "stopping", "stopped")

TOKEN_PARAMETER = "/facade/prod/processor-token"
HOST_PARAMETER = "/facade/prod/host"

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_SRC = "/opt/facade/src"
BOOTSTRAP_REL = "deploy/bootstrap.sh"
BOOTSTRAP_UNIT = "facade-bootstrap.service"
#: Files the server cannot run without; deploy refuses a checkout that lacks one.
REQUIRED_FILES = (
    BOOTSTRAP_REL,
    "deploy/Caddyfile",
    "server/worker/facade_photoguard_service.py",
    "server/photoguard_demo/app.py",
    "server/models/fetch_models.py",
    "server/health.py",
    "web/package.json",
    "web/package-lock.json",
)
#: Built on the server by bootstrap.sh; `rsync --delete` must leave them alone.
REMOTE_BUILD_OUTPUTS = ("/web/node_modules/", "/web/.next/")
MANIFEST_NAME = ".deploy-sha256sums"
INFO_NAME = ".deploy-info.json"
# Run on the server after a tar-stream push. Manifest lines are "<64 hex>  <path>",
# so the path starts at column 67; build_bundle refuses names with newlines.
PRUNE_UNLISTED_FILES = (
    f"set -eu; cd {REMOTE_SRC}; keep=$(mktemp); trap 'rm -f \"$keep\"' EXIT; "
    f"{{ cut -c67- {MANIFEST_NAME}; printf '%s\\n' {MANIFEST_NAME} {INFO_NAME}; }} | LC_ALL=C sort > \"$keep\"; "
    "find . \\( -path ./web/node_modules -o -path ./web/.next \\) -prune -o ! -type d -print | cut -c3- "
    "| LC_ALL=C sort | LC_ALL=C comm -23 - \"$keep\" | tr '\\n' '\\0' | xargs -0 -r rm -f --; "
    "find . -mindepth 1 -type d -empty ! -path './web/node_modules*' ! -path './web/.next*' -delete"
)
MAX_FILE_BYTES = 10 * 1024 * 1024  # a model weight never belongs in the code bundle
MAX_BUNDLE_BYTES = 100 * 1024 * 1024

#: What the website requires of /v1/capabilities before it sends a job.
EXPECTED_CAPABILITIES = {
    "ready": True,
    "gpu_smoke_validated": True,
    "recipe": "photoguard-encoder-linf-v1",
    "image_side": 512,
    "total_iterations": 200,
    "retention_seconds": 1800,
    "experimental": True,
}

#: `--service` names and the systemd units they mean.
SERVICES = {
    "worker": ["facade-worker"],
    "demo": ["facade-demo"],
    "web": ["facade-web"],
    "caddy": ["caddy"],
    "models": ["facade-models"],
    "health": ["facade-health"],
    "bootstrap": ["facade-bootstrap"],
}
ALL_SERVICES = ["facade-worker", "facade-demo", "facade-web", "caddy"]
RESTARTABLE = ("worker", "demo", "web", "caddy", "all")
FORWARD_PORTS = (3000, 7860, 8000)
POLL_SECONDS = 10.0

CREDENTIALS_HINT = (
    "the facade access key is invalid or expired: rerun `aws configure --profile facade` "
    "(region us-east-1) with a current access key of the IAM user facade"
)
ERROR_HINTS = {
    "ExpiredToken": CREDENTIALS_HINT,
    "InvalidClientTokenId": CREDENTIALS_HINT,
    "SignatureDoesNotMatch": CREDENTIALS_HINT,
    "AccessDenied": "the facade IAM policy refused this call; check that you use --profile facade",
}


def _sleep(seconds: float) -> None:  # tests replace this
    time.sleep(seconds)


def _monotonic() -> float:  # tests may replace this
    return time.monotonic()


_exec = os.execvp  # `ssh` hands the terminal to the Session Manager client


class Refused(RuntimeError):
    """A deliberate refusal: nothing was changed."""


class RemoteFailed(RuntimeError):
    """A command on the server failed."""


class AwsCallFailed(RuntimeError):
    def __init__(self, operation: str, stderr: str, *, timed_out: bool = False) -> None:
        self.operation = operation
        self.stderr = (stderr or "").strip()
        self.timed_out = timed_out
        hint = next((text for code, text in ERROR_HINTS.items() if code in self.stderr), "")
        super().__init__(
            f"aws {operation} failed: {self.stderr[-1500:] or 'no error text'}"
            + (f"\n  hint: {hint}" if hint else "")
        )

    def has(self, *codes: str) -> bool:
        return any(code in self.stderr for code in codes)


# ---------------------------------------------------------------------------
# 2. AWS CLI access: always the facade profile and us-east-1, stdin detached.
# ---------------------------------------------------------------------------


def aws(service: str, operation: str, *args: str, timeout: float = 300) -> dict[str, Any]:
    """Run one AWS CLI operation and return its parsed JSON output."""
    argv = ["aws", service, operation, *args, "--output", "json", "--profile", PROFILE, "--region", REGION]
    try:
        completed = subprocess.run(
            argv, stdin=subprocess.DEVNULL, check=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.CalledProcessError as error:
        raise AwsCallFailed(f"{service} {operation}", error.stderr or "") from None
    except subprocess.TimeoutExpired:
        raise AwsCallFailed(f"{service} {operation}", f"timed out after {timeout:.0f}s", timed_out=True) from None
    text = completed.stdout.strip()
    return json.loads(text) if text else {}


def require_tools(*names: str) -> None:
    hints = {
        "aws": "install AWS CLI v2 (macOS: brew install awscli)",
        "session-manager-plugin": "macOS: brew install --cask session-manager-plugin",
        "ssh": "OpenSSH client",
        "ssh-keygen": "OpenSSH client",
        "curl": "curl",
        "git": "git",
    }
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise Refused("missing tools: " + ", ".join(f"{name} ({hints.get(name, 'install it')})" for name in missing))


def require_identity() -> dict[str, Any]:
    """Refuse unless the facade profile is exactly the facade IAM user in the FACADE account."""
    identity = aws("sts", "get-caller-identity")
    arn = str(identity.get("Arn", ""))
    account = str(identity.get("Account", ""))
    if arn != EXPECTED_ARN or account != ACCOUNT_ID:
        raise Refused(
            f"AWS profile '{PROFILE}' is {arn or 'an unknown principal'} (account {account or '?'}). "
            f"This tool only runs as {EXPECTED_ARN}: never as root, a role or another account. "
            "Fix it with `aws configure --profile facade` (region us-east-1) and the facade user's access key."
        )
    return identity


def tag_map(resource: Mapping[str, Any]) -> dict[str, str]:
    return {str(tag.get("Key")): str(tag.get("Value")) for tag in resource.get("Tags") or []}


def require_project_tag(instance: Mapping[str, Any]) -> None:
    if tag_map(instance).get(PROJECT_KEY) != PROJECT_VALUE:
        raise Refused(
            f"refusing to act on instance {instance.get('InstanceId', '?')}: "
            f"it lacks the tag {PROJECT_KEY}={PROJECT_VALUE}"
        )


def find_instance() -> dict[str, Any] | None:
    """The one live instance tagged Project=FACADE and Name=facade-box, or None."""
    filters = [
        {"Name": f"tag:{PROJECT_KEY}", "Values": [PROJECT_VALUE]},
        {"Name": "tag:Name", "Values": [INSTANCE_NAME]},
        {"Name": "instance-state-name", "Values": list(LIVE_STATES)},
    ]
    data = aws("ec2", "describe-instances", "--filters", json.dumps(filters))
    instances = [item for reservation in data.get("Reservations", []) for item in reservation.get("Instances", [])]
    for instance in instances:
        require_project_tag(instance)
    if len(instances) > 1:
        ids = ", ".join(str(item.get("InstanceId")) for item in instances)
        raise Refused(f"more than one live {INSTANCE_NAME} instance ({ids}); the owner must resolve this first")
    return instances[0] if instances else None


def running_instance() -> dict[str, Any]:
    instance = find_instance()
    if instance is None:
        raise Refused(f"no live {PROJECT_KEY}={PROJECT_VALUE} {INSTANCE_NAME} instance (the owner launches it)")
    state = instance_state(instance)
    if state != "running":
        raise Refused(f"{instance['InstanceId']} is {state}, not running (the owner starts it)")
    return instance


def instance_state(instance: Mapping[str, Any]) -> str:
    return str(instance.get("State", {}).get("Name", "unknown"))


def get_parameter(name: str, *, decrypt: bool = False) -> dict[str, Any] | None:
    args = ["--name", name, *(["--with-decryption"] if decrypt else [])]
    try:
        return aws("ssm", "get-parameter", *args).get("Parameter")
    except AwsCallFailed as error:
        if error.has("ParameterNotFound"):
            return None
        raise


def read_processor_token() -> str:
    """The worker bearer, kept in memory only."""
    parameter = get_parameter(TOKEN_PARAMETER, decrypt=True)
    token = str((parameter or {}).get("Value", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise Refused(f"SSM {TOKEN_PARAMETER} is missing or not 64 lowercase hex")
    return token


_HOSTNAME = re.compile(r"(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+")


def read_host() -> str:
    parameter = get_parameter(HOST_PARAMETER)
    host = str((parameter or {}).get("Value", "")).strip().lower()
    if not _HOSTNAME.fullmatch(host):
        raise Refused(f"SSM {HOST_PARAMETER} is missing or not a DNS hostname")
    return host


def wait_for(description: str, probe: Callable[[], Any], *, timeout: float, interval: float = POLL_SECONDS) -> Any:
    deadline = _monotonic() + timeout
    while True:
        result = probe()
        if result:
            return result
        if _monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {description}")
        _sleep(interval)


def ssm_online(instance_id: str) -> bool:
    data = aws(
        "ssm", "describe-instance-information",
        "--filters", json.dumps([{"Key": "InstanceIds", "Values": [instance_id]}]),
    )
    items = data.get("InstanceInformationList", [])
    return bool(items) and items[0].get("PingStatus") == "Online"


def run_command(instance_id: str, script: str, *, comment: str, timeout: float = 180) -> dict[str, Any]:
    """Run a shell script on the server as root through SSM Run Command; return the invocation."""
    sent = aws(
        "ssm", "send-command",
        "--instance-ids", instance_id,
        "--document-name", "AWS-RunShellScript",
        "--comment", comment,
        "--parameters", json.dumps({"commands": [script]}),
    )
    command_id = str(sent["Command"]["CommandId"])

    def finished() -> dict[str, Any] | None:
        try:
            status = aws("ssm", "get-command-invocation", "--command-id", command_id, "--instance-id", instance_id)
        except AwsCallFailed as error:
            if error.has("InvocationDoesNotExist"):
                return None
            raise
        done = status.get("Status") in {"Success", "Failed", "Cancelled", "TimedOut"}
        return status if done else None

    return wait_for(f"'{comment}' on {instance_id}", finished, timeout=timeout, interval=2)


# ---------------------------------------------------------------------------
# 3. The code bundle: the repository's tracked files (`git ls-files`).
# ---------------------------------------------------------------------------


@dataclass
class Bundle:
    stage: Path
    files: list[str]
    info: dict[str, Any]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False,
    )
    if completed.returncode != 0:
        raise Refused(f"git {' '.join(args)} failed in {root}: {completed.stderr.strip()[-300:]}")
    return completed.stdout


def build_bundle(root: Path, stage: Path, *, allow_dirty: bool = False) -> Bundle:
    """Copy every tracked file into ``stage`` and write a sha256 manifest for the server to check."""
    root = Path(root).resolve()
    dirty = [line for line in _git(root, "status", "--porcelain", "--untracked-files=no").splitlines() if line]
    if dirty and not allow_dirty:
        raise Refused(
            f"{len(dirty)} tracked file(s) have uncommitted changes. Commit them first, or pass "
            "--allow-dirty to deploy the working copy as it is:\n  " + "\n  ".join(dirty[:20])
        )
    files: list[str] = []
    lines: list[str] = []
    total = 0
    for rel in sorted(set(filter(None, _git(root, "ls-files", "-z").split("\0")))):
        if "\n" in rel or "\\" in rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise Refused(f"refusing to deploy an unusual file name {rel!r}")
        source = root / rel
        if source.is_symlink():
            raise Refused(f"refusing to deploy the symlink {rel}")
        if not source.is_file():
            if allow_dirty:
                continue  # deleted in the working copy
            raise Refused(f"{rel} is tracked but missing")
        payload = source.read_bytes()
        if len(payload) > MAX_FILE_BYTES:
            raise Refused(f"{rel} is {len(payload)} bytes; large files (model weights?) never go to the server")
        total += len(payload)
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o755 if os.access(source, os.X_OK) else 0o644)
        files.append(rel)
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {rel}\n")
    missing = [rel for rel in REQUIRED_FILES if rel not in files]
    if missing:
        raise Refused("this checkout is missing files the server needs:\n  " + "\n  ".join(missing))
    if total > MAX_BUNDLE_BYTES:
        raise Refused(f"the code bundle is {total} bytes; refusing anything above {MAX_BUNDLE_BYTES}")
    manifest = "".join(lines).encode("utf-8")
    (stage / MANIFEST_NAME).write_bytes(manifest)
    info = {
        "commit": _git(root, "rev-parse", "HEAD").strip(),
        "uncommitted_changes": bool(dirty),
        "files": len(files),
        "bytes": total,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    (stage / INFO_NAME).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"bundle: {len(files)} files, {total} bytes, commit {info['commit'][:12]}"
        + (" plus uncommitted changes" if dirty else "")
    )
    return Bundle(stage=stage, files=files, info=info)


def bundle_tarball(bundle: Bundle) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for rel in [*bundle.files, MANIFEST_NAME, INFO_NAME]:
            path = bundle.stage / rel
            info = tarfile.TarInfo(rel)
            status = path.stat()
            info.size = status.st_size
            info.mode = status.st_mode & 0o777
            info.mtime = int(status.st_mtime)
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 4. ssh to the server over Session Manager with a throwaway key.
# ---------------------------------------------------------------------------


class BoxShell:
    """ssh/rsync to ``ubuntu@<instance-id>`` through an SSM AWS-StartSSHSession proxy.

    A fresh ed25519 key is made in a private temp dir and pushed with EC2
    Instance Connect (valid 60 s) to open one multiplexed ssh connection; it is
    deleted on exit. If the image refuses Instance Connect keys, the key is added
    for this session with SSM Run Command and removed again on exit.
    """

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.workdir = Path(tempfile.mkdtemp(prefix="facade-ssh-"))
        self.key = self.workdir / "id_ed25519"
        self.comment = f"facade-deploy-{secrets.token_hex(4)}"
        self.known_hosts = self.workdir / "known_hosts"
        self.control = self.workdir / "cm"
        self._socket_dir: Path | None = None
        if len(str(self.control)) > 80:  # unix socket paths are limited to ~104 bytes
            self._socket_dir = Path(tempfile.mkdtemp(prefix="fcm-", dir="/tmp"))
            self.control = self._socket_dir / "cm"
        self._fallback_key = False

    @property
    def target(self) -> str:
        return f"{OS_USER}@{self.instance_id}"

    def options(self) -> list[str]:
        proxy = (
            "aws ssm start-session --target %h --document-name AWS-StartSSHSession "
            f"--parameters portNumber=%p --profile {PROFILE} --region {REGION}"
        )
        return [
            "-i", str(self.key),
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
            "-o", f"ProxyCommand={proxy}",
            "-o", f"ControlPath={self.control}",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
            "-o", "ConnectTimeout=90",
            "-o", "LogLevel=ERROR",
        ]

    def __enter__(self) -> "BoxShell":
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", self.comment, "-f", str(self.key)],
            stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=60,
        )
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _push_key(self) -> None:
        aws(
            "ec2-instance-connect", "send-ssh-public-key",
            "--instance-id", self.instance_id,
            "--instance-os-user", OS_USER,
            "--ssh-public-key", f"file://{self.key}.pub",
        )

    def _start_master(self) -> bool:
        with (self.workdir / "master.err").open("wb") as errors:
            try:
                completed = subprocess.run(
                    # A leaked master exits after 15 idle minutes instead of holding an SSM session open.
                    ["ssh", *self.options(), "-o", "ControlMaster=yes", "-o", "ControlPersist=900", "-f", "-N", self.target],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors, timeout=240, check=False,
                )
            except subprocess.TimeoutExpired:
                return False
        return completed.returncode == 0

    def _master_error(self) -> str:
        path = self.workdir / "master.err"
        return path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else ""

    def _install_key_by_run_command(self) -> None:
        public = (self.workdir / "id_ed25519.pub").read_text(encoding="utf-8").strip()
        keys = f"/home/{OS_USER}/.ssh/authorized_keys"
        script = (
            f"install -d -m 700 -o {OS_USER} -g {OS_USER} /home/{OS_USER}/.ssh && touch {keys} && "
            f"chown {OS_USER}:{OS_USER} {keys} && chmod 600 {keys} && "
            f"(grep -qxF {shlex.quote(public)} {keys} || echo {shlex.quote(public)} >> {keys})"
        )
        result = run_command(self.instance_id, script, comment="FACADE temporary ssh key")
        if result.get("Status") != "Success":
            raise RemoteFailed("could not add the temporary ssh key through SSM Run Command")
        self._fallback_key = True

    def connect(self, attempts: int = 4) -> None:
        key_refused = False
        for attempt in range(1, attempts + 1):
            self._push_key()  # an Instance Connect key is valid for 60 s
            if self._start_master():
                return
            if "Permission denied" in self._master_error():
                key_refused = True
                break
            print(f"  ssh: attempt {attempt} failed ({self._master_error()[-160:] or 'no detail'}); retrying")
            _sleep(15)
        if key_refused:
            print("  ssh: the Instance Connect key was not accepted; adding a session key through SSM Run Command")
            self._install_key_by_run_command()
            if self._start_master():
                return
        raise RemoteFailed(
            f"cannot open ssh over SSM to {self.instance_id}: {self._master_error()[-800:] or 'no error text'}\n"
            "  check: session-manager-plugin installed, instance Online in SSM, sshd running"
        )

    def run(self, command: str, *, input_bytes: bytes | None = None, timeout: float = 600,
            check: bool = True, retries: int = 1) -> subprocess.CompletedProcess:
        """Run ``command`` on the server. Exit 255 (the connection dropped) reconnects and retries."""
        argv = ["ssh", *self.options(), "-o", "ControlMaster=no", self.target, command]
        attempt = 0
        while True:
            try:
                if input_bytes is None:
                    completed = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                               timeout=timeout, check=False)
                else:
                    completed = subprocess.run(argv, input=input_bytes, capture_output=True,
                                               timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                raise RemoteFailed(f"remote command timed out after {timeout:.0f}s: {command[:120]}") from None
            if completed.returncode != 255 or attempt >= retries:
                break
            attempt += 1
            print("  ssh: connection dropped; reconnecting")
            self._stop_master()
            self.connect()
        if check and completed.returncode != 0:
            detail = (completed.stderr or b"").decode("utf-8", "replace").strip()
            raise RemoteFailed(f"remote command failed (exit {completed.returncode}): {command[:160]}\n{detail[-1200:]}")
        return completed

    def text(self, command: str, **kwargs: Any) -> str:
        return (self.run(command, **kwargs).stdout or b"").decode("utf-8", "replace")

    def _stop_master(self) -> None:
        subprocess.run(["ssh", *self.options(), "-O", "exit", self.target],
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=False)

    def close(self) -> None:
        try:
            if self._fallback_key:
                self.run(f"sed -i '/ {self.comment}$/d' ~/.ssh/authorized_keys", check=False, retries=0, timeout=60)
            self._stop_master()
        except (RemoteFailed, OSError, subprocess.SubprocessError):
            pass
        finally:
            shutil.rmtree(self.workdir, ignore_errors=True)
            if self._socket_dir is not None:
                shutil.rmtree(self._socket_dir, ignore_errors=True)

    def push(self, bundle: Bundle) -> None:
        """Mirror the bundle to /opt/facade/src (rsync --delete; a tar stream if rsync fails), then verify it."""
        self.run(f"sudo install -d -m 0755 {REMOTE_SRC}")
        pushed = False
        if shutil.which("rsync"):
            remote_shell = shlex.join([
                "ssh", "-i", str(self.key),
                "-o", "IdentitiesOnly=yes",
                "-o", "BatchMode=yes",
                "-o", f"UserKnownHostsFile={self.known_hosts}",
                "-o", f"ControlPath={self.control}",
                "-o", "ControlMaster=no",
            ])
            excludes = [f"--exclude={path}" for path in REMOTE_BUILD_OUTPUTS]
            completed = subprocess.run(
                ["rsync", "-rt", "--delete", *excludes, "--rsync-path", "sudo rsync", "-e", remote_shell,
                 f"{bundle.stage}/", f"{self.target}:{REMOTE_SRC}/"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900, check=False,
            )
            pushed = completed.returncode == 0
            if not pushed:
                print(f"  rsync failed ({(completed.stderr or '').strip()[-300:]}); sending a tar stream instead")
        if not pushed:
            self.run(f"sudo tar -xzf - -C {REMOTE_SRC} --no-same-owner", input_bytes=bundle_tarball(bundle), timeout=900)
            # tar only adds and overwrites. Delete what rsync --delete would have:
            # every file the manifest does not list, keeping the server-built
            # web/node_modules/ and web/.next/.
            self.run(f"sudo sh -c {shlex.quote(PRUNE_UNLISTED_FILES)}", timeout=300)
        self.run(f"cd {REMOTE_SRC} && sudo sha256sum --check --strict --quiet {MANIFEST_NAME}", timeout=300)
        print(f"push: {len(bundle.files)} files verified on the server against {MANIFEST_NAME}")


# ---------------------------------------------------------------------------
# 5. bootstrap.sh runs in a transient systemd unit, so a laptop that sleeps or
#    loses its connection cannot kill it; this side only follows the journal.
# ---------------------------------------------------------------------------


def _unit_state(shell: BoxShell) -> dict[str, str]:
    text = shell.text(
        f"systemctl show {BOOTSTRAP_UNIT} -p ActiveState -p SubState -p Result -p ExecMainStatus -p InvocationID",
        timeout=120,
    )
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _journal(shell: BoxShell, invocation: str, cursor: str | None, *, limit: int = 25) -> str | None:
    command = f"sudo journalctl _SYSTEMD_INVOCATION_ID={shlex.quote(invocation)} -o cat --no-pager --show-cursor"
    if cursor:
        command += f" --after-cursor={shlex.quote(cursor)}"
    lines = shell.text(command, timeout=120, check=False).splitlines()
    new_cursor = cursor
    if lines and lines[-1].startswith("-- cursor: "):
        new_cursor = lines.pop()[len("-- cursor: "):]
    if len(lines) > limit:
        print(f"  [server] ... {len(lines) - limit} lines")
    for line in lines[-limit:]:
        print(f"  [server] {line}")
    return new_cursor


def run_bootstrap(shell: BoxShell, *, timeout: float) -> None:
    state = _unit_state(shell)
    if state.get("ActiveState") == "activating":
        print("bootstrap: one is already running on the server; following it")
    else:
        shell.run(
            f"sudo systemctl stop {BOOTSTRAP_UNIT} >/dev/null 2>&1 || true; "
            f"sudo systemctl reset-failed {BOOTSTRAP_UNIT} >/dev/null 2>&1 || true"
        )
        start = (
            "sudo systemd-run --no-block --unit=facade-bootstrap --description='FACADE bootstrap' "
            "--service-type=oneshot --property=RemainAfterExit=yes --property=TimeoutStartSec=infinity "
            "--setenv=HOME=/root --setenv=DEBIAN_FRONTEND=noninteractive "
            "--setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin "
            f"/bin/bash {REMOTE_SRC}/{BOOTSTRAP_REL}"
        )
        started = None
        for _ in range(3):  # a just-stopped transient unit can take a moment to unload
            started = shell.run(start, check=False, retries=0)
            if started.returncode == 0 or _unit_state(shell).get("ActiveState") in {"activating", "active"}:
                break
            _sleep(3)
        else:
            detail = ((started.stderr if started else b"") or b"").decode("utf-8", "replace").strip()
            raise RemoteFailed(f"could not start the bootstrap unit: {detail[-800:]}")
        print(f"bootstrap: started sudo bash {REMOTE_SRC}/{BOOTSTRAP_REL} as {BOOTSTRAP_UNIT}")
    deadline = _monotonic() + timeout
    cursor: str | None = None
    invocation = ""
    idle_polls = 0
    while True:
        state = _unit_state(shell)
        invocation = state.get("InvocationID") or invocation
        if invocation:
            cursor = _journal(shell, invocation, cursor)
        active, sub, result = state.get("ActiveState"), state.get("SubState"), state.get("Result")
        if active == "active" and sub == "exited":
            if result != "success":
                raise RemoteFailed(f"bootstrap finished with result {result}")
            print("bootstrap: finished OK")
            return
        if active == "failed":
            if invocation:
                tail = shell.text(
                    f"sudo journalctl _SYSTEMD_INVOCATION_ID={shlex.quote(invocation)} -o cat --no-pager -n 60",
                    check=False,
                )
                print("\n".join(f"  [server] {line}" for line in tail.splitlines()))
            raise RemoteFailed(
                f"bootstrap failed (result {result}, exit {state.get('ExecMainStatus')}). The full log is "
                "/var/log/facade-bootstrap.log on the server (`logs --service bootstrap`). Fix it and deploy again."
            )
        if active == "inactive":
            idle_polls += 1
            if idle_polls > 12:
                raise RemoteFailed("the bootstrap unit never started; deploy again")
        if _monotonic() >= deadline:
            raise TimeoutError(
                f"bootstrap still running after {timeout / 60:.0f} min; it keeps going on the server. "
                "Follow it with `logs --service bootstrap`, or deploy again to attach to it"
            )
        _sleep(POLL_SECONDS)


# ---------------------------------------------------------------------------
# 6. Public checks: the worker's capabilities, and that basic auth guards the rest.
# ---------------------------------------------------------------------------


def _curl(config_lines: Sequence[str], *, max_time: int) -> tuple[str, str]:
    """Run curl with its whole config (URL, headers) on stdin; return (body, http_code)."""
    config = "\n".join([*config_lines, 'proto = "=https"', "silent", "show-error", f"max-time = {max_time}", ""])
    try:
        completed = subprocess.run(
            ["curl", "--config", "-", "--write-out", "\\n%{http_code}"],
            input=config, capture_output=True, text=True, timeout=max_time + 15, check=False,
        )
    except subprocess.TimeoutExpired:
        return "", "curl timed out"
    if completed.returncode != 0:
        return "", f"curl exit {completed.returncode}: {(completed.stderr or '').strip()[-200:]}"
    body, _, code = completed.stdout.rpartition("\n")
    return body, code.strip()


def fetch_capabilities(host: str, token: str, *, max_time: int = 20) -> tuple[dict[str, Any] | None, str]:
    """GET https://<host>/v1/capabilities; the bearer reaches curl only through stdin."""
    body, code = _curl(
        [f'url = "https://{host}/v1/capabilities"', f'header = "Authorization: Bearer {token}"'], max_time=max_time
    )
    if code != "200":
        return None, code if code.startswith("curl") else f"HTTP {code}: {body.strip()[:160]}"
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None, "HTTP 200 with a non-JSON body"
    return (value, "") if isinstance(value, dict) else (None, "HTTP 200 with a non-object body")


def page_status(url: str, *, max_time: int = 20) -> str:
    """HTTP status of an unauthenticated GET; 401 means Caddy is up and basic auth guards the page."""
    _, code = _curl([f'url = "{url}"'], max_time=max_time)
    return code


def capabilities_ready(capabilities: Mapping[str, Any] | None) -> bool:
    return bool(capabilities) and all(
        key in capabilities and type(capabilities[key]) is type(value) and capabilities[key] == value
        for key, value in EXPECTED_CAPABILITIES.items()
    )


def wait_ready(host: str, token: str, *, timeout: float) -> dict[str, Any]:
    deadline = _monotonic() + timeout
    last = ""
    while True:
        capabilities, problem = fetch_capabilities(host, token)
        if capabilities_ready(capabilities):
            print(f"ready: https://{host}/v1/capabilities reports {json.dumps(capabilities, sort_keys=True)}")
            return dict(capabilities or {})
        message = problem or "worker answered but is not ready: " + ", ".join(
            f"{key}={(capabilities or {}).get(key)!r}"
            for key, value in EXPECTED_CAPABILITIES.items()
            if (capabilities or {}).get(key) != value
        )
        if message != last:
            print(f"  waiting for https://{host}/v1/capabilities: {message}")
            last = message
        if _monotonic() >= deadline:
            raise TimeoutError(f"https://{host}/v1/capabilities not ready after {timeout / 60:.0f} min: {message}")
        _sleep(POLL_SECONDS * 1.5)


def print_urls(host: str) -> None:
    print(
        f"\n  website (basic auth):          https://{host}/\n"
        f"  PhotoGuard demo (basic auth):  https://{host}/photoguard/\n"
        f"  worker API (bearer):           https://{host}/v1/capabilities"
    )


# ---------------------------------------------------------------------------
# 7. Subcommands.
# ---------------------------------------------------------------------------


def _source_root(args: argparse.Namespace) -> Path:
    return Path(args.source_root).resolve() if getattr(args, "source_root", None) else REPO_ROOT


def cmd_deploy(args: argparse.Namespace) -> int:
    require_tools("aws", "session-manager-plugin", "ssh", "ssh-keygen", "curl", "git")
    with tempfile.TemporaryDirectory(prefix="facade-bundle-") as scratch:
        bundle = build_bundle(_source_root(args), Path(scratch) / "stage", allow_dirty=args.allow_dirty)
        require_identity()
        instance = running_instance()
        instance_id = str(instance["InstanceId"])
        host = read_host()
        read_processor_token()  # fail before touching the server if the secret is missing
        print(f"deploy: {instance_id} ({host})")
        wait_for("the SSM agent to be online", lambda: ssm_online(instance_id), timeout=600)
        with BoxShell(instance_id) as shell:
            shell.push(bundle)
            run_bootstrap(shell, timeout=args.bootstrap_minutes * 60)
    if not args.no_wait:
        wait_ready(host, read_processor_token(), timeout=args.ready_minutes * 60)
    print("\nDeployed.")
    print_urls(host)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    require_identity()
    instance = find_instance()
    parameter = get_parameter(HOST_PARAMETER)
    host = str((parameter or {}).get("Value", "")).strip().lower()
    if instance is None:
        print("instance: none (the owner launches it)")
    else:
        print(
            f"instance: {instance['InstanceId']} {instance_state(instance)} {instance.get('InstanceType')} "
            f"{instance.get('Placement', {}).get('AvailabilityZone')} launched {instance.get('LaunchTime')}"
        )
        print(f"public ip: {instance.get('PublicIpAddress') or '-'}")
    print(f"host: {host or '(SSM ' + HOST_PARAMETER + ' is not set)'}")
    if instance is None or instance_state(instance) != "running" or not _HOSTNAME.fullmatch(host):
        return 0
    try:
        token = read_processor_token()
    except Refused as error:
        print(f"worker: not checked ({error})")
    else:
        capabilities, problem = fetch_capabilities(host, token)
        verdict = "READY" if capabilities_ready(capabilities) else "NOT READY"
        print(f"worker: {verdict} {json.dumps(capabilities, sort_keys=True) if capabilities else problem}")
    for name, url in (("website", f"https://{host}/"), ("demo", f"https://{host}/photoguard/")):
        code = page_status(url)
        meaning = "up, password protected" if code == "401" else "unexpected: expected 401 (basic auth)"
        print(f"{name}: HTTP {code} ({meaning})")
    print_urls(host)
    return 0


def _units(service: str) -> list[str]:
    return ALL_SERVICES if service == "all" else SERVICES[service]


def cmd_logs(args: argparse.Namespace) -> int:
    require_identity()
    instance_id = str(running_instance()["InstanceId"])
    units = " ".join(f"-u {shlex.quote(unit)}" for unit in _units(args.service))
    script = f"journalctl --no-pager -o short-iso -n {int(args.lines)} {units}"
    if args.service == "bootstrap":
        script = f"tail -n {int(args.lines)} /var/log/facade-bootstrap.log"
    result = run_command(instance_id, script, comment=f"FACADE logs {args.service}")
    output = str(result.get("StandardOutputContent", ""))
    print(output.rstrip("\n") or "(no output)")
    if len(output) >= 24000:
        print("(SSM returns at most 24,000 characters; use --lines to see less, or `ssh` for everything)")
    if result.get("Status") != "Success":
        print(f"logs: the command ended with {result.get('Status')}: "
              f"{str(result.get('StandardErrorContent', '')).strip()[-400:]}", file=sys.stderr)
        return 1
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    require_identity()
    instance_id = str(running_instance()["InstanceId"])
    units = " ".join(shlex.quote(f"{unit}.service") for unit in _units(args.service))
    script = f"systemctl restart {units} && systemctl is-active {units}"
    result = run_command(instance_id, script, comment=f"FACADE restart {args.service}")
    print(str(result.get("StandardOutputContent", "")).rstrip("\n"))
    if result.get("Status") != "Success":
        print(f"restart failed: {str(result.get('StandardErrorContent', '')).strip()[-400:]}", file=sys.stderr)
        return 1
    print(f"restarted: {units}")
    if args.service in {"worker", "all"}:
        print("The worker reloads the model and reruns its GPU smoke test (a minute or two); check with `status`.")
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    require_tools("aws", "session-manager-plugin")
    require_identity()
    instance_id = str(running_instance()["InstanceId"])
    argv = ["aws", "ssm", "start-session", "--target", instance_id]
    if args.forward:
        argv += [
            "--document-name", "AWS-StartPortForwardingSession",
            "--parameters", json.dumps({"portNumber": [str(args.forward)], "localPortNumber": [str(args.forward)]}),
        ]
        print(f"forwarding localhost:{args.forward} -> server 127.0.0.1:{args.forward} (Ctrl-C to stop)")
    argv += ["--profile", PROFILE, "--region", REGION]
    sys.stdout.flush()
    _exec("aws", argv)
    return 0  # pragma: no cover - exec does not return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("deploy", help="copy this checkout to the server, run bootstrap.sh, wait until ready")
    p.add_argument("--allow-dirty", action="store_true", help="deploy uncommitted changes to tracked files too")
    p.add_argument("--no-wait", action="store_true", help="do not wait for the worker to report ready")
    p.add_argument("--bootstrap-minutes", type=float, default=120.0)
    p.add_argument("--ready-minutes", type=float, default=45.0)
    p.add_argument("--source-root", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("status", help="instance state, URL and whether the worker is ready")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("logs", help="recent journal lines from the server")
    p.add_argument("--service", choices=[*SERVICES, "all"], default="all")
    p.add_argument("--lines", type=int, default=200)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("restart", help="restart one service (or all of them)")
    p.add_argument("--service", choices=RESTARTABLE, required=True)
    p.set_defaults(func=cmd_restart)

    p = sub.add_parser("ssh", help="a shell on the server through Session Manager, or a port forward")
    p.add_argument("--forward", type=int, choices=FORWARD_PORTS,
                   help="forward this loopback port (3000 website, 7860 demo, 8000 worker) instead of a shell")
    p.set_defaults(func=cmd_ssh)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "lines", 1) < 1:
        parser.error("--lines must be at least 1")
    try:
        return int(args.func(args) or 0)
    except Refused as error:
        print(f"\nREFUSED: {error}", file=sys.stderr)
        return 2
    except (AwsCallFailed, RemoteFailed, TimeoutError) as error:
        label = "AWS error" if isinstance(error, AwsCallFailed) else "ERROR"
        print(f"\n{label}: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
