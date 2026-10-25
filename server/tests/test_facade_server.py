"""deploy/facade_server.py with no AWS call leaving this computer.

Fake `aws`, `ssh`, `rsync` and `curl` programs go first on PATH. They record
their arguments and stdin, so the tests can check what the tool would send:
the identity and tag guards, the git-tracked bundle, that it never creates or
deletes AWS resources, and that the worker bearer never reaches argv or output.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "deploy" / "facade_server.py"

_spec = importlib.util.spec_from_file_location("facade_server", TOOL)
tool = importlib.util.module_from_spec(_spec)
sys.modules["facade_server"] = tool
_spec.loader.exec_module(tool)

ACCOUNT = "669707689345"
GOOD_ARN = f"arn:aws:iam::{ACCOUNT}:user/facade"
TOKEN = "ef" * 32
HOST = "198-51-100-7.sslip.io"
CAPABILITIES = {
    "ready": True, "recipe": "photoguard-encoder-linf-v1", "image_side": 512, "total_iterations": 200,
    "retention_seconds": 1800, "experimental": True, "gpu_smoke_validated": True,
    "engine_smoke_validated": True, "execution_device": "cuda:0", "efficacy_evaluated": False,
}
#: The only AWS calls this tool may make that change anything, and what they change.
ALLOWED_WRITES = {"send-ssh-public-key", "send-command"}

FAKE_AWS = r'''#!__PYTHON__
import json, os, sys
argv = sys.argv[1:]
with open(os.environ["FAKE_AWS_STATE"], encoding="utf-8") as handle:
    state = json.load(handle)
with open(os.environ["FAKE_AWS_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": argv}) + "\n")
service, op = argv[0], argv[1]
opts, key = {}, None
for token in argv[2:]:
    if token.startswith("--"):
        key = token[2:]
        opts.setdefault(key, [])
    elif key is not None:
        opts[key].append(token)
one = lambda name: (opts.get(name) or [None])[0]

def out(value):
    print(json.dumps(value)); sys.exit(0)

def err(code):
    sys.stderr.write(f"An error occurred ({code}) when calling the {op} operation: fake\n"); sys.exit(254)

name = f"{service} {op}"
if name == "sts get-caller-identity":
    out({"Account": state["account"], "Arn": state["arn"]})
if name == "ec2 describe-instances":
    filters = json.loads(one("filters"))
    found = []
    for instance in state["instances"]:
        tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
        ok = state.get("ignore_filters", False) or all(
            (tags.get(f["Name"][4:]) if f["Name"].startswith("tag:") else instance["State"]["Name"]) in f["Values"]
            for f in filters)
        if ok:
            found.append(instance)
    out({"Reservations": [{"Instances": found}] if found else []})
if name == "ssm get-parameter":
    item = state["params"].get(one("name"))
    if item is None:
        err("ParameterNotFound")
    hidden = item["Type"] == "SecureString" and "with-decryption" not in opts
    out({"Parameter": {"Name": one("name"), "Value": "AQICAHciphertext" if hidden else item["Value"]}})
if name == "ssm describe-instance-information":
    out({"InstanceInformationList": [{"PingStatus": "Online"}]})
if name == "ssm send-command":
    out({"Command": {"CommandId": "cmd-0001"}})
if name == "ssm get-command-invocation":
    out({"Status": state.get("command_status", "Success"), "StandardOutputContent": state.get("command_output", ""),
         "StandardErrorContent": ""})
if name == "ec2-instance-connect send-ssh-public-key":
    out({"Success": True})
err("FakeUnsupported")
'''

FAKE_SSH = r'''#!__PYTHON__
import json, os, sys
argv = sys.argv[1:]
data = b"" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.buffer.read()
with open(os.environ["FAKE_SSH_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": argv, "stdin_bytes": len(data)}) + "\n")
if "-O" in argv or "-N" in argv:
    sys.exit(0)
command = argv[-1]
if "systemctl show" in command:
    path = os.environ["FAKE_SSH_STATE"]
    with open(path, encoding="utf-8") as handle:
        states = json.load(handle)
    print(states[0] if len(states) == 1 else states.pop(0))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(states, handle)
    sys.exit(0)
if "journalctl" in command:
    print("[facade-bootstrap] done\n-- cursor: s=abc;i=1")
sys.exit(0)
'''

FAKE_RSYNC = r'''#!__PYTHON__
import json, os, sys
with open(os.environ["FAKE_RSYNC_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:]}) + "\n")
sys.exit(int(os.environ.get("FAKE_RSYNC_EXIT", "0")))
'''

FAKE_CURL = r'''#!__PYTHON__
import json, os, sys
config = sys.stdin.read()
with open(os.environ["FAKE_CURL_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:], "stdin": config}) + "\n")
if "/v1/capabilities" in config:
    print(os.environ["FAKE_CURL_BODY"]); print("200", end="")
elif "/photoguard/" in config:
    print("Unauthorized"); print("401", end="")
else:
    print("<!doctype html>"); print("200", end="")
'''


def _show(active: str, sub: str, result: str = "success") -> str:
    """Fake `systemctl show` output for a unit in the given state."""
    return f"ActiveState={active}\nSubState={sub}\nResult={result}\nExecMainStatus=0\nInvocationID=0123abcd"


def _box(tags: list[dict] | None = None, state: str = "running") -> dict:
    """A fake EC2 instance description, tagged Project=FACADE unless other tags are given."""
    return {
        "InstanceId": "i-0facade000000001", "InstanceType": "g6.xlarge", "State": {"Name": state},
        "Placement": {"AvailabilityZone": "us-east-1a"}, "LaunchTime": "2026-09-23T12:00:00+00:00",
        "PublicIpAddress": "198.51.100.7",
        "Tags": tags if tags is not None else [{"Key": "Project", "Value": "FACADE"}, {"Key": "Name", "Value": "facade-box"}],
    }


def _exe(path: Path, text: str) -> None:
    """Write a fake command-line program that runs with this test's Python, and make it executable."""
    path.write_text(text.replace("__PYTHON__", sys.executable), encoding="utf-8")
    path.chmod(0o755)


def _git(root: Path, *args: str) -> None:
    """Run a git command in `root`, failing the test if git fails."""
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, stdin=subprocess.DEVNULL)


def make_checkout(root: Path) -> Path:
    """A tiny git repository with the files the tool requires, plus some it must skip."""
    src = root / "checkout"
    files = {rel: f"# {rel}\n" for rel in tool.REQUIRED_FILES}
    files["deploy/bootstrap.sh"] = "#!/usr/bin/env bash\necho bootstrap\n"
    files["README.md"] = "# FACADE\n"
    for rel, text in files.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (src / "deploy/bootstrap.sh").chmod(0o755)
    _git(src.parent, "init", "-q", str(src))
    _git(src, "config", "user.email", "test@example.invalid")
    _git(src, "config", "user.name", "test")
    _git(src, "add", "-A")
    _git(src, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture")
    (src / "web/.env.local").write_text("FACADE_PROCESSOR_TOKEN=never-deployed\n", encoding="utf-8")  # untracked
    (src / "notes.txt").write_text("untracked\n", encoding="utf-8")
    return src


@dataclass
class World:
    root: Path
    src: Path
    state_path: Path
    logs: dict

    def set_state(self, **changes) -> None:
        """Change the fake AWS account's state for the rest of the test."""
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state.update(changes)
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    def calls(self, name: str = "aws") -> list[dict]:
        """Every recorded call to one fake program, oldest first."""
        path = self.logs[name]
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    def operations(self) -> list[str]:
        """The names of the AWS operations called so far, such as describe-instances."""
        return [call["argv"][1] for call in self.calls("aws")]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """Fake aws, ssh, rsync and curl on PATH, a fake AWS account with one server, and a tiny git checkout."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "aws", FAKE_AWS)
    _exe(bin_dir / "ssh", FAKE_SSH)
    _exe(bin_dir / "rsync", FAKE_RSYNC)
    _exe(bin_dir / "curl", FAKE_CURL)
    _exe(bin_dir / "session-manager-plugin", "#!/bin/sh\nexit 0\n")
    logs = {name: tmp_path / f"{name}.log" for name in ("aws", "ssh", "rsync", "curl")}
    state_path = tmp_path / "aws-state.json"
    state_path.write_text(json.dumps({
        "account": ACCOUNT, "arn": GOOD_ARN, "instances": [_box()],
        "params": {
            "/facade/prod/processor-token": {"Type": "SecureString", "Value": TOKEN},
            "/facade/prod/host": {"Type": "String", "Value": HOST},
            "/facade/prod/basic-auth-hash": {"Type": "SecureString", "Value": "$2y$10$" + "A" * 53},
        },
    }), encoding="utf-8")
    ssh_state = tmp_path / "ssh-state.json"
    ssh_state.write_text(json.dumps([_show("inactive", "dead"), _show("activating", "start"),
                                     _show("active", "exited")]), encoding="utf-8")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_AWS_STATE", str(state_path))
    monkeypatch.setenv("FAKE_AWS_LOG", str(logs["aws"]))
    monkeypatch.setenv("FAKE_SSH_STATE", str(ssh_state))
    monkeypatch.setenv("FAKE_SSH_LOG", str(logs["ssh"]))
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(logs["rsync"]))
    monkeypatch.setenv("FAKE_CURL_LOG", str(logs["curl"]))
    monkeypatch.setenv("FAKE_CURL_BODY", json.dumps(CAPABILITIES))
    monkeypatch.setattr(tool, "_sleep", lambda seconds: None)
    return World(tmp_path, make_checkout(tmp_path), state_path, logs)


def deploy(world: World, *extra: str) -> list[str]:
    """Arguments for `deploy` from the fixture's checkout."""
    return ["deploy", "--source-root", str(world.src), *extra]


def assert_no_secret_leaked(world: World, output: str) -> None:
    """The worker bearer appears in no output and no argv; it may only travel on curl's stdin."""
    assert TOKEN not in output
    for name in ("aws", "ssh", "rsync"):
        for call in world.calls(name):
            assert TOKEN not in json.dumps(call["argv"]), (name, call["argv"])
    for call in world.calls("curl"):
        assert TOKEN not in json.dumps(call["argv"])  # it may appear on stdin only


def test_deploy_pushes_tracked_files_runs_bootstrap_and_waits_for_ready(world: World, capsys) -> None:
    """A deploy copies the files, checks the manifest on the server, runs bootstrap.sh and waits for the worker over HTTPS."""
    assert tool.main(deploy(world)) == 0
    out = capsys.readouterr()
    assert "Deployed." in out.out and f"https://{HOST}/photoguard/" in out.out
    assert set(world.operations()) - {"get-caller-identity", "describe-instances", "get-parameter",
                                      "describe-instance-information"} <= ALLOWED_WRITES
    assert "send-ssh-public-key" in world.operations()
    (rsync,) = world.calls("rsync")
    argv = rsync["argv"]
    assert argv[:2] == ["-rt", "--delete"] and "--exclude=/web/node_modules/" in argv and "--exclude=/web/.next/" in argv
    assert argv[-1] == f"ubuntu@i-0facade000000001:{tool.REMOTE_SRC}/"
    ssh_commands = [call["argv"][-1] for call in world.calls("ssh")]
    assert any(f"sha256sum --check --strict --quiet {tool.MANIFEST_NAME}" in c for c in ssh_commands)
    assert any("systemd-run" in c and f"/bin/bash {tool.REMOTE_SRC}/deploy/bootstrap.sh" in c for c in ssh_commands)
    (curl,) = [c for c in world.calls("curl") if "/v1/capabilities" in c["stdin"]]
    assert f"Authorization: Bearer {TOKEN}" in curl["stdin"] and 'proto = "=https"' in curl["stdin"]
    assert_no_secret_leaked(world, out.out + out.err)


def test_bundle_is_exactly_the_tracked_files_with_a_manifest(world: World, tmp_path: Path) -> None:
    """The bundle holds exactly the git-tracked files (no .env.local, no untracked notes) and a valid sha256 manifest."""
    bundle = tool.build_bundle(world.src, tmp_path / "stage")
    assert sorted(bundle.files) == sorted([*tool.REQUIRED_FILES, "README.md"])
    assert not (tmp_path / "stage/web/.env.local").exists() and not (tmp_path / "stage/notes.txt").exists()
    manifest = (tmp_path / "stage" / tool.MANIFEST_NAME).read_text()
    assert len(manifest.splitlines()) == len(bundle.files)
    check = subprocess.run(["shasum", "-a", "256", "-c", tool.MANIFEST_NAME] if sys.platform == "darwin"
                           else ["sha256sum", "--check", "--strict", tool.MANIFEST_NAME],
                           cwd=tmp_path / "stage", capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    assert os.access(tmp_path / "stage/deploy/bootstrap.sh", os.X_OK)
    info = json.loads((tmp_path / "stage" / tool.INFO_NAME).read_text())
    assert info["files"] == len(bundle.files) and info["uncommitted_changes"] is False


def test_deploy_refuses_uncommitted_changes_unless_asked(world: World, tmp_path: Path, capsys) -> None:
    """Uncommitted changes stop a deploy before any AWS call, unless explicitly allowed."""
    (world.src / "README.md").write_text("# changed\n", encoding="utf-8")
    assert tool.main(deploy(world)) == 2
    assert "uncommitted changes" in capsys.readouterr().err
    assert world.calls("aws") == [], "refused before any AWS call"
    bundle = tool.build_bundle(world.src, tmp_path / "stage", allow_dirty=True)
    assert (tmp_path / "stage/README.md").read_text() == "# changed\n" and bundle.info["uncommitted_changes"]


def test_bundle_refuses_large_files_and_missing_server_files(world: World, tmp_path: Path) -> None:
    """A tracked file over the size limit, or a missing required file, stops the bundle."""
    (world.src / "weights.bin").write_bytes(b"\0" * (tool.MAX_FILE_BYTES + 1))
    _git(world.src, "add", "weights.bin")
    _git(world.src, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "big")
    with pytest.raises(tool.Refused, match="large files"):
        tool.build_bundle(world.src, tmp_path / "a")
    _git(world.src, "rm", "-q", "weights.bin", "web/package-lock.json")
    _git(world.src, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "drop")
    with pytest.raises(tool.Refused, match="web/package-lock.json"):
        tool.build_bundle(world.src, tmp_path / "b")


@pytest.mark.parametrize("arn, account", [
    (f"arn:aws:iam::{ACCOUNT}:root", ACCOUNT),
    (f"arn:aws:sts::{ACCOUNT}:assumed-role/admin/x", ACCOUNT),
    ("arn:aws:iam::111122223333:user/facade", "111122223333"),
])
@pytest.mark.parametrize("command", [["status"], ["logs"], ["restart", "--service", "web"], ["ssh"]])
def test_identity_guard_refuses_root_roles_and_other_accounts(world: World, capsys, arn, account, command) -> None:
    """Every command refuses to run as the root user, an assumed role or another AWS account."""
    world.set_state(arn=arn, account=account)
    assert tool.main(command) == 2
    assert "only runs as" in capsys.readouterr().err
    assert world.operations() == ["get-caller-identity"]


def test_refuses_an_instance_without_the_project_tag(world: World, capsys) -> None:
    """A server without the Project=FACADE tag is never touched."""
    world.set_state(instances=[_box(tags=[{"Key": "Name", "Value": "facade-box"}])], ignore_filters=True)
    assert tool.main(["restart", "--service", "web"]) == 2
    assert "lacks the tag Project=FACADE" in capsys.readouterr().err
    assert "send-command" not in world.operations()


def test_refuses_when_the_server_is_not_running(world: World, capsys) -> None:
    """A stopped server is reported and not deployed to."""
    world.set_state(instances=[_box(state="stopped")])
    assert tool.main(deploy(world)) == 2
    assert "is stopped, not running" in capsys.readouterr().err
    assert "send-ssh-public-key" not in world.operations()


def test_status_reports_readiness_and_basic_auth_without_printing_the_token(world: World, capsys) -> None:
    """status shows the worker as ready, the public website and the password-protected demo as up, without sending or printing the bearer elsewhere."""
    assert tool.main(["status"]) == 0
    out = capsys.readouterr()
    assert "worker: READY" in out.out
    assert "website: HTTP 200 (up, public)" in out.out
    assert "demo: HTTP 401 (up, password protected)" in out.out
    unauthenticated = [c["stdin"] for c in world.calls("curl") if "/v1/" not in c["stdin"]]
    assert unauthenticated and all("Authorization" not in text for text in unauthenticated)
    assert set(world.operations()) <= {"get-caller-identity", "describe-instances", "get-parameter"}
    assert_no_secret_leaked(world, out.out + out.err)


def test_logs_runs_journalctl_for_the_chosen_units(world: World, capsys) -> None:
    """logs runs journalctl for the chosen unit, or for all four FACADE units by default."""
    world.set_state(command_output="Sep 23 facade-web: ready\n")
    assert tool.main(["logs", "--service", "web", "--lines", "50"]) == 0
    assert "facade-web: ready" in capsys.readouterr().out
    (send,) = [c["argv"] for c in world.calls() if c["argv"][1] == "send-command"]
    script = json.loads(send[send.index("--parameters") + 1])["commands"][0]
    assert script == "journalctl --no-pager -o short-iso -n 50 -u facade-web"
    assert tool.main(["logs"]) == 0
    sends = [c["argv"] for c in world.calls() if c["argv"][1] == "send-command"]
    script = json.loads(sends[-1][sends[-1].index("--parameters") + 1])["commands"][0]
    for unit in ("facade-worker", "facade-demo", "facade-web", "caddy"):
        assert f"-u {unit}" in script


def test_restart_restarts_only_the_named_service(world: World, capsys) -> None:
    """restart restarts exactly the named unit and reports a failed restart with exit code 1."""
    world.set_state(command_output="active\n")
    assert tool.main(["restart", "--service", "demo"]) == 0
    (send,) = [c["argv"] for c in world.calls() if c["argv"][1] == "send-command"]
    script = json.loads(send[send.index("--parameters") + 1])["commands"][0]
    assert script == "systemctl restart facade-demo.service && systemctl is-active facade-demo.service"
    world.set_state(command_status="Failed")
    assert tool.main(["restart", "--service", "worker"]) == 1


def test_ssh_hands_the_terminal_to_session_manager(world: World, monkeypatch) -> None:
    """ssh replaces itself with an AWS Session Manager session, with port forwarding when asked."""
    seen = []
    monkeypatch.setattr(tool, "_exec", lambda file, argv: seen.append(argv))
    tool.main(["ssh", "--forward", "3000"])
    (argv,) = seen
    assert argv[:5] == ["aws", "ssm", "start-session", "--target", "i-0facade000000001"]
    assert "AWS-StartPortForwardingSession" in argv and argv[-4:] == ["--profile", "facade", "--region", "us-east-1"]


def test_rsync_failure_falls_back_to_a_verified_tar_stream(world: World, capsys, monkeypatch) -> None:
    """If rsync fails, the files go over as a tar stream, are checked against the manifest, and stale files are pruned."""
    monkeypatch.setenv("FAKE_RSYNC_EXIT", "12")
    assert tool.main(deploy(world, "--no-wait")) == 0
    ssh = world.calls("ssh")
    tar = [c for c in ssh if "tar -xzf -" in c["argv"][-1]]
    assert tar and tar[0]["stdin_bytes"] > 0
    assert any("sha256sum --check" in c["argv"][-1] for c in ssh)
    # tar never deletes, so the fallback prunes what the manifest does not list,
    # like rsync --delete, keeping the server-built web output.
    commands = [c["argv"][-1] for c in ssh]
    prune = [i for i, command in enumerate(commands) if "comm -23" in command]
    assert prune and prune[0] > commands.index(tar[0]["argv"][-1])
    assert "./web/node_modules" in commands[prune[0]] and "./web/.next" in commands[prune[0]]


def test_the_tar_fallback_prune_keeps_listed_files_and_build_output(tmp_path) -> None:
    """The prune step deletes unlisted files but keeps listed ones and the web build output."""
    shell = shutil.which("sh")
    for tool_name in ("comm", "cut", "xargs", "find", "sort", "tr"):
        if shutil.which(tool_name) is None:
            pytest.skip(f"{tool_name} not available")
    src = tmp_path / "src"
    for rel, text in {"server/a.py": "a", "web/app/page.tsx": "b", "research/old/stale.py": "old",
                      "web/stale.ts": "old", "web/node_modules/x/i.js": "keep", "web/.next/BUILD_ID": "keep"}.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    (src / "web/.next/cache").mkdir()
    lines = "".join(f"{hashlib.sha256((src / rel).read_bytes()).hexdigest()}  {rel}\n"
                    for rel in ("server/a.py", "web/app/page.tsx"))
    (src / tool.MANIFEST_NAME).write_text(lines)
    (src / tool.INFO_NAME).write_text("{}\n")
    script = tool.PRUNE_UNLISTED_FILES.replace(tool.REMOTE_SRC, shlex.quote(str(src)), 1)
    subprocess.run([shell, "-c", script], check=True, stdin=subprocess.DEVNULL)
    left = sorted(str(path.relative_to(src)) for path in src.rglob("*"))
    assert left == sorted([
        tool.INFO_NAME, tool.MANIFEST_NAME, "server", "server/a.py", "web", "web/.next", "web/.next/BUILD_ID",
        "web/.next/cache", "web/app", "web/app/page.tsx", "web/node_modules", "web/node_modules/x",
        "web/node_modules/x/i.js",
    ])


def test_the_tool_never_creates_or_deletes_aws_resources() -> None:
    """The deploy tool's source contains no AWS command that creates, starts, stops or deletes resources."""
    source = TOOL.read_text()
    for forbidden in ("run-instances", "terminate-instances", "stop-instances", "start-instances",
                      "create-security-group", "delete-security-group", "allocate-address", "release-address",
                      "put-parameter", "delete-parameter"):
        assert forbidden not in source, forbidden
