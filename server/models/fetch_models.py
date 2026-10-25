#!/usr/bin/env python3
"""Fetch the pinned SD1.5-inpainting snapshot for the FACADE server.

Downloads exactly the files listed in MODEL_SHA256SUMS from
stable-diffusion-v1-5/stable-diffusion-inpainting at the pinned revision,
verifies every SHA-256, then publishes the tree in one directory rename:

    <dest parent>/.staging-<name>/   resumable downloads (hash-checked on reuse)
    <dest parent>/.assemble-<name>/  verified files only + .verified marker
    <dest>                           read-only (files 0444, dirs 0555)

The published directory therefore either does not exist or is complete and
verified. It holds only the listed files plus `.verified`: no hub cache
metadata, so the encoder/diffusion loaders' inventory checks (which reject
unlisted files inside vae/, unet/, text_encoder/, tokenizer/, scheduler/) and
StableDiffusionInpaintPipeline.from_pretrained all accept the same directory.

Download path: huggingface_hub.hf_hub_download (the lock pins 1.29.0) when it
is importable, falling back to plain HTTPS from the hub's resolve URL. Either
way the bytes must match the checked-in digest or nothing is published.
Stdlib only apart from the optional huggingface_hub import.

The weights are under the CreativeML OpenRAIL-M license
(licenses/CreativeML-OpenRAIL-M.txt); they are downloaded here, never committed.
On the server, facade-models.service runs this once; to fetch a copy for local
development:

    python server/models/fetch_models.py --dest ~/facade-models/sd15-inpaint-8a4288

Why so careful? These files are the model that every protected photo depends
on, and they come from the internet. Pinning a git revision says WHICH files
are wanted; the SHA-256 digests in MODEL_SHA256SUMS prove the bytes received
are those files, whatever server or mirror sent them. A truncated download, a
changed upload on the hub, or a tampered copy all fail the hash and are never
published. Publishing with one final rename means a power cut at any moment
leaves either no model folder or a complete, verified one, never a
half-written folder that the worker or the demo could load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
from typing import Callable
import urllib.parse
import urllib.request

REPO_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
DEFAULT_DEST = Path("/srv/facade/models/sd15-inpaint-8a4288")
DEFAULT_SUMS = Path(__file__).resolve().with_name("MODEL_SHA256SUMS")
# Written last, only after every file matched. facade-models.service skips its
# run when this file exists, and the demo logs a warning when it is missing.
MARKER = ".verified"
# One line of MODEL_SHA256SUMS: 64 hex digits, two spaces, a plain relative
# path. The path may not start with "." or "/", so no hidden or absolute paths.
_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_][A-Za-z0-9_./-]*)")
# Read and hash files 8 MiB at a time, so a 3 GB UNet never sits in memory.
_CHUNK = 8 * 1024 * 1024

# download(relative_path, staging_dir, force) must leave the file at
# staging_dir/relative_path. Tests pass fake downloaders of this shape.
Downloader = Callable[[str, Path, bool], None]


def log(message: str) -> None:
    """One progress line on stdout, which systemd sends to the journal."""
    print(f"[fetch_models] {message}", flush=True)


def parse_sums(path: Path) -> dict[str, str]:
    """Strict `sha256sum` format: '<64 lowercase hex>  <relative path>' per line."""
    result: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        match = _LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"MODEL_SHA256SUMS line {number} is malformed")
        digest, relative = match.groups()
        parts = PurePosixPath(relative).parts
        # Refuse any path that could point outside the destination folder.
        if relative.startswith("/") or ".." in parts or "." in parts or relative.endswith("/"):
            raise ValueError(f"MODEL_SHA256SUMS line {number} has an unsafe path")
        if relative in result:
            raise ValueError(f"MODEL_SHA256SUMS lists {relative} twice")
        if relative == MARKER or parts[0].startswith("."):
            raise ValueError(f"MODEL_SHA256SUMS line {number} uses a reserved path")
        result[relative] = digest
    if not result:
        raise ValueError("MODEL_SHA256SUMS is empty")
    return result


def sha256_file(path: Path) -> str:
    """SHA-256 of a file as 64 lowercase hex digits, read in 8 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def hf_downloader(relative: str, staging: Path, force: bool) -> None:
    """Download one file with the official huggingface_hub client into the staging folder.

    token=False: no Hugging Face account is used or sent; the model is public.
    If the client hands back a path somewhere else (its cache), the file is
    copied to where `obtain` will look for it.
    """
    os.environ.pop("HF_HUB_OFFLINE", None)  # this unit alone is allowed to go online
    from huggingface_hub import hf_hub_download

    fetched = Path(hf_hub_download(repo_id=REPO_ID, filename=relative, revision=REVISION,
                                   local_dir=str(staging), force_download=force, token=False))
    target = staging / relative
    if fetched.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fetched, target)


def https_downloader(relative: str, staging: Path, force: bool) -> None:
    """Download one file over plain HTTPS from the hub's resolve URL at the pinned revision.

    The bytes go to "<name>.part" first and are renamed into place only once
    the download finished, so an interrupted download never looks complete.
    `force` is unused: this downloader always fetches the whole file.
    """
    url = (f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/"
           + urllib.parse.quote(relative, safe="/"))
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "facade-fetch-models/1"})
    # timeout=120 bounds each wait for data, not the whole multi-gigabyte download.
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, _CHUNK)
    os.replace(partial, target)


def default_downloaders() -> list[tuple[str, Downloader]]:
    """The retry plan for one file: three tries with huggingface_hub, then three over HTTPS.

    Without huggingface_hub installed, five tries over HTTPS.
    """
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        return [("https", https_downloader)] * 5
    return [("hf", hf_downloader)] * 3 + [("https", https_downloader)] * 3


def _discard(path: Path) -> None:
    """Delete a file if it exists; a missing file is fine."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def obtain(relative: str, digest: str, staging: Path, attempts: list[tuple[str, Downloader]],
           *, sleep: Callable[[float], None] = time.sleep) -> Path:
    """Return staging/relative once its bytes match; reuse a verified earlier download.

    A run that was interrupted keeps its finished files in the staging folder,
    so the next run re-hashes them and skips the download when they match.
    Otherwise the attempts are tried in order with a growing pause between
    them (5, 10, 15 ... seconds, at most 60). Every try's file is hashed; a
    wrong file is deleted and the next try begins.
    """
    target = staging / relative
    # A symlink is never trusted as a finished download: it could point anywhere.
    if target.is_file() and not target.is_symlink():
        if sha256_file(target) == digest:
            log(f"reuse {relative} (verified)")
            return target
        _discard(target)
    for number, (name, download) in enumerate(attempts, 1):
        try:
            # force=True on every retry, so a bad cached copy is not reused.
            download(relative, staging, number > 1)
            if target.is_symlink() or not target.is_file():
                raise OSError("download did not produce a regular file")
            actual = sha256_file(target)
            if actual == digest:
                log(f"fetched {relative} via {name} ({target.stat().st_size} B, sha256 ok)")
                return target
            log(f"sha256 mismatch for {relative} via {name} (attempt {number}); discarding")
        except Exception as error:  # network/hub errors: retry with the next attempt
            log(f"download of {relative} via {name} failed (attempt {number}): {type(error).__name__}")
        _discard(target)
        if number < len(attempts):
            sleep(min(60.0, 5.0 * number))
    raise RuntimeError(f"could not obtain a verified copy of {relative}")


def _make_writable(path: Path) -> None:
    """Undo make_read_only, so that a folder tree can be deleted."""
    for root, dirs, files in os.walk(path):
        os.chmod(root, 0o755)
        for name in files:
            full = os.path.join(root, name)
            if not os.path.islink(full):
                os.chmod(full, 0o644)


def force_rmtree(path: Path) -> None:
    """Delete a file or a whole folder tree, even one that make_read_only locked."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        _make_writable(path)
        shutil.rmtree(path)


def make_read_only(path: Path) -> None:
    """Make every file 0444 and every folder 0555, so nothing can change the model by accident.

    The walk goes bottom-up (topdown=False): a folder is locked only after
    everything inside it has been changed.
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.chmod(os.path.join(root, name), 0o444)
        os.chmod(root, 0o555)


def _fsync_dir(path: Path) -> None:
    """Ask the OS to write a folder's entries to disk, so the final rename survives a power cut.

    Some systems do not allow this for folders; that is ignored.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def verify_tree(dest: Path, sums: dict[str, str]) -> None:
    """Exact inventory (listed files + marker) and digests; raises on any difference.

    `--verify-only` runs this on the published folder to re-hash every file.
    """
    actual = {str(path.relative_to(dest)) for path in dest.rglob("*") if path.is_file() or path.is_symlink()}
    expected = set(sums) | {MARKER}
    if actual != expected:
        raise ValueError("published model tree inventory differs from MODEL_SHA256SUMS")
    for relative, digest in sums.items():
        if (dest / relative).is_symlink() or sha256_file(dest / relative) != digest:
            raise ValueError(f"published model file differs: {relative}")


def publish(sums: dict[str, str], dest: Path, *, attempts: list[tuple[str, Downloader]] | None = None,
            sleep: Callable[[float], None] = time.sleep, force: bool = False,
            sums_sha256: str = "") -> Path:
    """Download, verify and publish the whole snapshot at `dest`; return `dest`.

    Steps: every file into the staging folder (reusing verified ones), then
    all of them moved into a fresh assemble folder, the marker written, the
    tree checked once more and made read-only, and finally the assemble
    folder renamed to `dest` in one step. An existing `dest` is first moved
    aside and deleted. If anything fails before that point, `dest` is left
    exactly as it was.
    """
    dest = Path(dest)
    parent = dest.parent
    if (dest / MARKER).is_file() and not force:
        log(f"{dest} already verified; nothing to do")
        return dest
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".staging-{dest.name}"
    assemble = parent / f".assemble-{dest.name}"
    force_rmtree(assemble)
    staging.mkdir(parents=True, exist_ok=True)
    plan = attempts if attempts is not None else default_downloaders()
    total = 0
    for relative in sorted(sums):
        path = obtain(relative, sums[relative], staging, plan, sleep=sleep)
        total += path.stat().st_size
    for relative in sorted(sums):
        target = assemble / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging / relative, target)
    # The marker records what was verified and against which sums file.
    marker = {"repo_id": REPO_ID, "revision": REVISION, "files": len(sums), "bytes": total,
              "sums_sha256": sums_sha256, "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (assemble / MARKER).write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="ascii")
    verify_tree(assemble, sums)  # renames only; cheap insurance against a stray file
    make_read_only(assemble)
    # os.replace cannot rename onto a non-empty folder, so an old copy is moved
    # aside under a unique name and deleted first.
    if dest.exists() or dest.is_symlink():
        stale = parent / f".stale-{dest.name}-{int(time.time())}"
        os.replace(dest, stale)
        force_rmtree(stale)
    os.replace(assemble, dest)
    _fsync_dir(parent)
    force_rmtree(staging)
    log(f"published {len(sums)} verified files ({total} B) at {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    """Fetch and publish (default), or with --verify-only re-hash a published copy; returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sums", type=Path, default=DEFAULT_SUMS)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="re-fetch even if .verified exists")
    parser.add_argument("--verify-only", action="store_true",
                        help="re-hash an already published tree and exit")
    args = parser.parse_args(argv)
    sums = parse_sums(args.sums)
    if args.verify_only:
        verify_tree(args.dest, sums)
        log(f"{args.dest}: {len(sums)} files verified")
        return 0
    publish(sums, args.dest, force=args.force, sums_sha256=sha256_file(args.sums))
    return 0


if __name__ == "__main__":
    # Any failure prints one FAILED line and exits 1, so systemd marks the
    # unit failed and nothing that needs the model starts.
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"FAILED: {type(error).__name__}: {error}")
        raise SystemExit(1)
