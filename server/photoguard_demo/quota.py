"""Daily run limits for the public sample mode, saved to disk so a restart does not reset them.

The public sample mode (sample.py) lets anyone run PhotoGuard on the team's
sample faces without a password. Every run costs about 47 seconds of GPU time,
so this module allows each visitor RUNS_PER_VISITOR_PER_DAY runs and the whole
site RUNS_PER_DAY runs per day. A day is a calendar day in UTC: everything
starts again at midnight UTC (7 pm in New York in winter, 8 pm in summer).

What is stored, and why it is safe to store
-------------------------------------------
One small JSON file (mode 0600, readable by the `facade` user only):

    {"version": 1, "day": "2026-11-17", "secret": "<64 hex>", "site": 12,
     "visitors": {"3fa9c1": 2, "07be44": 3}}

* No IP address is ever stored or logged. A visitor's entry is keyed by a
  6-character code: the first 24 bits of an HMAC-SHA256 of their address
  under a random secret. About 256 IPv4 addresses (and a vast number of IPv6
  ones) share each code, so a code cannot be turned back into one address.
  With at most RUNS_PER_DAY codes stored, two visitors share a code by
  chance less than once in a thousand days, which costs them nothing worse
  than sharing three runs.
* The secret is drawn fresh every day and the file is rewritten with empty
  counts when the day changes (also by a timer, so it happens within minutes
  of midnight even if nobody visits). Codes from different days cannot be
  matched to each other.
* IPv6 visitors are counted by their /64 network, because one phone or one
  home usually gets a whole /64 and could otherwise switch addresses at will.

Where the address comes from
----------------------------
The demo listens on 127.0.0.1 only, so every request arrives through Caddy.
Caddy (trusting no proxies in front of it) replaces any X-Forwarded-For a
client sent with the client's real address. client_address() takes the last
X-Forwarded-For entry, and only when the TCP peer is a loopback address, that
is, the local Caddy. Uvicorn, which Gradio runs on, already does the same
rewrite for peers on 127.0.0.1, so the peer it reports may itself be the
visitor's address; both cases end up with the same answer.

This module uses only the Python standard library; it never imports gradio
or torch, so the tests run anywhere.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

LOG = logging.getLogger("photoguard_demo.quota")

# Runs one visitor may start per UTC day.
RUNS_PER_VISITOR_PER_DAY = 3
# Runs the whole public sample mode may start per UTC day: about two hours of GPU time.
RUNS_PER_DAY = 150
# Length of a visitor code in hex characters (6 = 24 bits; see the module docstring).
CODE_CHARS = 6
# The key used for requests whose address cannot be read (e.g. a local test without a proxy).
UNKNOWN = "unknown"
FORMAT_VERSION = 1


def today(now: float) -> str:
    """The UTC calendar day of a Unix time, as YYYY-MM-DD."""
    return time.strftime("%Y-%m-%d", time.gmtime(now))


def client_address(peer: str | None, forwarded_for: str | None) -> str:
    """The visitor's IP address as text, or UNKNOWN.

    `peer` is the TCP peer of the request and `forwarded_for` its
    X-Forwarded-For header. The header is trusted only from a loopback peer
    (the local Caddy); from anyone else it could be made up, so the peer is
    used instead.
    """
    peer = (peer or "").strip()
    last = (forwarded_for or "").split(",")[-1].strip()
    if last and _is_loopback(peer):
        return last
    return peer or UNKNOWN


def _is_loopback(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def visitor_network(address: str) -> str:
    """What one visitor is counted as: an IPv4 address, or an IPv6 /64 network.

    An IPv4 address written as IPv6 (::ffff:1.2.3.4) counts as the IPv4
    address. Anything that is not an IP address counts as UNKNOWN, so all
    such requests share one allowance.
    """
    try:
        parsed = ipaddress.ip_address(address.strip().split("%")[0])
    except ValueError:
        return UNKNOWN
    if isinstance(parsed, ipaddress.IPv6Address):
        if parsed.ipv4_mapped is not None:
            return str(parsed.ipv4_mapped)
        return str(ipaddress.IPv6Network((parsed, 64), strict=False))
    return str(parsed)


@dataclass(frozen=True)
class Ticket:
    """One counted run: the day it was counted on and the visitor code it was counted for."""
    day: str
    code: str


@dataclass(frozen=True)
class Decision:
    """The answer to reserve(): a ticket, or the reason for refusing ("visitor" or "site")."""
    ticket: Ticket | None
    reason: str | None
    left: int  # runs this visitor still has today, after this one

    @property
    def ok(self) -> bool:
        return self.ticket is not None


class DailyQuota:
    """Per-visitor and site-wide daily run counts, persisted to `path` (or kept in memory if None).

    Thread-safe: Gradio calls it from several worker threads at once.
    """

    def __init__(self, path: Path | str | None, *, per_visitor: int = RUNS_PER_VISITOR_PER_DAY,
                 per_day: int = RUNS_PER_DAY, clock: Callable[[], float] = time.time):
        if per_visitor < 1 or per_day < 1:
            raise ValueError("limits must be at least 1")
        self.path = None if path is None else Path(path)
        self.per_visitor = per_visitor
        self.per_day = per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._state = self._load()
        self._roll_over_locked()

    # --- public ---------------------------------------------------------------

    def reserve(self, address: str) -> Decision:
        """Count one run for this visitor and for the site, or refuse without counting anything."""
        with self._lock:
            self._roll_over_locked()
            code = self._code(address)
            used = self._state["visitors"].get(code, 0)
            if self._state["site"] >= self.per_day:
                return Decision(None, "site", max(0, self.per_visitor - used))
            if used >= self.per_visitor:
                return Decision(None, "visitor", 0)
            self._state["visitors"][code] = used + 1
            self._state["site"] += 1
            self._save_locked()
            return Decision(Ticket(self._state["day"], code), None, self.per_visitor - used - 1)

    def release(self, ticket: Ticket) -> None:
        """Give back a run that was counted but did not happen because the server failed.

        A ticket from an earlier day is ignored: that day's counts are gone.
        """
        with self._lock:
            self._roll_over_locked()
            if ticket.day != self._state["day"]:
                return
            visitors = self._state["visitors"]
            if visitors.get(ticket.code, 0) > 0:
                visitors[ticket.code] -= 1
                if not visitors[ticket.code]:
                    del visitors[ticket.code]
            if self._state["site"] > 0:
                self._state["site"] -= 1
            self._save_locked()

    def left(self, address: str) -> int:
        """Runs this visitor may still start today (0 also when the whole site has used its runs)."""
        with self._lock:
            self._roll_over_locked()
            if self._state["site"] >= self.per_day:
                return 0
            return max(0, self.per_visitor - self._state["visitors"].get(self._code(address), 0))

    def site_used(self) -> int:
        """Runs the whole sample mode has started today."""
        with self._lock:
            self._roll_over_locked()
            return self._state["site"]

    def roll_over(self) -> None:
        """Start a new day if the UTC date has changed (called by a timer as well as on every use)."""
        with self._lock:
            self._roll_over_locked()

    # --- internals --------------------------------------------------------------

    def _code(self, address: str) -> str:
        """The visitor code for an address under today's secret (see the module docstring)."""
        secret = bytes.fromhex(self._state["secret"])
        digest = hmac.new(secret, visitor_network(address).encode(), hashlib.sha256).hexdigest()
        return digest[:CODE_CHARS]

    def _fresh(self) -> dict:
        return {"version": FORMAT_VERSION, "day": today(self._clock()), "secret": secrets.token_hex(32),
                "site": 0, "visitors": {}}

    def _roll_over_locked(self) -> None:
        if self._state["day"] != today(self._clock()):
            self._state = self._fresh()
            self._save_locked()

    def _load(self) -> dict:
        """Today's counts from the file; a missing, unreadable or malformed file starts a fresh day."""
        if self.path is None or not self.path.exists():
            return self._fresh()
        try:
            state = json.loads(self.path.read_text())
            if not _valid_state(state):
                raise ValueError("unexpected contents")
        except (OSError, ValueError) as error:
            # Losing the counts only lets everyone run again today; refusing to start would take
            # the password-protected demo down with it.
            LOG.warning("ignoring the sample-mode counts file %s (%s); starting from zero",
                        self.path, type(error).__name__)
            return self._fresh()
        return state

    def _save_locked(self) -> None:
        """Write the counts atomically (a temporary file, then rename), readable by this user only."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(prefix=".counts-", dir=self.path.parent)
            try:
                with os.fdopen(handle, "w") as stream:
                    json.dump(self._state, stream, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        except OSError:
            # The counts in memory are still enforced; only a restart would forget them.
            LOG.exception("could not save the sample-mode counts to %s", self.path)


def _valid_state(state) -> bool:
    """True if `state` has exactly the shape this module writes."""
    if not isinstance(state, dict) or state.get("version") != FORMAT_VERSION:
        return False
    day, secret, site, visitors = (state.get(key) for key in ("day", "secret", "site", "visitors"))
    try:
        valid_secret = isinstance(secret, str) and len(bytes.fromhex(secret)) == 32
    except ValueError:
        valid_secret = False
    return (isinstance(day, str) and len(day) == 10 and valid_secret
            and type(site) is int and site >= 0 and isinstance(visitors, dict)
            and all(isinstance(code, str) and len(code) == CODE_CHARS and type(count) is int and count > 0
                    for code, count in visitors.items()))
