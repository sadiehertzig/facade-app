"""Tests for photoguard_demo/quota.py: the public sample mode's daily run limits.

Standard library only, like the module: no gradio, torch or network.
"""
from __future__ import annotations

import json
import stat
import threading

import pytest

from photoguard_demo import quota
from photoguard_demo.quota import DailyQuota, client_address, visitor_network

DAY = 1_763_337_600.0  # 2025-11-17 00:00:00 UTC
HOUR = 3600.0


class Clock:
    """A settable stand-in for time.time."""

    def __init__(self, now=DAY + 10 * HOUR):
        self.now = now

    def __call__(self):
        return self.now


@pytest.mark.parametrize("peer, forwarded, expected", [
    ("127.0.0.1", "203.0.113.9", "203.0.113.9"),  # from the local Caddy: its X-Forwarded-For
    ("::1", "2001:db8::5", "2001:db8::5"),
    ("127.0.0.1", "10.0.0.1, 203.0.113.9", "203.0.113.9"),  # only the last entry, the one Caddy wrote
    ("127.0.0.1", " 203.0.113.9 ", "203.0.113.9"),
    ("198.51.100.4", "203.0.113.9", "198.51.100.4"),  # not from Caddy: the header could be made up
    ("198.51.100.4", None, "198.51.100.4"),  # uvicorn already put the visitor's address in the peer
    ("127.0.0.1", None, "127.0.0.1"),  # a local request without Caddy
    ("127.0.0.1", "", "127.0.0.1"),
    (None, None, quota.UNKNOWN),
    ("not-an-ip", "203.0.113.9", "not-an-ip"),
])
def test_client_address_trusts_forwarded_for_only_from_loopback(peer, forwarded, expected):
    """X-Forwarded-For counts only when the request came from the local Caddy; otherwise the peer does."""
    assert client_address(peer, forwarded) == expected


@pytest.mark.parametrize("address, expected", [
    ("203.0.113.9", "203.0.113.9"),
    ("2001:db8:1:2:aaaa:bbbb:cccc:dddd", "2001:db8:1:2::/64"),
    ("2001:db8:1:2::1", "2001:db8:1:2::/64"),
    ("::ffff:203.0.113.9", "203.0.113.9"),
    ("fe80::1%en0", "fe80::/64"),
    ("not-an-ip", quota.UNKNOWN),
    ("", quota.UNKNOWN),
])
def test_visitor_network_counts_ipv6_by_its_64(address, expected):
    """An IPv6 visitor is one /64 network; an IPv4 address written as IPv6 is the IPv4 address."""
    assert visitor_network(address) == expected


def test_each_visitor_gets_three_runs_a_day_and_refusals_cost_nothing():
    """The fourth run of a visitor is refused; other visitors are unaffected; a refusal is not counted."""
    counts = DailyQuota(None, per_visitor=3, per_day=150, clock=Clock())
    lefts = [counts.reserve("203.0.113.9").left for _ in range(3)]
    assert lefts == [2, 1, 0]
    refused = counts.reserve("203.0.113.9")
    assert not refused.ok and refused.reason == "visitor" and refused.left == 0
    assert counts.site_used() == 3 and counts.left("203.0.113.9") == 0
    assert counts.reserve("198.51.100.4").ok and counts.left("198.51.100.4") == 2
    # Two addresses in the same IPv6 /64 share one allowance.
    for _ in range(3):
        assert counts.reserve("2001:db8:1:2::1").ok
    assert counts.reserve("2001:db8:1:2::ffff").reason == "visitor"
    assert counts.reserve("2001:db8:1:3::1").ok


def test_the_whole_site_stops_at_its_daily_cap():
    """Once the site has used its runs, everyone is refused with reason "site", and nothing more is counted."""
    counts = DailyQuota(None, per_visitor=3, per_day=4, clock=Clock())
    for address in ("10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.3"):
        assert counts.reserve(address).ok
    refused = counts.reserve("10.0.0.4")
    assert not refused.ok and refused.reason == "site" and refused.left == 3
    assert counts.left("10.0.0.4") == 0 and counts.site_used() == 4


def test_release_gives_back_a_run_but_not_across_days():
    """A run the server failed is given back to the visitor and the site, on the day it was counted only."""
    clock = Clock()
    counts = DailyQuota(None, per_visitor=1, per_day=10, clock=clock)
    ticket = counts.reserve("10.0.0.1").ticket
    assert counts.left("10.0.0.1") == 0
    counts.release(ticket)
    assert counts.left("10.0.0.1") == 1 and counts.site_used() == 0
    ticket = counts.reserve("10.0.0.1").ticket
    clock.now += 24 * HOUR
    assert counts.reserve("10.0.0.1").ok  # a new day
    counts.release(ticket)  # yesterday's ticket: ignored
    assert counts.left("10.0.0.1") == 0 and counts.site_used() == 1


def test_midnight_utc_starts_a_new_day_with_a_new_secret():
    """At midnight UTC the counts start again and the visitor codes change."""
    clock = Clock(DAY + 23 * HOUR + 3599)
    counts = DailyQuota(None, per_visitor=3, per_day=10, clock=clock)
    code = counts.reserve("10.0.0.1").ticket.code
    assert counts.left("10.0.0.1") == 2
    clock.now = DAY + 24 * HOUR  # midnight UTC
    assert counts.left("10.0.0.1") == 3 and counts.site_used() == 0
    new = counts.reserve("10.0.0.1").ticket
    assert new.day == "2025-11-18" and new.code != code  # (a 1-in-16-million coincidence otherwise)


def test_counts_survive_a_restart_and_the_file_holds_no_address(tmp_path):
    """A second DailyQuota on the same file (a restarted server) sees today's counts; the file has no IP address."""
    path = tmp_path / "state" / "counts.json"
    clock = Clock()
    first = DailyQuota(path, per_visitor=3, per_day=150, clock=clock)
    for _ in range(3):
        first.reserve("203.0.113.9")
    first.reserve("2001:db8:1:2::1")
    restarted = DailyQuota(path, per_visitor=3, per_day=150, clock=clock)
    assert restarted.left("203.0.113.9") == 0 and restarted.left("2001:db8:1:2::7") == 2
    assert restarted.site_used() == 4
    text = path.read_text()
    assert "203.0.113" not in text and "2001" not in text and "db8" not in text
    state = json.loads(text)
    assert set(state) == {"version", "day", "secret", "site", "visitors"}
    assert all(len(code) == quota.CODE_CHARS for code in state["visitors"])
    assert sorted(state["visitors"].values()) == [1, 3]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert [p.name for p in path.parent.iterdir()] == ["counts.json"], "no temporary files left behind"


def test_a_new_day_overwrites_yesterdays_file(tmp_path):
    """Starting after midnight (or the timer's roll_over) rewrites the file with empty counts and a new secret."""
    path = tmp_path / "counts.json"
    clock = Clock()
    DailyQuota(path, clock=clock).reserve("10.0.0.1")
    yesterday = json.loads(path.read_text())
    clock.now += 24 * HOUR
    counts = DailyQuota(path, clock=clock)
    today = json.loads(path.read_text())
    assert today["day"] != yesterday["day"] and today["secret"] != yesterday["secret"]
    assert today["site"] == 0 and today["visitors"] == {}
    clock.now += 24 * HOUR
    counts.roll_over()
    assert json.loads(path.read_text())["day"] == "2025-11-19"


@pytest.mark.parametrize("contents", [
    "not json", "[]", '{"version": 2}',
    '{"version": 1, "day": "2025-11-17", "secret": "zz", "site": 0, "visitors": {}}',
    '{"version": 1, "day": "2025-11-17", "secret": "' + "ab" * 32 + '", "site": -1, "visitors": {}}',
    '{"version": 1, "day": "2025-11-17", "secret": "' + "ab" * 32 + '", "site": 1, "visitors": {"x": 1}}',
])
def test_an_unreadable_counts_file_starts_from_zero(tmp_path, contents):
    """A damaged file must not stop the demo (it shares a process with the full demo); it starts a fresh day."""
    path = tmp_path / "counts.json"
    path.write_text(contents)
    counts = DailyQuota(path, clock=Clock())
    assert counts.site_used() == 0 and counts.reserve("10.0.0.1").ok
    assert json.loads(path.read_text())["site"] == 1


def test_an_unwritable_counts_file_still_enforces_the_limit_in_memory(tmp_path):
    """If the file cannot be written, the counts in memory still apply."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    counts = DailyQuota(blocker / "counts.json", per_visitor=1, per_day=5, clock=Clock())
    assert counts.reserve("10.0.0.1").ok
    assert counts.reserve("10.0.0.1").reason == "visitor"


def test_limits_hold_under_concurrent_requests():
    """Many threads reserving at once never get more runs than the limits allow."""
    counts = DailyQuota(None, per_visitor=3, per_day=20, clock=Clock())
    granted = []

    def worker(index):
        for _ in range(5):
            if counts.reserve(f"10.0.{index % 4}.1").ok:
                granted.append(index % 4)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(granted) == 12 and all(granted.count(v) == 3 for v in range(4))
    assert counts.site_used() == 12


def test_limits_must_be_positive():
    with pytest.raises(ValueError):
        DailyQuota(None, per_visitor=0)
    with pytest.raises(ValueError):
        DailyQuota(None, per_day=0)
