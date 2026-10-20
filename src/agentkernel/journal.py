"""The journal: an append-only, hash-chained event log in SQLite. It is the source of truth.

Nothing about a run lives in memory between steps; the run's state is *derived* by replaying
its events. That single decision is what makes crash recovery, durable approvals, deterministic
replay and audit all fall out of the same mechanism.

Each event stores a hash over (previous hash, sequence number, kind, data, timestamp), so
editing or removing an event is detectable (`verify`). Note the limit: truncating the *tail* of
a log cannot be detected by the chain alone; anchor the head hash externally if you need that.

Leases give single-writer semantics per run: a worker must hold the run's lease to advance it,
and a crashed worker's lease simply expires.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import canonical_json

GENESIS = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_ts REAL NOT NULL, status TEXT NOT NULL, task TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL, kind TEXT NOT NULL,
    data TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY (run_id, seq));
CREATE TABLE IF NOT EXISTS leases (
    run_id TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL);
"""


def _chain_hash(prev: str, seq: int, kind: str, data_json: str, ts: float) -> str:
    return hashlib.sha256(f"{prev}|{seq}|{kind}|{data_json}|{ts!r}".encode()).hexdigest()


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int
    ts: float
    kind: str
    data: dict[str, Any]
    hash: str


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    created_ts: float
    status: str
    task: str


class Journal:
    def __init__(self, path: str | Path = ":memory:", *, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.RLock()
        target = str(path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        # autocommit mode; explicit BEGIN IMMEDIATE makes each write atomic across processes too
        self._conn = sqlite3.connect(
            target, timeout=30, check_same_thread=False, isolation_level=None
        )
        if target != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ runs
    def create_run(self, run_id: str, task: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, created_ts, status, task) VALUES (?, ?, 'running', ?)",
                (run_id, self._clock(), task),
            )

    def set_status(self, run_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id, created_ts, status, task FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return RunRecord(*row) if row else None

    def runs(self, limit: int = 50) -> list[RunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, created_ts, status, task FROM runs "
                "ORDER BY created_ts DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [RunRecord(*r) for r in rows]

    # ------------------------------------------------------------------ events
    def append(self, run_id: str, kind: str, data: dict[str, Any]) -> Event:
        data_json = canonical_json(data)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT seq, hash FROM events WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
                seq, prev = (row[0] + 1, row[1]) if row else (1, GENESIS)
                ts = self._clock()
                h = _chain_hash(prev, seq, kind, data_json, ts)
                self._conn.execute(
                    "INSERT INTO events (run_id, seq, ts, kind, data, hash) VALUES (?,?,?,?,?,?)",
                    (run_id, seq, ts, kind, data_json, h),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return Event(run_id, seq, ts, kind, json.loads(data_json), h)

    def events(self, run_id: str, after: int = 0) -> list[Event]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, ts, kind, data, hash FROM events WHERE run_id = ? AND seq > ? "
                "ORDER BY seq",
                (run_id, after),
            ).fetchall()
        return [Event(run_id, s, ts, k, json.loads(d), h) for s, ts, k, d, h in rows]

    def verify(self, run_id: str) -> list[str]:
        """Recompute the hash chain. Returns human-readable problems (empty list = intact)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, ts, kind, data, hash FROM events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        problems: list[str] = []
        prev, expected = GENESIS, 1
        for seq, ts, kind, data_json, stored in rows:
            if seq != expected:
                problems.append(f"event {expected} is missing (next event is {seq})")
            if _chain_hash(prev, seq, kind, data_json, ts) != stored:
                problems.append(f"event {seq} ({kind}) was modified, or its predecessor removed")
            prev, expected = stored, seq + 1
        return problems

    # ------------------------------------------------------------------ leases
    def claim(self, run_id: str, owner: str, ttl: float) -> bool:
        """Acquire or renew the run's lease. False if another live worker holds it."""
        now = self._clock()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT owner, expires FROM leases WHERE run_id = ?", (run_id,)
                ).fetchone()
                ok = row is None or row[1] <= now or row[0] == owner
                if ok:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO leases (run_id, owner, expires) VALUES (?, ?, ?)",
                        (run_id, owner, now + ttl),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return bool(ok)

    def release(self, run_id: str, owner: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM leases WHERE run_id = ? AND owner = ?", (run_id, owner))
