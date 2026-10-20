import sqlite3

import pytest

from agentkernel import Journal


def test_append_assigns_gapless_sequence_numbers_per_run():
    j = Journal()
    j.create_run("a", "t")
    j.create_run("b", "t")
    for i in range(3):
        j.append("a", "x", {"i": i})
    j.append("b", "x", {})
    assert [e.seq for e in j.events("a")] == [1, 2, 3]
    assert [e.seq for e in j.events("b")] == [1]
    assert [e.data["i"] for e in j.events("a", after=1)] == [1, 2]


def test_chain_verifies_when_untouched():
    j = Journal()
    j.create_run("a", "t")
    for i in range(5):
        j.append("a", "x", {"i": i})
    assert j.verify("a") == []


def test_editing_an_event_is_detected_and_localised():
    j = Journal()
    j.create_run("a", "t")
    for i in range(5):
        j.append("a", "x", {"i": i})
    j._conn.execute("UPDATE events SET data = '{\"i\":99}' WHERE run_id = 'a' AND seq = 3")
    problems = j.verify("a")
    assert len(problems) == 1 and "event 3" in problems[0]


def test_removing_a_middle_event_is_detected():
    j = Journal()
    j.create_run("a", "t")
    for i in range(5):
        j.append("a", "x", {"i": i})
    j._conn.execute("DELETE FROM events WHERE run_id = 'a' AND seq = 2")
    text = " ".join(j.verify("a"))
    assert "event 2 is missing" in text and "event 3" in text


def test_persists_to_disk_and_survives_reopening(tmp_path):
    path = tmp_path / "sub" / "journal.db"
    j = Journal(path)
    j.create_run("a", "task")
    j.append("a", "run_started", {"task": "task"})
    j.close()
    again = Journal(path)
    assert again.get_run("a").task == "task" and len(again.events("a")) == 1
    assert again.verify("a") == []


def test_runs_listing_and_status():
    j = Journal()
    j.create_run("a", "first")
    j.create_run("b", "second")
    j.set_status("a", "completed")
    runs = j.runs()
    assert [r.run_id for r in runs] == ["b", "a"] and runs[1].status == "completed"
    assert j.get_run("zzz") is None


def test_lease_blocks_a_second_worker_until_it_expires_or_is_released():
    now = [1000.0]
    j = Journal(clock=lambda: now[0])
    assert j.claim("r", "w1", ttl=60)
    assert j.claim("r", "w1", ttl=60)  # the owner may renew
    assert not j.claim("r", "w2", ttl=60)
    now[0] += 61  # w1 "crashed": its lease runs out
    assert j.claim("r", "w2", ttl=60)
    j.release("r", "w1")  # a stale release by the old owner must not drop the new lease
    assert not j.claim("r", "w1", ttl=60)
    j.release("r", "w2")
    assert j.claim("r", "w1", ttl=60)


def test_a_failed_write_rolls_back_cleanly():
    j = Journal()
    j.create_run("a", "t")
    with pytest.raises(sqlite3.Error):
        j._conn.execute("INSERT INTO events (run_id) VALUES ('a')")
    j.append("a", "x", {})
    assert [e.seq for e in j.events("a")] == [1]
