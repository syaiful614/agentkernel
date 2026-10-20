import json
import types

import pytest

from agentkernel import Journal, cli
from agentkernel.cli import format_event
from fakes import ScriptedLLM, calls, text

EMAIL = {"to": "manager@example.com", "subject": "s", "body": "b"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    holder = {"llm": None}
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: holder["llm"])
    db, ws = tmp_path / "journal.db", tmp_path / "ws"
    return types.SimpleNamespace(
        holder=holder,
        db=db,
        ws=ws,
        base=["--db", str(db), "--workspace", str(ws)],
        llm=["--provider", "openai", "--model", "test-model"],
    )


def run_id_of(env):
    return Journal(env.db).runs()[0].run_id


def test_run_completes_and_prints_the_answer(env, capsys):
    env.holder["llm"] = ScriptedLLM([text("hello there")])
    assert cli.main(["run", "say hi", *env.base, *env.llm]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "hello there" and "[completed | run " in out.err


def test_the_full_human_in_the_loop_flow_across_separate_invocations(env, capsys):
    env.holder["llm"] = ScriptedLLM([calls(("send_email", EMAIL))])
    assert cli.main(["run", "email my manager", *env.base, *env.llm]) == 3  # needs a human
    err = capsys.readouterr().err
    run_id = run_id_of(env)
    assert f"agentkernel approve {run_id} 1.0" in err and f"agentkernel resume {run_id}" in err
    assert not (env.ws / "outbox.jsonl").exists()

    assert cli.main(["approve", run_id, "1.0", "--note", "ok", *env.base]) == 0
    assert "approved call 1.0" in capsys.readouterr().out

    env.holder["llm"] = ScriptedLLM([text("sent")])
    assert cli.main(["resume", run_id, *env.base, *env.llm]) == 0
    assert capsys.readouterr().out.strip() == "sent"
    assert json.loads((env.ws / "outbox.jsonl").read_text()) == EMAIL


def test_deny_blocks_the_call(env, capsys):
    env.holder["llm"] = ScriptedLLM([calls(("send_email", EMAIL))])
    cli.main(["run", "email", *env.base, *env.llm])
    run_id = run_id_of(env)
    assert cli.main(["deny", run_id, "1.0", *env.base]) == 0
    assert "rejected call 1.0" in capsys.readouterr().out
    env.holder["llm"] = ScriptedLLM([text("not sent")])
    assert cli.main(["resume", run_id, *env.base, *env.llm]) == 0
    assert not (env.ws / "outbox.jsonl").exists()


def completed_run(env):
    env.holder["llm"] = ScriptedLLM([calls(("calculator", {"expression": "6*7"})), text("42")])
    cli.main(["run", "what is 6*7", *env.base, *env.llm])
    return run_id_of(env)


def test_list_and_show(env, capsys):
    run_id = completed_run(env)
    capsys.readouterr()
    assert cli.main(["list", *env.base]) == 0
    listing = capsys.readouterr().out
    assert run_id in listing and "completed" in listing and "what is 6*7" in listing

    assert cli.main(["show", run_id, *env.base]) == 0
    timeline = capsys.readouterr().out
    for word in ("run_started", "policy_decision", "ALLOW", "tool_result", "run_finished"):
        assert word in timeline

    assert cli.main(["show", run_id, "--json", *env.base]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["seq"] for r in rows] == list(range(1, len(rows) + 1))


def test_an_empty_journal_lists_nothing(env, capsys):
    assert cli.main(["list", *env.base]) == 0
    assert "(no runs yet)" in capsys.readouterr().out


def test_verify_detects_tampering(env, capsys):
    run_id = completed_run(env)
    capsys.readouterr()
    assert cli.main(["verify", run_id, *env.base]) == 0
    assert "journal intact" in capsys.readouterr().out

    journal = Journal(env.db)
    journal._conn.execute(
        "UPDATE events SET data = replace(data, '42', '43') "
        "WHERE run_id = ? AND kind = 'tool_result'",
        (run_id,),
    )
    journal.close()
    assert cli.main(["verify", run_id, *env.base]) == 1
    assert "TAMPERED" in capsys.readouterr().out


def test_replay_matches_and_then_catches_a_policy_change(env, capsys, tmp_path):
    run_id = completed_run(env)
    capsys.readouterr()
    assert cli.main(["replay", run_id, *env.base]) == 0
    assert "replay matches the recording" in capsys.readouterr().out

    stricter = tmp_path / "strict.json"
    stricter.write_text(json.dumps({"default": "deny", "rules": []}))
    assert cli.main(["replay", run_id, "--policy", str(stricter), *env.base]) == 1
    assert "DIVERGES" in capsys.readouterr().out


def test_redteam_command(capsys, tmp_path):
    assert cli.main(["redteam"]) == 0
    out = capsys.readouterr().out
    assert "[PASS] exfil_via_poisoned_inbox" in out and "attack scenarios contained" in out

    failing = {
        "name": "bad",
        "task": "x",
        "turns": [{"say": "nothing"}],
        "expect": {"outbox_count": 5},
    }
    (tmp_path / "bad.json").write_text(json.dumps(failing))
    assert cli.main(["redteam", "--scenarios", str(tmp_path)]) == 1
    assert "[FAIL] bad" in capsys.readouterr().out


def test_errors_are_one_clean_line(env, capsys, tmp_path):
    assert cli.main(["show", "nope", *env.base]) == 1
    assert "error: unknown run 'nope'" in capsys.readouterr().err

    assert cli.main(["run", "x", *env.base, "--provider", "openai"]) == 1
    assert "Set --model" in capsys.readouterr().err

    env.holder["llm"] = ScriptedLLM([])
    missing = str(tmp_path / "missing.json")
    assert cli.main(["run", "x", *env.base, *env.llm, "--policy", missing]) == 1
    assert "cannot read policy" in capsys.readouterr().err


def test_verbose_streams_events_to_stderr(env, capsys):
    env.holder["llm"] = ScriptedLLM([text("hi")])
    cli.main(["run", "x", "-v", *env.base, *env.llm])
    err = capsys.readouterr().err
    assert "run_started" in err and "llm_response" in err and "run_finished" in err


def test_every_event_kind_has_a_readable_line(env):
    env.holder["llm"] = ScriptedLLM([calls(("send_email", EMAIL)), text("sent")])
    cli.main(["run", "email", *env.base, *env.llm])
    run_id = run_id_of(env)
    cli.main(["approve", run_id, "1.0", *env.base])
    cli.main(["resume", run_id, *env.base, *env.llm])
    lines = [format_event(e) for e in Journal(env.db).events(run_id)]
    text_ = "\n".join(lines)
    for word in ("needs a human", "APPROVED", "started", "waiting for approval", "completed"):
        assert word in text_


def test_the_human_sees_the_whole_input_they_are_asked_to_approve(env, capsys):
    hidden = "ATTACKER-PAYLOAD-AFTER-A-LONG-PREFIX"
    long_email = {**EMAIL, "body": "x" * 400 + hidden}
    env.holder["llm"] = ScriptedLLM([calls(("send_email", long_email))])
    assert cli.main(["run", "email", *env.base, *env.llm]) == 3
    assert hidden in capsys.readouterr().err  # nothing is hidden behind a preview limit
