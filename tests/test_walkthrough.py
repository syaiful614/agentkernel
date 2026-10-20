"""The narrated demo is part of the contract: it must keep telling the truth."""

import importlib.util
from pathlib import Path


def test_the_walkthrough_demonstrates_every_guarantee(capsys):
    path = Path(__file__).parent.parent / "examples" / "walkthrough.py"
    spec = importlib.util.spec_from_file_location("walkthrough", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    out = capsys.readouterr().out
    for expected in (
        "send_email -> DENY by 'trifecta'; emails sent: 0",
        "after approval: completed; emails sent: 1 (exactly one)",
        "recorded as ['uncertain']; times charged: 1 (not repeated)",
        "before: intact",
        "was modified, or its predecessor removed",
        "same code: matches the recording",
        "looser policy: policy decision for call 1.0 changed: recorded ask, now allow",
    ):
        assert expected in out
