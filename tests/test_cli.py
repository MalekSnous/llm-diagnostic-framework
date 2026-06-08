"""The console entry points must resolve to callables."""

from llm_diagnostic import cli


def test_entry_points_exist():
    for name in ("diagnose", "improve", "report"):
        assert callable(getattr(cli, name))
