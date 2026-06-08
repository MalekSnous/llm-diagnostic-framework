"""
Console entry points for the LLM Diagnostic Framework.

These are thin wrappers around the argparse-based scripts in ``scripts/`` so
that the ``[project.scripts]`` declarations in ``pyproject.toml`` resolve to
real callables:

    llm-diagnose -> llm_diagnostic.cli:diagnose
    llm-improve  -> llm_diagnostic.cli:improve
    llm-report   -> llm_diagnostic.cli:report

Each wrapper simply delegates to the corresponding ``main()`` which parses
``sys.argv`` itself.
"""

from __future__ import annotations


def diagnose() -> None:
    """Run the diagnostic suite (``llm-diagnose``)."""
    from scripts.run_diagnostics import main

    main()


def improve() -> None:
    """Run improvement strategies against a baseline (``llm-improve``)."""
    from scripts.run_improvements import main

    main()


def report() -> None:
    """Generate an HTML report from results (``llm-report``)."""
    from scripts.generate_report import main

    main()
