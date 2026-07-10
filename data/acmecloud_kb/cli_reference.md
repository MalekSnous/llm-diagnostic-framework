# CLI Reference (`acme`)

## Installation

The recommended installation method for the CLI is `pipx install acme-cli`.
The CLI requires Python 3.10 or newer. The CLI major version is pinned to the
platform major version (a v3.x CLI talks to a v3.x platform).

## Configuration

The CLI reads its configuration from `~/.acme/config.toml`. The default output
format is a human-readable table; pass `--output json` for machine-readable
output.

## Common commands

- `acme auth rotate` — rotate the current AKT without downtime.
- `acme pipeline deploy --env <name>` — deploy a pipeline to an environment.
- `acme dataset ls` — list datasets in the current project.
- `acme logs tail <pipeline>` — stream live logs for a running pipeline.
- `acme doctor` — diagnose connectivity and configuration problems.
- `acme doctor --bundle` — produce a diagnostics bundle to attach to a support
  ticket.
