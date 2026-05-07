# 3MET / TMET

Mother Machine Microscopy Experiments Tools.

The GitHub repository is named `3MET`, while the Python package and command are
named `TMET` because Python import packages cannot start with a digit.

## Tools

### `moma-movie-maker`

Create annotated MP4 movies from mother-machine microscopy TIFF frame folders.
The tool can use ROI metadata when available, or fall back to image-derived crop
boundaries.

```bash
TMET moma-movie-maker -a /path/to/experiment
```

The command also has a lowercase alias:

```bash
tmet moma-movie-maker -a /path/to/experiment
```

## Development Layout

New tools should live under `src/TMET/<tool_name>/` and expose a
`register_parser(subparsers)` function from their `cli.py`. The top-level
`src/TMET/cli.py` imports each tool's registration function and attaches it as a
subcommand.

For example, a future growth-rate tool could be added as:

```text
src/TMET/growth_rate/
    __init__.py
    cli.py
    processor.py
```

and exposed as:

```bash
TMET growth-rate -a /path/to/experiment
```
