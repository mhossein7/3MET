"""Command-line entry point for TMET."""

import argparse

from . import __version__
from .OLP.cli import register_parser as register_olp
from .PDIP.cli import register_parser as register_pdip
from .moma_movie_maker.cli import register_parser as register_moma_movie_maker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TMET",
        description="Mother Machine Microscopy Experiments Tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        title="tools",
        dest="command",
        metavar="<tool>",
        required=True,
    )
    register_moma_movie_maker(subparsers)
    register_olp(subparsers)
    register_pdip(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
