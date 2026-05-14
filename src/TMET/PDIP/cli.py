"""Command-line interface for Pre-DeLTA image processing tools."""

import argparse
from pathlib import Path

from .utils import batch_organizer


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "pdip",
        aliases=["PDIP"],
        help="Run Pre-DeLTA image processing tools.",
        description="Run Pre-DeLTA image processing tools.",
    )
    tool_subparsers = parser.add_subparsers(
        title="PDIP tools",
        dest="pdip_command",
        metavar="<pdip-tool>",
        required=True,
    )
    register_manual_organizer(tool_subparsers)
    return parser


def register_manual_organizer(subparsers):
    parser = subparsers.add_parser(
        "manual-organizer",
        help="Organize manually prepared image batches into DeLTA-compatible files.",
        description="Organize manually prepared image batches into DeLTA-compatible files.",
    )
    parser.add_argument("-a", "--address", required=True, help="Experiment root folder")
    parser.add_argument(
        "--imaged",
        default="imaged",
        help="Name of the subdirectory containing imaged files.",
    )
    parser.add_argument(
        "--unimaged",
        default="unimaged",
        help="Name of the subdirectory containing unimaged files.",
    )
    parser.add_argument(
        "--constant",
        default="_",
        help="Prefix for the subdirectory containing TIFF files.",
    )
    parser.set_defaults(func=run_manual_organizer)
    return parser


def run_manual_organizer(args):
    batch_organizer(
        str(Path(args.address)),
        imaged=args.imaged,
        unimaged=args.unimaged,
        constant=args.constant,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description="PDIP CLI")
    subparsers = parser.add_subparsers(
        title="PDIP tools",
        dest="pdip_command",
        metavar="<pdip-tool>",
        required=True,
    )
    register_manual_organizer(subparsers)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
