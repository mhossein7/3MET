"""Command-line interface for open-loop experiment processing."""

import argparse

from .processor import process_experiment


def add_arguments(parser):
    parser.add_argument("-a", "--address", required=True, help="Experiment root folder")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Extra features to plot. Commas are supported; fluo-growth_rate means fluo and growth_rate.",
    )
    parser.add_argument(
        "--feature-names",
        "--feature_names",
        nargs="*",
        default=None,
        help="Feature names matching the measurement rows in mothers.pkl.",
    )
    parser.add_argument(
        "--features-address",
        "--features_address",
        default=None,
        help="Path to feature_list.json or a directory containing it.",
    )
    parser.add_argument(
        "--plot-modes",
        "--plot_modes",
        nargs="*",
        default=None,
        help="Plot modes: all, channel, group, channel_group.",
    )
    parser.add_argument(
        "--plot-inputs",
        action="store_true",
        help="Draw red/green input backgrounds on group-resolved plots.",
    )
    parser.add_argument(
        "--fluorescence-label",
        "--fluorescence_label",
        default="GFP",
        help="Fluorophore name for fluorescence y-axis labels, e.g. CFP.",
    )
    parser.add_argument(
        "--interval-minutes",
        "--interval_minutes",
        type=float,
        default=5,
        help="Minutes between time-series measurements.",
    )
    parser.add_argument(
        "--tick-hours",
        "--tick_hours",
        type=float,
        default=4,
        help="Hour spacing between x-axis tick labels.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        action="append",
        metavar="FILTER",
        help="Include selected values, e.g. --include channel 1,2 group 3,4.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        action="append",
        metavar="FILTER",
        help="Exclude selected values, e.g. --exclude channel 1,2 group 3,4.",
    )
    parser.add_argument(
        "--channel-labels",
        "--channel_labels",
        nargs="*",
        default=None,
        metavar="CHANNEL:LABEL",
        help="Display labels for channels, e.g. --channel-labels 1:Control 2:Treated.",
    )
    parser.add_argument(
        "--group-labels",
        "--group_labels",
        nargs="*",
        default=None,
        metavar="GROUP:LABEL",
        help="Display labels for groups, e.g. --group-labels 1:Low 2:High.",
    )
    parser.set_defaults(func=run)
    return parser


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "OLP",
        aliases=["olp"],
        help="Process open-loop mother-machine experiments.",
        description="Process open-loop mother-machine experiments.",
    )
    return add_arguments(parser)


def run(args):
    process_experiment(
        args.address,
        features=_parse_features(args.features),
        plot_modes=args.plot_modes,
        plot_inputs=args.plot_inputs,
        feature_names=args.feature_names,
        features_address=args.features_address,
        include=_parse_filters(args.include),
        exclude=_parse_filters(args.exclude),
        channel_labels=_parse_labels(args.channel_labels),
        group_labels=_parse_labels(args.group_labels),
        fluorescence_label=args.fluorescence_label,
        interval_minutes=args.interval_minutes,
        tick_hours=args.tick_hours,
    )
    return 0


def _parse_filters(items):
    if not items:
        return None

    filters = {}
    for item in items:
        filters.update(_parse_filter_tokens(item, existing=filters))
    return filters


def _parse_filter_tokens(tokens, existing=None):
    filters = existing or {}
    idx = 0

    while idx < len(tokens):
        field = tokens[idx]
        normalized_field = field.strip().lower().replace("-", "_")
        if normalized_field not in {"channel", "channels", "group", "groups"}:
            raise ValueError(
                f"Expected filter field 'channel' or 'group', got '{field}'."
            )

        idx += 1
        values = []
        while idx < len(tokens):
            next_token = tokens[idx]
            normalized_next = next_token.strip().lower().replace("-", "_")
            if normalized_next in {"channel", "channels", "group", "groups"}:
                break
            values.extend(_split_values(next_token))
            idx += 1

        if not values:
            raise ValueError(f"Filter field '{field}' needs at least one value.")

        filters.setdefault(field, []).extend(values)

    return filters


def _parse_features(features):
    if not features:
        return None

    parsed = []
    for feature in features:
        parsed.extend(_split_feature_value(feature))
    return parsed


def _parse_labels(items):
    if not items:
        return None

    labels = {}
    for item in items:
        if ":" not in item:
            raise ValueError(
                f"Label '{item}' must use KEY:LABEL format, such as 1:Control."
            )
        key, label = item.split(":", 1)
        labels[key.strip()] = label.strip()
    return labels


def _split_feature_value(value):
    value = value.strip()
    if not value:
        return []

    parts = []
    for comma_part in _split_values(value):
        if "-" in comma_part and comma_part.lower() not in {"growth-rate"}:
            parts.extend(part for part in comma_part.split("-") if part)
        else:
            parts.append(comma_part)
    return parts


def _split_values(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="OLP CLI")
    add_arguments(parser)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
