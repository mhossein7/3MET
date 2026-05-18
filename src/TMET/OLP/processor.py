"""Processing workflow for open-loop mother-machine experiments."""

from pathlib import Path

import numpy as np

from .utils import compute_growth, mothers_to_df_loader, resolve_groups
from .visualization import (
    plot_cell_median_violin,
    plot_feature_cycle_comparison,
    plot_feature_group_comparison,
    plot_feature_summary,
    plot_feature_summary_grid,
)


DEFAULT_PLOT_MODES = ("all", "channel", "group", "channel_group")
DEFAULT_FEATURE = "fluorescence"
FLUORESCENCE_ALIASES = ("fluorescence", "fluorescecne", "fluo")
GROWTH_RATE_FEATURE = "growth_rate"
GROWTH_RATE_ALIASES = ("growth_rate", "growth rate", "growth-rate", "growthrate")
AREA_ALIASES = ("area",)
PLOT_CHANNEL_COLUMN = "plot_channel"
PLOT_GROUP_COLUMN = "plot_group"


def process_experiment(
    root,
    features=None,
    plot_modes=DEFAULT_PLOT_MODES,
    plot_inputs=False,
    feature_names=None,
    features_address=None,
    include=None,
    exclude=None,
    merge=None,
    channel_labels=None,
    group_labels=None,
    fluorescence_label="GFP",
    interval_minutes=5,
    tick_hours=4,
    default_feature=DEFAULT_FEATURE,
):
    """
    Load an open-loop experiment directory and save summary plots.

    Parameters
    ----------
    root : str or pathlib.Path
        Experiment directory containing mothers.pkl, cells_stims.npy, and usually
        group_config.json.
    features : str or list[str], optional
        Additional feature column(s) to plot besides all fluorescence-like
        readouts. If no fluorescence-like readout exists, at least one explicit
        feature must be provided.
    plot_modes : iterable[str]
        Any of: 'all', 'channel', 'group', 'channel_group'.
    plot_inputs : bool
        If True, draw the corresponding red/green input sequence behind
        group-resolved plots.
    feature_names, features_address
        Passed through to mothers_to_df_loader for naming feature columns.
    include, exclude : dict, optional
        Optional filters with 'channel' and/or 'group' keys. Filtered plots are
        saved under plots/custom.
    merge : dict, optional
        Optional channel/group merge spec. Matching channels and/or groups are
        plotted as one user-defined channel/group.
    channel_labels, group_labels : dict or list, optional
        Optional display labels for plot legends and subplot titles.
    fluorescence_label : str
        Fluorophore name used for fluorescence y-axis labels.
    interval_minutes : int or float
        Minutes between consecutive time-series measurements.
    tick_hours : int or float
        Hour spacing for x-axis ticks.

    Returns
    -------
    pandas.DataFrame
        Cell-level table generated from mothers.pkl.
    """
    root = Path(root)
    channel_labels = _normalize_label_mapping(channel_labels, "channel")
    group_labels = _normalize_label_mapping(group_labels, "group")
    df = mothers_to_df_loader(
        root,
        feature_names=feature_names,
        features_address=features_address,
    )
    group_config = resolve_groups(root)

    df = _add_requested_derived_features(df, features)
    df = _filter_dataframe(df, include=include, exclude=exclude)
    df = _prepare_plot_columns(df, merge=merge)
    requested_features = _resolve_requested_features(df, default_feature, features)
    if plot_modes is None:
        plot_modes = DEFAULT_PLOT_MODES
    normalized_modes = _normalize_plot_modes(plot_modes)
    plots_root = root / "plots"
    if _has_merge(merge):
        plots_root = plots_root / "custom" / "merged" / _merge_plot_dir_name(merge)
    elif _has_filters(include) or _has_filters(exclude):
        plots_root = plots_root / "custom" / _custom_plot_dir_name(df)

    for feature in requested_features:
        if "all" in normalized_modes:
            _plot_all(
                df,
                feature,
                plots_root,
                fluorescence_label,
                interval_minutes,
                tick_hours,
            )

        if "channel" in normalized_modes:
            _plot_by_channel(
                df,
                feature,
                plots_root,
                channel_labels,
                fluorescence_label,
                interval_minutes,
                tick_hours,
            )

        if "group" in normalized_modes:
            _plot_by_group(
                df,
                feature,
                plots_root,
                group_config,
                plot_inputs,
                group_labels,
                fluorescence_label,
                interval_minutes,
                tick_hours,
            )

        if "channel_group" in normalized_modes:
            _plot_by_channel_and_group(
                df,
                feature,
                plots_root,
                group_config,
                plot_inputs,
                channel_labels,
                group_labels,
                fluorescence_label,
                interval_minutes,
                tick_hours,
            )

    return df


def process_xpt(*args, **kwargs):
    """Backward-compatible alias for process_experiment."""
    return process_experiment(*args, **kwargs)


def _plot_all(
    df,
    feature,
    plots_root,
    fluorescence_label,
    interval_minutes,
    tick_hours,
):
    output_path = plots_root / "all" / f"{_safe_name(feature)}.png"
    plot_feature_summary(
        df,
        feature,
        output_path,
        title=f"{feature} - all cells",
        ylabel=_ylabel_for_feature(feature, fluorescence_label),
        interval_minutes=interval_minutes,
        tick_hours=tick_hours,
    )


def _plot_by_channel(
    df,
    feature,
    plots_root,
    channel_labels=None,
    fluorescence_label="GFP",
    interval_minutes=5,
    tick_hours=4,
):
    subsets = [
        (_channel_display_label(channel, channel_labels), channel_df)
        for channel, channel_df in df.groupby(_channel_column(df), sort=True)
    ]
    output_path = plots_root / "channel-based" / f"{_safe_name(feature)}.png"
    plot_feature_summary_grid(
        subsets,
        feature,
        output_path,
        title=f"{feature} by channel",
        ylabel=_ylabel_for_feature(feature, fluorescence_label),
        interval_minutes=interval_minutes,
        tick_hours=tick_hours,
    )
    if _is_growth_rate_feature(feature):
        comparison_path = (
            plots_root / "channel-based" / f"{_safe_name(feature)}_comparison.png"
        )
        plot_feature_cycle_comparison(
            subsets,
            feature,
            comparison_path,
            title=f"{feature} channel comparison",
            ylabel=_ylabel_for_feature(feature, fluorescence_label),
            interval_minutes=interval_minutes,
            tick_hours=tick_hours,
        )
        violin_path = (
            plots_root
            / "channel-based"
            / f"{_safe_name(feature)}_cell_median_violin.png"
        )
        plot_cell_median_violin(
            subsets,
            feature,
            violin_path,
            title=f"{feature} cell-median by channel",
        )


def _plot_by_group(
    df,
    feature,
    plots_root,
    group_config,
    plot_inputs,
    group_labels=None,
    fluorescence_label="GFP",
    interval_minutes=5,
    tick_hours=4,
):
    subsets = [
        (_group_display_label(group, group_labels), group_df)
        for group, group_df in df.groupby(_group_column(df), sort=True)
    ]
    group_stim_sequences = {
        _group_display_label(group, group_labels): _stim_for_group(group_config, group)
        for group, _ in df.groupby(_group_column(df), sort=True)
    }
    stim_sequences = {}
    if plot_inputs:
        stim_sequences = group_stim_sequences
    output_path = plots_root / "group-based" / f"{_safe_name(feature)}.png"
    plot_feature_summary_grid(
        subsets,
        feature,
        output_path,
        title=f"{feature} by group",
        stim_sequences=stim_sequences,
        ylabel=_ylabel_for_feature(feature, fluorescence_label),
        interval_minutes=interval_minutes,
        tick_hours=tick_hours,
    )
    comparison_path = (
        plots_root / "group-based" / f"{_safe_name(feature)}_comparison.png"
    )
    if _is_growth_rate_feature(feature):
        plot_feature_cycle_comparison(
            subsets,
            feature,
            comparison_path,
            title=f"{feature} group comparison",
            ylabel=_ylabel_for_feature(feature, fluorescence_label),
            interval_minutes=interval_minutes,
            tick_hours=tick_hours,
        )
        violin_path = (
            plots_root / "group-based" / f"{_safe_name(feature)}_cell_median_violin.png"
        )
        plot_cell_median_violin(
            subsets,
            feature,
            violin_path,
            title=f"{feature} cell-median by group",
        )
    else:
        plot_feature_group_comparison(
            subsets,
            feature,
            comparison_path,
            group_stim_sequences,
            title=f"{feature} group comparison",
            ylabel=_ylabel_for_feature(feature, fluorescence_label),
            interval_minutes=interval_minutes,
            tick_hours=tick_hours,
        )


def _plot_by_channel_and_group(
    df,
    feature,
    plots_root,
    group_config,
    plot_inputs,
    channel_labels=None,
    group_labels=None,
    fluorescence_label="GFP",
    interval_minutes=5,
    tick_hours=4,
):
    for channel, channel_df in df.groupby(_channel_column(df), sort=True):
        subsets = [
            (_group_display_label(group, group_labels), group_df)
            for group, group_df in channel_df.groupby(_group_column(df), sort=True)
        ]
        group_stim_sequences = {
            _group_display_label(group, group_labels): _stim_for_group(
                group_config,
                group,
            )
            for group, _ in channel_df.groupby(_group_column(df), sort=True)
        }
        stim_sequences = {}
        if plot_inputs:
            stim_sequences = group_stim_sequences
        output_path = (
            plots_root
            / "channel-group-based"
            / f"channel_{channel}"
            / f"{_safe_name(feature)}.png"
        )
        plot_feature_summary_grid(
            subsets,
            feature,
            output_path,
            title=(
                f"{feature} by group - "
                f"{_channel_display_label(channel, channel_labels)}"
            ),
            stim_sequences=stim_sequences,
            ylabel=_ylabel_for_feature(feature, fluorescence_label),
            interval_minutes=interval_minutes,
            tick_hours=tick_hours,
        )
        comparison_path = (
            plots_root
            / "channel-group-based"
            / f"channel_{channel}"
            / f"{_safe_name(feature)}_comparison.png"
        )
        if _is_growth_rate_feature(feature):
            plot_feature_cycle_comparison(
                subsets,
                feature,
                comparison_path,
                title=(
                    f"{feature} group comparison - "
                    f"{_channel_display_label(channel, channel_labels)}"
                ),
                ylabel=_ylabel_for_feature(feature, fluorescence_label),
                interval_minutes=interval_minutes,
                tick_hours=tick_hours,
            )
            violin_path = (
                plots_root
                / "channel-group-based"
                / f"channel_{channel}"
                / f"{_safe_name(feature)}_cell_median_violin.png"
            )
            plot_cell_median_violin(
                subsets,
                feature,
                violin_path,
                title=(
                    f"{feature} cell-median by group - "
                    f"{_channel_display_label(channel, channel_labels)}"
                ),
            )
        else:
            plot_feature_group_comparison(
                subsets,
                feature,
                comparison_path,
                group_stim_sequences,
                title=(
                    f"{feature} group comparison - "
                    f"{_channel_display_label(channel, channel_labels)}"
                ),
                ylabel=_ylabel_for_feature(feature, fluorescence_label),
                interval_minutes=interval_minutes,
                tick_hours=tick_hours,
            )


def _resolve_requested_features(df, default_feature, extra_features):
    selected = _find_fluorescence_columns(df, default_feature)

    for feature in _as_list(extra_features):
        columns = _find_feature_columns(df, feature)
        if not columns:
            raise ValueError(
                f"Feature '{feature}' was not found. Available features: "
                f"{', '.join(_feature_columns(df))}"
            )
        selected.extend(columns)

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError(
            "No fluorescence feature was detected. Please provide at least one "
            "feature to plot with the features argument. Available features: "
            f"{', '.join(_feature_columns(df))}"
        )
    return selected


def _add_requested_derived_features(df, extra_features):
    if any(_is_growth_rate_name(feature) for feature in _as_list(extra_features)):
        area_column = _find_area_column(df)
        if area_column is None:
            raise ValueError(
                "Growth rate was requested, but no area feature was found. "
                "Add an 'area' feature column before requesting growth_rate. "
                f"Available features: {', '.join(_feature_columns(df))}"
            )
        df = df.copy()
        df[GROWTH_RATE_FEATURE] = df[area_column].apply(compute_growth)
    return df


def _filter_dataframe(df, include=None, exclude=None):
    filtered = df

    if _has_filters(include):
        filtered = _apply_filter(filtered, include, keep=True)

    if _has_filters(exclude):
        filtered = _apply_filter(filtered, exclude, keep=False)

    if filtered.empty:
        raise ValueError("No cells remain after applying include/exclude filters.")

    return filtered


def _prepare_plot_columns(df, merge=None):
    df = df.copy()
    df[PLOT_CHANNEL_COLUMN] = df["channel"].astype(object)
    df[PLOT_GROUP_COLUMN] = df["group"].astype(object)

    if not _has_merge(merge):
        return df

    channels = _merge_values(merge, "channel")
    groups = _merge_values(merge, "group")

    if channels:
        normalized_channels = {_normalize_channel_value(value) for value in channels}
        mask = df["channel"].apply(_normalize_channel_value).isin(normalized_channels)
        df.loc[mask, PLOT_CHANNEL_COLUMN] = _merged_value("channels", channels)

    if groups:
        normalized_groups = {_normalize_group_value(value) for value in groups}
        mask = df["group"].apply(_normalize_group_value).isin(normalized_groups)
        df.loc[mask, PLOT_GROUP_COLUMN] = _merged_value("groups", groups)

    return df


def _apply_filter(df, filters, keep):
    filtered = df
    for field, values in filters.items():
        normalized_field = str(field).strip().lower().replace("-", "_")
        values = _as_list(values)

        if normalized_field in {"channel", "channels"}:
            channels = {_normalize_channel_value(value) for value in values}
            mask = filtered["channel"].apply(_normalize_channel_value).isin(channels)
        elif normalized_field in {"group", "groups"}:
            groups = {_normalize_group_value(value) for value in values}
            mask = filtered["group"].apply(_normalize_group_value).isin(groups)
        else:
            raise ValueError(
                f"Unknown filter field '{field}'. Use 'channel' or 'group'."
            )

        filtered = filtered[mask] if keep else filtered[~mask]

    return filtered


def _has_filters(filters):
    return bool(filters and any(_as_list(values) for values in filters.values()))


def _has_merge(merge):
    return _has_filters(merge)


def _merge_values(merge, field):
    if not merge:
        return []

    matches = []
    field_names = {field, f"{field}s"}
    for key, values in merge.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in field_names:
            matches.extend(_as_list(values))
    return matches


def _merged_value(prefix, values):
    return f"merged_{prefix}_{'_'.join(_safe_name(value) for value in values)}"


def _merge_plot_dir_name(merge):
    channels = _merge_values(merge, "channel")
    groups = _merge_values(merge, "group")
    parts = []
    if channels:
        parts.append(_merged_value("channels", channels))
    if groups:
        parts.append(_merged_value("groups", groups))
    return "_".join(parts) if parts else "merged"


def _channel_column(df):
    return PLOT_CHANNEL_COLUMN if PLOT_CHANNEL_COLUMN in df.columns else "channel"


def _group_column(df):
    return PLOT_GROUP_COLUMN if PLOT_GROUP_COLUMN in df.columns else "group"


def _normalize_channel_value(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _normalize_group_value(value):
    text = str(value).strip().lower().replace("_", " ")
    if text.startswith("group "):
        text = text.split("group ", 1)[1]
    elif text.startswith("group"):
        text = text.split("group", 1)[1]
    return text.strip()


def _custom_plot_dir_name(df):
    channels = [
        _safe_name(value)
        for value in sorted(df[_channel_column(df)].unique(), key=lambda item: str(item))
    ]
    groups = [
        _safe_name(_normalize_group_value(value))
        for value in sorted(df[_group_column(df)].unique(), key=lambda item: str(item))
    ]

    channel_part = "channels_" + "_".join(channels) if channels else "channels_none"
    group_part = "groups_" + "_".join(groups) if groups else "groups_none"
    return f"{channel_part}_{group_part}"


def _normalize_label_mapping(labels, kind):
    if labels is None:
        return {}

    if isinstance(labels, dict):
        items = labels.items()
    else:
        items = enumerate(_as_list(labels), start=1)

    normalized = {}
    for key, label in items:
        if kind == "channel":
            normalized[_normalize_channel_value(key)] = str(label)
        elif kind == "group":
            normalized[_normalize_group_value(key)] = str(label)
        else:
            raise ValueError(f"Unknown label kind '{kind}'.")

    return normalized


def _channel_display_label(channel, channel_labels=None):
    labels = channel_labels or {}
    normalized = _normalize_channel_value(channel)
    if normalized in labels:
        return labels[normalized]
    if _is_merged_value(channel, "channels"):
        return _merged_display_label(channel, "channels")
    return f"Channel {channel}"


def _group_display_label(group, group_labels=None):
    labels = group_labels or {}
    normalized = _normalize_group_value(group)
    if normalized in labels:
        return labels[normalized]
    if _is_merged_value(group, "groups"):
        return _merged_display_label(group, "groups")
    return str(group)


def _is_merged_value(value, prefix):
    return str(value).startswith(f"merged_{prefix}_")


def _merged_display_label(value, prefix):
    text = str(value).split(f"merged_{prefix}_", 1)[1]
    label_prefix = "Merged channels" if prefix == "channels" else "Merged groups"
    return f"{label_prefix} {text.replace('_', ',')}"


def _find_feature_columns(df, feature):
    if feature in df.columns:
        return [feature]

    if _is_fluorescence_name(feature):
        return _find_fluorescence_columns(df, feature)

    if _is_growth_rate_name(feature):
        return [GROWTH_RATE_FEATURE] if GROWTH_RATE_FEATURE in df.columns else []

    casefolded = {str(column).lower(): column for column in df.columns}
    column = casefolded.get(str(feature).lower())
    return [column] if column is not None else []


def _find_fluorescence_columns(df, default_feature):
    aliases = set(FLUORESCENCE_ALIASES)
    aliases.add(str(default_feature).lower())
    return [
        column
        for column in _feature_columns(df)
        if _is_fluorescence_name(column, aliases)
    ]


def _is_fluorescence_name(value, aliases=FLUORESCENCE_ALIASES):
    text = str(value).lower()
    return any(alias in text for alias in aliases)


def _ylabel_for_feature(feature, fluorescence_label):
    if _is_fluorescence_name(feature):
        return f"{fluorescence_label} (A. U.)"
    return feature


def _find_area_column(df):
    for column in _feature_columns(df):
        normalized = _normalize_feature_name(column)
        if normalized in AREA_ALIASES:
            return column
    return None


def _is_growth_rate_name(value):
    normalized = _normalize_feature_name(value)
    return normalized in {
        _normalize_feature_name(alias) for alias in GROWTH_RATE_ALIASES
    }


def _is_growth_rate_feature(feature):
    return _normalize_feature_name(feature) == GROWTH_RATE_FEATURE


def _normalize_feature_name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_plot_modes(plot_modes):
    aliases = {
        "all": "all",
        "channel": "channel",
        "channels": "channel",
        "channel-based": "channel",
        "group": "group",
        "groups": "group",
        "group-based": "group",
        "channel_group": "channel_group",
        "channel-group": "channel_group",
        "channel-group-based": "channel_group",
        "channel_and_group": "channel_group",
    }

    modes = []
    for mode in _as_list(plot_modes):
        normalized = aliases.get(str(mode).strip().lower())
        if normalized is None:
            raise ValueError(
                f"Unknown plot mode '{mode}'. Use one of: all, channel, group, channel_group."
            )
        modes.append(normalized)

    return set(modes)


def _stim_for_group(group_config, group):
    if group not in group_config:
        return None

    value = np.asarray(group_config[group])
    if value.ndim == 1:
        return value

    return None


def _feature_columns(df):
    metadata_columns = {"cell_number", "channel", "frame", "chamber", "group"}
    return [column for column in df.columns if column not in metadata_columns]


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _safe_name(value):
    text = str(value).strip().replace(" ", "_")
    return "".join(char for char in text if char.isalnum() or char in ("_", "-", "."))
