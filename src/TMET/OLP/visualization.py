from pathlib import Path

import numpy as np


SINGLE_LINE_COLOR = "#000000"
SINGLE_IQR_COLOR = "#D9D9D9"
RED_INPUT_COLOR = np.array([214, 39, 40]) / 255
GREEN_INPUT_COLOR = np.array([44, 160, 44]) / 255
MISSING_INPUT_COLOR = "#7F7F7F"


def plot_feature_summary(
    df,
    feature,
    output_path,
    title=None,
    stim_sequence=None,
    ylabel=None,
    interval_minutes=5,
    tick_hours=4,
):
    """
    Plot nanmedian and the interquartile range for a time-series feature.
    """
    import matplotlib.pyplot as plt

    summary = _summarize_time_series(df[feature])
    x = np.arange(summary["median"].size)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    if stim_sequence is not None:
        _plot_stim_background(ax, stim_sequence, summary["median"].size)

    ax.fill_between(
        x,
        summary["q1"],
        summary["q3"],
        color=SINGLE_IQR_COLOR,
        alpha=0.65,
        linewidth=0,
    )
    ax.plot(x, summary["median"], color=SINGLE_LINE_COLOR, linewidth=2.0)
    _format_time_axis(ax, summary["median"].size, interval_minutes, tick_hours)
    ax.set_ylabel(ylabel or feature)
    ax.set_title(title or feature)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_group_comparison(
    subsets,
    feature,
    output_path,
    group_stim_sequences,
    title=None,
    ylabel=None,
    interval_minutes=5,
    tick_hours=4,
):
    """
    Plot group median traces in one shared axis, colored by green input fraction.
    """
    import matplotlib.pyplot as plt

    summaries = []
    for label, subset in subsets:
        if subset.empty:
            continue
        summaries.append((label, _summarize_time_series(subset[feature])))

    if not summaries:
        raise ValueError(f"No data available for feature '{feature}'.")

    colors = _group_colors(
        {label: summary["median"].size for label, summary in summaries},
        group_stim_sequences,
    )

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

    for label, summary in summaries:
        x = np.arange(summary["median"].size)
        color = colors[label]
        ax.fill_between(
            x,
            summary["q1"],
            summary["q3"],
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        ax.plot(x, summary["median"], color=color, linewidth=2.0, label=str(label))

    y_limits = _shared_ylim([summary for _, summary in summaries])
    if y_limits is not None:
        ax.set_ylim(y_limits)
    max_timepoints = max(summary["median"].size for _, summary in summaries)
    _format_time_axis(ax, max_timepoints, interval_minutes, tick_hours)
    ax.set_ylabel(ylabel or feature)
    ax.set_title(title or f"{feature} group comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_cycle_comparison(
    subsets,
    feature,
    output_path,
    title=None,
    ylabel=None,
    interval_minutes=5,
    tick_hours=4,
):
    """
    Plot subset median traces in one shared axis using Matplotlib's color cycle.
    """
    import matplotlib.pyplot as plt

    summaries = []
    for label, subset in subsets:
        if subset.empty:
            continue
        summaries.append((label, _summarize_time_series(subset[feature])))

    if not summaries:
        raise ValueError(f"No data available for feature '{feature}'.")

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

    for label, summary in summaries:
        x = np.arange(summary["median"].size)
        line = ax.plot(x, summary["median"], linewidth=2.0, label=str(label))[0]
        ax.fill_between(
            x,
            summary["q1"],
            summary["q3"],
            color=line.get_color(),
            alpha=0.10,
            linewidth=0,
        )

    y_limits = _shared_ylim([summary for _, summary in summaries])
    if y_limits is not None:
        ax.set_ylim(y_limits)
    max_timepoints = max(summary["median"].size for _, summary in summaries)
    _format_time_axis(ax, max_timepoints, interval_minutes, tick_hours)
    ax.set_ylabel(ylabel or feature)
    ax.set_title(title or f"{feature} comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cell_median_violin(
    subsets,
    feature,
    output_path,
    title=None,
    ylabel=None,
):
    """
    Plot one violin per subset using each cell's median value over time.
    """
    import matplotlib.pyplot as plt

    labels = []
    distributions = []
    for label, subset in subsets:
        if subset.empty:
            continue

        cell_medians = _cell_medians(subset[feature])
        if cell_medians.size == 0:
            continue

        labels.append(str(label))
        distributions.append(cell_medians)

    if not distributions:
        raise ValueError(f"No finite cell-median data available for feature '{feature}'.")

    fig, ax = plt.subplots(
        figsize=(max(6, 1.2 * len(distributions)), 4.8),
        constrained_layout=True,
    )
    parts = ax.violinplot(
        distributions,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(f"C{idx % 10}")
        body.set_edgecolor("black")
        body.set_alpha(0.55)
        body.set_linewidth(0.8)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel or feature)
    ax.set_title(title or f"{feature} cell-median comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_summary_grid(
    subsets,
    feature,
    output_path,
    title=None,
    stim_sequences=None,
    ylabel=None,
    interval_minutes=5,
    tick_hours=4,
):
    """
    Plot one summary subplot per labeled subset.
    """
    import matplotlib.pyplot as plt

    subsets = [(label, subset) for label, subset in subsets if not subset.empty]
    if not subsets:
        raise ValueError(f"No data available for feature '{feature}'.")

    n_panels = len(subsets)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    summaries = [
        (label, subset, _summarize_time_series(subset[feature]))
        for label, subset in subsets
    ]
    y_limits = _shared_ylim([summary for _, _, summary in summaries])
    stim_sequences = stim_sequences or {}
    for ax, (label, subset, summary) in zip(axes.ravel(), summaries):
        _plot_feature_on_axis(
            ax,
            summary,
            feature,
            title=str(label),
            stim_sequence=stim_sequences.get(label),
            ylabel=ylabel,
            y_limits=y_limits,
            interval_minutes=interval_minutes,
            tick_hours=tick_hours,
        )

    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_feature_on_axis(
    ax,
    summary,
    feature,
    title=None,
    stim_sequence=None,
    ylabel=None,
    y_limits=None,
    interval_minutes=5,
    tick_hours=4,
):
    x = np.arange(summary["median"].size)

    if stim_sequence is not None:
        _plot_stim_background(ax, stim_sequence, summary["median"].size)

    ax.fill_between(
        x,
        summary["q1"],
        summary["q3"],
        color=SINGLE_IQR_COLOR,
        alpha=0.65,
        linewidth=0,
    )
    ax.plot(x, summary["median"], color=SINGLE_LINE_COLOR, linewidth=2.0)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    _format_time_axis(ax, summary["median"].size, interval_minutes, tick_hours)
    ax.set_ylabel(ylabel or feature)
    ax.set_title(title or feature)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _summarize_time_series(series):
    values = _stack_time_series(series)
    if values.size == 0:
        raise ValueError("No time-series data available.")

    return {
        "median": np.nanmedian(values, axis=0),
        "q1": np.nanpercentile(values, 25, axis=0),
        "q3": np.nanpercentile(values, 75, axis=0),
    }


def _shared_ylim(summaries):
    values = []
    for summary in summaries:
        values.extend(summary["q1"][np.isfinite(summary["q1"])])
        values.extend(summary["q3"][np.isfinite(summary["q3"])])

    if not values:
        return None

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if y_min == y_max:
        margin = abs(y_min) * 0.05 if y_min else 1.0
    else:
        margin = (y_max - y_min) * 0.05

    return y_min - margin, y_max + margin


def _group_colors(label_timepoints, group_stim_sequences):
    green_fractions = {}
    for label, n_timepoints in label_timepoints.items():
        stim = group_stim_sequences.get(label)
        if stim is None:
            green_fractions[label] = None
            continue

        stim = np.asarray(stim, dtype=float).ravel()[:n_timepoints]
        finite = stim[np.isfinite(stim)]
        green_fractions[label] = float(np.mean(finite)) if finite.size else None

    valid_values = [
        fraction for fraction in green_fractions.values() if fraction is not None
    ]
    if not valid_values:
        return {label: MISSING_INPUT_COLOR for label in label_timepoints}

    min_fraction = min(valid_values)
    max_fraction = max(valid_values)
    use_relative_scale = max_fraction > min_fraction

    colors = {}
    for label in label_timepoints:
        fraction = green_fractions[label]
        if fraction is None:
            colors[label] = MISSING_INPUT_COLOR
            continue

        if use_relative_scale:
            position = (fraction - min_fraction) / (max_fraction - min_fraction)
        else:
            position = fraction

        rgb = RED_INPUT_COLOR + position * (GREEN_INPUT_COLOR - RED_INPUT_COLOR)
        colors[label] = tuple(rgb)

    return colors


def _stack_time_series(series):
    arrays = [np.asarray(value, dtype=float).ravel() for value in series]
    arrays = [arr for arr in arrays if arr.size > 0]
    if not arrays:
        return np.empty((0, 0))

    max_len = max(arr.size for arr in arrays)
    stacked = np.full((len(arrays), max_len), np.nan)
    for idx, arr in enumerate(arrays):
        stacked[idx, : arr.size] = arr

    return stacked


def _cell_medians(series):
    medians = []
    for value in series:
        arr = np.asarray(value, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            medians.append(float(np.median(arr)))

    return np.asarray(medians, dtype=float)


def _format_time_axis(ax, n_timepoints, interval_minutes=5, tick_hours=4):
    if n_timepoints <= 0:
        ax.set_xlabel("Time (hour)")
        return

    frames_per_tick = max(1, int(round((tick_hours * 60) / interval_minutes)))
    ticks = np.arange(0, n_timepoints, frames_per_tick)
    if ticks.size == 0 or ticks[0] != 0:
        ticks = np.insert(ticks, 0, 0)

    labels = []
    for tick in ticks:
        hour = tick * interval_minutes / 60
        labels.append(f"{int(hour)}" if hour.is_integer() else f"{hour:g}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Time (hour)")


def _plot_stim_background(ax, stim_sequence, n_timepoints):
    stim = np.asarray(stim_sequence).ravel()[:n_timepoints]
    if stim.size == 0 or n_timepoints == 0:
        return

    for idx, value in enumerate(stim):
        color = "#2CA02C" if int(value) == 1 else "#D62728"
        ax.axvspan(idx, idx + 1, color=color, alpha=0.08, linewidth=0)
