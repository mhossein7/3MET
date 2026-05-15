# 3MET / TMET

Mother Machine Microscopy Experiments Tools.

This package is under active development. APIs, command names, expected file
formats, and outputs may change as the toolkit grows.

## Installation

From the repository root:

```bash
pip install -e .
```

The command is available as either `TMET` or `tmet`.

```bash
TMET --help
```

## Tools

### Movie Maker

Create annotated MP4 movies from mother-machine microscopy TIFF frame folders.
With ROI metadata, movies are cropped around mother-machine chambers and can
show chamber stimulation state over time using `cells_stims.npy`.

```bash
TMET moma-movie-maker --address /path/to/experiment
```

Common options:

```bash
TMET moma-movie-maker \
  --address /path/to/experiment \
  --fps 15 \
  --minutes-per-frame 5 \
  --n-series 30
```

Use `--no-roi` to detect crop boundaries from the images instead of using
`roi_boxes.pkl`.

### OLP

Open-loop processing (`OLP`) loads mother-machine experiment measurements,
assigns cells to stimulation groups, and saves summary plots.

Expected experiment files:

```text
/path/to/experiment/
  mothers.pkl
  cells_stims.npy
  group_config.json
  feature_list.json              # optional
  experiment_settings.json        # optional, can contain {"features": [...]}
```

Feature names are resolved in this order:

1. `--feature-names`
2. `--features-address`
3. `feature_list.json`
4. `experiment_settings.json` key `features`
5. generated names like `Feature_0`, `Feature_1`

Basic use:

```bash
TMET OLP --address /path/to/experiment
```

By default, OLP plots all fluorescence-like features, including names containing
`fluorescence`, `fluorescecne`, or `fluo`. If no fluorescence feature is found,
provide at least one feature:

```bash
TMET OLP --address /path/to/experiment --features length area
```

Request growth-rate plots from the `area` feature:

```bash
TMET OLP --address /path/to/experiment \
  --features growth_rate \
  --feature-names fluorescence area length
```

Growth-rate requests accept variants such as `growth rate`, `growth-rate`,
`growthrate`, and `growth_rate`.

Plot outputs are saved under:

```text
plots/
  all/
  channel-based/
  group-based/
  channel-group-based/
```

For growth rate, OLP also saves comparison plots and cell-median violin plots.

#### Filtering

Use compact include/exclude filters to plot selected channels or groups:

```bash
TMET OLP --address /path/to/experiment \
  --include channel 1,2 group 3,4
```

```bash
TMET OLP --address /path/to/experiment \
  --exclude channel 4 group 1
```

Filtered outputs are saved under a descriptive custom directory, for example:

```text
plots/custom/channels_1_2_groups_2_3_4/
```

#### Labels And Plot Styling

Use channel and group labels to make legends and subplot titles more readable:

```bash
TMET OLP --address /path/to/experiment \
  --channel-labels 1:Control 2:Drug \
  --group-labels 1:Red 2:Green
```

Fluorescence plots use `GFP (A. U.)` as the default y-axis label. Change the
fluorophore name with:

```bash
TMET OLP --address /path/to/experiment --fluorescence-label CFP
```

Time-series plots assume 5-minute acquisition intervals by default and label the
x-axis in hours at 4-hour intervals:

```bash
TMET OLP --address /path/to/experiment \
  --interval-minutes 5 \
  --tick-hours 4
```

Group-resolved plots can show the red/green input sequence in the background:

```bash
TMET OLP --address /path/to/experiment --plot-inputs
```

### PDIP

Pre-DeLTA Image Processing (`PDIP`) hosts utilities for organizing and preparing
image data before sending it to DeLTA for segmentation.

The current PDIP tool is `manual-organizer`, which converts manually prepared
OME-TIFF batches into DeLTA-compatible channel files.

```bash
tmet pdip manual-organizer --address /path/to/files
```

Optional directory-name arguments:

```bash
tmet pdip manual-organizer \
  --address /path/to/files \
  --imaged imaged \
  --unimaged unimaged \
  --constant _
```

The organizer expects image batches under:

```text
/path/to/files/
  imaged/_1/
  unimaged/_1/
```

and writes DeLTA-compatible TIFFs into:

```text
imaged/DeLTA_compatible/
unimaged/DeLTA_compatible/
```

## Development Layout

New tools should live under `src/TMET/<tool_name>/` and expose a
`register_parser(subparsers)` function from their `cli.py`. The top-level
`src/TMET/cli.py` imports each tool's registration function and attaches it as a
subcommand.

```text
src/TMET/
  cli.py
  moma_movie_maker/
  OLP/
  PDIP/
```
