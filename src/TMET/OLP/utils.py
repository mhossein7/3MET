import pickle
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict



def mothers_to_df_loader(root,feature_names=None,features_address = None):
    """
    Input: root folder where MM experiments is saved
    
    Convert mothers list structure:

        data[c][f][n][p][t]

    into a compact DataFrame where each feature column
    stores the full time series.

    Adds a global 'cell_number' column that increases
    sequentially for every added row.

    Returns
    -------
    pd.DataFrame
    """
    import pandas as pd

    mothers = mother_loader(root)
    group_config = resolve_groups(root)
    cells_stims = cells_stims_loader(root)
    cell_identities = assign_cells_to_groups(group_config, cells_stims)
    
    features = _resolve_feature_names(root, feature_names, features_address)

    rows = []
    cell_counter = 1

    for c_idx, c_item in enumerate(mothers):
        for f_idx, f_item in enumerate(c_item):
            for n_idx, chamber in enumerate(f_item):

                chamber = np.asarray(chamber)
                chamber = chamber.T

                if chamber.ndim != 2:
                    raise ValueError(
                        f"Expected shape (p,t), got {chamber.shape}"
                    )

                p, t = chamber.shape

                row = {
                    "cell_number": cell_counter,
                    "channel": c_idx+1,
                    "frame": f_idx+1,
                    "chamber": n_idx+1,
                    "group":cell_identities[cell_counter-1]
                }

                if features is not None and len(features) < p:
                    raise ValueError(
                        f"Expected at least {p} feature names for chamber data, "
                        f"but received {len(features)}."
                    )

                for p_idx in range(p):

                    feature_name = (
                        features[p_idx]
                        if features is not None
                        else f"Feature_{p_idx}"
                    )

                    # store full time series
                    row[feature_name] = chamber[p_idx]

                rows.append(row)

                cell_counter += 1

    return pd.DataFrame(rows)


def _resolve_feature_names(root, feature_names=None, features_address=None):
    if feature_names is not None:
        return feature_names

    if features_address is not None:
        return feature_list_loader(features_address)

    root = Path(root)
    if root.is_dir() and (root / "feature_list.json").is_file():
        return feature_list_loader(root)

    if root.is_dir() and (root / "experiment_settings.json").is_file():
        return experiment_settings_feature_loader(root)

    return None


def experiment_settings_feature_loader(path):
    path = Path(path)
    settings_path = path
    if path.is_dir():
        settings_path = path / "experiment_settings.json"

    if settings_path.name != "experiment_settings.json" or not settings_path.is_file():
        raise ValueError(
            "Path must be an experiment_settings.json file or a directory containing it: "
            f"{path}"
        )

    with open(settings_path, "r") as f:
        settings = json.load(f)

    features = settings.get("features")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError(
            f"{settings_path} must contain a 'features' key with a list of strings."
        )

    return features



def count_channels(df):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        - 'channel'
        - 'frame'

    Returns
    -------
    dict
        Example:
        {
            'total_n_channel': 3,
            'n_frame_channel_1': 30,
            'n_frame_channel_2': 30,
            'n_frame_channel_3': 20
        }
    """

    result = {}

    # number of unique channels
    unique_channels = sorted(df["channel"].unique())
    result["total_n_channel"] = len(unique_channels)

    # maximum frame for each channel
    max_frames = df.groupby("channel")["frame"].max()

    for channel, max_frame in max_frames.items():
        result[f"n_frame_channel_{channel}"] = int(max_frame)

    return result




def mother_loader(address):
    address = Path(address)
    if address.name == "mothers.pkl":
        if address.is_file():
            with open(address, "rb") as f:
                mothers = pickle.load(f)
            return mothers
        raise FileNotFoundError(f"'mothers.pkl' does not exist: {address}")

    # Case 2: path is a directory containing mothers.pkl
    if address.is_dir():
        candidate = address / "mothers.pkl"
        if candidate.is_file():
            with open(candidate, "rb") as f:
                mothers = pickle.load(f)
            return mothers

    # Neither case matched
    raise ValueError(
        f"Path must either be a 'mothers.pkl' file or a directory containing it: {address}"
    )


def cells_stims_loader(address):
    """
    Load cells_stims.npy from either a direct file path or a directory containing it.
    """
    address = Path(address)

    if address.name == "cells_stims.npy":
        if address.is_file():
            return np.load(address)
        raise FileNotFoundError(f"'cells_stims.npy' does not exist: {address}")

    if address.is_dir():
        candidate = address / "cells_stims.npy"
        if candidate.is_file():
            return np.load(candidate)

    raise ValueError(
        f"Path must either be a 'cells_stims.npy' file or a directory containing it: {address}"
    )


def save_feature_list(root, features):
    """
    Save feature names to feature_list.json in the experiment root directory.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if not isinstance(features, tuple):
        raise TypeError("features must be a tuple of strings.")

    if not all(isinstance(feature, str) for feature in features):
        raise TypeError("features must be a tuple of strings.")

    output_path = root / "feature_list.json"
    with open(output_path, "w") as f:
        json.dump(list(features), f, indent=4)

    return output_path


def resolve_groups(path):
    path = Path(path)
    
    if path.name == "group_config.json":
        with open(path,'rb') as f:
            group_config = json.load(f)
        return group_config
        
    if path.name == "cells_stims.npy":
        stims = cells_stims_loader(path)
        group_config = _resolve_cells_stims(stims, path.parent)
        return group_config

    if path.is_dir():
        group_config_file = path / "group_config.json"
        if group_config_file.is_file():
            with open(group_config_file,'rb') as f:
                group_config = json.load(f)
            return group_config
        
        cells_stims = path / "cells_stims.npy"
        if cells_stims.is_file():
            stims = cells_stims_loader(cells_stims)
            group_config = _resolve_cells_stims(stims, path)
            return group_config
    
    raise ValueError(f"Path must either be a 'group_config.json' or a 'cells_stims.npy' file or a directory containing either of them: {path}")
    
def _resolve_cells_stims(cells_stims,root):
    """
    Groups identical rows of a binary NumPy matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Binary matrix of shape (n, t).
    output_file : str
        Path to output JSON file.

    Returns
    -------
    dict
        Dictionary of grouped rows.
    """
   
    matrix = np.asarray(cells_stims)

    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")

    unique_rows, inverse, counts = np.unique(
        matrix,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    grouped_output = OrderedDict()
    singleton_rows = []
    group_id = 1

    for i, row in enumerate(unique_rows):
        if counts[i] > 1:
            grouped_output[f"Group {group_id}"] = row.tolist()
            group_id += 1
        else:
            singleton_rows.append(row.tolist())

    # If no repeated rows exist, put everything into Group 1
    if group_id == 1:
        grouped_output["Group 1"] = matrix.tolist()

    # Otherwise, put all non-identical rows into the last group
    elif singleton_rows:
        grouped_output[f"Group {group_id}"] = singleton_rows

    output_file = Path(root) / "group_config.json"
    with open(output_file, "w") as f:
        json.dump(grouped_output, f, indent=4)

    return grouped_output



def assign_cells_to_groups(groups_config, cells_stims_matrix):
    """
    Assign each row of cells_stim matrix to a group from groups_dict.

    Rows matching a group whose value contains multiple input rows
    are labeled as 'Random'.

    Parameters
    ----------
    groups_dict : dict
        Example:
        {
            "Group 1": [0, 0, 1],
            "Group 2": [1, 0, 1],
            "Group 3": [[0, 1, 0], [1, 1, 0]]
        }

    matrix : np.ndarray
        Binary matrix of shape (n, t).

    Returns
    -------
    list
        Group identity for each row.
    """
    matrix = cells_stims_matrix
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D NumPy array.")

    row_to_group = {}

    for group_name, group_value in groups_config.items():
        arr = np.asarray(group_value)

        # Case 1: group has one representative input, e.g. [0, 0, 1]
        if arr.ndim == 1:
            row_to_group[tuple(arr.tolist())] = group_name

        # Case 2: group has multiple inputs, e.g. [[0, 1, 0], [1, 1, 0]]
        elif arr.ndim == 2:
            for row in arr:
                row_to_group[tuple(row.tolist())] = "Random"

        else:
            raise ValueError(f"{group_name} has invalid shape.")

    assignments = []

    for row in matrix:
        row_tuple = tuple(row.tolist())

        if row_tuple not in row_to_group:
            raise ValueError(f"Row {row.tolist()} was not found in groups_dict.")

        assignments.append(row_to_group[row_tuple])

    return assignments



def _identify_group(group_config, stim):
    """
    Assign a group identity to a candidate binary sequence.

    Parameters
    ----------
    groups_dict : dict
        Group classification dictionary.

    candidate_sequence : array-like
        Candidate input sequence.

    Returns
    -------
    str
        Group name or 'Random'.

    Raises
    ------
    ValueError
        If the sequence does not belong to any group.
    """

    candidate = tuple(np.asarray(stim).tolist())

    for group_name, group_value in group_config.items():

        arr = np.asarray(group_value)

        # Case 1:
        # Single representative sequence
        if arr.ndim == 1:

            if tuple(arr.tolist()) == candidate:
                return group_name

        # Case 2:
        # Multiple sequences -> Random group
        elif arr.ndim == 2:

            for row in arr:
                if tuple(row.tolist()) == candidate:
                    return "Random"

    # If no match found
    raise ValueError(
        f"Sequence {list(candidate)} does not belong to any group."
    )


def feature_list_loader(path):
    path = Path(path)
    features = []
    
    if path.name == 'feature_list.json':
        with open(path,'rb') as f:
            feature_list = json.load(f)
        
        if type(feature_list) == list:
            features = feature_list
        if type(feature_list) == dict:
            for name in feature_list.values():
                features.append(name)
        return features
    
    if path.is_dir():
        feature_list_file = path / 'feature_list.json'
        if feature_list_file.is_file():
            with open(feature_list_file,'rb') as f:
                feature_list = json.load(f)
            if type(feature_list) == list:
                features = feature_list
            if type(feature_list) == dict:
                for name in feature_list.values():
                    features.append(name)
            return features
    
    raise ValueError(f'Either the path to feature_list.json file or the directory including it should be given. {path}')



def compute_growth(area):
    
    # Run through time:
    growth = []
    for t in range(len(area)-1):
    
        cont_flag = False
    
        # If current area is None or too small, append a NaN (empty chamber)
        if area[t] is None or area[t] < 100 :
            growth.append(np.nan)
            continue
        
        # Same thing for time t+1:
        if area[t+1] is None or area[t+1] < 100:
            growth.append(np.nan)
            continue
        
        # Otherwise compute growth as delta_area / area:
        growth.append((area[t+1] - area[t])/area[t])
    
    # Filter out divisions and glitches:
    growth = np.array(growth)
    growth[growth<-.2] = np.nan
    growth[growth>.3] = np.nan
    
    # Convert to 1/hour:
    growth*=12
    
    return growth
