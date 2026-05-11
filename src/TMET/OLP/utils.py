import pickle
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict
import pandas as pd

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
   
def resolve_groups(path):
    path = Path(path)
    
    if path.name == "group_config.json":
        with open(path,'rb') as f:
            group_config = json.load(f)
        return group_config
        
    if path.name == "cells_stims.npy":
        stims = np.load(cells_stims)
        group_config = _resolve_cells_stims(stims)
        return group_config

    if path.is_dir():
        group_config_file = path / "group_config.json"
        if group_config_file.is_file():
            with open(path,'rb') as f:
                group_config = json.load(f)
            return group_config
        
        cells_stims = path / "cells_stims.npy"
        if cells_stims.is_file():
            stims = np.load(cells_stims)
            group_config = _resolve_cells_stims(stims)
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

    with open(root+'group_config.json', "w") as f:
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



import numpy as np

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


def mothers_to_df_loader(root,feature_names=None):
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
    mothers = mother_loader(root)
    group_config = resolve_groups(root)
    cell_identities = assign_cells_to_groups(group_config)
    rows = []
    cell_counter = 1

    for c_idx, c_item in enumerate(mothers):
        for f_idx, f_item in enumerate(c_item):
            for n_idx, chamber in enumerate(f_item):

                chamber = np.asarray(chamber)

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

                for p_idx in range(p):

                    feature_name = (
                        feature_names[p_idx]
                        if feature_names is not None
                        else f"Feature_{p_idx}"
                    )

                    # store full time series
                    row[feature_name] = chamber[p_idx]

                rows.append(row)

                cell_counter += 1

    return pd.DataFrame(rows)

