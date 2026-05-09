import pickle
import numpy as np
from pathlib import Path

def mother_loader(address):
    address = Path(address)
    if address.name == "mothers.pkl":
        if address.is_file():
            with open(address, "rb") as f:
                mothers = pickle.load(f)
            return mothers
        raise FileNotFoundError(f"'mothers.pkl' does not exist: {path}")

    # Case 2: path is a directory containing mothers.pkl
    if address.is_dir():
        candidate = address / "mothers.pkl"
        if candidate.is_file():
            with open(candidate, "rb") as f:
                mothers = pickle.load(f)
            return mothers

    # Neither case matched
    raise ValueError(
        f"Path must either be a 'mothers.pkl' file or a directory containing it: {path}"
    )
   