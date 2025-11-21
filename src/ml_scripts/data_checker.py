"""
data_utils.py

Minimal shared helpers for ASL training.
"""

import pandas as pd


def validate_target_column(signData, target_col):
    """
    Check that the target column exists in the dataset.
    Returns (True, "") if ok, otherwise (False, reason).
    """
    if target_col not in signData.columns:
        msg = f"Target {target_col} not found in data, skipping."
        return False, msg

    return True, ""
