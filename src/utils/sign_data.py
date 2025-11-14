"""
sign_data.py

Helpers for loading ASL sign data.
"""

import pandas as pd
import data_prep.parseData


def _ensure_canonical_columns(signData: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the DataFrame has canonical columns:
    Movement, MajorLocation, MinorLocation, Handshape.

    Uses parseData.attemptRowGet on each row if they are missing.
    """
    df = signData

    # Only make a copy if we actually need to add something
    needs_copy = any(
        col not in df.columns
        for col in ["Movement", "MajorLocation", "MinorLocation", "Handshape"]
    )
    if needs_copy:
        df = df.copy()

    # Add each canonical column only if it doesn't already exist
    if "Movement" not in df.columns:
        df["Movement"] = df.apply(
            lambda row: data_prep.parseData.attemptRowGet(row, "Movement"), axis=1
        )

    if "MajorLocation" not in df.columns:
        df["MajorLocation"] = df.apply(
            lambda row: data_prep.parseData.attemptRowGet(row, "MajorLocation"), axis=1
        )

    if "MinorLocation" not in df.columns:
        df["MinorLocation"] = df.apply(
            lambda row: data_prep.parseData.attemptRowGet(row, "MinorLocation"), axis=1
        )

    if "Handshape" not in df.columns:
        df["Handshape"] = df.apply(
            lambda row: data_prep.parseData.attemptRowGet(row, "Handshape"), axis=1
        )

    return df


def get_sign_data() -> pd.DataFrame:
    """
    Return the parsed ASL sign DataFrame, enriched with canonical columns:
    Movement, MajorLocation, MinorLocation, Handshape.
    """
    signData = data_prep.parseData.signData
    if signData is None:
        raise RuntimeError(
            "Data not loaded. Make sure parseData.py can see ../Data/... "
            "and signdata.csv exists."
        )

    signData = _ensure_canonical_columns(signData)

    print(f"[INFO] Data shape: {signData.shape}")
    return signData
