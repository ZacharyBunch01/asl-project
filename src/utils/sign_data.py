"""
sign_data.py

Helpers for loading ASL sign data.
"""

import pandas as pd
import parseData


def get_sign_data() -> pd.DataFrame:
    """
    return the parsed Data
    """
    signData = parseData.signData
    if signData is None:
        raise RuntimeError("Data not loaded. Make sure parseData.py can see ../Data/...")

    print(f"[INFO] Data shape: {signData.shape}")
    return signData
