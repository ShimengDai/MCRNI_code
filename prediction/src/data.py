# src/data.py
# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
# from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold


import numpy as np
import pandas as pd


def load_data(data_cfg: dict) -> pd.DataFrame:
    return pd.read_csv(data_cfg["raw_path"])


def clean_data(df: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    df = df.copy()

    # drop junk column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # outcome: listwise deletion on outcome
    outcome = data_cfg["outcome"]
    df = df.dropna(subset=[outcome])

    return df


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.3,  # 70/30 split by default
    seed: int = 42,
):
    """
    Random row split (NOT group-based).
    Returns df_train, df_test.
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    n_test = int(np.ceil(len(df) * test_size))
    df_test = df.iloc[:n_test].copy()
    df_train = df.iloc[n_test:].copy()
    return df_train, df_test
