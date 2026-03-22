from typing import List, Tuple
import pandas as pd


def get_feature_columns(
    df: pd.DataFrame, *, data_cfg: dict, feature_cfg: dict
) -> List[str]:
    outcome = data_cfg["outcome"]
    school_col = data_cfg["school_id_col"]

    exclude = set(feature_cfg.get("exclude_cols", []))
    exclude.update([outcome, school_col])

    feat_cols = [c for c in df.columns if c not in exclude]
    feat_cols = [
        c
        for c in feat_cols
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
    ]
    return feat_cols


def make_xy(
    df: pd.DataFrame, *, data_cfg: dict, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    outcome = data_cfg["outcome"]
    school_col = data_cfg["school_id_col"]

    X = df[feature_cols].copy()
    y = df[outcome].copy()
    groups = df[school_col].copy()

    if hasattr(groups.dtype, "categories"):
        groups = groups.cat.remove_unused_categories()
    return X, y, groups
