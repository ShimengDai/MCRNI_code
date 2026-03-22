# src/predict.py
from pathlib import Path
import pandas as pd
import yaml
from src.io import load_model
from src.features import get_feature_columns


def predict_one(model, X: pd.DataFrame, pred_type: str = "label"):
    if pred_type == "proba":
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
    return model.predict(X)


def add_pred_column(df: pd.DataFrame, pred, col_name: str) -> pd.DataFrame:
    df_out = df.copy()
    df_out[col_name] = pred
    return df_out


def run_predict_over_models(
    df: pd.DataFrame, *, models_dir, models_cfg, data_cfg, feature_cfg
):
    # 1) derive feature columns (ideally from train rules/config)
    feature_cols = get_feature_columns(df, data_cfg=data_cfg, feature_cfg=feature_cfg)
    X = df[feature_cols].copy()

    df_out = df.copy()
    models_dir = Path(models_dir)

    for m in models_cfg:
        model_path = models_dir / m["file"]
        model = load_model(model_path)

        # 2) Optional but recommended: enforce feature alignment if available
        if hasattr(model, "feature_names_in_"):
            needed = list(model.feature_names_in_)

            missing = [c for c in needed if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing features for model {m.get('name', m['file'])}: {missing[:10]}"
                )

            # reorder columns to match training-time order
            X_use = X[needed]
        else:
            X_use = X

        # 3) predict (label/proba)
        pred = predict_one(model, X_use, pred_type=m.get("type", "label"))

        # 4) append prediction column
        df_out = add_pred_column(df_out, pred, m["out_col"])

    return df_out
