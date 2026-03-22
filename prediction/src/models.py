# src/models.py
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_model(
    model_name: str,
    model_cfg: dict,
    seed: int | None = None,
    missing_cfg: dict | None = None,
):
    params = dict(model_cfg.get("params", {}))

    # -------------------------
    # Missing handling
    # -------------------------
    use_imputer = False
    impute_method = "median"
    if missing_cfg is not None and missing_cfg.get("strategy") == "impute":
        use_imputer = True
        impute_method = missing_cfg.get("imputer", {}).get("method", "median")

    if impute_method not in {"median", "mean", "most_frequent"}:
        raise ValueError(f"Unknown imputation method: {impute_method}")

    # -------------------------
    # Random state
    # -------------------------
    if seed is not None and "random_state" not in params:
        # LR, RF, SVC all accept random_state (SVC uses it mainly when probability=True)
        if model_name in {"logistic_regression", "random_forest", "svm"}:
            params["random_state"] = seed

    # -------------------------
    # Build
    # -------------------------
    if model_name == "logistic_regression":
        model = LogisticRegression(**params)
        steps: list[tuple[str, object]] = []
        if use_imputer:
            steps.append(("imputer", SimpleImputer(strategy=impute_method)))
        steps += [("scaler", StandardScaler()), ("model", model)]
        return Pipeline(steps)

    if model_name == "svm":
        model = SVC(**params)
        steps: list[tuple[str, object]] = []
        if use_imputer:
            steps.append(("imputer", SimpleImputer(strategy=impute_method)))
        steps += [("scaler", StandardScaler()), ("model", model)]
        return Pipeline(steps)

    if model_name == "random_forest":
        model = RandomForestClassifier(**params)
        steps: list[tuple[str, object]] = []
        if use_imputer:
            steps.append(("imputer", SimpleImputer(strategy=impute_method)))
        steps.append(("model", model))
        return Pipeline(steps)

    raise ValueError(f"Unknown model: {model_name}")
