# src/train.py

# from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV

from src.models import build_model

from pathlib import Path


def run_grid_search(
    X,
    y,
    groups,
    *,
    model_name: str,
    model_cfg: dict,
    cv_cfg: dict,
    seed: int | None = None,
    missing_cfg: dict | None = None,
):
    # 1) build estimator (pipeline/model)
    estimator = build_model(
        model_name=model_name,
        model_cfg=model_cfg,
        seed=seed,
        missing_cfg=missing_cfg,  # <-- NEW
    )

    # 2) param grid
    param_grid = model_cfg.get("param_grid", {}) or {}

    # If we wrapped a "model" step in a Pipeline, prefix params with "model__"
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        param_grid = {f"model__{k}": v for k, v in param_grid.items()}

    # 3) CV strategy (you are using GroupKFold)
    n_splits = cv_cfg.get("n_splits", 10)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 4) grid search
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=cv_cfg.get("scoring", "roc_auc"),  # "roc_auc" / "average_precision"
        cv=cv,
        refit=True,
        n_jobs=cv_cfg.get("n_jobs", -1),
        error_score="raise",
    )

    search.fit(X, y)
    return search
