import joblib

from pathlib import Path


def save_model(estimator, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)


def save_best_model(search, out_dir: str | Path, model_name: str) -> Path:
    """
    Save GridSearchCV.best_estimator_ for a given model.
    Returns the saved path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{model_name}_best.joblib"
    save_model(search.best_estimator_, path)
    return path


def load_model(path: str | Path):
    path = Path(path)
    return joblib.load(path)
