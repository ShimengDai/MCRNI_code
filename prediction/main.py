import yaml
from pathlib import Path
from src.io import save_best_model
import joblib

from src.data import (
    load_data,
    clean_data,
    make_splits,
)

from src.features import get_feature_columns, make_xy
from src.train import run_grid_search
from src.io import save_best_model


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        data_cfg = cfg["data"]
        feature_cfg = cfg["feature"]
        models_cfg = cfg["models"]
        cv_cfg = cfg["cv"]
        run_cfg = cfg.get("run", {})
        models_to_run = run_cfg.get("models", list(models_cfg.keys()))
        missing_cfg = cfg.get("missing", None)
    # =========================
    #  test data.py
    # =========================

    df = load_data(data_cfg)

    df = clean_data(df, data_cfg)

    df_train, df_test = make_splits(
        df,
        test_size=data_cfg["test_size"],
        seed=data_cfg["seed"],
    )

    split_dir = Path("output/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "df_train.csv"
    test_csv = split_dir / "df_test.csv"

    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"[Split] Saved train -> {train_csv}")
    print(f"[Split] Saved test  -> {test_csv}")

    # (temporary) sanity checks
    print("Train rows   :", len(df_train))
    print("Test rows    :", len(df_test))

    # =========================
    # test features.py
    # =========================
    # 1) feature columns should be derived from TRAIN ONLY
    feature_cols = get_feature_columns(
        df_train,
        data_cfg=data_cfg,
        feature_cfg=feature_cfg,
    )

    # 2) build X/y/groups for train and test
    X_train, y_train, g_train = make_xy(
        df_train,
        data_cfg=data_cfg,
        feature_cols=feature_cols,
    )
    X_test, y_test, g_test = make_xy(
        df_test,
        data_cfg=data_cfg,
        feature_cols=feature_cols,
    )

    # 3) sanity checks for shapes / alignment
    print("\n[Feature test]")
    print("n_features  :", len(feature_cols))
    print(
        "X_train     :",
        X_train.shape,
        "y_train:",
        y_train.shape,
        "groups:",
        g_train.nunique(),
    )
    print(
        "X_test      :",
        X_test.shape,
        "y_test :",
        y_test.shape,
        "groups:",
        g_test.nunique(),
    )

    assert X_train.shape[0] == y_train.shape[0] == len(g_train)
    assert X_test.shape[0] == y_test.shape[0] == len(g_test)
    assert X_train.shape[1] == X_test.shape[1] == len(feature_cols)
    assert list(X_train.columns) == list(X_test.columns) == feature_cols

    # 5) optional: quick target balance
    print("y_train value counts:\n", y_train.value_counts(dropna=False))
    print("y_test  value counts:\n", y_test.value_counts(dropna=False))

    # =========================
    #  test model.py & train.py
    # =========================

    missing_cfg = cfg.get("missing", None)
    output_dir = Path("output") / "models"

    for model_name in models_to_run:
        print("\n" + "=" * 50)
        print(f"Running model: {model_name}")
        print("=" * 50)

        search = run_grid_search(
            X_train,
            y_train,
            g_train,
            model_name=model_name,
            model_cfg=models_cfg[model_name],
            cv_cfg=cv_cfg,
            seed=data_cfg["seed"],
            missing_cfg=missing_cfg,
        )
        print(f"[{model_name}] Best CV score : {search.best_score_:.4f}")
        print(f"[{model_name}] Best params   : {search.best_params_}")

        saved_path = save_best_model(search, out_dir=output_dir, model_name=model_name)
        print(f"[{model_name}] Saved best model -> {saved_path}")


if __name__ == "__main__":
    main()
