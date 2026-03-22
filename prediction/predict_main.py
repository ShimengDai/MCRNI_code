# src/predict.py
from pathlib import Path
import pandas as pd
import yaml

# from src.io import load_model
# from src.features import get_feature_columns
from src.predict import predict_one, add_pred_column, run_predict_over_models


# =========================
# Script entry
# =========================


def main():
    # 1) load config (single fixed yaml)
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    feature_cfg = cfg["feature"]
    predict_cfg = cfg["predict"]

    # 2) resolve paths
    input_path = Path(predict_cfg["input_path"])
    output_path = Path(predict_cfg["output_path"])
    models_dir = Path(predict_cfg["models_dir"])

    models_cfg = predict_cfg.get("models", [])
    if not models_cfg:
        raise ValueError("predict.models is empty in yaml.")

    # 3) load dataset
    df = pd.read_csv(input_path)

    # 4) predict over all models
    df_out = run_predict_over_models(
        df,
        models_dir=models_dir,
        models_cfg=models_cfg,
        data_cfg=data_cfg,
        feature_cfg=feature_cfg,
    )

    # 5) save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
