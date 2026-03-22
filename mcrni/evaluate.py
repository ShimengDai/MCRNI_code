import pandas as pd
from MCRNI import MCRNI
import os

def evaluate_models(input_csv='predictions.csv', output_csv='results/metrics_summary.csv', threshold=0.5, auc_target=0.5):
    """
    Evaluate multiple models from a prediction CSV file using MCRNI and standard metrics.

    Parameters:
    - input_csv: str, path to the input CSV file with true labels and model predictions.
    - output_csv: str, path to save the metrics summary CSV.
    - threshold: float, threshold for binary classification.
    - auc_target: float, AUC value to use as the nullifying threshold (typically 0.5).
    """

    # Load data
    df = pd.read_csv(input_csv)
    y_true = df['label'].values

    # Prepare output list
    all_metrics = []

    # Iterate through prediction columns
    for col in df.columns:
        if col.endswith('_prob'):
            model_name = col.replace('_prob', '')
            y_scores = df[col].values
            
            # Instantiate MCRNI object
            model = MCRNI(y_true, y_scores)
            model.__rank__()
            model.compute_mcrni_with_auc(auc_target)

            # Gather metrics
            metrics = model.compute_standard_metrics(threshold)
            metrics['Model'] = model_name
            metrics['MCRNI'] = model.mcrni
            metrics['Rank Sum Pos'] = model.rank_sum_pos
            metrics['Rank Sum Neg'] = model.rank_sum_neg
            metrics['U Stat Pos'] = model.u_pos
            metrics['U Stat Neg'] = model.u_neg

            all_metrics.append(metrics)

    # Convert to DataFrame
    summary_df = pd.DataFrame(all_metrics)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save to CSV
    summary_df.to_csv(output_csv, index=False)
    print(f"✅ Metrics summary saved to: {output_csv}")

if __name__ == "__main__":
    evaluate_models()
