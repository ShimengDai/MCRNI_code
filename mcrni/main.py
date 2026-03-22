import pandas as pd
from MCRNI import MCRNI


import os


def main():
    df = pd.read_csv("input_data/predictions_wide_full.csv")

    group_col = "urbanicity"
    label_col = "dropout"
    score_col = "pred_svm_proba"  # pred_logreg_proba, pred_rf_proba, pred_svm_proba

    #  a = AUC Threshold
    auc_a = 0.5  #  compute_mcrni_with_auc(a)
    cls_threshold = 0.2  #  compute_standard_metrics(threshold)

    rows = []

    for area, g in df.groupby(group_col, sort=False):
        y_true = g[label_col]
        y_scores = g[score_col]

        if y_true.nunique() < 2:
            rows.append(
                {
                    "urbanicity": area,
                    "n": len(g),
                    "note": "only one class in labels; ROC-AUC/MCRNI may be undefined",
                }
            )
            continue

        model = MCRNI(y_true, y_scores)

        # 先算 MCRNI（内部会确保 rank 已经算好）
        mcrni = model.compute_mcrni_with_auc(auc_a)

        # 再算标准分类指标（Accuracy/Precision/Recall/F1/ROC-AUC）
        metrics = model.compute_standard_metrics(threshold=cls_threshold)

        # 组装“一行结果”（列名风格跟 save_metrics 一致）
        row = {
            "urbanicity": area,
            "n": len(g),
            "Rank Sum Pos": model.rank_sum_pos,
            "Rank Sum Neg": model.rank_sum_neg,
            "U Stat Pos": model.u_pos,
            "U Stat Neg": model.u_neg,
            "MCRNI": mcrni,
            "AUC": metrics["ROC-AUC"],
            "Average Precision": metrics["Average Precision"],
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            # confusion matrix，
            # "Confusion Matrix": metrics["Confusion Matrix"],
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    out_path = "results/pred_svm_area.csv"  # pred_log_area.csv/ pred_rf_area.csv/ pred_svm_area.csv
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ Saved grouped metrics to: {out_path}")


if __name__ == "__main__":
    main()
