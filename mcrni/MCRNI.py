import numpy as np
import pandas as pd
from scipy.stats import rankdata

import os

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


class MCRNI:
    def __init__(self, y_true, y_scores):
        """
        Initialize the MCRNI class

        Parameters:
        - y_true: list or numpy array of binary labels (0s and 1s)
        - y_scores: list or numpy array of model scores or probabilities.
        - threshold: float, optional threshold to binarize predictions if needed

        """

        self.y_true = np.array(y_true)
        self.y_scores = np.array(y_scores)

        # Validate inputs
        if self.y_true.shape != self.y_scores.shape:
            raise ValueError("Shapes of y_true and y_scores must match.")
        if not set(np.unique(self.y_true)).issubset({0, 1}):
            raise ValueError("y_true must contain only 0 and 1.")

        # Validate inputs
        self.n_pos = np.sum(self.y_true)
        self.n_neg = len(self.y_true) - self.n_pos

    def __rank__(self, verbose=False):
        """
        Assign ordinal ranks to scores and compute rank sums and U statistics.
        Ensures that ranks align with original score/label order.
        """
        # Assign ordinal ranks (unique, order-preserving)
        self.ranks = rankdata(self.y_scores, method="ordinal")

        # Compute rank sums
        self.rank_sum_pos = np.sum(self.ranks[self.y_true == 1])
        self.rank_sum_neg = np.sum(self.ranks[self.y_true == 0])

        # Compute Mann-Whitney U statistics
        self.u_pos = self.rank_sum_pos - (self.n_pos * (self.n_pos + 1)) / 2
        self.u_neg = self.rank_sum_neg - (self.n_neg * (self.n_neg + 1)) / 2

        # Optional verbose output
        if verbose:
            print("\n--- Ranking Debug ---")
            print("Scores:         ", self.y_scores)
            print("Ranks:          ", self.ranks)
            print("True Labels:    ", self.y_true)
            print("Positive Ranks: ", self.ranks[self.y_true == 1])
            print("Rank Sum (pos): ", self.rank_sum_pos)
            print("U-Stat (pos):   ", self.u_pos)

        return self.ranks

    def __R__(self):
        """
        Return rank sums for the positive and negative group.
        Assumes __rank__() has been called.
        """
        return self.rank_sum_pos, self.rank_sum_neg

    def __U__(self):
        """
        Return Mann-Whitney U statistics for each group.
        Assumes __rank__() has been called.
        """
        return self.u_pos, self.u_neg

    def compute_mcrni_with_auc(self, a):
        """
        Compute the Mean Change in Rank to Nullify the Inference (MCRNI),
        for a specified AUC threshold 'a'.

        Parameters:
        - a: float, AUC threshold (e.g., 0.5 for chance level)

        Returns:
        - mcrni: float, average change in rank needed to reduce AUC to 'a'
        """

        # Make sure ranks and rank sums are available
        if not hasattr(self, "ranks"):
            self.__rank__()

        # Sizes of the positive (A) and negative (B) groups
        n_pos = self.n_pos
        n_neg = self.n_neg

        # Observed rank sum for group A (positives)
        r_pos = self.rank_sum_pos

        # Threshold rank sum R_A^#
        r_thresh = a * n_pos * n_neg + (n_pos * (n_pos + 1)) / 2

        # Compute MCRNI
        self.mcrni = (r_pos - r_thresh) / n_pos

        return self.mcrni

    def compute_mcrni_with_R_pos(self, r_thresh):
        """

        # Load and rank both models
        # model1 = MCRNI(y_true1, y_scores1)
        # model2 = MCRNI(y_true2, y_scores2)

        # model1.__rank__()
        # model2.__rank__()

        # # Use model 2 to get the R_A^# threshold
        # r_thresh = model2.rank_sum_pos

        # # Now compute how much model 1 would have to change to hit that threshold
        # mcrni_relative = model1.compute_mcrni_with_R_pos(r_thresh)

        # print(f"Comparative MCRNI (Model 1 vs Model 2 threshold): {mcrni_relative:.3f}")


        """
        # Group sizes

        n_pos = self.n_pos

        # Compute MCRNI

        self.mcrni = (self.rank_sum_pos - r_thresh) / n_pos

        return self.mcrni

    def scrni(self):
        """
        Sum Change in Rank to Nullify the Inference
        """

        self.scrni = self.mcrni * self.n_pos

        return self.scrni

    def __add__(self, other):
        if isinstance(other, MCRNI):
            return self.mcrni + other.mcrni
        elif isinstance(other, (int, float)):  # support sum()
            return self.mcrni + other
        else:
            raise ValueError("Unsupported type for addition")

    def __radd__(self, other):
        return self.__add__(other)

    def compute_standard_metrics(self, threshold=0.5):
        """
        Compute standard classification metrics using a fixed threshold
        to binarize predicted scores.

        Parameters:
        - threshold: float, cutoff for converting scores to binary predictions

        Returns:
        Returns:
        - metrics: dict containing confusion matrix, accuracy, precision, recall, F1, AUC, and average precision
        """

        # Binarize predictions
        y_pred = (self.y_scores >= threshold).astype(int)

        # Compute metrics
        metrics = {
            "Confusion Matrix": confusion_matrix(self.y_true, y_pred).tolist(),
            "Accuracy": accuracy_score(self.y_true, y_pred),
            "Precision": precision_score(self.y_true, y_pred, zero_division=0),
            "Recall": recall_score(self.y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(self.y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(self.y_true, self.y_scores),
            "Average Precision": average_precision_score(self.y_true, self.y_scores),
        }

        return metrics

    def print_report(self, threshold=0.5):
        """
        Print a detailed report including:
        - Rank statistics
        - U statistics
        - MCRNI
        - ROC-AUC
        - Standard classification metrics
        """

        print("📊 === Model Evaluation Report ===\n")

        # Ensure ranks and U are computed
        if not hasattr(self, "ranks"):
            self.__rank__()

        # Compute standard metrics
        metrics = self.compute_standard_metrics(threshold)

        # Print Rank Sums and U stats
        print(f"Rank Sum (Positive group): {self.rank_sum_pos:.2f}")
        print(f"Rank Sum (Negative group): {self.rank_sum_neg:.2f}")
        print(f"U Statistic (Positive group): {self.u_pos:.2f}")
        print(f"U Statistic (Negative group): {self.u_neg:.2f}")

        # Print MCRNI if available
        if hasattr(self, "mcrni"):
            print(f"MCRNI (compared to threshold or reference): {self.mcrni:.4f}")
        else:
            print(
                "MCRNI not computed yet. Run compute_mcrni_with_auc() or compute_mcrni_with_R_pos()."
            )

        # Print standard classification metrics
        print("\n🔍 Classification Metrics (threshold = {:.2f}):".format(threshold))
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        print(
            "\n📈 ROC-AUC Score (probability ranking): {:.4f}".format(
                metrics["ROC-AUC"]
            )
        )

    def save_metrics(
        self, filename="model_metrics.csv", folder="results", threshold=0.5
    ):
        """
        Save metrics to a CSV file in the specified folder.

        Parameters:
        - filename: str, the name of the output CSV file
        - folder: str, subdirectory where results are saved
        - threshold: float, cutoff used for binary classification
        """

        # Ensure rank/U/MCRNI are computed
        if not hasattr(self, "ranks"):
            self.__rank__()

        if not hasattr(self, "mcrni"):
            self.compute_mcrni_with_auc(0.5)  # Default to chance level if not set

        metrics = self.compute_standard_metrics(threshold)

        # Build the record
        data = {
            "Rank Sum Pos": [self.rank_sum_pos],
            "Rank Sum Neg": [self.rank_sum_neg],
            "U Stat Pos": [self.u_pos],
            "U Stat Neg": [self.u_neg],
            "MCRNI": [self.mcrni],
            "AUC": [metrics["ROC-AUC"]],
            "Average Precision": [metrics["Average Precision"]],
            "Accuracy": [metrics["Accuracy"]],
            "Precision": [metrics["Precision"]],
            "Recall": [metrics["Recall"]],
            "F1 Score": [metrics["F1 Score"]],
        }

        df = pd.DataFrame(data)

        # Ensure output folder exists
        os.makedirs(folder, exist_ok=True)

        # Full path
        full_path = os.path.join(folder, filename)

        # Save
        if os.path.exists(full_path):
            df.to_csv(full_path, mode="a", header=False, index=False)
        else:
            df.to_csv(full_path, index=False)

        print(f"✅ Metrics saved to: {full_path}")

    @staticmethod
    def plot_roc_curves(models, labels, colors=None, save_path=None):
        """
        Plot ROC curves for one or multiple MCRNI models.

        Parameters:
        - models: list of MCRNI instances
        - labels: list of labels for each model
        - colors: optional list of colors for each curve
        - save_path: optional path to save the plot as a file
        """
        fig, ax = plt.subplots()

        # Plot ROC for each model
        for idx, model in enumerate(models):
            fpr, tpr, _ = roc_curve(model.y_true, model.y_scores)
            auc = roc_auc_score(model.y_true, model.y_scores)
            color = colors[idx] if colors else None
            ax.plot(fpr, tpr, label=f"{labels[idx]} (AUC = {auc:.2f})", color=color)

        # Add random classifier line just once
        ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"📁 ROC curve saved to {save_path}")
        else:
            plt.show()
