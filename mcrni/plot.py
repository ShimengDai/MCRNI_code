import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np



def compute_mcrni_array(self, auc_grid):
    """
    Compute MCRNI values over a fixed AUC grid.

    Parameters:
    - auc_grid: numpy array of AUC thresholds (shared across models)

    Returns:
    - mcrni_arr: numpy array
    """

    mcrni_arr = np.array([
        self.compute_mcrni_with_auc(a) for a in auc_grid
    ])

    return mcrni_arr



def init_mcrni_plot(self, auc_grid, mcrni_arr, label=None):
    """
    Initialize plot with first MCRNI curve.
    """

    self._auc_grid = auc_grid
    self._fig, self._ax = plt.subplots()

    if label is None:
        label = getattr(self, "name", "Model 1")

    self._ax.plot(auc_grid, mcrni_arr, marker='o', label=label)

    self._ax.set_xlabel("AUC Threshold (a)")
    self._ax.set_ylabel("MCRNI")
    self._ax.set_title("MCRNI Robustness Curves")
    self._ax.grid(True)






def add_mcrni_curve(self, mcrni_arr, label=None):
    """
    Add another MCRNI curve using the same AUC grid.
    """

    if not hasattr(self, "_ax"):
        raise ValueError("Call init_mcrni_plot first.")

    if label is None:
        label = getattr(self, "name", f"Model {len(self._ax.lines)+1}")

    self._ax.plot(self._auc_grid, mcrni_arr, marker='o', label=label)




    
if __name__ == "__main__":
    df = load_metrics()

    # Generate plots
    # Generate plots
    plot_mcrni_bars(df)
    plot_metric_bars(df, metric="AUC")
    plot_metric_bars(df, metric="Accuracy")
    plot_scatter_mcrni_vs_metric(df, metric="AUC")
