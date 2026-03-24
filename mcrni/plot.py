# import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
# import os
import numpy as np


def compute_mcrni_array(self, auc_grid):
    """
    Compute MCRNI values across a shared AUC grid.

    Parameters:
    - auc_grid: array-like of AUC thresholds

    Returns:
    - mcrni_arr: numpy array
    """

    # Ensure ranks are computed once
    if not hasattr(self, "ranks"):
        self.__rank__()

    auc_grid = np.asarray(auc_grid)

    mcrni_arr = np.array([self.compute_mcrni_with_auc(a) for a in auc_grid])

    return mcrni_arr


def init_mcrni_plot(mcrni_arr, auc_grid, label=None):
    """
    Initialize MCRNI plot using precomputed arrays.

    Parameters:
    - mcrni_arr: numpy array (y-axis)
    - auc_grid: numpy array (x-axis)
    - label: optional label

    Returns:
    - fig, ax
    """

    auc_grid = np.asarray(auc_grid)

    fig, ax = plt.subplots()

    ax.plot(auc_grid, mcrni_arr, label=label)

    ax.set_xlabel("AUC Threshold (a)")
    ax.set_ylabel("MCRNI")
    ax.set_title("MCRNI Robustness Curves")
    ax.grid(True)
    ax.legend()

    return fig, ax


def add_mcrni_curve(self, other_model, label=None):
    """
    Add another MCRNI curve from a different model.

    Parameters:
    - other_model: another MCRNI instance
    - label: optional label
    """

    if not hasattr(self, "_ax"):
        raise ValueError("Call init_mcrni_plot() first.")

    if not isinstance(other_model, MCRNI):
        raise ValueError("Input must be an MCRNI instance.")

    # Compute curve using SAME auc grid
    mcrni_arr = other_model.compute_mcrni_array(self._auc_grid)

    if label is None:
        label = getattr(other_model, "name", f"Model {len(self._ax.lines) + 1}")

    self._ax.plot(self._auc_grid, mcrni_arr, marker="o", label=label)


def show_mcrni_plot(self):
    """
    Display the MCRNI plot.
    """
    if not hasattr(self, "_ax"):
        raise ValueError("No plot initialized.")

    self._ax.legend()
    plt.show()
