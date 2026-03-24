import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np







if __name__ == "__main__":
    df = load_metrics()

    # Generate plots
    # Generate plots
    plot_mcrni_bars(df)
    plot_metric_bars(df, metric="AUC")
    plot_metric_bars(df, metric="Accuracy")
    plot_scatter_mcrni_vs_metric(df, metric="AUC")
