import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_metrics(csv_path='results/metrics_summary.csv'):
    """Load model evaluation metrics from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    return pd.read_csv(csv_path)


def plot_mcrni_bars(df, save_path='results/mcrni_barplot.png'):
    """Plot MCRNI values as a bar plot sorted by model order."""
    # Extract numerical index for sorting
    df['Model_Index'] = df['Model'].str.extract('(\d+)').astype(int)
    sorted_df = df.sort_values('Model_Index')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=sorted_df, x='Model', y='MCRNI', palette='coolwarm')
    plt.title('MCRNI per Model')
    plt.xlabel('Model')
    plt.ylabel('MCRNI')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # Clean up
    df.drop('Model_Index', axis=1, inplace=True)



def plot_metric_bars(df, metric='AUC', save_path=None):
    """Plot bar plot for a selected metric with consistent model order (model_0, model_1, ...)."""
    # Sort models numerically if named as model_0, model_1, ...
    df['Model_Index'] = df['Model'].str.extract('(\d+)').astype(int)
    sorted_df = df.sort_values('Model_Index')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=sorted_df, x='Model', y=metric, palette='viridis')
    plt.title(f'{metric} per Model')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # Clean up
    df.drop('Model_Index', axis=1, inplace=True)



def plot_scatter_mcrni_vs_metric(df, metric='AUC', save_path=None):
    """Scatter plot of MCRNI vs another metric."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='MCRNI', y=metric, hue='Model', s=100, palette='deep')
    plt.title(f'MCRNI vs {metric}')
    plt.xlabel('MCRNI')
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    df = load_metrics()

    # Generate plots
    # Generate plots
    plot_mcrni_bars(df)
    plot_metric_bars(df, metric='AUC')
    plot_metric_bars(df, metric='Accuracy')
    plot_scatter_mcrni_vs_metric(df, metric='AUC')
