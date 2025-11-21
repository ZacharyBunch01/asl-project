
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .export import save_plot

def plot_handshape_movement_heatmap(df, handshape_col="Handshape", movement_col="Movement", folder="../Figures"):
    #Plot a heatmap of handshape vs movement frequency

    if handshape_col not in df.columns or movement_col not in df.columns:
        print("Columns not found: {handshape_col}, {movement_col}. Skipping.")
        return

    crosstab = pd.crosstab(df[handshape_col], df[movement_col])

    if crosstab.empty:
        print("[heatmaps] Crosstab is empty; no heatmap generated.")
        return

    # Dynamic figure sized Based on category counts
    size = max(10, 0.4 * max(len(crosstab.index), len(crosstab.columns)))
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
      crosstab,
      cmap="Blues",
      cbar=True,
      ax=ax,
      square=True,
      cbar_kws={"shrink": 0.6},
    )

    ax.set_title("Handshape vs. Movement Frequency", fontsize=16, pad=14)
    ax.set_xlabel(movement_col)
    ax.set_ylabel(handshape_col)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_plot(fig, "heatmap_handshape_movement", folder=folder)
    plt.close(fig)


def plot_numeric_correlation(df, max_features=None, folder="../Figures"):
    #Plot a correlation heatmap for numeric columns.

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        print("[heatmaps] Not enough numeric columns for correlation heatmap.")
        return

    # If there are too many features, pick the most variable ones
    if max_features is not None and len(num_cols) > max_features:
        stds = df[num_cols].std().sort_values(ascending=False)
        num_cols = stds.head(max_features).index.tolist()

    corr = df[num_cols].corr()

    if corr.empty:
        print("[heatmaps] Corrleation matrix is empty; skipping heatmap")
        return

    size = max(8, 0.4 * len(num_cols))
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.7},
    )

    ax.set_title("Numeric Feature Correlation Heatmap", fontsize=16, pad=14)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_plot(fig, "numeric_correlation", folder=folder)
    plt.close(fig)
