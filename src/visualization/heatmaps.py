
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .export import save_plot

def plot_handshape_movement_heatmap(df, handshape_col="Handshape", movement_col="Movement", folder="../Figures"):
    # Ensure required columns exist before plotting
    if handshape_col not in df.columns or movement_col not in df.columns:
        print("Columns not found: {handshape_col}, {movement_col}. Skipping.")
        return

    # Build contingency table between handshape and movement
    crosstab = pd.crosstab(df[handshape_col], df[movement_col])

    # Skip if there is no usable cross-tab data
    if crosstab.empty:
        print("[heatmaps] Crosstab is empty; no heatmap generated.")
        return

    # Dynamically scale figure size based on number of categories
    size = max(10, 0.4 * max(len(crosstab.index), len(crosstab.columns)))
    fig, ax = plt.subplots(figsize=(size, size))

    # Draw heatmap for category relationships
    sns.heatmap(
      crosstab,
      cmap="Blues",
      cbar=True,
      ax=ax,
      square=True,               # enforce square cells for aesthetics
      cbar_kws={"shrink": 0.6},  # shrink colorbar for readability
    )

    # Title + axis labels
    ax.set_title("Handshape vs. Movement Frequency", fontsize=16, pad=14)
    ax.set_xlabel(movement_col)
    ax.set_ylabel(handshape_col)

    # Rotate tick labels for visibility
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Save and close figure
    plt.tight_layout()
    save_plot(fig, "heatmap_handshape_movement", folder=folder)
    plt.close(fig)


def plot_numeric_correlation(df, max_features=None, folder="../Figures"):
    # Extract numeric columns only

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Need at least 2 numeric columns to compute a correlation matrix
    if len(num_cols) < 2:
        print("[heatmaps] Not enough numeric columns for correlation heatmap.")
        return

    # If too many features, keep the ones with the greatest variability
    if max_features is not None and len(num_cols) > max_features:
        stds = df[num_cols].std().sort_values(ascending=False)
        num_cols = stds.head(max_features).index.tolist()

    # Compute correlation matrix
    corr = df[num_cols].corr()

    # Skip plotting if correlation matrix is empty
    if corr.empty:
        print("[heatmaps] Corrleation matrix is empty; skipping heatmap")
        return

    # Scale figure size with number of features
    size = max(8, 0.4 * len(num_cols))
    fig, ax = plt.subplots(figsize=(size, size))

    # Draw correlation heatmap
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
