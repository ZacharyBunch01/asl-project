
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .utils import save_plot

def plot_handshape_movement_heatmap(
    df: pd.DataFrame,
    handshape_col: str = "Handshape.2.0",
    movement_col: str = "Movement.2.0",
    folder: str = "../Figures",
) -> None:
    """Plot a heatmap of handshape vs movement frequency."""
    if handshape_col not in df.columns or movement_col not in df.columns:
        print("[heatmaps] Handshape/Movement columns not found; skipping heatmap.")
        return

    crosstab = pd.crosstab(df[handshape_col], df[movement_col])

    size = max(12, 0.3 * max(len(crosstab.index), len(crosstab.columns)))
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
      crosstab,
      cmap="Blues",
      cbar=True,
      ax=ax,
      square=True,
      cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Handshape vs. Movement Frequency", fontsize=14, pad=12)
    plt.xticks(rotation=90, ha="center")
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_plot(fig, "heatmap_handshape_movement", folder=folder)
    plt.close(fig)


def plot_numeric_correlation(
    df: pd.DataFrame,
    max_features: int | None = None,
    folder: str = "../Figures",
) -> None:
    """Plot a correlation heatmap for numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        print("[heatmaps] Not enough numeric columns for correlation heatmap.")
        return

    if max_features is not None and len(num_cols) > max_features:
        stds = df[num_cols].std().sort_values(ascending=False)
        num_cols = stds.head(max_features).index.tolist()

    corr = df[num_cols].corr(numeric_only=True)

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
    ax.set_title("Numeric Feature Correlation Heatmap", fontsize=14, pad=12)
    plt.xticks(rotation=90, ha="center")
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_plot(fig, "numeric_correlation", folder=folder)
    plt.close(fig)
