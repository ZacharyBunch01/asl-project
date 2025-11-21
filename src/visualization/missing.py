import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .export import save_plot

def plot_missing_values(df, top_n=15, folder="../Figures"):
    # Plot the top-N columns by proportion of missing values.
    if df is None or df.empty:
        print("[missing] DataFrame is empty, skipping missing-value plot")
        return
    
    missing = df.isnull().mean().sort_values(ascending=False)

    if missing.empty or missing.max() == 0:
        print("[missing] No missing values to plot.")
        return

    missing_top = missing.head(top_n)

    height = max(5, 0.35 * len(missing_top))
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.barplot(
        x=missing_top,
        y=missing_top.index,
        color="royalblue",
        ax=ax,
    )

    ax.set_title("Top Columns by Missing Value Percentage", fontsize=14, pad=10)
    ax.set_xlabel("Proportion Missing", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)

    plt.tight_layout()
    save_plot(fig, "missing_values", folder=folder)
    plt.close(fig)