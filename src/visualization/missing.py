import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .export import save_plot

def plot_missing_values(df, top_n=15, folder="../Figures"):
    # If the DataFrame is empty or None, there's nothing to plot
    if df is None or df.empty:
        print("[missing] DataFrame is empty, skipping missing-value plot")
        return
    
    # Compute proportion of missing values per column and sort descending
    missing = df.isnull().mean().sort_values(ascending=False)

    # If no column has missing values, skip plotting
    if missing.empty or missing.max() == 0:
        print("[missing] No missing values to plot.")
        return

    # Select only the top N columns with the most missing data
    missing_top = missing.head(top_n)

    # Determine figure height dynamically based on number of columns
    height = max(5, 0.35 * len(missing_top))

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the missing proportions as a horizontal bar chart
    sns.barplot(
        x=missing_top,
        y=missing_top.index,
        color="royalblue",
        ax=ax,
    )

    # Labeling and title for readability
    ax.set_title("Top Columns by Missing Value Percentage", fontsize=14, pad=10)
    ax.set_xlabel("Proportion Missing", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)

    # Tight layout prevents label cutoff
    plt.tight_layout()

    # Save the figure to the output folder
    save_plot(fig, "missing_values", folder=folder)

    # Close the figure to free memory
    plt.close(fig)