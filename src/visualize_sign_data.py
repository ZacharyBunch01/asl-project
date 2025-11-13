"""
Entry point for running visualizations on parsed sign data.
"""

import parseData
from visualization.config import configure_plots
from visualization import (
    plot_categorical_distributions,
    plot_handshape_movement_heatmap,
    plot_missing_values,
    plot_numeric_correlation,
    save_summary_stats,
)

def main():
    configure_plots()

    signData = parseData.signData
    if signData is None:
        raise RuntimeError(
            "Sign data not available; run parseData.py first or check dataset paths."
        )

    print(f"Data loaded: {signData.shape[0]} rows, {signData.shape[1]} columns.")

    output_folder = "../Figures"

    # Run visualization steps
    plot_categorical_distributions(signData, folder=output_folder)
    plot_handshape_movement_heatmap(signData, folder=output_folder)
    plot_missing_values(signData, folder=output_folder)
    plot_numeric_correlation(signData, max_features=30, folder=output_folder)
    save_summary_stats(signData, folder=output_folder)

if __name__ == "__main__":
    main()