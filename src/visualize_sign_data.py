"""
Entry point for running visualizations on parsed sign data.
"""

from sign_helpers.sign_data import get_sign_data

# Apply global plot styling (Seaborn + Matplotlib)
from visualization.config import configure_plots

# Visualization functions collected in __init__.py
from visualization import (
    plot_categorical_distributions,
    plot_handshape_movement_heatmap,
    plot_missing_values,
    plot_numeric_correlation,
    save_summary_stats,
)

def main():
    # Initialize global plotting settings (only runs once)
    configure_plots()

    # Load processed DataFrame with canonical ASL features
    signData = get_sign_data()
    if signData is None or signData.empty:
        raise RuntimeError("Sign data not available or empty")

    print(f"[runner] Data loaded: {signData.shape[0]} rows x {signData.shape[1]} columns.")

    # All outputs will be saved into ../Figures
    output_folder = "../Figures"

    # Run visualization steps
    print("[runner] Creating categorical distribution plots...")
    plot_categorical_distributions(signData, folder=output_folder)
    
    print("[runner] Creating handshape x movement heatmap...")
    plot_handshape_movement_heatmap(signData, folder=output_folder)

    print("[runner] Creating missing values plot...")
    plot_missing_values(signData, folder=output_folder)

    print("[runner] Creating numeric correlation heatmap...")
    plot_numeric_correlation(signData, max_features=30, folder=output_folder)

    print("[runner] Saving summary statistics...")
    save_summary_stats(signData, folder=output_folder)

    print(f"[runner] All visualizations saved in: {output_folder}")

if __name__ == "__main__":
    main()
