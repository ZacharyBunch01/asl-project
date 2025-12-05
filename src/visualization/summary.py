import pandas as pd
from .export import get_output_dir

def save_summary_stats(df, folder="../Figures", filename="signData_summary.csv"):
    # If the DataFrame is empty or missing, do not attempt to export
    if df is None or df.empty:
        print("[summary] DataFrame is empty, skipping summary export.")
        return
    
    # Ensure output directory exists (creates it if needed)
    out_dir = get_output_dir(folder)

    # Build full file path for the summary CSV
    summary_path = out_dir / filename

    # Compute summary statistics; transpose for better readability
    summary_df = df.describe(include="all").transpose()

    # Save the summary statistics to CSV
    summary_df.to_csv(summary_path)

    print(f"[summary] Saved summary statistics to: {summary_path.resolve()}")
    print(f"[summary] Output directory:          {out_dir.resolve()}")

