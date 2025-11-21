import pandas as pd
from .export import get_output_dir

def save_summary_stats(df, folder="../Figures", filename="signData_summary.csv"):
    """Save a descriptive statistics table for the dataset."""
    if df is None or df.empty:
        print("[summary] DataFrame is empty, skipping summary export.")
        return
    
    out_dir = get_output_dir(folder)
    summary_path = out_dir / filename

    summary_df = df.describe(include="all").transpose()
    summary_df.to_csv(summary_path)

    print(f"[summary] Saved summary statistics to: {summary_path.resolve()}")
    print(f"[summary] Output directory:          {out_dir.resolve()}")

