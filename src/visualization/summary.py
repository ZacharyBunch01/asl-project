import pandas as pd
from .utils import get_output_dir

def save_summary_stats(
    df: pd.DataFrame,
    folder: str = "../Figures",
    filename: str = "signData_summary.csv",
) -> None:
    """Save a descriptive statistics table for the dataset."""
    out_dir = get_output_dir(folder)
    summary_path = out_dir / filename

    df.describe(include="all").transpose().to_csv(summary_path)
    print("Saved data summary to", summary_path.resolve())
    print("All figures and summary in:", out_dir.resolve())
