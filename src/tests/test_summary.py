import pandas as pd
import visualization.summary as summary
from pathlib import Path


def test_summary_skips_empty_df(tmp_path, capsys):
    """If the DataFrame is empty, summary should not be saved."""
    empty_df = pd.DataFrame()

    summary.save_summary_stats(empty_df, folder=str(tmp_path), filename="test.csv")

    # Ensure no file created
    assert not (tmp_path / "test.csv").exists()

    # Capture printed message
    captured = capsys.readouterr()
    assert "DataFrame is empty" in captured.out


def test_summary_writes_csv(tmp_path):
    """A non-empty DataFrame should produce a .csv summary file."""
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })

    out_file = tmp_path / "summary.csv"

    summary.save_summary_stats(df, folder=str(tmp_path), filename="summary.csv")

    # Check file exists
    assert out_file.exists()

    # Validate content is not empty
    text = out_file.read_text()
    assert "A" in text
    assert "B" in text


def test_summary_creates_folder_if_missing(tmp_path):
    """Summary writer should create folder automatically if it does not exist."""
    df = pd.DataFrame({"A": [1, 2, 3]})

    new_folder = tmp_path / "new_output"
    out_file = new_folder / "stats.csv"

    # Folder doesn't exist before test
    assert not new_folder.exists()

    summary.save_summary_stats(df, folder=str(new_folder), filename="stats.csv")

    # Folder must exist now
    assert new_folder.exists()
    assert out_file.exists()


def test_summary_uses_correct_filename(tmp_path):
    """Ensure filename argument is respected."""
    df = pd.DataFrame({"A": [10, 20]})
    custom_name = "custom_summary.csv"

    summary.save_summary_stats(df, folder=str(tmp_path), filename=custom_name)

    assert (tmp_path / custom_name).exists()
