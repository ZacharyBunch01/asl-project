import pandas as pd
import visualization.summary as summary
from pathlib import Path


def test_summary_skips_empty_df(tmp_path, capsys):
    # Empty DataFrame should cause summary export to be skipped
    empty_df = pd.DataFrame()

    summary.save_summary_stats(empty_df, folder=str(tmp_path), filename="test.csv")

    # Ensure no file was created
    assert not (tmp_path / "test.csv").exists()

    # Capture printed output and confirm the warning message
    captured = capsys.readouterr()
    assert "DataFrame is empty" in captured.out


def test_summary_writes_csv(tmp_path):
    # Non-empty DataFrame should result in a summary CSV being written
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })

    out_file = tmp_path / "summary.csv"

    summary.save_summary_stats(df, folder=str(tmp_path), filename="summary.csv")

    # Check that the file was created
    assert out_file.exists()

    # Validate content is not empty
    text = out_file.read_text()
    assert "A" in text
    assert "B" in text


def test_summary_creates_folder_if_missing(tmp_path):
    # Summary function should create the output folder if it does not exist
    df = pd.DataFrame({"A": [1, 2, 3]})

    new_folder = tmp_path / "new_output"
    out_file = new_folder / "stats.csv"

    # Confirm folder does not exist before the call
    assert not new_folder.exists()

    summary.save_summary_stats(df, folder=str(new_folder), filename="stats.csv")

    # Folder and file should now exist
    assert new_folder.exists()
    assert out_file.exists()


def test_summary_uses_correct_filename(tmp_path):
    # Ensure the provided filename is respected by the writer
    df = pd.DataFrame({"A": [10, 20]})
    custom_name = "custom_summary.csv"

    summary.save_summary_stats(df, folder=str(tmp_path), filename=custom_name)

    # File with custom name should exist in the output folder
    assert (tmp_path / custom_name).exists()
