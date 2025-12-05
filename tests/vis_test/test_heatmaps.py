import pandas as pd
import visualization.heatmaps as heatmaps
from pathlib import Path


def test_handshape_movement_missing_columns(monkeypatch, tmp_path):
    # Capture calls to save_plot
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    # Replace save_plot with fake version
    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    # DataFrame missing required columns
    df = pd.DataFrame({
        "Other": [1, 2, 3],
        "NotMovement": ["a", "b", "c"],
    })

    # Run function
    heatmaps.plot_handshape_movement_heatmap(df, folder=str(tmp_path))

    # No plot should be created
    assert calls == []


def test_handshape_movement_empty_crosstab(monkeypatch, tmp_path):
    # Capture calls to save_plot
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    # Columns exist but are empty → crosstab empty
    df = pd.DataFrame({
        "Handshape": [],
        "Movement": [],
    })

    heatmaps.plot_handshape_movement_heatmap(df, folder=str(tmp_path))

    # No figure should be saved
    assert calls == []


def test_handshape_movement_creates_plot(monkeypatch, tmp_path):
    # Track calls to save_plot
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    # Valid dataset
    df = pd.DataFrame({
        "Handshape": ["A", "B", "A"],
        "Movement": ["up", "down", "up"],
    })

    heatmaps.plot_handshape_movement_heatmap(df, folder=str(tmp_path))

    # Expect exactly one saved figure
    assert len(calls) == 1

    _, name, folder_used = calls[0]
    assert name == "heatmap_handshape_movement"
    assert str(folder_used) == str(tmp_path)


# ========= Numeric correlation tests ========= #

def test_numeric_corr_insufficient_columns(monkeypatch, tmp_path):
   # Only one numeric column → should skip
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    df = pd.DataFrame({"a": [1, 2, 3]})  # only one numeric col

    heatmaps.plot_numeric_correlation(df, folder=str(tmp_path))

     # No plot should be generated
    assert calls == []


def test_numeric_corr_empty_corr_matrix(monkeypatch, tmp_path):
    # Two numeric columns but no rows → empty correlation matrix
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    df = pd.DataFrame({"a": [], "b": []})  # 2 numeric cols but no rows

    heatmaps.plot_numeric_correlation(df, folder=str(tmp_path))

    # No figure expected
    assert calls == []


def test_numeric_corr_creates_plot(monkeypatch, tmp_path):
    # Valid numeric dataset
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": ["not numeric", "x", "y"]
    })

    heatmaps.plot_numeric_correlation(df, folder=str(tmp_path))

    # Expect exactly one plot generated
    assert len(calls) == 1
    
    _, name, folder_used = calls[0]
    assert name == "numeric_correlation"
    assert str(folder_used) == str(tmp_path)
