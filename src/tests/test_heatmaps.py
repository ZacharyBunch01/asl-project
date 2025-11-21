import pandas as pd
import visualization.heatmaps as heatmaps
from pathlib import Path


def test_handshape_movement_missing_columns(monkeypatch, tmp_path):
    """
    If required columns are missing, no plot should be saved.
    """
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    df = pd.DataFrame({
        "Other": [1, 2, 3],
        "NotMovement": ["a", "b", "c"],
    })

    heatmaps.plot_handshape_movement_heatmap(df, folder=str(tmp_path))

    assert calls == []


def test_handshape_movement_empty_crosstab(monkeypatch, tmp_path):
    """
    If the crosstab is empty (e.g., columns exist but no overlap), skip plotting.
    """
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    df = pd.DataFrame({
        "Handshape": [],
        "Movement": [],
    })

    heatmaps.plot_handshape_movement_heatmap(df, folder=str(tmp_path))

    assert calls == []


def test_handshape_movement_creates_plot(monkeypatch, tmp_path):
    """
    When valid handshape and movement data exist, exactly one plot should be saved.
    """
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    df = pd.DataFrame({
        "Handshape": ["A", "B", "A"],
        "Movement": ["up", "down", "up"],
    })

    heatmaps.plot_handshape_movement_heatmap(df, folder=str(tmp_path))

    assert len(calls) == 1
    _, name, folder_used = calls[0]
    assert name == "heatmap_handshape_movement"
    assert str(folder_used) == str(tmp_path)


# ========= Numeric correlation tests ========= #

def test_numeric_corr_insufficient_columns(monkeypatch, tmp_path):
    """
    If fewer than 2 numeric cols exist, skip plotting.
    """
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    df = pd.DataFrame({"a": [1, 2, 3]})  # only one numeric col

    heatmaps.plot_numeric_correlation(df, folder=str(tmp_path))

    assert calls == []


def test_numeric_corr_empty_corr_matrix(monkeypatch, tmp_path):
    """
    If the resulting correlation matrix is empty, skip plotting.
    """
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(heatmaps, "save_plot", fake_save_plot)

    df = pd.DataFrame({"a": [], "b": []})  # 2 numeric cols but no rows

    heatmaps.plot_numeric_correlation(df, folder=str(tmp_path))

    assert calls == []


def test_numeric_corr_creates_plot(monkeypatch, tmp_path):
    """
    If >=2 numeric columns exist, save_plot should be called once.
    """
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

    assert len(calls) == 1
    _, name, folder_used = calls[0]
    assert name == "numeric_correlation"
    assert str(folder_used) == str(tmp_path)
