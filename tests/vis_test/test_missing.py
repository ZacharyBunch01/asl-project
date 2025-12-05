import pandas as pd
import visualization.missing as missing


def test_missing_empty_df(monkeypatch, tmp_path):
    # Tracks calls to save_plot
    calls = []

    # Fake replacement for save_plot
    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(missing, "save_plot", fake_save_plot)

    # Empty DataFrame → no missing values plot expected
    df = pd.DataFrame()
    missing.plot_missing_values(df, folder=tmp_path)

     # Should not save any plot
    assert calls == []


def test_missing_no_missing_values(monkeypatch, tmp_path):
    # Tracks calls to save_plot
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(missing, "save_plot", fake_save_plot)

    # No missing values → nothing to plot
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    missing.plot_missing_values(df, folder=tmp_path)

    # No plot should be saved
    assert calls == []


def test_missing_creates_plot(monkeypatch, tmp_path):
    # Tracks calls to save_plot
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(missing, "save_plot", fake_save_plot)

    # DataFrame with missing values → should generate one plot
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [None, None, 1],
        "c": [1, 2, 3]
    })

    missing.plot_missing_values(df, top_n=2, folder=tmp_path)

    # Exactly one plot expected
    assert len(calls) == 1

    # Validate output naming and folder
    fig, name, folder_used = calls[0]
    assert name == "missing_values"
    assert str(folder_used) == str(tmp_path)
