import pandas as pd
import visualization.categorical as categorical


def test_categorical_skips_missing_columns(monkeypatch, tmp_path):
    """If none of the requested features exist, save_plot should not be called."""
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    # Patch save_plot in the categorical module
    monkeypatch.setattr(categorical, "save_plot", fake_save_plot)

    # DataFrame without the default categorical columns
    df = pd.DataFrame({
        "OtherCol": [1, 2, 3],
        "SomethingElse": ["a", "b", "c"],
    })

    categorical.plot_categorical_distributions(df, folder=str(tmp_path))

    # No feature matched => no plots saved
    assert calls == []


def test_categorical_uses_default_features(monkeypatch, tmp_path):
    """With default features present, save_plot should be called for each."""
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(categorical, "save_plot", fake_save_plot)

    # Include the default columns: Movement.2.0, Location.2.0, Handshape.2.0
    df = pd.DataFrame({
        "Movement.2.0": ["up", "down", "up"],
        "Location.2.0": ["head", "torso", "head"],
        "Handshape.2.0": ["5", "B", "5"],
    })

    categorical.plot_categorical_distributions(df, folder=str(tmp_path))

    # We expect 3 plots: one per default feature
    assert len(calls) == 3

    # Check that filenames are correct (order may follow features)
    names = {name for (_, name, _) in calls}
    assert "distribution_Movement.2.0" in names
    assert "distribution_Location.2.0" in names
    assert "distribution_Handshape.2.0" in names


def test_categorical_custom_features_and_top_n(monkeypatch, tmp_path):
    """
    With a custom feature list and top_n provided, save_plot should be
    called only for existing features, and at least once.
    (We mainly verify that it doesn't crash and calls save_plot correctly.)
    """
    calls = []

    def fake_save_plot(fig, name, folder="../Figures"):
        calls.append((fig, name, folder))

    monkeypatch.setattr(categorical, "save_plot", fake_save_plot)

    df = pd.DataFrame({
        "Movement.2.0": ["up", "down", "up", "left"],
        "Extra": ["x", "y", "z", "w"],
    })

    features = ["Movement.2.0", "NonExistentFeature"]
    categorical.plot_categorical_distributions(
        df,
        features=features,
        top_n=2,
        folder=str(tmp_path),
    )

    # Only Movement.2.0 exists, so exactly one plot saved
    assert len(calls) == 1
    _, name, folder_used = calls[0]
    assert name == "distribution_Movement.2.0"
    assert str(folder_used) == str(tmp_path)
