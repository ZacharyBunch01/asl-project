import pandas as pd
import builtins
import visualization
import visualize_sign_data


def test_runner_calls_all_steps(monkeypatch):
    # Fake DataFrame to be returned by get_sign_data
    fake_df = pd.DataFrame({
        "Movement": ["A", "B"],
        "Handshape": ["X", "Y"]
    })

    # Replace get_sign_data so the runner doesn't depend on real data
    monkeypatch.setattr(
        "visualize_sign_data.get_sign_data",
        lambda: fake_df
    )

    # Counters to check each visualization function is called exactly once
    calls = {
        "categorical": 0,
        "heatmap": 0,
        "missing": 0,
        "correlation": 0,
        "summary": 0,
    }

    # Replace each visualization function with a lambda that bumps the counter
    monkeypatch.setattr(
        "visualize_sign_data.plot_categorical_distributions",
        lambda *args, **kwargs: calls.__setitem__("categorical", calls["categorical"] + 1)
    )
    monkeypatch.setattr(
        "visualize_sign_data.plot_handshape_movement_heatmap",
        lambda *args, **kwargs: calls.__setitem__("heatmap", calls["heatmap"] + 1)
    )
    monkeypatch.setattr(
        "visualize_sign_data.plot_missing_values",
        lambda *args, **kwargs: calls.__setitem__("missing", calls["missing"] + 1)
    )
    monkeypatch.setattr(
        "visualize_sign_data.plot_numeric_correlation",
        lambda *args, **kwargs: calls.__setitem__("correlation", calls["correlation"] + 1)
    )
    monkeypatch.setattr(
        "visualize_sign_data.save_summary_stats",
        lambda *args, **kwargs: calls.__setitem__("summary", calls["summary"] + 1)
    )

    # Avoid global style side effects during tests
    monkeypatch.setattr(
        "visualize_sign_data.configure_plots",
        lambda: None
    )

    # Run the visualization runner
    visualize_sign_data.main()

   # Verify each visualization step was executed exactly once
    assert calls["categorical"] == 1
    assert calls["heatmap"] == 1
    assert calls["missing"] == 1
    assert calls["correlation"] == 1
    assert calls["summary"] == 1


def test_runner_errors_on_empty_data(monkeypatch):
    # get_sign_data returns an empty DataFrame → should trigger RuntimeError

    monkeypatch.setattr(
        "visualize_sign_data.get_sign_data",
        lambda: pd.DataFrame()  # empty dataset triggers error
    )

    # Avoid styling side effects again
    monkeypatch.setattr(
        "visualize_sign_data.configure_plots",
        lambda: None
    )

    # main() should raise RuntimeError if dataset is empty
    import pytest
    with pytest.raises(RuntimeError):
        visualize_sign_data.main()
