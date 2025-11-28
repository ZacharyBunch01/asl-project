import pytest
import matplotlib.pyplot as plt
import visualization.config as config


def test_configure_plots_applies_settings(monkeypatch):
    """
    Test that configure_plots() correctly updates matplotlib rcParams.
    """

    # Reset the module's config_applied flag
    config.config_applied = False

    # Call the function
    config.configure_plots(style="whitegrid", dpi=123)

    # Verify rcParams were set
    assert plt.rcParams["figure.dpi"] == 123
    assert plt.rcParams["savefig.dpi"] == 123
    assert plt.rcParams["axes.titlesize"] == 14
    assert plt.rcParams["axes.labelsize"] == 12
    assert plt.rcParams["xtick.labelsize"] == 10
    assert plt.rcParams["ytick.labelsize"] == 10
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["grid.alpha"] == 0.4
    assert plt.rcParams["figure.facecolor"] == "white"


def test_configure_plots_runs_only_once(monkeypatch):
    """
    Test that calling configure_plots() twice does NOT reapply settings.
    """

    # Reset state
    config.config_applied = False

    # First call — should apply settings
    config.configure_plots(style="whitegrid", dpi=150)
    first_dpi = plt.rcParams["figure.dpi"]

    # Modify rcParams manually to detect if second call resets it
    plt.rcParams["figure.dpi"] = 999

    # Second call — should NOT override rcParams
    config.configure_plots(style="whitegrid", dpi=50)

    # Because config_applied == True, function should NOT overwrite our manual value
    assert plt.rcParams["figure.dpi"] == 999
