import pytest
import matplotlib.pyplot as plt
import visualization.config as config


def test_configure_plots_applies_settings(monkeypatch):
   
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
    # Reset the state so the first call applies configuration
    config.config_applied = False

    # First call should apply settings
    config.configure_plots(style="whitegrid", dpi=150)
    first_dpi = plt.rcParams["figure.dpi"]

    # Manually override a value to detect if a second call changes it
    plt.rcParams["figure.dpi"] = 999

    # Second call should be a no-op because config_applied is already True
    config.configure_plots(style="whitegrid", dpi=50)

    # The value we set manually should remain unchanged
    assert plt.rcParams["figure.dpi"] == 999
