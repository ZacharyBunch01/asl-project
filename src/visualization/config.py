"""
config.py

Global plotting configuration.
Use this module to define visual defaults for all analysis plots
using matplotlib and seaborn.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

# Private flag to make sure config only runs once
config_applied = False

"""
Sets up the global style for all plots

style: The seaborn theme to apply
dpi: The dots-per-inch resolution
"""
def configure_plots(style: str = "whitegrid", dpi: int = 150) -> None:
    global config_applied

    # Don't re run if it's already done
    if config_applied:
        return
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "figure.facecolor": "white"
    })

    config_applied = True
    print(f"Plot styling initialized! (Theme: '{style})")