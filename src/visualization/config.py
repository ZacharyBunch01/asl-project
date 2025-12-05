
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

# Flag to ensure plot configuration is only applied once per session
config_applied = False

def configure_plots(style: str = "whitegrid", dpi: int = 150) -> None:
    global config_applied

    # Prevent re-applying the configuration if it’s already been set
    if config_applied:
        return
    
    # Apply global matplotlib settings for consistent plot appearance
    plt.rcParams.update({
        "figure.dpi": dpi, # resolution for interactive figures
        "savefig.dpi": dpi, # resolution for saved images
        "axes.titlesize": 14, # title text size
        "axes.labelsize": 12, # axis label size
        "xtick.labelsize": 10, # x-axis tick label size
        "ytick.labelsize": 10, # y-axis tick label size
        "axes.grid": True, # enable gridlines
        "grid.alpha": 0.4, # grid transparency
        "figure.facecolor": "white" # background color of figures
    })
    # Apply seaborn styling theme
    sns.set_theme(style=style)

    # Mark configuration as done
    config_applied = True
    print(f"Plot styling initialized! (Theme: '{style})")