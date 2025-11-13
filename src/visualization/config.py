import matplotlib.pyplot as plt
import seaborn as sns

def configure_plots():
    """Set global plotting style"""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9
    })