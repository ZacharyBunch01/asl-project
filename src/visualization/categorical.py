from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .utils import save_plot

def plot_categorical_distributions(
        df: pd.DataFrame,
        features: Iterable[str] | None = None,
        top_n: int | None = None,
        folder: str = "../Figures",
) -> None:
    """Plot count distributions for selected categorical features."""
    if features is None:
        features = ["Movement.2.0", "Location.2.0", "Handshape.2.0"]

    for feature in features:
        if feature not in df.columns:
            print(f"[categorical] Skipping {feature}: not in DataFrame.")
            continue

        vc = df[feature].value_counts(dropna=False)
        if top_n is not None:
            vc = vc.head(top_n)
        order = vc.index

        num_categories = len(order)
        height = max(6, min(0.35 * num_categories, 20))  # dynamic height

        fig, ax = plt.subplots(figsize=(11, height))
        sns.countplot(
            y=feature,
            data=df,
            order=order,
            ax=ax,
            color="steelblue",
        )

        ax.set_title(f"Distribution of {feature}", fontsize=14, pad=10)
        ax.set_xlabel("Count", fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        ax.xaxis.get_major_locator().set_params(integer=True)

        plt.tight_layout()
        save_plot(fig, f"distribution_{feature}", folder=folder)
        plt.close(fig)