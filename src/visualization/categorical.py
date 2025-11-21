
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .export import save_plot

def plot_categorical_distributions(df, features=None, top_n=None, folder="../Figures"):
    
    # Set default features if none are provided
    if features is None:
        features = ["Movement", "MajorLocation", "MinorLocation", "Handshape"]

    for feature in features:
        # check if the feature exists before processing
        if feature not in df.columns:
            print(f"Skipping '{feature}': not in DataFrame.")
            continue

        # Data prep
        value_counts = df[feature].value_counts(dropna=False)
        if top_n is not None:
            value_counts = value_counts.head(top_n)

        category_order =value_counts.index
        num_categories = len(category_order)
        
        #Plot config
        BASE_HEIGHT = 6
        HEIGHT_PER_CATEGORY = 0.35
        MAX_HEIGHT = 20
        height = min(MAX_HEIGHT, max(BASE_HEIGHT, HEIGHT_PER_CATEGORY* num_categories))

        # Create Plot
        fig, ax = plt.subplots(figsize=(11, height))
        sns.countplot(
            y=feature,
            data=df,
            order=category_order,
            ax=ax,
            color="steelblue",
        )

         # Clean labels for readability
        cleaned_labels = [
            lbl.get_text().replace("_", " ")[:18] + "…" 
            if len(lbl.get_text()) > 18 
            else lbl.get_text().replace("_", " ")
            for lbl in ax.get_yticklabels()
        ]
        
        # Customization
        ax.set_title(f"Distribution of **{feature}**", fontsize=14, pad=10)
        ax.set_xlabel("Count", fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        # Ensure x-axis ticks are integers for count data
        ax.xaxis.get_major_locator().set_params(integer=True)

        # Save and Cleanup
        plt.tight_layout()
        save_plot(fig, f"distribution_{feature}", folder=folder)
        plt.close(fig)