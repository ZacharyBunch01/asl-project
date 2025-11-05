"""
visualize_data.py

Load the parsed ASL sign data from parseData.py and produce exploratory visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

import parseData

# Helper: save plot to Figures/, creating it if needed
def save_plot(fig, name, folder="Figures"):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")

# Load parsed data
signData = parseData.signData
if signData is None:
    raise RuntimeError("Sign data not available, run parseData.py first or check dataset paths.")

print(f"Data loaded: {signData.shape[0]} rows, {signData.shape[1]} columns.")

# 1) Distributions for main categorical features
cat_features = ["Movement.2.0", "Location.2.0", "Handshape.2.0"]
for feature in cat_features:
    if feature in signData.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=feature, data=signData,
                      order=signData[feature].value_counts(dropna=False).index, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel("Count")
        ax.set_ylabel(feature)
        plt.tight_layout()
        save_plot(fig, f"distribution_{feature}")
        plt.close(fig)

# 2) Handshape x Movement heatmap
if all(f in signData.columns for f in ["Handshape.2.0", "Movement.2.0"]):
    crosstab = pd.crosstab(signData["Handshape.2.0"], signData["Movement.2.0"])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(crosstab, cmap="Blues", cbar=True, ax=ax)
    ax.set_title("Handshape vs. Movement Frequency")
    plt.tight_layout()
    save_plot(fig, "heatmap_handshape_movement")
    plt.close(fig)

# 3) Missing value overview
missing = signData.isnull().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=missing.head(15), y=missing.head(15).index, color="royalblue", ax=ax)
ax.set_title("Top 15 Columns by Missing Value Percentage")
ax.set_xlabel("Proportion Missing")
ax.set_ylabel("Feature")
plt.tight_layout()
save_plot(fig, "missing_values")
plt.close(fig)

# 4) Numeric correlation heatmap
num_cols = signData.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 2:
    corr = signData[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    save_plot(fig, "numeric_correlation")
    plt.close(fig)

# 5) Summary stats CSV
out_dir = Path("Figures")
out_dir.mkdir(parents=True, exist_ok=True)
summary_path = out_dir / "signData_summary.csv"
signData.describe(include="all").transpose().to_csv(summary_path)
print("Saved data summary to", summary_path.resolve())
print("All figures saved under:", out_dir.resolve())
