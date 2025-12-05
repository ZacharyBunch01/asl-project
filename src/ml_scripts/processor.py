"""
processor.py

Build sklearn preprocessors for numeric + categorical columns.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes + encodes numeric and categorical features.
    This function looks at the dataset and automatically creates the correct
    preprocessing steps for each type of column.
    """

    # Identify which columns are numeric (integers or floats)
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Identify which columns are categorical (strings or labeled categories)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # -------------------------
    # Numeric preprocessing
    # -------------------------
    # This pipeline defines what to do with numeric columns:
    # 1. Fill missing values with the *mean* of the column.
    # 2. Scale all numeric values so they are centered and have similar ranges.
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),   # Replace missing numbers
        ("scale", StandardScaler()),                 # Normalize numeric values
    ])

    # -------------------------
    # Categorical preprocessing
    # -------------------------
    # This pipeline defines what to do with categorical columns:
    # 1. Fill missing values with the most common value in the column.
    # 2. Convert categories (text labels) into binary columns using One-Hot Encoding.
    #    handle_unknown="ignore" prevents errors if the model later sees new categories.
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),   # Fill missing categories
        ("onehot", OneHotEncoder(handle_unknown="ignore")),     # Turn categories into numbers
    ])

    # -------------------------
    # Combine both pipelines
    # -------------------------
    # ColumnTransformer applies different preprocessing pipelines to different
    # sets of columns (numeric → numeric_pipe, categorical → categorical_pipe).
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),  # Apply numeric steps to numeric columns
            ("cat", categorical_pipe, cat_cols),  # Apply categorical steps to categorical columns
        ]
    )

    # Return the full preprocessing object so it can be used in a model pipeline
    return preprocessor
