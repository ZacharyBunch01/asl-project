"""
model_pipeline.py

Model building, training, evaluation, and saving utilities.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_scripts.processor import build_preprocessor


def get_stratify_arg(y):
    """
    Return y if all classes have >= 2 samples, else None.
    Stratified: when the data split into train and test,
    each class keeps about the same proportion in both sets.



    """
    # Count how many samples each class has
    if y.value_counts().min() >= 2:
        # If every class appears at least twice, we are safe to stratify
        return y
    # skip stratifying and just warn the user
    print("[WARN] Not stratifying because at least one class has <2 samples")
    return None


def build_pipeline(X, class_weight=None):
    """
    Build preprocessing + model pipeline
    Preprocess the input features (X) using build_preprocessor
    Train a Random Forest classifier on the processed data 
    """
    
    # handles all data preprocessing steps
    preprocessor = build_preprocessor(X)
    
    # RandomForestClassifier = a model that builds many decision trees and combines their predictions
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=class_weight)
    # n_estimator: number of trees in the forest
    # random_state: seed for reproducibility
    # class_weight: handle imbalanced classes


    # Create the pipeline: first preprocess, then model
    return Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Shared stratified train/test split wrapper.
    Training set: sed to teach the model
    Test set: used to check how well the model learned on unseen data
    """
    stratify_arg = get_stratify_arg(y)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,  # straify:  keep class proportions similar when possible
    )


def fit_model(pipe, X_train, y_train):
    """
    fit the pipleline on the training data
    after pipeline has "learned" patterns from X_train and y_train
    """
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe, X_test, y_test):
    """
    evaluate how well the model is on the test set
    """
    # Use the trained pipeline to predict labels for the test data
    y_pred = pipe.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred), # Accuracy = (correct predictions) / (total predictions)
        "f1_macro": f1_score(y_test, y_pred, average="macro"), # Macro F1 = average F1 across all classes, treating each class equally
    }


def save_model(pipe, models_dir, target_col):
    """
    Save the trained models as a .pkl file
    """
    # Make sure the folder exists
    models_dir.mkdir(exist_ok=True)
    # Replace '.' in column names so the filename is safe
    safe_target = target_col.replace(".", "_")
    # Build full path
    model_path = models_dir / f"model_{safe_target}.pkl"
    # Save the pipeline using joblib
    joblib.dump(pipe, model_path)
    return model_path

