"""
trainer.py

High-level training orchestration for ASL models.
"""

from pathlib import Path
import pandas as pd
import warnings
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from ml_scripts.config import DEFAULT_TARGETS, DEFAULT_MODELS_DIR
from ml_scripts.data_checker import validate_target_column
from ml_scripts.model_pipeline import (
    build_pipeline,
    train_test_split_data,
    fit_model,
    evaluate_model,
    save_model,
)

# Hide certain warnings so the output is easier to read.
warnings.filterwarnings("ignore", message="Skipping features without any observed values*",)


def train_one_target(signData, target_col, models_dir):
    """
    Train a model for a single target column.
    

    """
    # make sure that there is a model dir
    if isinstance(models_dir, str):
        models_dir = Path(models_dir)

    # make sure that the targets we are training are actually in the Dataset
    valid, msg = validate_target_column(signData, target_col)
    if not valid:
        print(f"[SKIP] {msg}")
        return {"target": target_col, "status": "skip", "reason": msg}

    print("\n==============================")
    print(f"[INFO] Training for target: {target_col}")
    print("==============================")

    # Remove rows that have no label for this target to ensure we are only training on the target we want
    data_trained = signData.dropna(subset=[target_col]).copy()
    # if the target we are training on is empty skip ove it
    if data_trained.empty:
        msg = f"No data for {target_col}"
        print(f"[SKIP] {msg}")
        return {"target": target_col, "status": "skip", "reason": msg}

    # y the data the model tries to learn
    # X is all other info that helps predict y
    y = data_trained[target_col]
    X = data_trained.drop(columns=[target_col])

    #Compute class weights using sklearn
    #
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
        )
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}


    print("[WEIGHTS] Using class weights:")
    # Commented out to reduce noise, but can be turned on if needed shows how the data has been weighted for each target
    """
    for k, v in class_weight_dict.items():
        print(f"   {k}: {v:.2f}")
    """


    # Train/test split
    # split data: Trains on 80% of the data and 20% will be used for testing
    # train set: data the model learns from
    # test set: data used to check how well the model learned
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Build and fit pipeline with weights included
    pipe = build_pipeline(X, class_weight=class_weight_dict)
    print(f"[INFO] Fitting model for {target_col} on {len(X_train)} rows...")
    pipe = fit_model(pipe, X_train, y_train)
    print("[INFO] Done fitting.")

    # Evaluate
    metrics = evaluate_model(pipe, X_test, y_test)
    # accuracy: how many predictions are correct
    acc = metrics["accuracy"]
    # F1_score: how well the model handles all classes fairly
    f1m = metrics["f1_macro"]
    print(f"[INFO] {target_col:20s} | acc: {acc:.3f} | f1: {f1m:.3f}")

    # Save model to a file
    model_path = save_model(pipe, models_dir, target_col)

    return {
        "target": target_col,
        "status": "ok",
        "acc": acc,
        "f1_macro": f1m,
        "path": str(model_path),
    }


def train_all_targets(signData, targets=None, models_dir=None):
    """
    Train models for a list of target columns on the given dataset.

    Steps:
    - Use default targets if none are given
    - Make sure the models folder exists
    - Train each target one-by-one
    - Collect results and return them
    """

    # Use default folder if none is provided
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
    elif isinstance(models_dir, str):
        models_dir = Path(models_dir)

    # Make folder if missing
    models_dir.mkdir(exist_ok=True)

     # Use default target list if none given
    if targets is None:
        targets = DEFAULT_TARGETS

    results = []
    # Train each target one at a time
    for target_col in targets:
        res = train_one_target(signData, target_col, models_dir)
        results.append(res)

    return results


