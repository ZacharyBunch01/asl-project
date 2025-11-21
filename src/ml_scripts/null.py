"""
null_baseline.py

Compute null baselines (majority + stratified) for ASL targets.
This mirrors the training flow but uses DummyClassifier instead of real ML.
"""

import warnings

import pandas as pd
from sklearn.dummy import DummyClassifier

from ml_scripts.config import DEFAULT_TARGETS
from ml_scripts.data_checker import validate_target_column
from ml_scripts.model_pipeline import train_test_split_data, evaluate_model

# Hide noisy warnings 
warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values*",
)


def null_one_target(signData, target_col):
    """
    Compute null baselines (majority + stratified) for a single target.

    Major: Always predicts the most common class
    Stratified random model: redicts classes at random uses
    the same class proportions as in the training data.
    """
    valid, msg = validate_target_column(signData, target_col)
    if not valid:
        print(f"[SKIP] {msg}")
        return {"target": target_col, "status": "skip", "reason": msg}

    print("\n" + "=" * 40)
    print(f"[NULL TARGET] {target_col}")
    print("=" * 40)

    # Drop rows with missing labels for this target
    data_trained = signData.dropna(subset=[target_col]).copy()
    if data_trained.empty:
        msg = f"No data for {target_col}"
        print(f"[SKIP] {msg}")
        return {"target": target_col, "status": "skip", "reason": msg}

    # Separate features and target
    y = data_trained[target_col]
    X = data_trained.drop(columns=[target_col])

    """
    #show class distribution
    # helps understand class imbalance
    
    class_counts = y.value_counts()
    print("[INFO] Class distribution:")
    print(class_counts.to_string())
    """
    # Use the SAME train/test split logic as our real model
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # --- Null model 1: Majority-class baseline ---
    maj_clf = DummyClassifier(strategy="most_frequent")
    maj_clf.fit(X_train, y_train)
    maj_metrics = evaluate_model(maj_clf, X_test, y_test)
    maj_acc = maj_metrics["accuracy"]
    maj_f1 = maj_metrics["f1_macro"]

    print(f"[BASELINE] Majority   | acc={maj_acc:.3f} | f1_macro={maj_f1:.3f}")

    # --- Null model 2: Stratified random baseline ---
    strat_clf = DummyClassifier(strategy="stratified", random_state=42)
    strat_clf.fit(X_train, y_train)
    strat_metrics = evaluate_model(strat_clf, X_test, y_test)
    strat_acc = strat_metrics["accuracy"]
    strat_f1 = strat_metrics["f1_macro"]

    print(f"[BASELINE] Stratified | acc={strat_acc:.3f} | f1_macro={strat_f1:.3f}")

    return {
        "target": target_col,
        "status": "ok",
        "acc_majority": maj_acc,
        "f1_majority": maj_f1,
        "acc_stratified": strat_acc,
        "f1_stratified": strat_f1,
    }


def null_all_targets(signData, targets=None):
    """
    Run null baselines for multiple targets, similar to train_all_targets.
    """
    if targets is None:
        targets = DEFAULT_TARGETS

    results = []
    for target_col in targets:
        res = null_one_target(signData, target_col)
        results.append(res)

    return results
