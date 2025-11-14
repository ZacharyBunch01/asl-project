"""
null.py

test if the ml is actually learning 
"""

import warnings
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils.sign_data import get_sign_data
from ml_scripts.training import DEFAULT_TARGETS, ID_LIKE_COLS

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def null_one_target(signData: pd.DataFrame, target_col: str) -> Dict:
    """
    Compute null baselines (majority + stratified) for a single target.
    Mirrors training.train_one_target, but without real ML.
    """
    original_target = target_col

    # --- same target handling as training.py ---
    if target_col not in signData.columns:
        if target_col == "Location.2.0" and "MajorLocation.2.0" in signData.columns:
            print("[WARN] Location.2.0 not found, using MajorLocation.2.0 instead.")
            target_col = "MajorLocation.2.0"
        else:
            msg = f"Target {target_col} not found in data, skipping."
            print(f"[SKIP] {msg}")
            return {"target": original_target, "status": "skip", "reason": msg}

    print("\n==============================")
    print(f"[INFO] NULL baselines for target: {target_col}")
    print("==============================")

    # Drop rows with missing labels for this target
    data_trained = signData.dropna(subset=[target_col]).copy()
    if data_trained.empty:
        msg = f"No data for {target_col}"
        print(f"[SKIP] {msg}")
        return {"target": target_col, "status": "skip", "reason": msg}

    y = data_trained[target_col]
    X = data_trained.drop(columns=[target_col])

    # Drop ID / big-text columns (optional but consistent)
    for col in ID_LIKE_COLS:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Stratify only if all classes have >= 2 samples (reusing your logic)
    min_class_size = y.value_counts().min()
    stratify_arg = y if min_class_size >= 2 else None
    if stratify_arg is None:
        print(f"[WARN] Not stratifying {target_col} because at least one class has <2 samples")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    # --- Null model 1: majority class ---
    maj_clf = DummyClassifier(strategy="most_frequent")
    maj_clf.fit(X_train, y_train)   # X is ignored, but needed for API
    y_pred_maj = maj_clf.predict(X_test)

    acc_maj = accuracy_score(y_test, y_pred_maj)
    f1_maj = f1_score(y_test, y_pred_maj, average="macro")

    print("\n[NULL] Majority-class baseline")
    print(f"  acc:     {acc_maj:.3f}")
    print(f"  f1_macro:{f1_maj:.3f}")

    # --- Null model 2: stratified random ---
    strat_clf = DummyClassifier(strategy="stratified", random_state=42)
    strat_clf.fit(X_train, y_train)
    y_pred_strat = strat_clf.predict(X_test)

    acc_strat = accuracy_score(y_test, y_pred_strat)
    f1_strat = f1_score(y_test, y_pred_strat, average="macro")

    print("\n[NULL] Stratified random baseline")
    print(f"  acc:     {acc_strat:.3f}")
    print(f"  f1_macro:{f1_strat:.3f}")

    print("\n[DEBUG] Stratified null – per-class report (optional)")
    print(classification_report(y_test, y_pred_strat))

    return {
        "target": target_col,
        "status": "ok",
        "acc_majority": acc_maj,
        "f1_majority": f1_maj,
        "acc_stratified": acc_strat,
        "f1_stratified": f1_strat,
    }


def null_all_targets(signData: pd.DataFrame, targets: List[str] | None = None) -> List[Dict]:
    """
    Run null baselines for multiple targets, mirroring train_all_targets.
    """
    if targets is None:
        targets = DEFAULT_TARGETS

    results: List[Dict] = []
    for target_col in targets:
        res = null_one_target(signData, target_col)
        results.append(res)
    return results


if __name__ == "__main__":
    df = get_sign_data()
    results = null_all_targets(df)
    print("\nSUMMARY:")
    for r in results:
        print(r)
