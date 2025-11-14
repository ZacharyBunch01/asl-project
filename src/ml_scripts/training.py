"""
training.py

Train ML models on ASL data, one target at a time.
"""

import warnings
from pathlib import Path
from typing import List, Dict

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


from ml_scripts.processor import build_preprocessor

# hide unnecessary warnings 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# default targets to train
DEFAULT_TARGETS = ["Handshape", "Movement", "MajorLocation", "MinorLocation"]

# columns that are ID-ish or huge text that we don't want to one-hot encode
ID_LIKE_COLS = ["LemmaID", "SignBankEnglishTranslations", "SignBankLemmaID", "EntryID", "Item"]

def train_one_target(
    signData: pd.DataFrame,
    target_col: str,
    models_dir: Path,
) -> Dict:
    """
    Train a model for a single target column.

    Parameters
    ----------
    signData : pd.DataFrame
        Full ASL sign dataset.
    target_col : str
        Name of the column to predict.
    models_dir : Path
        Directory where the trained model will be saved.

    Returns
    -------
    dict
        Dictionary with status, metrics, and model path.
    """
    original_target = target_col

    if target_col not in signData.columns:
        msg = f"Target {target_col} not found in data, skipping."
        print(f"[SKIP] {msg}")
        return {"target": original_target, "status": "skip", "reason": msg}

    print("\n==============================")
    print(f"[INFO] Training for target: {target_col}")
    print("==============================")

    # Drop rows with no label / no data for this target
    data_trained = signData.dropna(subset=[target_col]).copy()
    if data_trained.empty:
        msg = f"No data for {target_col}"
        print(f"[SKIP] {msg}")
        return {"target": target_col, "status": "skip", "reason": msg}

    y = data_trained[target_col]
    X = data_trained.drop(columns=[target_col])

    # Drop ID / big text columns (if present)
    for col in ID_LIKE_COLS:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Build pipeline
    preprocessor = build_preprocessor(X)

    # Handle tiny classes (stratify only if all classes have >= 2 samples)
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

    # Model
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    print(f"[INFO] Fitting model for {target_col} on {len(X_train)} rows...")
    pipe.fit(X_train, y_train)
    print("[INFO] Done fitting.")

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    print(f"[INFO] {target_col:20s} | acc: {acc:.3f} | f1: {f1m:.3f}")

    # Save model
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"model_{target_col.replace('.', '_')}.pkl"
    joblib.dump(pipe, model_path)

    return {
        "target": target_col,
        "status": "ok",
        "acc": acc,
        "f1_macro": f1m,
        "path": str(model_path),
    }


def train_all_targets(
    signData: pd.DataFrame,
    targets: List[str] | None = None,
    models_dir: Path | None = None,
) -> List[Dict]:
    """
    Train models for a list of target columns on the given dataset.

    Parameters
    ----------
    signData : pd.DataFrame
        Full ASL sign dataset.
    targets : list[str] or None
        Target columns to train on. If None, uses DEFAULT_TARGETS.
    models_dir : Path or None
        Where to save models. Defaults to 'models/'.

    Returns
    -------
    list[dict]
        One dict of results per target.
    """
    if models_dir is None:
        models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    if targets is None:
        targets = DEFAULT_TARGETS

    results: List[Dict] = []
    for target_col in targets:
        res = train_one_target(signData, target_col, models_dir)
        results.append(res)

    return results
