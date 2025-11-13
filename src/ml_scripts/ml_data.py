'''

ml_data.py

Purpose:
    train ML models on ASL data, one target at a time.


'''

# Import packages
import warnings
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import joblib

#load parsed data
import parseData
signData = parseData.signData
if signData is None:
    raise RuntimeError("Data not loaded. Make sure parseData.py can see ../Data/...")

print(f"[INFO] Data shape: {signData.shape}")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#visualize data
candidate_targets = [
    "Handshape.2.0",
    "Movement.2.0",
    "Location.2.0",
]


# helper to build preprocessors
def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    return preprocessor

# make folder/ access folder that will hold our models
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

#train model per target
for target_col in candidate_targets:
    # handle the “maybe it's called something else” case
    if target_col not in signData.columns:
        if target_col == "Location.2.0" and "MajorLocation.2.0" in signData.columns:
            print("[WARN] Location.2.0 not found, using MajorLocation.2.0 instead.")
            target_col = "MajorLocation.2.0"
        else:
            print(f"[SKIP] Target {target_col} not found in data, skipping.")
            continue

    print(f"\n==============================")
    print(f"[INFO] Training for target: {target_col}")
    print(f"==============================")

    #drop rows that do not have labels/ no data
    dataTrained = signData.dropna(subset=[target_col]).copy()
    if dataTrained.empty:
        print(f"[SKIP] No data for {target_col}")
        continue

    
    y = dataTrained[target_col]
    X = dataTrained.drop(columns=[target_col])

    # drop identifiers
    for col in ["LemmaID", "SignBankEnglishTranslations", "SignBankLemmaID", "EntryID", "Item"]:
        if col in X.columns:
            X = X.drop(columns=[col]) 

    # build preprocess
    preprocessor = build_preprocessor(X)

    # handle tiny classes
    min_class_size = y.value_counts().min()
    stratify_arg = y if min_class_size >= 2 else None
    if stratify_arg is None:
        print(f"[WARN] Not stratifying {target_col} because at least one class has <2 samples")

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg,)

    # model
    model = RandomForestClassifier(n_estimators=300, random_state=42)

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model),])

    print(f"[INFO] Fitting model for {target_col} on {len(X_train)} rows...")
    pipe.fit(X_train, y_train)
    print("[INFO] Done fitting.")

    #test the accuracry of the models
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    # Save the model and vectorizer for later use
    model_path = models_dir / f"model_{target_col.replace('.', '_')}.pkl"
    joblib.dump(pipe, model_path)
    print(f"[INFO] {target_col:20s} | acc: {acc:.3f} | f1: {f1m:.3f}")

