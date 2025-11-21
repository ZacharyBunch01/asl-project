# tests/ml_test/test_ml_pipeline.py
#
# This file contains tests machine learning training code.
# The goal of these tests is to:
#   - make sure training runs normally with a small fake dataset
#   - ensure class weights are applied correctly
#   - test that null baseline metrics are created properly


import os
from pathlib import Path

import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer

from ml_scripts import model_pipeline
from ml_scripts.training import train_one_target
from ml_scripts.null import null_one_target


# ---------------------------------------------------------------
# FIXTURE 1 — tiny fake dataset
# ---------------------------------------------------------------
@pytest.fixture
def dummy_sign_data():
    """
    Create a tiny fake ASL dataset for testing.

    A "fixture" in pytest is a function that provides reusable data.
     create a tiny DataFrame with:
      - feat1, feat2: pretend features
      - Handshape: simple labels to predict
    """
    df = pd.DataFrame({
        "feat1": [1, 2, 3, 4, 5, 6],
        "feat2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "Handshape": ["a", "b", "a", "b", "a", "b"],
    })
    return df


# ---------------------------------------------------------------
# Helper: identity transform so pipeline works without preprocessing
# ---------------------------------------------------------------
def identity_transform(x):
    """
    Simple function that returns its input unchanged.

    FunctionTransformer to act as a "dummy preprocessor".
    It must be defined at the top level so joblib can pickle it.
    """
    return x


# ---------------------------------------------------------------
# FIXTURE 2 — monkeypatch the real build_preprocessor
# ---------------------------------------------------------------
@pytest.fixture(autouse=True)
def monkeypatch_preprocessor(monkeypatch):
    """
    Automatically replace the real build_preprocessor function
    with a simple one that does nothing.

    Why monkeypatch?
    ----------------
    real preprocessing pipeline may include:
      - one-hot encoding
      - imputers
      - complex transformations

    replace the real preprocessor
    with a simple transformer that just returns the data unchanged.

    FunctionTransformer(identity_transform) acts as our “fake preprocessor”.
    """

    def _dummy_build_preprocessor(X):
        # Wrap the identity_transform so sklearn thinks it's a real transformer
        return FunctionTransformer(identity_transform)

    # Replace the build_preprocessor function inside ml_scripts.model_pipeline
    monkeypatch.setattr(
        model_pipeline,
        "build_preprocessor",
        _dummy_build_preprocessor,
    )


# ---------------------------------------------------------------
# TEST 1 — Does train_one_target work end-to-end?
# ---------------------------------------------------------------
def test_train_one_target_creates_model_file(tmp_path, dummy_sign_data):
    """
    Test that train_one_target:
      - runs without crashing
      - returns correct result fields
      - saves a model file to disk
    """

    # Create a temporary folder (pytest cleans it up afterwards)
    models_dir = tmp_path / "models"

    # Run the training function using fake dataset
    result = train_one_target(dummy_sign_data, "Handshape", models_dir)

    # Basic result checks
    assert result["status"] == "ok"
    assert result["target"] == "Handshape"
    assert 0.0 <= result["acc"] <= 1.0
    assert 0.0 <= result["f1_macro"] <= 1.0

    # Confirm the model file was written to disk
    model_path = Path(result["path"])
    assert model_path.exists()
    assert model_path.is_file()


# ---------------------------------------------------------------
# TEST 2 — Does the RandomForest contain class weights?
# ---------------------------------------------------------------
def test_trained_model_has_class_weights(tmp_path, dummy_sign_data):
    """
    Test that the RandomForest inside the trained pipeline
    actually receives and stores class_weight.

    ensures training code is passing class_weight dict
    correctly into the model.
    """
    import joblib

    # Temporary folder to save test models
    models_dir = tmp_path / "models"

    # Train model
    result = train_one_target(dummy_sign_data, "Handshape", models_dir)

    # Load trained pipeline
    model_path = Path(result["path"])
    pipe = joblib.load(model_path)

    # Inside the pipeline, the model is stored under step name "model"
    rf = pipe.named_steps["model"]

    # Check that class_weight is present and correct
    assert rf.class_weight is not None
    assert isinstance(rf.class_weight, dict)
    assert set(rf.class_weight.keys()) == {"a", "b"}


# ---------------------------------------------------------------
# TEST 3 — Null should run and return valid metrics
# ---------------------------------------------------------------
def test_null_runs_and_returns_metrics(dummy_sign_data):
    """
    Test that the null baseline:
      - runs successfully
      - returns metrics for both majority and stratified classifiers
      - each returned metric is a number between 0 and 1
    """

    result = null_one_target(dummy_sign_data, "Handshape")

    # Basic checks
    assert result["status"] == "ok"
    assert result["target"] == "Handshape"

    # Verify metrics look valid
    for key in ["acc_majority", "f1_majority", "acc_stratified", "f1_stratified"]:
        assert key in result               # Key exists
        assert 0.0 <= result[key] <= 1.0   # Metric is valid
