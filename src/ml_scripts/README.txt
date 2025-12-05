# `ml_scripts/` – ASL Machine Learning Pipeline


Dependecies:
python3 -m pip install pandas scikit-learn joblib

This folder contains the core machine learning code for the ASL project.

The goal of this code is to:

- Train **tabular ML models** to predict ASL features  
  - `Handshape`
  - `Movement`
  - `MajorLocation`
  - `MinorLocation`
- Evaluate those models against **null baselines**
- Save trained models to disk for later use (e.g. in a runner script or notebook)

The models are currently based on **RandomForestClassifier** from scikit-learn, with **class weights** to handle label imbalance.

---


### 'data_checker.py'

Make sure that we are using the correct data and signed

### `config.py`

Central configuration and constants used by the rest of the ML code.

Typical contents:

- `DEFAULT_TARGETS`  
  List of column names to train models on, e.g.:
  ```python
  DEFAULT_TARGETS = ["Handshape", "Movement", "MajorLocation", "MinorLocation"]

### 'model_pipeline.py'

stores most of the main functions that build the pipeline that are used in training.py to actually build the ML models using the pipeline
build_preprocessor(X)

Builds a preprocessing step for your features X

	Often handles missing values, categorical encoding, etc.

	build_pipeline(X, class_weight=None)

Creates a full scikit-learn Pipeline that:

	first runs the preprocessor

	then uses a RandomForestClassifier

	Accepts a class_weight dict so the model pays more attention to rare labels.

train_test_split_data(X, y)

	Wraps train_test_split to create training and test sets with a consistent random seed.

fit_model(pipe, X_train, y_train)

	Fits the pipeline on training data.

evaluate_model(pipe, X_test, y_test)

	Computes metrics like:

	accuracy

	macro F1

	Returns them in a simple dict.

save_model(pipe, models_dir, target_col)

	Saves a trained pipeline as a .pkl file in models_dir.

	File name typically looks like model_<Target>.pkl.

### 'training.py'

Puts all the funtions made in model_pipeline to build one ml on a target and then has a seperate function that goes through the rest of the targets and shows the overall results from each target