# Credit Risk ML Pipeline — Detailed Project Guide

## 1) What this project is about

This project is an end-to-end machine learning system for **credit card default prediction**.  
It trains classification models to estimate whether a customer is likely to default on their next month’s credit card payment, then serves that model through a Streamlit web app for interactive inference.

At a high level, it covers:

- Data ingestion from MongoDB
- Data cleaning and feature engineering
- Feature preprocessing + train/test split + class balancing
- Multi-model training with hyperparameter tuning
- Best-model selection and artifact persistence
- Interactive prediction UI with top feature importance display

---

## 2) Problem it solves

Financial institutions need to estimate repayment risk to reduce losses and make better credit decisions.

This project addresses that by:

- Predicting default probability from demographic, repayment, billing, and payment history variables
- Providing a reusable training pipeline for retraining and model refresh
- Offering a UI for quick what-if scoring without writing code
- Surfacing top features driving a prediction to support interpretability

Business value:

- Better risk segmentation
- Early intervention for likely defaulters
- Better credit limit and policy decisions

---

## 3) Basic workflow (end-to-end)

### A. Data bootstrap (optional one-time step)

1. `upload_data.py` reads `notebooks/datasets/UCI_Credit_Card.csv`
2. Converts rows to JSON records
3. Inserts into MongoDB collection

### B. Training pipeline (`src/pipeline/train_pipeline.py`)

1. **Data ingestion** (`DataIngestion`)  
   Pulls records from MongoDB and writes `artifacts/credit_data.csv`
2. **Data transformation** (`DataTransformation`)
   - Loads CSV
   - Renames target column to internal name
   - Cleans known value issues (`MARRIAGE`, `EDUCATION`)
   - Feature engineering (age groups, averaged behavior features, utilization ratio)
   - Splits train/test
   - Preprocesses with `ColumnTransformer`
   - Applies SMOTE on training split
   - Saves:
     - `artifacts/preprocessor.pkl`
     - `artifacts/train.csv`
     - `artifacts/test.csv`
3. **Model training** (`ModelTrainer`)
   - Trains 4 candidate models with `GridSearchCV` using `config/model.yaml`
   - Evaluates via ROC-AUC on test split
   - Selects best model
   - Validates score threshold (expected ROC-AUC >= 0.7)
   - Saves `artifacts/model.pkl`

### C. Inference pipeline (`src/pipeline/predict_pipeline.py`)

1. Receives one-row feature input from UI
2. Applies same cleaning + feature engineering logic
3. Loads `preprocessor.pkl` and transforms input
4. Loads `model.pkl` and computes `predict_proba`
5. Returns probabilities + top 5 important features

### D. UI layer (`streamlit_app.py`)

1. Collects feature input in sidebar
2. Calls `PredictionPipeline`
3. Displays:
   - Default vs no-default risk message
   - Predicted probability
   - Bar chart of top 5 influential features

---

## 4) Project architecture map

The code follows a modular pattern:

- `components/` = reusable ML stages
- `pipeline/` = orchestrators (training & prediction)
- `utils/` = shared helpers (I/O, YAML, serialization)
- `constant/` = central constants
- `logger.py` and `exception.py` = cross-cutting concerns
- `streamlit_app.py` = presentation layer

---

## 5) File-by-file responsibilities

## Root files

### `README.md`

Primary project documentation: purpose, setup, usage, and high-level structure.

### `requirements.txt`

Pinned Python dependencies needed to run training and app.

### `setup.py`

Package metadata + dependency parser (`get_requirements`) so project can be installed as a package (`-e .`).

### `streamlit_app.py`

Interactive Streamlit UI for manual inference:

- Defines input widgets for all model features
- Creates input DataFrame
- Calls prediction pipeline
- Renders risk outcome and feature-importance chart

### `upload_data.py`

Dataset bootstrap script:

- Reads local CSV from notebooks dataset folder
- Connects to MongoDB
- Inserts records into configured DB/collection

### `3-model_training_and_evaluation.ipynb`

Notebook version of model training/evaluation experimentation (root copy).

### `info.md`

This detailed project guide.

---

## `config/`

### `config/model.yaml`

Hyperparameter search spaces for each candidate model used by `GridSearchCV` in `ModelTrainer`.

---

## `src/`

### `src/__init__.py`

Marks `src` as a Python package.

### `src/logger.py`

Configures Python logging output path and format.

- Creates timestamp-based log directory/file under `logs/`
- Sets global logging format and level

### `src/exception.py`

Defines custom exception wrapper (`CustomException`) that enriches errors with script name and line number.

---

## `src/constant/`

### `src/constant/__init__.py`

Central constants:

- Mongo DB names and URL
- Target column internal name
- Artifact folder name
- Model filename metadata

---

## `src/utils/`

### `src/utils/__init__.py`

Package marker.

### `src/utils/main_utils.py`

General helper utilities:

- Read YAML config files
- Save Python objects (pickle)
- Load Python objects (pickle)
- (Includes `read_schema_config_file` helper, though `schema.yaml` is not present in current tree)

---

## `src/components/`

### `src/components/__init__.py`

Package marker.

### `src/components/data_ingestion.py`

Data ingestion component:

- Connects to MongoDB
- Exports collection to pandas DataFrame
- Drops Mongo `_id`
- Writes feature-store CSV to `artifacts/credit_data.csv`

### `src/components/data_transformation.py`

Data transformation and preprocessing component:

- Loads feature-store CSV
- Renames target to `default_payment_next_month`
- Corrects known categorical anomalies
- Performs feature engineering:
  - `Age_Groups`
  - `Avg_Bill_Amt`
  - `Avg_Pay_Amt`
  - `Avg_Delay_Score`
  - `Average_Credit_Utilization_Ratio`
- Splits train/test
- Builds preprocessing pipelines:
  - Numerical: impute + scale
  - Nominal: impute + one-hot
  - Ordinal: impute + ordinal encode
- Applies SMOTE to training split
- Saves transformed datasets and preprocessor artifact

### `src/components/model_trainer.py`

Model training/evaluation component:

- Defines candidate models:
  - XGBoost
  - Gradient Boosting
  - KNN
  - Random Forest
- Runs `GridSearchCV` based on `config/model.yaml`
- Compares by test ROC-AUC
- Fits best model and computes ROC-AUC/F1/Precision/Recall
- Enforces minimum ROC-AUC threshold
- Saves best model to `artifacts/model.pkl`

---

## `src/pipeline/`

### `src/pipeline/__init__.py`

Package marker.

### `src/pipeline/train_pipeline.py`

Orchestration layer for training:

- Calls data ingestion
- Calls data transformation
- Calls model training
- Prints final ROC-AUC

### `src/pipeline/predict_pipeline.py`

Orchestration layer for inference:

- Applies preprocessing path used in training
- Loads saved preprocessor + model from `artifacts`
- Returns prediction probabilities
- Computes top 5 feature importances for UI display

---

## `artifacts/`

Generated runtime/training artifacts (some may already exist):

- `artifacts/credit_data.csv` — exported raw feature store from MongoDB
- `artifacts/train.csv` — transformed training dataset
- `artifacts/test.csv` — transformed test dataset
- `artifacts/preprocessor.pkl` — fitted preprocessing pipeline
- `artifacts/model.pkl` — selected trained model

---

## `notebooks/`

Exploration and offline analysis area:

- `notebooks/1-exploratory_data_analysis-EDA.ipynb` — EDA and understanding distributions
- `notebooks/2-data_preprocessing.ipynb` — preprocessing experiments
- `notebooks/3-model_training_and_evaluation.ipynb` — model comparison experiments
- `notebooks/datasets/UCI_Credit_Card.csv` — source dataset used for local exploration/bootstrap
- `notebooks/csv_outputs/` — performance summary CSVs
- `notebooks/feature_importance_outputs/` — importance export artifacts
- `notebooks/test_performance_outputs/` — test visual outputs
- `notebooks/validation_performance_outputs/` — validation visual outputs
- `notebooks/visualization_outputs/` — saved charts by category/relationship

---

## 6) Inputs/outputs summary

### Input data expected

- Credit card customer profile + recent payment/billing behavior fields

### Model output

- Binary risk probabilities from `predict_proba`:
  - Probability of no default
  - Probability of default

### Persisted outputs

- Trained model artifact
- Preprocessor artifact
- Transformed train/test data
- Logs

---

## 7) Operational notes

- App entrypoint: `streamlit_app.py`
- Training entrypoint: `python src/pipeline/train_pipeline.py` (or import and call `TrainingPipeline().run_pipeline()`)
- This project currently stores Mongo credentials in constants; for production, move these to environment variables or a secrets manager.

---

## 8) Quick mental model

Think of this project as two connected tracks:

1. **Offline training track**: MongoDB → feature engineering/preprocessing → tuned model selection → saved artifacts
2. **Online inference track**: User input → same feature engineering/preprocessing → load saved model → probability + explanation

That consistency between training and inference preprocessing is what keeps predictions reliable in deployment.
