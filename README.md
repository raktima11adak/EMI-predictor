# EMIPredict AI - Intelligent Financial Risk Assessment Platform

A runnable capstone project based on the supplied project brief. It provides a dual-task machine-learning workflow (three classifier candidates and three regressor candidates), engineered affordability indicators, EDA, optional MLflow tracking, and a multipage Streamlit interface with prediction, exploration, and session-level CRUD.

## Important dataset note

The project currently includes a **404,800-row dataset** with the 27 expected columns (25 application inputs and two targets), and the saved model was trained on 303,600 rows with 101,200 held out for evaluation. The dataset contains 17,488 `High_Risk` examples. Treat the metrics as project evaluation results, not as a validated lending decision system.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Train six model candidates, compare a held-out split, save the selected models, and produce reports:

```powershell
.\.venv\Scripts\python.exe src\train.py
```

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The explicit virtual-environment Python command keeps Streamlit and its compiled dependencies (including PyArrow) inside .venv, avoiding accidental launch through a separately active Anaconda base environment.

Generate data quality summary and sample EDA visualization:

```powershell
.\.venv\Scripts\python.exe src\eda.py
```

When MLflow is installed, model parameters, metrics, and model artifacts are logged to the local `mlruns` tracking store. View it with `mlflow ui`. The project runs without MLflow and clearly reports when tracking is unavailable.

## Project layout

- `app.py` - Streamlit interface: risk assessment, data explorer, session CRUD, model results
- `src/pipeline.py` - loading, validation, feature engineering, six model candidates, evaluation, model persistence, optional MLflow
- `src/train.py` - command-line training entry point
- `src/eda.py` - quality summary and EDA chart
- `data/emi_database.csv` - supplied small sample
- `reports/` - generated metrics, quality report, and EDA output
- `artifacts/` - selected classifier and regressor (generated during training)

## Method

Rows missing either target and exact duplicate rows are excluded from model training. Missing predictor values are imputed inside each training pipeline. Categorical predictors are one-hot encoded; numeric predictors are median-imputed. Features include expense-to-income, current-EMI-to-income, requested-loan-to-income, emergency-savings months, and employment-tenure-to-age ratios. A fixed 75/25 held-out split is used; where possible, classification is stratified. The top classifier is selected by weighted F1 and the top regressor by lowest RMSE. See the generated report tables for all metrics and use the held-out scores rather than training fit as model evidence.

The source targets are retained only as labels and excluded from predictors. Model outputs are estimates from the sample's patterns, not causal or validated credit decisions. Validate against representative, consented, production data and conduct fairness, privacy, security, and regulatory review before deployment. Streamlit Cloud deployment requires publishing this folder to a GitHub repository and configuring the app entry point as `app.py`.





