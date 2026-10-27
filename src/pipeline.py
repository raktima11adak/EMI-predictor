from __future__ import annotations

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'emi_database.csv'
ARTIFACT_DIR = ROOT / 'artifacts'
REPORT_DIR = ROOT / 'reports'
TARGET_CLASS = 'emi_eligibility'
TARGET_REG = 'max_monthly_emi'
DROP_FROM_FEATURES = {TARGET_CLASS, TARGET_REG}


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load CSV with normalized headers and numeric coercion for known measures."""
    df = pd.read_csv(path)
    df.columns = [re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_') for c in df.columns]
    numeric = ['age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size',
               'dependents', 'school_fees', 'college_fees', 'travel_expenses',
               'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount',
               'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
               'requested_tenure', TARGET_REG]
    for col in numeric:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in df.select_dtypes(include='object'):
        cleaned = df[col].astype('string').str.strip().replace({'': pd.NA, 'nan': pd.NA})
        # scikit-learn imputers need np.nan; pandas.NA can raise ambiguous boolean errors.
        df[col] = cleaned.astype(object).mask(cleaned.isna(), np.nan)
    return df


def assess_quality(df: pd.DataFrame) -> dict:
    return {
        'rows': int(len(df)), 'columns': int(len(df.columns)),
        'duplicate_rows': int(df.duplicated().sum()),
        'missing_cells': int(df.isna().sum().sum()),
        'missing_by_column': {k: int(v) for k, v in df.isna().sum().items() if v},
        'target_class_counts': df[TARGET_CLASS].value_counts(dropna=False).to_dict() if TARGET_CLASS in df else {},
        'scenario_counts': df['emi_scenario'].value_counts(dropna=False).to_dict() if 'emi_scenario' in df else {},
    }


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    if TARGET_CLASS in df:
        df = df.dropna(subset=[TARGET_CLASS])
    if TARGET_REG in df:
        df = df.dropna(subset=[TARGET_REG])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive affordability indicators available at application time."""
    x = df.copy()
    salary = pd.to_numeric(x.get('monthly_salary', 0), errors='coerce').replace(0, np.nan)
    expenses_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses',
                     'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount']
    expenses = pd.DataFrame({c: pd.to_numeric(x[c], errors='coerce').fillna(0)
                              for c in expenses_cols if c in x}).sum(axis=1)
    x['total_monthly_obligations'] = expenses
    x['expense_to_income_ratio'] = (expenses / salary).replace([np.inf, -np.inf], np.nan)
    x['existing_emi_to_income_ratio'] = (
        pd.to_numeric(x.get('current_emi_amount', 0), errors='coerce') / salary
    ).replace([np.inf, -np.inf], np.nan)
    x['requested_amount_to_income'] = (
        pd.to_numeric(x.get('requested_amount', 0), errors='coerce') / salary
    ).replace([np.inf, -np.inf], np.nan)
    x['savings_months'] = (
        pd.to_numeric(x.get('emergency_fund', 0), errors='coerce') /
        pd.to_numeric(x.get('monthly_salary', 0), errors='coerce').replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    x['employment_tenure_ratio'] = (
        pd.to_numeric(x.get('years_of_employment', 0), errors='coerce') /
        pd.to_numeric(x.get('age', 0), errors='coerce').replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    return x.drop(columns=list(DROP_FROM_FEATURES), errors='ignore')


def make_preprocessor(X: pd.DataFrame, scale: bool = False) -> ColumnTransformer:
    cats = list(X.select_dtypes(include=['object', 'string', 'category', 'bool']).columns)
    nums = [c for c in X.columns if c not in cats]
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        numeric_steps.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), nums),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cats),
    ], remainder='drop')


def build_models(X: pd.DataFrame):
    class_models = {
        'Logistic Regression': (LogisticRegression(max_iter=2000, class_weight='balanced'), True),
        'Random Forest': (RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
                             class_weight='balanced', random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (HistGradientBoostingClassifier(max_iter=150, learning_rate=.08,
                                  l2_regularization=1.0, random_state=42), False),
    }
    reg_models = {
        'Linear Regression': (LinearRegression(), True),
        'Random Forest': (RandomForestRegressor(n_estimators=350, min_samples_leaf=2,
                            random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (HistGradientBoostingRegressor(max_iter=150, learning_rate=.08,
                                 l2_regularization=1.0, random_state=42), False),
    }
    return class_models, reg_models


def safe_auc(y_true, probabilities, classes):
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, multi_class='ovr', average='weighted', labels=classes))
    except Exception:
        return None


def fit_task(X_train, X_test, y_train, y_test, candidates, task):
    rows, fitted = [], {}
    for name, (estimator, scale) in candidates.items():
        pipe = Pipeline([('preprocess', make_preprocessor(X_train, scale)), ('model', estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        if task == 'classification':
            prob = pipe.predict_proba(X_test)
            classes = pipe.named_steps['model'].classes_
            rows.append({'model': name, 'accuracy': accuracy_score(y_test, pred),
                         'precision_weighted': precision_score(y_test, pred, average='weighted', zero_division=0),
                         'recall_weighted': recall_score(y_test, pred, average='weighted', zero_division=0),
                         'f1_weighted': f1_score(y_test, pred, average='weighted', zero_division=0),
                         'roc_auc_ovr_weighted': safe_auc(y_test, prob, classes)})
        else:
            rmse = mean_squared_error(y_test, pred) ** 0.5
            nonzero = np.abs(y_test) > 1e-8
            mape = float(np.mean(np.abs((y_test[nonzero] - pred[nonzero]) / y_test[nonzero])) * 100) if nonzero.any() else None
            rows.append({'model': name, 'rmse': rmse, 'mae': mean_absolute_error(y_test, pred),
                         'r2': r2_score(y_test, pred), 'mape_percent': mape})
    return pd.DataFrame(rows).sort_values('f1_weighted' if task == 'classification' else 'rmse',
                                         ascending=(task != 'classification')), fitted


def log_mlflow(results: dict, best_class, best_reg, trained_class, trained_reg):
    try:
        import mlflow
        mlflow.set_experiment('EMIPredict AI')
        for task, rows, models in [('classification', results['classification'], trained_class),
                                   ('regression', results['regression'], trained_reg)]:
            for row in rows:
                with mlflow.start_run(run_name=f"{task} - {row['model']}"):
                    mlflow.log_param('task', task)
                    mlflow.log_param('model', row['model'])
                    mlflow.log_param('training_rows', results['training_rows'])
                    mlflow.log_param('evaluation_rows', results['evaluation_rows'])
                    mlflow.log_param('sample_mode', results['sample_mode'])
                    metrics = {k: float(v) for k, v in row.items()
                               if k != 'model' and v is not None and pd.notna(v)}
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(models[row['model']], artifact_path='model', serialization_format='cloudpickle')
        with mlflow.start_run(run_name='Selected production candidates'):
            mlflow.log_param('classification_selection', best_class)
            mlflow.log_param('regression_selection', best_reg)
    except ImportError:
        results['mlflow_status'] = 'MLflow is not installed; metrics are saved locally. Install requirements and rerun to enable tracking.'
    except Exception as exc:
        results['mlflow_status'] = f'MLflow tracking skipped: {exc}'
    return results


def train_all(path: str | Path = DATA_PATH, sample_mode: bool = False):
    raw = load_data(path)
    quality = assess_quality(raw)
    df = clean_training_data(raw)
    if sample_mode and len(df) > 20000:
        df = df.sample(20000, random_state=42)
    X = engineer_features(df.drop(columns=list(DROP_FROM_FEATURES)))
    # Use one shared split so both tasks are compared on the same profiles.
    class_counts = df[TARGET_CLASS].value_counts()
    stratify = df[TARGET_CLASS] if class_counts.min() >= 2 else None
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=.25, random_state=42, stratify=stratify)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yc_train, yc_test = df[TARGET_CLASS].iloc[train_idx], df[TARGET_CLASS].iloc[test_idx]
    yr_train, yr_test = df[TARGET_REG].iloc[train_idx], df[TARGET_REG].iloc[test_idx]
    class_candidates, reg_candidates = build_models(X_train)
    class_metrics, fitted_class = fit_task(X_train, X_test, yc_train, yc_test, class_candidates, 'classification')
    reg_metrics, fitted_reg = fit_task(X_train, X_test, yr_train, yr_test, reg_candidates, 'regression')
    best_class = str(class_metrics.iloc[0]['model'])
    best_reg = str(reg_metrics.iloc[0]['model'])
    results = {'classification': class_metrics.to_dict(orient='records'),
               'regression': reg_metrics.to_dict(orient='records'),
               'classification_report': classification_report(yc_test, fitted_class[best_class].predict(X_test), zero_division=0),
               'quality': quality, 'training_rows': int(len(train_idx)),
               'evaluation_rows': int(len(test_idx)), 'sample_mode': bool(sample_mode),
               'best_classification_model': best_class, 'best_regression_model': best_reg}
    results = log_mlflow(results, best_class, best_reg, fitted_class, fitted_reg)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({'classifier': fitted_class[best_class], 'regressor': fitted_reg[best_reg],
                 'classifier_name': best_class, 'regressor_name': best_reg,
                 'feature_columns': list(X.columns), 'class_labels': list(fitted_class[best_class].named_steps['model'].classes_)},
                ARTIFACT_DIR / 'emipredict_models.joblib')
    class_metrics.to_csv(REPORT_DIR / 'classification_model_comparison.csv', index=False)
    reg_metrics.to_csv(REPORT_DIR / 'regression_model_comparison.csv', index=False)
    (REPORT_DIR / 'training_summary.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
    return results



