"""
Model Training Script for Airline Customer Churn Prediction
Trains LightGBM and Logistic Regression with SMOTE, threshold optimization, and SHAP.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Will use alternative model.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Explainability features will be limited.")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: Optuna not installed. Using GridSearchCV for hyperparameter tuning.")


def load_processed_data(data_dir: Path = None):
    """Load processed features."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    df = pd.read_csv(data_dir / "processed_features.csv")
    return df


def get_feature_columns():
    """Return the list of feature columns used in the model."""
    return [
        'total_flights_24m',
        'avg_monthly_flights',
        'flight_consistency',
        'total_flights_booked',
        'total_distance_24m',
        'points_accumulated_24m',
        'points_redeemed_24m',
        'dollar_points_redeemed_24m',
        'flight_trend_yoy',
        'companion_ratio',
        'avg_distance_per_flight',
        'points_utilization',
        'months_active',
        'recency_last_flight',
        'redemption_frequency',
        'seasonal_peak_ratio',
        'tenure_months',
        'loyalty_tier_numeric',
        'is_promotion_enrollee',
        'clv_per_year',
        'gender_encoded',
        'education_encoded',
        'Salary',
        'CLV',
        'marital_Married',
        'marital_Single',
        'country_United States',
        'clv_per_flight',
        'distance_per_salary',
        'points_per_flight',
        'miles_per_dollar_clv',
    ]


def prepare_data(df: pd.DataFrame):
    """Prepare features and target for modeling."""
    feature_cols = get_feature_columns()

    # Ensure all columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].copy()
    y = df['churn'].copy()

    # Handle NaN and inf values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    return X, y, feature_cols


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1 score for minority class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find threshold that maximizes F1
    optimal_idx = np.argmax(f1_scores[:-1])  # Last values are for threshold=1
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


def train_logistic_regression(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Train Logistic Regression baseline model."""
    print("\nTraining Logistic Regression...")

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train model with class weights
    model = LogisticRegression(
        class_weight={0: 1, 1: scale_pos_weight},
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)

    # Predict
    y_proba = model.predict_proba(X_val_scaled)[:, 1]

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_val, y_proba)
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc_roc': roc_auc_score(y_val, y_proba),
        'auc_pr': average_precision_score(y_val, y_proba),
        'threshold': optimal_threshold
    }

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")

    return model, scaler, metrics


def train_lightgbm_optuna(X_train, y_train, X_val, y_val, scale_pos_weight, n_trials=50):
    """Train LightGBM with Optuna hyperparameter tuning."""
    print("\nTraining LightGBM with Optuna optimization...")

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight,
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 300, 500, 700]),
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 9, -1]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1]),
            'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63, 127]),
            'min_child_samples': trial.suggest_categorical('min_child_samples', [10, 20, 50]),
            'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 0.9]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.7, 0.8, 0.9]),
            'reg_alpha': trial.suggest_categorical('reg_alpha', [0, 0.1, 1.0]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0, 0.1, 1.0]),
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        return f1_score(y_val, y_pred)

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best parameters
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'binary_logloss'
    best_params['boosting_type'] = 'gbdt'
    best_params['verbosity'] = -1
    best_params['random_state'] = 42
    best_params['scale_pos_weight'] = scale_pos_weight

    print(f"\nBest F1 Score: {study.best_value:.4f}")
    print(f"Best Parameters: {best_params}")

    # Train final model with best parameters
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    return final_model, best_params


def train_lightgbm_grid(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Train LightGBM with manual grid search (fallback if Optuna unavailable)."""
    print("\nTraining LightGBM with grid search...")

    best_f1 = 0
    best_params = None
    best_model = None

    # Simplified grid for faster training
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'min_child_samples': [20],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.1],
        'reg_lambda': [0.1],
    }

    from itertools import product

    keys = param_grid.keys()
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    print(f"Total combinations: {total_combos}")

    combo_count = 0
    for values in product(*param_grid.values()):
        combo_count += 1
        params = dict(zip(keys, values))
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params['boosting_type'] = 'gbdt'
        params['verbosity'] = -1
        params['random_state'] = 42
        params['scale_pos_weight'] = scale_pos_weight

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        f1 = f1_score(y_val, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params.copy()
            best_model = model
            print(f"  [{combo_count}/{total_combos}] New best F1: {best_f1:.4f}")

    print(f"\nBest F1 Score: {best_f1:.4f}")
    print(f"Best Parameters: {best_params}")

    return best_model, best_params


def evaluate_model(model, X, y, threshold=0.5, scaler=None):
    """Evaluate model and return all metrics."""
    if scaler is not None:
        X = scaler.transform(X)

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc_roc': roc_auc_score(y, y_proba),
        'auc_pr': average_precision_score(y, y_proba),
    }

    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred, output_dict=True)

    return metrics, conf_matrix, class_report, y_proba, y_pred


def compute_shap_values(model, X, feature_names):
    """Compute SHAP values for model explainability."""
    print("\nComputing SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values is a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]  # Get values for churned class
    else:
        shap_values_churn = shap_values

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value_churn = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    else:
        expected_value_churn = expected_value

    print(f"  SHAP values shape: {shap_values_churn.shape}")

    return explainer, shap_values_churn, expected_value_churn


def save_artifacts(model_dir, lgbm_model, lr_model, lr_scaler, lgbm_threshold, lr_threshold,
                   lgbm_metrics, lr_metrics, feature_names, shap_values=None, expected_value=None,
                   X_test=None, y_test=None, lgbm_params=None):
    """Save all model artifacts."""
    print("\nSaving artifacts...")

    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    # Save LightGBM model
    joblib.dump(lgbm_model, model_dir / "lgbm_churn_model.joblib")
    print(f"  Saved LightGBM model")

    # Save Logistic Regression model and scaler
    joblib.dump(lr_model, model_dir / "lr_model.joblib")
    joblib.dump(lr_scaler, model_dir / "lr_scaler.joblib")
    print(f"  Saved Logistic Regression model and scaler")

    # Save thresholds
    thresholds = {
        'lgbm': float(lgbm_threshold),
        'lr': float(lr_threshold)
    }
    with open(model_dir / "threshold.json", 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"  Saved thresholds")

    # Save feature names
    with open(model_dir / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Saved feature names")

    # Save metrics
    all_metrics = {
        'lgbm': lgbm_metrics,
        'lr': lr_metrics
    }
    with open(model_dir / "metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved metrics")

    # Save best params
    if lgbm_params:
        # Convert any numpy types to Python types
        lgbm_params_clean = {}
        for k, v in lgbm_params.items():
            if isinstance(v, np.integer):
                lgbm_params_clean[k] = int(v)
            elif isinstance(v, np.floating):
                lgbm_params_clean[k] = float(v)
            else:
                lgbm_params_clean[k] = v
        with open(model_dir / "lgbm_params.json", 'w') as f:
            json.dump(lgbm_params_clean, f, indent=2)
        print(f"  Saved LightGBM parameters")

    # Save SHAP values
    if shap_values is not None:
        np.save(model_dir / "shap_values_test.npy", shap_values)
        print(f"  Saved SHAP values")

    if expected_value is not None:
        np.save(model_dir / "shap_expected_value.npy", np.array([expected_value]))
        print(f"  Saved SHAP expected value")

    # Save test data for reference
    if X_test is not None and y_test is not None:
        X_test.to_csv(model_dir / "X_test.csv", index=False)
        y_test.to_csv(model_dir / "y_test.csv", index=False)
        print(f"  Saved test data")

    print("\nAll artifacts saved!")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("AIRLINE CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    model_dir = project_dir / "model"

    # Load data
    print("\n1. Loading processed data...")
    df = load_processed_data(data_dir)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Churn rate: {df['churn'].mean():.2%}")

    # Prepare features
    print("\n2. Preparing features...")
    X, y, feature_names = prepare_data(df)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of features: {len(feature_names)}")

    # Train/val/test split (70/15/15)
    print("\n3. Splitting data (70/15/15)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
    )  # 0.176 of 85% = ~15%

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    print(f"   Train churn rate: {y_train.mean():.2%}")

    # Apply SMOTE to training set only
    print("\n4. Applying SMOTE to training set...")
    smote = SMOTE(sampling_strategy=0.4, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"   Before SMOTE: {X_train.shape[0]} samples, churn rate: {y_train.mean():.2%}")
    print(f"   After SMOTE: {X_train_resampled.shape[0]} samples, churn rate: {y_train_resampled.mean():.2%}")

    # Calculate scale_pos_weight (after SMOTE)
    scale_pos_weight = len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1])
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    # Train Logistic Regression baseline
    print("\n5. Training Logistic Regression baseline...")
    lr_model, lr_scaler, lr_val_metrics = train_logistic_regression(
        X_train_resampled, y_train_resampled, X_val, y_val, scale_pos_weight
    )
    lr_threshold = lr_val_metrics['threshold']

    # Train LightGBM
    print("\n6. Training LightGBM...")
    if HAS_LIGHTGBM:
        if HAS_OPTUNA:
            lgbm_model, lgbm_params = train_lightgbm_optuna(
                X_train_resampled, y_train_resampled, X_val, y_val, scale_pos_weight, n_trials=30
            )
        else:
            lgbm_model, lgbm_params = train_lightgbm_grid(
                X_train_resampled, y_train_resampled, X_val, y_val, scale_pos_weight
            )

        # Find optimal threshold on validation set
        y_val_proba = lgbm_model.predict_proba(X_val)[:, 1]
        lgbm_threshold = find_optimal_threshold(y_val, y_val_proba)
        print(f"   Optimal threshold: {lgbm_threshold:.4f}")
    else:
        print("   LightGBM not available, skipping...")
        lgbm_model = None
        lgbm_params = None
        lgbm_threshold = 0.5

    # Evaluate on test set
    print("\n7. Final evaluation on test set...")

    print("\n   Logistic Regression:")
    lr_metrics, lr_conf_matrix, lr_class_report, lr_y_proba, lr_y_pred = evaluate_model(
        lr_model, X_test, y_test, lr_threshold, lr_scaler
    )
    print(f"   Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"   Precision: {lr_metrics['precision']:.4f}")
    print(f"   Recall: {lr_metrics['recall']:.4f}")
    print(f"   F1 Score: {lr_metrics['f1']:.4f}")
    print(f"   AUC-ROC: {lr_metrics['auc_roc']:.4f}")
    print(f"   AUC-PR: {lr_metrics['auc_pr']:.4f}")

    if lgbm_model:
        print("\n   LightGBM:")
        lgbm_metrics, lgbm_conf_matrix, lgbm_class_report, lgbm_y_proba, lgbm_y_pred = evaluate_model(
            lgbm_model, X_test, y_test, lgbm_threshold
        )
        print(f"   Accuracy: {lgbm_metrics['accuracy']:.4f}")
        print(f"   Precision: {lgbm_metrics['precision']:.4f}")
        print(f"   Recall: {lgbm_metrics['recall']:.4f}")
        print(f"   F1 Score: {lgbm_metrics['f1']:.4f}")
        print(f"   AUC-ROC: {lgbm_metrics['auc_roc']:.4f}")
        print(f"   AUC-PR: {lgbm_metrics['auc_pr']:.4f}")
    else:
        lgbm_metrics = lr_metrics.copy()
        lgbm_model = lr_model
        lgbm_threshold = lr_threshold

    # Compute SHAP values
    shap_values = None
    expected_value = None
    if HAS_SHAP and lgbm_model and HAS_LIGHTGBM:
        print("\n8. Computing SHAP values...")
        explainer, shap_values, expected_value = compute_shap_values(
            lgbm_model, X_test, feature_names
        )

    # Save artifacts
    print("\n9. Saving all artifacts...")
    save_artifacts(
        model_dir=model_dir,
        lgbm_model=lgbm_model,
        lr_model=lr_model,
        lr_scaler=lr_scaler,
        lgbm_threshold=lgbm_threshold,
        lr_threshold=lr_threshold,
        lgbm_metrics=lgbm_metrics,
        lr_metrics=lr_metrics,
        feature_names=feature_names,
        shap_values=shap_values,
        expected_value=expected_value,
        X_test=X_test,
        y_test=y_test,
        lgbm_params=lgbm_params
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel Performance Summary:")
    print(f"  LightGBM F1 Score: {lgbm_metrics['f1']:.4f}")
    print(f"  Logistic Regression F1 Score: {lr_metrics['f1']:.4f}")
    print(f"\nArtifacts saved to: {model_dir}")

    return lgbm_model, lr_model, lgbm_metrics, lr_metrics


if __name__ == "__main__":
    main()
