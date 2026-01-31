"""
Model Utilities for SkyGuard Streamlit App
Handles model loading, predictions, and SHAP explanations.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@st.cache_resource
def load_lgbm_model():
    """Load the LightGBM model."""
    model_path = get_project_root() / "model" / "lgbm_churn_model.joblib"
    return joblib.load(model_path)


@st.cache_resource
def load_lr_model():
    """Load the Logistic Regression model."""
    model_path = get_project_root() / "model" / "lr_model.joblib"
    return joblib.load(model_path)


@st.cache_resource
def load_lr_scaler():
    """Load the scaler for Logistic Regression."""
    scaler_path = get_project_root() / "model" / "lr_scaler.joblib"
    return joblib.load(scaler_path)


@st.cache_resource
def load_shap_explainer():
    """Load or create SHAP explainer."""
    if not HAS_SHAP:
        return None
    model = load_lgbm_model()
    return shap.TreeExplainer(model)


def get_feature_names() -> list:
    """Get list of feature names."""
    path = get_project_root() / "model" / "feature_names.json"
    with open(path, 'r') as f:
        return json.load(f)


def get_threshold() -> float:
    """Get the optimal decision threshold."""
    path = get_project_root() / "model" / "threshold.json"
    with open(path, 'r') as f:
        thresholds = json.load(f)
    return thresholds.get('lgbm', 0.5)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction."""
    feature_names = get_feature_names()

    # Ensure all features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Select only required features in correct order
    X = df[feature_names].copy()

    # Handle NaN and inf values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    return X


def predict_churn_probability(df: pd.DataFrame) -> np.ndarray:
    """Predict churn probability for given data."""
    model = load_lgbm_model()
    X = prepare_features(df)
    return model.predict_proba(X)[:, 1]


def predict_churn(df: pd.DataFrame, threshold: float = None) -> np.ndarray:
    """Predict churn class for given data."""
    if threshold is None:
        threshold = get_threshold()
    probabilities = predict_churn_probability(df)
    return (probabilities >= threshold).astype(int)


def get_risk_level(probability: float) -> str:
    """Get risk level string from probability."""
    if probability >= 0.6:
        return 'High'
    elif probability >= 0.3:
        return 'Medium'
    else:
        return 'Low'


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        'High': '#FF3B30',
        'Medium': '#FF9500',
        'Low': '#34C759'
    }
    return colors.get(risk_level, '#86868B')


def compute_shap_values(X: pd.DataFrame):
    """Compute SHAP values for given features."""
    if not HAS_SHAP:
        return None, None

    explainer = load_shap_explainer()
    shap_values = explainer.shap_values(X)

    # For binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Get values for churned class

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    return shap_values, expected_value


def get_top_risk_factors(shap_values: np.ndarray, feature_names: list, n: int = 5) -> list:
    """Get top risk factors from SHAP values."""
    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(shap_values))[::-1][:n]

    factors = []
    for idx in indices:
        factor = {
            'feature': feature_names[idx],
            'shap_value': shap_values[idx],
            'direction': 'increases' if shap_values[idx] > 0 else 'decreases'
        }
        factors.append(factor)

    return factors


def generate_risk_explanation(shap_values: np.ndarray, feature_values: pd.Series,
                               feature_names: list) -> list:
    """Generate human-readable explanations for risk factors."""
    explanations = []

    # Get top factors
    top_factors = get_top_risk_factors(shap_values, feature_names, n=5)

    feature_explanations = {
        'flight_trend_yoy': lambda v, s: f"Flight activity {'decreased' if v < 0 else 'increased'} by {abs(v):.1f} flights year-over-year",
        'recency_last_flight': lambda v, s: f"Last flight was {int(v)} months ago" if v > 0 else "Recently active flyer",
        'total_flights_24m': lambda v, s: f"{'Only ' if v < 10 else ''}{int(v)} total flights in 24 months",
        'is_promotion_enrollee': lambda v, s: "Enrolled through 2018 promotion (higher churn risk segment)" if v == 1 else "Standard enrollment",
        'points_utilization': lambda v, s: f"{'Low' if v < 0.3 else 'High'} points utilization ({v:.0%})",
        'CLV': lambda v, s: f"Customer Lifetime Value: ${v:,.2f}",
        'clv_per_year': lambda v, s: f"Annualized CLV: ${v:,.2f}",
        'tenure_months': lambda v, s: f"Member for {int(v)} months",
        'companion_ratio': lambda v, s: f"{v:.0%} of flights with companions",
        'months_active': lambda v, s: f"Active in {int(v)} of 24 months",
        'loyalty_tier_numeric': lambda v, s: f"{'Star' if v == 1 else 'Nova' if v == 2 else 'Aurora'} tier member",
        'redemption_frequency': lambda v, s: f"Redeems points {v:.0%} of months",
    }

    for factor in top_factors:
        feature = factor['feature']
        shap_val = factor['shap_value']

        if feature in feature_values.index:
            value = feature_values[feature]
        else:
            value = 0

        # Get explanation
        if feature in feature_explanations:
            explanation = feature_explanations[feature](value, shap_val)
        else:
            explanation = f"{feature}: {value:.2f}"

        # Add impact indicator
        impact = "risk factor" if shap_val > 0 else "retention factor"

        explanations.append({
            'text': explanation,
            'impact': impact,
            'shap_value': shap_val
        })

    return explanations


def get_retention_recommendations(shap_values: np.ndarray, feature_values: pd.Series,
                                   feature_names: list, probability: float) -> list:
    """Generate retention recommendations based on risk factors."""
    recommendations = []

    # Map features to retention strategies
    strategies = {
        'flight_trend_yoy': {
            'condition': lambda v: v < 0,
            'action': "Targeted fare discount (15% off next booking)",
            'priority': 'High',
            'cost': '$50/customer',
            'reason': 'Declining flight activity'
        },
        'points_utilization': {
            'condition': lambda v: v < 0.3,
            'action': "Points expiry warning + bonus redemption catalog",
            'priority': 'Medium',
            'cost': '$10/customer',
            'reason': 'Low points engagement'
        },
        'is_promotion_enrollee': {
            'condition': lambda v: v == 1,
            'action': "Welcome series + personal onboarding call",
            'priority': 'High',
            'cost': '$25/customer',
            'reason': 'Promotion enrollee (high churn segment)'
        },
        'companion_ratio': {
            'condition': lambda v: v < 0.2,
            'action': "Family/group booking incentive (2-for-1 companion)",
            'priority': 'Medium',
            'cost': '$30/customer',
            'reason': 'Low companion travel'
        },
        'recency_last_flight': {
            'condition': lambda v: v > 6,
            'action': "Win-back campaign with exclusive offer",
            'priority': 'Medium',
            'cost': '$15/customer',
            'reason': 'Lapsed traveler'
        },
        'redemption_frequency': {
            'condition': lambda v: v < 0.1,
            'action': "Points bonus multiplier for next redemption",
            'priority': 'Low',
            'cost': '$20/customer',
            'reason': 'Rare points redemption'
        }
    }

    # Check each strategy
    for feature, strategy in strategies.items():
        if feature in feature_values.index:
            value = feature_values[feature]
            if strategy['condition'](value):
                recommendations.append({
                    'action': strategy['action'],
                    'priority': strategy['priority'],
                    'cost': strategy['cost'],
                    'reason': strategy['reason']
                })

    # Add CLV-based priority
    if 'CLV' in feature_values.index:
        clv = feature_values['CLV']
        if clv > 5000 and probability > 0.5:
            recommendations.insert(0, {
                'action': f"Priority outreach - Assign personal account manager (CLV: ${clv:,.0f})",
                'priority': 'Critical',
                'cost': '$100/customer',
                'reason': 'High-value customer at risk'
            })

    # Sort by priority
    priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

    return recommendations[:5]  # Return top 5


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Run batch predictions on a DataFrame."""
    # Get predictions
    probabilities = predict_churn_probability(df)

    # Add results to dataframe
    result_df = df.copy()
    result_df['churn_probability'] = probabilities
    result_df['risk_level'] = result_df['churn_probability'].apply(get_risk_level)
    result_df['predicted_churn'] = (probabilities >= get_threshold()).astype(int)

    # Get top risk factor for each customer
    feature_names = get_feature_names()
    X = prepare_features(df)

    if HAS_SHAP:
        shap_values, _ = compute_shap_values(X)
        if shap_values is not None:
            top_factors = []
            for i in range(len(df)):
                factors = get_top_risk_factors(shap_values[i], feature_names, n=1)
                top_factors.append(factors[0]['feature'] if factors else 'N/A')
            result_df['top_risk_factor'] = top_factors
    else:
        result_df['top_risk_factor'] = 'N/A'

    return result_df
