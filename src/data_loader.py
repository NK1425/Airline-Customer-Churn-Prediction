"""
Data Loading Module for SkyGuard Streamlit App
Handles cached data loading for optimal performance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@st.cache_data
def load_processed_features() -> pd.DataFrame:
    """Load the processed features dataset."""
    data_path = get_project_root() / "data" / "processed_features.csv"
    return pd.read_csv(data_path)


@st.cache_data
def load_flight_activity() -> pd.DataFrame:
    """Load raw flight activity data."""
    data_path = get_project_root() / "data" / "Customer_Flight_Activity.csv"
    return pd.read_csv(data_path)


@st.cache_data
def load_loyalty_history() -> pd.DataFrame:
    """Load raw loyalty history data."""
    data_path = get_project_root() / "data" / "Customer_Loyalty_History.csv"
    return pd.read_csv(data_path)


@st.cache_data
def load_test_data():
    """Load test data used for model evaluation."""
    model_dir = get_project_root() / "model"
    X_test = pd.read_csv(model_dir / "X_test.csv")
    y_test = pd.read_csv(model_dir / "y_test.csv")
    return X_test, y_test['churn']


@st.cache_data
def load_feature_names() -> list:
    """Load feature names."""
    path = get_project_root() / "model" / "feature_names.json"
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> dict:
    """Load model metrics."""
    path = get_project_root() / "model" / "metrics.json"
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_thresholds() -> dict:
    """Load decision thresholds."""
    path = get_project_root() / "model" / "threshold.json"
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_shap_values():
    """Load pre-computed SHAP values."""
    model_dir = get_project_root() / "model"
    shap_values = np.load(model_dir / "shap_values_test.npy")
    expected_value = np.load(model_dir / "shap_expected_value.npy")
    return shap_values, expected_value[0]


@st.cache_data
def get_summary_stats(df: pd.DataFrame) -> dict:
    """Calculate summary statistics for the dataset."""
    stats = {
        'total_customers': len(df),
        'churned_customers': df['churn'].sum(),
        'active_customers': len(df) - df['churn'].sum(),
        'churn_rate': df['churn'].mean() * 100,
        'avg_clv': df['CLV'].mean(),
        'total_clv': df['CLV'].sum(),
        'churned_clv': df[df['churn'] == 1]['CLV'].sum(),
    }

    # Tier distribution
    if 'Loyalty Card' in df.columns:
        tier_col = 'Loyalty Card'
    else:
        tier_col = None

    if tier_col:
        for tier in ['Star', 'Nova', 'Aurora']:
            tier_df = df[df[tier_col] == tier]
            stats[f'{tier.lower()}_count'] = len(tier_df)
            stats[f'{tier.lower()}_churn_rate'] = tier_df['churn'].mean() * 100 if len(tier_df) > 0 else 0

    return stats


@st.cache_data
def get_churn_by_segment(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Calculate churn rate by segment."""
    return df.groupby(segment_col).agg({
        'churn': ['sum', 'count', 'mean'],
        'CLV': 'mean'
    }).reset_index()


def get_customer_by_id(df: pd.DataFrame, loyalty_number: int) -> pd.Series:
    """Get customer data by loyalty number."""
    customer = df[df['Loyalty Number'] == loyalty_number]
    if len(customer) == 0:
        return None
    return customer.iloc[0]


def get_sample_customers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get a sample of customers for demonstration."""
    # Get mix of churned and active
    churned = df[df['churn'] == 1].sample(min(n // 2, len(df[df['churn'] == 1])))
    active = df[df['churn'] == 0].sample(min(n // 2, len(df[df['churn'] == 0])))
    return pd.concat([churned, active]).sample(frac=1).reset_index(drop=True)


def get_high_risk_customers(df: pd.DataFrame, predictions: np.ndarray, threshold: float = 0.6) -> pd.DataFrame:
    """Get customers with high churn risk."""
    df_copy = df.copy()
    df_copy['churn_probability'] = predictions
    df_copy['risk_level'] = df_copy['churn_probability'].apply(
        lambda x: 'High' if x >= 0.6 else ('Medium' if x >= 0.3 else 'Low')
    )
    return df_copy[df_copy['churn_probability'] >= threshold].sort_values('churn_probability', ascending=False)
