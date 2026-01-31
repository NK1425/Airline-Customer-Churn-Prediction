"""
Customer Prediction Page - Single Customer Analysis
Enter or select a customer to get churn probability + SHAP explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_loader import load_processed_features, get_customer_by_id
from src.model_utils import (
    predict_churn_probability, compute_shap_values, prepare_features,
    get_feature_names, get_risk_level, get_risk_color,
    generate_risk_explanation, get_retention_recommendations
)
from src.ui_components import (
    load_css, section_header, gradient_divider, risk_badge, tier_badge,
    progress_ring, customer_card, COLORS, RISK_COLORS
)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Page config
st.set_page_config(page_title="Customer Prediction - SkyGuard", page_icon="Q", layout="wide")
load_css()

# Header
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="color: #1D1D1F; font-size: 32px; font-weight: 700; margin: 0;">
        Customer Churn Prediction
    </h1>
    <p style="color: #6E6E73; font-size: 16px; margin: 5px 0 0 0;">
        Individual customer risk analysis with SHAP explanations
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return load_processed_features()

df = load_data()

# Customer selection
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Select Customer")

    # Method selection
    selection_method = st.radio(
        "Selection method",
        ["Search by Loyalty Number", "Select from list"],
        label_visibility="collapsed"
    )

    if selection_method == "Search by Loyalty Number":
        loyalty_number = st.number_input(
            "Enter Loyalty Number",
            min_value=int(df['Loyalty Number'].min()),
            max_value=int(df['Loyalty Number'].max()),
            value=int(df['Loyalty Number'].iloc[0]),
            step=1
        )
    else:
        # Sample customers - mix of churned and active, high and low CLV
        sample_ids = df.nlargest(5, 'CLV')['Loyalty Number'].tolist()
        sample_ids += df[df['churn'] == 1].sample(min(5, len(df[df['churn'] == 1])))['Loyalty Number'].tolist()
        sample_ids = list(set(sample_ids))[:10]

        loyalty_number = st.selectbox(
            "Select a customer",
            options=sample_ids,
            format_func=lambda x: f"Customer #{x}"
        )

# Get customer data
customer = df[df['Loyalty Number'] == loyalty_number]

if len(customer) == 0:
    st.error(f"Customer #{loyalty_number} not found.")
    st.stop()

customer = customer.iloc[0]

with col2:
    # Customer profile card
    st.markdown("### Customer Profile")

    tier = customer.get('Loyalty Card', 'Star')
    tier_badge_html = tier_badge(tier)

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown(f"""
        <div style="padding: 10px;">
            <p style="color: #86868B; font-size: 12px; margin: 0;">Tier</p>
            {tier_badge_html}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="padding: 10px;">
            <p style="color: #86868B; font-size: 12px; margin: 0;">CLV</p>
            <p style="color: #1D1D1F; font-size: 18px; font-weight: 600; margin: 0;">${customer['CLV']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div style="padding: 10px;">
            <p style="color: #86868B; font-size: 12px; margin: 0;">Tenure</p>
            <p style="color: #1D1D1F; font-size: 18px; font-weight: 600; margin: 0;">{int(customer['tenure_months'])} months</p>
        </div>
        """, unsafe_allow_html=True)

    with col_d:
        enrollment = "Promotion" if customer.get('is_promotion_enrollee', 0) == 1 else "Standard"
        st.markdown(f"""
        <div style="padding: 10px;">
            <p style="color: #86868B; font-size: 12px; margin: 0;">Enrollment</p>
            <p style="color: #1D1D1F; font-size: 18px; font-weight: 600; margin: 0;">{enrollment}</p>
        </div>
        """, unsafe_allow_html=True)

gradient_divider()

# Prediction section
section_header("Prediction Result")

# Get prediction
X_customer = prepare_features(pd.DataFrame([customer]))
probability = predict_churn_probability(pd.DataFrame([customer]))[0]
risk_level = get_risk_level(probability)
risk_color = get_risk_color(risk_level)

col1, col2 = st.columns([1, 2])

with col1:
    # Risk gauge
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #E5E5EA;
        text-align: center;
    ">
        <p style="color: #86868B; font-size: 14px; margin: 0 0 20px 0; text-transform: uppercase; letter-spacing: 1px;">
            Churn Risk Score
        </p>
        {progress_ring(probability * 100, size=150, color=risk_color)}
        <div style="margin-top: -40px;">
            {risk_badge(risk_level)}
        </div>
        <p style="color: #6E6E73; font-size: 13px; margin: 20px 0 0 0;">
            The model predicts a <b>{probability*100:.1f}%</b> probability<br>
            of churn within the next quarter.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # SHAP waterfall plot
    if HAS_SHAP:
        st.markdown("#### SHAP Explanation")
        st.markdown("<p style='color: #6E6E73; font-size: 13px;'>Feature contributions to the prediction</p>", unsafe_allow_html=True)

        try:
            shap_values, expected_value = compute_shap_values(X_customer)

            if shap_values is not None:
                feature_names = get_feature_names()

                # Create waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))

                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=expected_value,
                        data=X_customer.iloc[0].values,
                        feature_names=feature_names
                    ),
                    max_display=10,
                    show=False
                )

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        except Exception as e:
            st.warning(f"Could not generate SHAP plot: {str(e)}")
    else:
        st.info("SHAP library not available. Install shap for explainability features.")

gradient_divider()

# Risk factors and recommendations
col1, col2 = st.columns(2)

with col1:
    section_header("Key Risk Factors", "Plain English explanations")

    if HAS_SHAP and shap_values is not None:
        explanations = generate_risk_explanation(shap_values[0], customer, feature_names)

        for exp in explanations[:5]:
            icon = "!" if exp['impact'] == 'risk factor' else "+"
            color = COLORS['red'] if exp['impact'] == 'risk factor' else COLORS['green']
            bg_color = 'rgba(255, 59, 48, 0.1)' if exp['impact'] == 'risk factor' else 'rgba(52, 199, 89, 0.1)'

            st.markdown(f"""
            <div style="
                background: {bg_color};
                border-left: 4px solid {color};
                padding: 12px 16px;
                border-radius: 8px;
                margin-bottom: 10px;
            ">
                <p style="color: {color}; margin: 0; font-size: 14px;">
                    <strong>{icon}</strong> {exp['text']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback without SHAP
        st.markdown("""
        <div style="background: rgba(0, 113, 227, 0.1); padding: 16px; border-radius: 8px;">
            <p style="color: #0071E3; margin: 0;">SHAP explanations require the shap library.</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    section_header("Recommended Retention Actions", "Personalized strategies")

    if HAS_SHAP and shap_values is not None:
        recommendations = get_retention_recommendations(shap_values[0], customer, feature_names, probability)

        for i, rec in enumerate(recommendations, 1):
            priority_color = {
                'Critical': COLORS['red'],
                'High': COLORS['orange'],
                'Medium': COLORS['primary'],
                'Low': COLORS['gray']
            }.get(rec['priority'], COLORS['gray'])

            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                border: 1px solid #E5E5EA;
                border-left: 4px solid {priority_color};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="
                        background: {priority_color}20;
                        color: {priority_color};
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: 600;
                    ">{rec['priority']}</span>
                    <span style="color: #86868B; font-size: 12px;">{rec['cost']}</span>
                </div>
                <p style="color: #1D1D1F; margin: 0; font-size: 14px; font-weight: 500;">
                    {rec['action']}
                </p>
                <p style="color: #86868B; margin: 4px 0 0 0; font-size: 12px;">
                    Reason: {rec['reason']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Recommendations require SHAP analysis.")

# Actual outcome (if available)
if 'churn' in customer.index:
    actual = "Churned" if customer['churn'] == 1 else "Active"
    actual_color = COLORS['red'] if customer['churn'] == 1 else COLORS['green']

    st.markdown(f"""
    <div style="
        background: {actual_color}10;
        border: 1px solid {actual_color};
        border-radius: 12px;
        padding: 16px;
        margin-top: 20px;
        text-align: center;
    ">
        <p style="color: {actual_color}; margin: 0; font-size: 14px;">
            <strong>Actual Outcome:</strong> This customer has {actual.lower()}
        </p>
    </div>
    """, unsafe_allow_html=True)
