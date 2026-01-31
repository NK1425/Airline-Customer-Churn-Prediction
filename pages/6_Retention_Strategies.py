"""
Retention Strategies Page - AI-Driven Recommendations
Translate predictions into business action with ROI analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_loader import load_processed_features
from src.model_utils import predict_churn_probability, get_risk_level
from src.ui_components import (
    load_css, section_header, gradient_divider, metric_card,
    COLORS, RISK_COLORS, CHART_LAYOUT
)

# Page config
st.set_page_config(page_title="Retention Strategies - SkyGuard", page_icon="$", layout="wide")
load_css()

# Header
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="color: #1D1D1F; font-size: 32px; font-weight: 700; margin: 0;">
        Retention Strategies
    </h1>
    <p style="color: #6E6E73; font-size: 16px; margin: 5px 0 0 0;">
        Data-driven recommendations with ROI analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Load data and predictions
@st.cache_data
def load_data_with_predictions():
    df = load_processed_features()
    df['churn_probability'] = predict_churn_probability(df)
    df['risk_level'] = df['churn_probability'].apply(get_risk_level)
    return df

df = load_data_with_predictions()

# Risk tier summary
section_header("Risk Tier Summary", "Current portfolio breakdown")

high_risk = df[df['risk_level'] == 'High']
medium_risk = df[df['risk_level'] == 'Medium']
low_risk = df[df['risk_level'] == 'Low']

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,59,48,0.1) 0%, rgba(255,59,48,0.05) 100%);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid {COLORS['red']};
    ">
        <h3 style="color: {COLORS['red']}; margin: 0 0 15px 0;">High Risk (>60%)</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <p style="color: #86868B; font-size: 12px; margin: 0;">Customers</p>
                <p style="color: #1D1D1F; font-size: 24px; font-weight: 700; margin: 0;">{len(high_risk):,}</p>
            </div>
            <div>
                <p style="color: #86868B; font-size: 12px; margin: 0;">CLV at Risk</p>
                <p style="color: #1D1D1F; font-size: 24px; font-weight: 700; margin: 0;">${high_risk['CLV'].sum():,.0f}</p>
            </div>
        </div>
        <p style="color: #86868B; font-size: 13px; margin: 15px 0 0 0;">
            Avg probability: {high_risk['churn_probability'].mean():.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,149,0,0.1) 0%, rgba(255,149,0,0.05) 100%);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid {COLORS['orange']};
    ">
        <h3 style="color: {COLORS['orange']}; margin: 0 0 15px 0;">Medium Risk (30-60%)</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <p style="color: #86868B; font-size: 12px; margin: 0;">Customers</p>
                <p style="color: #1D1D1F; font-size: 24px; font-weight: 700; margin: 0;">{len(medium_risk):,}</p>
            </div>
            <div>
                <p style="color: #86868B; font-size: 12px; margin: 0;">CLV at Risk</p>
                <p style="color: #1D1D1F; font-size: 24px; font-weight: 700; margin: 0;">${medium_risk['CLV'].sum():,.0f}</p>
            </div>
        </div>
        <p style="color: #86868B; font-size: 13px; margin: 15px 0 0 0;">
            Avg probability: {medium_risk['churn_probability'].mean():.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(52,199,89,0.1) 0%, rgba(52,199,89,0.05) 100%);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid {COLORS['green']};
    ">
        <h3 style="color: {COLORS['green']}; margin: 0 0 15px 0;">Low Risk (<30%)</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <p style="color: #86868B; font-size: 12px; margin: 0;">Customers</p>
                <p style="color: #1D1D1F; font-size: 24px; font-weight: 700; margin: 0;">{len(low_risk):,}</p>
            </div>
            <div>
                <p style="color: #86868B; font-size: 12px; margin: 0;">CLV at Risk</p>
                <p style="color: #1D1D1F; font-size: 24px; font-weight: 700; margin: 0;">${low_risk['CLV'].sum():,.0f}</p>
            </div>
        </div>
        <p style="color: #86868B; font-size: 13px; margin: 15px 0 0 0;">
            Avg probability: {low_risk['churn_probability'].mean():.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)

gradient_divider()

# Retention Strategy Matrix
section_header("Retention Strategy Matrix", "SHAP-informed recommendations by churn driver")

strategies = [
    {
        'driver': 'Declining Flights',
        'action': 'Targeted fare discount (15% off)',
        'priority': 'High',
        'cost': '$50/customer',
        'est_save': '$2,100 CLV',
        'target_segment': 'Negative flight_trend_yoy',
        'color': COLORS['red']
    },
    {
        'driver': 'Low Points Utilization',
        'action': 'Points expiry warning + bonus catalog',
        'priority': 'Medium',
        'cost': '$10/customer',
        'est_save': '$800 CLV',
        'target_segment': 'points_utilization < 0.3',
        'color': COLORS['orange']
    },
    {
        'driver': 'Promotion Enrollee',
        'action': 'Welcome series + onboarding call',
        'priority': 'High',
        'cost': '$25/customer',
        'est_save': '$1,500 CLV',
        'target_segment': 'is_promotion_enrollee = 1',
        'color': COLORS['red']
    },
    {
        'driver': 'Low Companion Travel',
        'action': 'Family/group booking incentive',
        'priority': 'Medium',
        'cost': '$30/customer',
        'est_save': '$950 CLV',
        'target_segment': 'companion_ratio < 0.2',
        'color': COLORS['orange']
    },
    {
        'driver': 'Lapsed Traveler',
        'action': 'Win-back campaign with exclusive offer',
        'priority': 'Low',
        'cost': '$15/customer',
        'est_save': '$400 CLV',
        'target_segment': 'recency_last_flight > 6',
        'color': COLORS['primary']
    },
]

# Display strategy table
for strategy in strategies:
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid {strategy['color']};
        display: grid;
        grid-template-columns: 2fr 3fr 1fr 1fr 1fr;
        gap: 20px;
        align-items: center;
    ">
        <div>
            <p style="color: #86868B; font-size: 11px; margin: 0; text-transform: uppercase;">Churn Driver</p>
            <p style="color: #1D1D1F; font-size: 14px; font-weight: 600; margin: 4px 0 0 0;">{strategy['driver']}</p>
        </div>
        <div>
            <p style="color: #86868B; font-size: 11px; margin: 0; text-transform: uppercase;">Recommended Action</p>
            <p style="color: #1D1D1F; font-size: 14px; margin: 4px 0 0 0;">{strategy['action']}</p>
        </div>
        <div>
            <p style="color: #86868B; font-size: 11px; margin: 0; text-transform: uppercase;">Priority</p>
            <span style="
                background: {strategy['color']}20;
                color: {strategy['color']};
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
            ">{strategy['priority']}</span>
        </div>
        <div>
            <p style="color: #86868B; font-size: 11px; margin: 0; text-transform: uppercase;">Est. Cost</p>
            <p style="color: #1D1D1F; font-size: 14px; font-weight: 500; margin: 4px 0 0 0;">{strategy['cost']}</p>
        </div>
        <div>
            <p style="color: #86868B; font-size: 11px; margin: 0; text-transform: uppercase;">Est. Save</p>
            <p style="color: {COLORS['green']}; font-size: 14px; font-weight: 600; margin: 4px 0 0 0;">{strategy['est_save']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

gradient_divider()

# ROI Calculator
section_header("ROI Calculator", "Estimate return on retention investment")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Campaign Parameters")

    target_segment = st.selectbox(
        "Target Segment",
        options=['High Risk Only', 'High + Medium Risk', 'All At-Risk Customers'],
        index=0
    )

    if target_segment == 'High Risk Only':
        target_count = len(high_risk)
        avg_clv = high_risk['CLV'].mean()
    elif target_segment == 'High + Medium Risk':
        target_count = len(high_risk) + len(medium_risk)
        avg_clv = pd.concat([high_risk, medium_risk])['CLV'].mean()
    else:
        target_count = len(df[df['risk_level'] != 'Low'])
        avg_clv = df[df['risk_level'] != 'Low']['CLV'].mean()

    cost_per_customer = st.slider(
        "Campaign Cost per Customer ($)",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

    success_rate = st.slider(
        "Expected Success Rate (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    ) / 100

    st.markdown(f"""
    <div style="
        background: rgba(0, 113, 227, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin-top: 20px;
    ">
        <p style="color: #0071E3; font-size: 13px; margin: 0;">
            <strong>Target Customers:</strong> {target_count:,}<br>
            <strong>Average CLV:</strong> ${avg_clv:,.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Projected Results")

    total_cost = target_count * cost_per_customer
    customers_saved = int(target_count * success_rate)
    clv_saved = customers_saved * avg_clv
    roi = (clv_saved - total_cost) / total_cost * 100 if total_cost > 0 else 0
    roi_multiplier = clv_saved / total_cost if total_cost > 0 else 0

    # Results display
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 15px;
        ">
            <p style="color: #86868B; font-size: 12px; margin: 0;">Total Campaign Cost</p>
            <p style="color: #1D1D1F; font-size: 28px; font-weight: 700; margin: 5px 0 0 0;">${total_cost:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        ">
            <p style="color: #86868B; font-size: 12px; margin: 0;">Customers Retained</p>
            <p style="color: {COLORS['green']}; font-size: 28px; font-weight: 700; margin: 5px 0 0 0;">{customers_saved:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 15px;
        ">
            <p style="color: #86868B; font-size: 12px; margin: 0;">CLV Saved</p>
            <p style="color: {COLORS['green']}; font-size: 28px; font-weight: 700; margin: 5px 0 0 0;">${clv_saved:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        roi_color = COLORS['green'] if roi > 0 else COLORS['red']
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {roi_color}20 0%, {roi_color}10 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 2px solid {roi_color};
        ">
            <p style="color: #86868B; font-size: 12px; margin: 0;">ROI</p>
            <p style="color: {roi_color}; font-size: 28px; font-weight: 700; margin: 5px 0 0 0;">{roi:.0f}%</p>
            <p style="color: #86868B; font-size: 12px; margin: 5px 0 0 0;">{roi_multiplier:.1f}x return</p>
        </div>
        """, unsafe_allow_html=True)

    # ROI formula explanation
    st.markdown(f"""
    <div style="
        background: rgba(134, 134, 139, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin-top: 15px;
    ">
        <p style="color: #86868B; font-size: 12px; margin: 0; font-family: monospace;">
            ROI = (CLV Saved - Campaign Cost) / Campaign Cost<br>
            ROI = (${clv_saved:,.0f} - ${total_cost:,.0f}) / ${total_cost:,.0f} = {roi:.0f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

gradient_divider()

# Download Action List
section_header("Download Action List", "Export high-risk customers with recommendations")

# Prepare download data
high_risk_export = high_risk[['Loyalty Number', 'CLV', 'churn_probability', 'risk_level']].copy()
high_risk_export['churn_probability'] = high_risk_export['churn_probability'].apply(lambda x: f"{x:.1%}")

# Add recommendations based on features
def get_recommendation(row):
    recommendations = []
    if row.get('flight_trend_yoy', 0) < 0:
        recommendations.append("Targeted fare discount")
    if row.get('points_utilization', 0) < 0.3:
        recommendations.append("Points engagement campaign")
    if row.get('is_promotion_enrollee', 0) == 1:
        recommendations.append("Personal onboarding call")
    if row.get('recency_last_flight', 0) > 6:
        recommendations.append("Win-back campaign")
    return "; ".join(recommendations[:2]) if recommendations else "General retention outreach"

high_risk_full = high_risk.copy()
high_risk_full['recommended_actions'] = high_risk_full.apply(get_recommendation, axis=1)

export_df = high_risk_full[['Loyalty Number', 'CLV', 'churn_probability', 'recommended_actions']].copy()
export_df.columns = ['Customer ID', 'CLV', 'Churn Risk', 'Recommended Actions']
export_df['Churn Risk'] = export_df['Churn Risk'].apply(lambda x: f"{x:.1%}")
export_df['CLV'] = export_df['CLV'].apply(lambda x: f"${x:,.2f}")

# Preview
st.markdown("#### Preview (Top 10)")
st.dataframe(export_df.head(10), use_container_width=True, hide_index=True)

# Download button
csv_data = export_df.to_csv(index=False)
st.download_button(
    label=f"Download Full Action List ({len(high_risk):,} customers)",
    data=csv_data,
    file_name="high_risk_action_list.csv",
    mime="text/csv",
    use_container_width=True
)

# Summary stats
st.markdown(f"""
<div style="
    background: rgba(0, 113, 227, 0.05);
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    text-align: center;
">
    <p style="color: #0071E3; font-size: 14px; margin: 0;">
        <strong>{len(high_risk):,}</strong> high-risk customers identified |
        <strong>${high_risk['CLV'].sum():,.0f}</strong> total CLV at risk |
        Average churn probability: <strong>{high_risk['churn_probability'].mean():.1%}</strong>
    </p>
</div>
""", unsafe_allow_html=True)
