"""
Dashboard Page - Executive Overview
Real-time loyalty program health monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_loader import load_processed_features, load_flight_activity, get_summary_stats
from src.model_utils import predict_churn_probability, get_feature_names
from src.ui_components import (
    load_css, metric_card, section_header, gradient_divider,
    COLORS, TIER_COLORS, CHURN_COLORS, CHART_LAYOUT, styled_plotly_chart,
    create_donut_chart, create_bar_chart
)

# Page config
st.set_page_config(page_title="Dashboard - SkyGuard", page_icon="*", layout="wide")
load_css()

# Header
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="color: #1D1D1F; font-size: 32px; font-weight: 700; margin: 0;">
        Dashboard
    </h1>
    <p style="color: #6E6E73; font-size: 16px; margin: 5px 0 0 0;">
        Real-time loyalty program health monitoring
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = load_processed_features()
    return df

df = load_data()

# Calculate metrics
total_customers = len(df)
churned_customers = df['churn'].sum()
active_customers = total_customers - churned_customers
churn_rate = (churned_customers / total_customers) * 100
total_clv = df['CLV'].sum()
churned_clv = df[df['churn'] == 1]['CLV'].sum()

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    metric_card("Total Members", f"{total_customers:,}", icon="U")

with col2:
    metric_card("Active Members", f"{active_customers:,}", delta=f"{(active_customers/total_customers)*100:.1f}%", delta_color="green", icon="G")

with col3:
    metric_card("Churned Members", f"{churned_customers:,}", delta=f"{churn_rate:.1f}%", delta_color="red", icon="E")

with col4:
    metric_card("Churn Rate", f"{churn_rate:.1f}%", icon="%")

with col5:
    metric_card("Revenue at Risk", f"${churned_clv/1000000:.1f}M", delta="CLV of churned", delta_color="red", icon="$")

st.markdown("<br>", unsafe_allow_html=True)

gradient_divider()

# Second row - Charts
col1, col2 = st.columns([2, 1])

with col1:
    section_header("Churn Rate by Loyalty Tier", "Star tier shows highest churn propensity")

    # Calculate tier stats
    if 'Loyalty Card' in df.columns:
        tier_stats = df.groupby('Loyalty Card').agg({
            'churn': ['sum', 'count', 'mean'],
            'CLV': 'mean'
        }).reset_index()
        tier_stats.columns = ['Tier', 'Churned', 'Total', 'Churn Rate', 'Avg CLV']
        tier_stats['Churn Rate'] = tier_stats['Churn Rate'] * 100

        # Sort by churn rate
        tier_stats = tier_stats.sort_values('Churn Rate', ascending=True)

        # Create horizontal bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tier_stats['Tier'],
            x=tier_stats['Churn Rate'],
            orientation='h',
            marker_color=[TIER_COLORS.get(t, COLORS['gray']) for t in tier_stats['Tier']],
            text=[f"{r:.1f}%" for r in tier_stats['Churn Rate']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Churn Rate: %{x:.1f}%<br>Total: %{customdata[0]:,}<br>Churned: %{customdata[1]:,}<extra></extra>',
            customdata=tier_stats[['Total', 'Churned']].values
        ))

        fig.update_layout(
            **CHART_LAYOUT,
            xaxis_title="Churn Rate (%)",
            yaxis_title="",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    section_header("Churn by Enrollment Type", "2018 Promotion shows higher churn")

    # Enrollment type distribution
    if 'Enrollment Type' in df.columns:
        enrollment_stats = df.groupby('Enrollment Type')['churn'].agg(['sum', 'count']).reset_index()
        enrollment_stats.columns = ['Type', 'Churned', 'Total']
        enrollment_stats['Churn Rate'] = enrollment_stats['Churned'] / enrollment_stats['Total'] * 100

        fig = create_donut_chart(
            values=enrollment_stats['Churned'].tolist(),
            labels=enrollment_stats['Type'].tolist(),
            colors=[COLORS['primary'], COLORS['orange']]
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

gradient_divider()

# Monthly flight trend
section_header("Monthly Flight Activity Trend", "Churned customers show declining activity 6+ months before cancellation")

# Load flight activity for trend analysis
@st.cache_data
def get_monthly_trend():
    flight_df = load_flight_activity()
    df_with_churn = load_processed_features()

    # Merge churn label
    flight_df = flight_df.merge(
        df_with_churn[['Loyalty Number', 'churn']],
        on='Loyalty Number',
        how='left'
    )

    # Create month-year column
    flight_df['Month_Year'] = flight_df['Year'].astype(str) + '-' + flight_df['Month'].astype(str).str.zfill(2)

    # Average flights by month and churn status
    monthly = flight_df.groupby(['Month_Year', 'churn'])['Total Flights'].mean().reset_index()
    monthly['Status'] = monthly['churn'].map({0: 'Active', 1: 'Churned'})

    return monthly

monthly_trend = get_monthly_trend()

# Create line chart
fig = go.Figure()

for status in ['Active', 'Churned']:
    data = monthly_trend[monthly_trend['Status'] == status]
    fig.add_trace(go.Scatter(
        x=data['Month_Year'],
        y=data['Total Flights'],
        name=status,
        mode='lines+markers',
        line=dict(color=CHURN_COLORS[status], width=3),
        marker=dict(size=6),
        hovertemplate=f'<b>{status}</b><br>%{{x}}<br>Avg Flights: %{{y:.2f}}<extra></extra>'
    ))

# Add annotation
fig.add_annotation(
    x='2018-06',
    y=monthly_trend[monthly_trend['Status'] == 'Active']['Total Flights'].max() * 0.9,
    text="Churn signal visible<br>6+ months early",
    showarrow=True,
    arrowhead=2,
    arrowcolor=COLORS['red'],
    font=dict(size=12, color=COLORS['red']),
    bgcolor='white',
    bordercolor=COLORS['red']
)

fig.update_layout(
    **CHART_LAYOUT,
    xaxis_title="Month",
    yaxis_title="Average Flights",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

gradient_divider()

# Bottom row - Feature importance and geography
col1, col2 = st.columns(2)

with col1:
    section_header("Top 5 Churn Risk Factors", "SHAP-based feature importance")

    # Load SHAP values
    try:
        shap_values = np.load(Path(__file__).parent.parent / "model" / "shap_values_test.npy")
        feature_names = get_feature_names()

        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=True).tail(5)

        # Clean feature names for display
        feature_importance['Display Name'] = feature_importance['Feature'].str.replace('_', ' ').str.title()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=feature_importance['Display Name'],
            x=feature_importance['Importance'],
            orientation='h',
            marker_color=COLORS['primary'],
            hovertemplate='<b>%{y}</b><br>SHAP Importance: %{x:.4f}<extra></extra>'
        ))

        fig.update_layout(
            **CHART_LAYOUT,
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.info("SHAP values not available. Run model training first.")

with col2:
    section_header("Churn Rate by Geography", "Provincial distribution (Canada)")

    if 'Province' in df.columns:
        # Get province stats (Canada only)
        canada_df = df[df['Country'] == 'Canada'] if 'Country' in df.columns else df

        province_stats = canada_df.groupby('Province').agg({
            'churn': ['sum', 'count', 'mean']
        }).reset_index()
        province_stats.columns = ['Province', 'Churned', 'Total', 'Churn Rate']
        province_stats['Churn Rate'] = province_stats['Churn Rate'] * 100

        # Top 10 provinces by total customers
        province_stats = province_stats.nlargest(10, 'Total')
        province_stats = province_stats.sort_values('Churn Rate', ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=province_stats['Province'],
            x=province_stats['Churn Rate'],
            orientation='h',
            marker_color=province_stats['Churn Rate'].apply(
                lambda x: COLORS['red'] if x > 18 else COLORS['orange'] if x > 15 else COLORS['green']
            ),
            text=[f"{r:.1f}%" for r in province_stats['Churn Rate']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Churn Rate: %{x:.1f}%<br>Total: %{customdata:,}<extra></extra>',
            customdata=province_stats['Total']
        ))

        fig.update_layout(
            **CHART_LAYOUT,
            xaxis_title="Churn Rate (%)",
            yaxis_title="",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Province data not available.")

# Footer
st.markdown("""
<div style="text-align: center; padding: 30px 0; color: #86868B; font-size: 12px;">
    Data refreshed in real-time | SkyGuard Churn Intelligence Platform
</div>
""", unsafe_allow_html=True)
