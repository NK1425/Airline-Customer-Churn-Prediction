"""
Segment Analysis Page - Deep Dive by Segment
Interactive filtering and analysis by tier, geography, demographics.
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
from src.ui_components import (
    load_css, section_header, gradient_divider, metric_card,
    COLORS, TIER_COLORS, CHURN_COLORS, CHART_LAYOUT
)

# Page config
st.set_page_config(page_title="Segment Analysis - SkyGuard", page_icon="@", layout="wide")
load_css()

# Header
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="color: #1D1D1F; font-size: 32px; font-weight: 700; margin: 0;">
        Segment Analysis
    </h1>
    <p style="color: #6E6E73; font-size: 16px; margin: 5px 0 0 0;">
        Deep-dive into churn patterns by customer segments
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return load_processed_features()

df = load_data()

# Sidebar filters
st.sidebar.markdown("### Filters")

# Tier filter
if 'Loyalty Card' in df.columns:
    tiers = st.sidebar.multiselect(
        "Loyalty Tier",
        options=df['Loyalty Card'].unique().tolist(),
        default=df['Loyalty Card'].unique().tolist()
    )
    df_filtered = df[df['Loyalty Card'].isin(tiers)]
else:
    df_filtered = df

# Enrollment type filter
if 'Enrollment Type' in df.columns:
    enrollment_types = st.sidebar.multiselect(
        "Enrollment Type",
        options=df['Enrollment Type'].unique().tolist(),
        default=df['Enrollment Type'].unique().tolist()
    )
    df_filtered = df_filtered[df_filtered['Enrollment Type'].isin(enrollment_types)]

# Province filter
if 'Province' in df.columns:
    provinces = st.sidebar.multiselect(
        "Province",
        options=sorted(df['Province'].dropna().unique().tolist()),
        default=[]
    )
    if provinces:
        df_filtered = df_filtered[df_filtered['Province'].isin(provinces)]

# Salary range filter
if 'Salary' in df.columns:
    salary_range = st.sidebar.slider(
        "Salary Range ($)",
        min_value=int(df['Salary'].min()),
        max_value=int(df['Salary'].max()),
        value=(int(df['Salary'].min()), int(df['Salary'].max()))
    )
    df_filtered = df_filtered[(df_filtered['Salary'] >= salary_range[0]) & (df_filtered['Salary'] <= salary_range[1])]

# Tenure range filter
if 'tenure_months' in df.columns:
    tenure_range = st.sidebar.slider(
        "Tenure (months)",
        min_value=int(df['tenure_months'].min()),
        max_value=int(df['tenure_months'].max()),
        value=(int(df['tenure_months'].min()), int(df['tenure_months'].max()))
    )
    df_filtered = df_filtered[(df_filtered['tenure_months'] >= tenure_range[0]) & (df_filtered['tenure_months'] <= tenure_range[1])]

# Show filter summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered customers:** {len(df_filtered):,} of {len(df):,}")

# Main content
if len(df_filtered) == 0:
    st.warning("No customers match the selected filters. Please adjust your filter criteria.")
    st.stop()

# Quick stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Customers", f"{len(df_filtered):,}", icon="#")

with col2:
    churn_rate = df_filtered['churn'].mean() * 100
    metric_card("Churn Rate", f"{churn_rate:.1f}%", icon="%")

with col3:
    avg_clv = df_filtered['CLV'].mean()
    metric_card("Avg CLV", f"${avg_clv:,.0f}", icon="$")

with col4:
    avg_flights = df_filtered['total_flights_24m'].mean()
    metric_card("Avg Flights", f"{avg_flights:.1f}", icon="P")

gradient_divider()

# Section 1: Churn by Loyalty Tier
section_header("Churn by Loyalty Tier", "Grouped analysis by tier with key metrics")

if 'Loyalty Card' in df_filtered.columns:
    col1, col2 = st.columns([2, 1])

    with col1:
        tier_stats = df_filtered.groupby('Loyalty Card').agg({
            'churn': ['sum', 'count', 'mean'],
            'CLV': 'mean',
            'total_flights_24m': 'mean'
        }).reset_index()
        tier_stats.columns = ['Tier', 'Churned', 'Total', 'Churn Rate', 'Avg CLV', 'Avg Flights']
        tier_stats['Active'] = tier_stats['Total'] - tier_stats['Churned']
        tier_stats['Churn Rate'] = tier_stats['Churn Rate'] * 100

        # Grouped bar chart
        fig = go.Figure()

        for status, color in [('Active', CHURN_COLORS['Active']), ('Churned', CHURN_COLORS['Churned'])]:
            fig.add_trace(go.Bar(
                name=status,
                x=tier_stats['Tier'],
                y=tier_stats[status],
                marker_color=color,
                hovertemplate=f'<b>%{{x}}</b><br>{status}: %{{y:,}}<extra></extra>'
            ))

        fig.update_layout(
            **CHART_LAYOUT,
            barmode='group',
            xaxis_title="Loyalty Tier",
            yaxis_title="Number of Customers",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Tier metrics table
        st.markdown("#### Tier Metrics")
        display_df = tier_stats[['Tier', 'Churn Rate', 'Avg CLV', 'Avg Flights']].copy()
        display_df['Churn Rate'] = display_df['Churn Rate'].apply(lambda x: f"{x:.1f}%")
        display_df['Avg CLV'] = display_df['Avg CLV'].apply(lambda x: f"${x:,.0f}")
        display_df['Avg Flights'] = display_df['Avg Flights'].apply(lambda x: f"{x:.1f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

gradient_divider()

# Section 2: Churn by Enrollment Type
section_header("Churn by Enrollment Type", "Standard vs 2018 Promotion comparison")

if 'Enrollment Type' in df_filtered.columns:
    col1, col2 = st.columns(2)

    with col1:
        enrollment_stats = df_filtered.groupby('Enrollment Type').agg({
            'churn': ['sum', 'count', 'mean'],
            'CLV': 'mean'
        }).reset_index()
        enrollment_stats.columns = ['Type', 'Churned', 'Total', 'Churn Rate', 'Avg CLV']
        enrollment_stats['Churn Rate'] = enrollment_stats['Churn Rate'] * 100

        # Donut chart
        fig = go.Figure(data=[go.Pie(
            labels=enrollment_stats['Type'],
            values=enrollment_stats['Churned'],
            hole=0.6,
            marker_colors=[COLORS['primary'], COLORS['orange']],
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Churned: %{value:,}<br>%{percent}<extra></extra>'
        )])

        fig.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Churned Customers by Enrollment", font=dict(size=14))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Comparison table
        st.markdown("#### Comparison Metrics")

        for _, row in enrollment_stats.iterrows():
            color = COLORS['red'] if row['Churn Rate'] > 20 else COLORS['orange'] if row['Churn Rate'] > 15 else COLORS['green']
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                border-left: 4px solid {color};
            ">
                <h4 style="margin: 0 0 10px 0; color: #1D1D1F;">{row['Type']}</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <p style="color: #86868B; font-size: 12px; margin: 0;">Churn Rate</p>
                        <p style="color: {color}; font-size: 20px; font-weight: 600; margin: 0;">{row['Churn Rate']:.1f}%</p>
                    </div>
                    <div>
                        <p style="color: #86868B; font-size: 12px; margin: 0;">Avg CLV</p>
                        <p style="color: #1D1D1F; font-size: 20px; font-weight: 600; margin: 0;">${row['Avg CLV']:,.0f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

gradient_divider()

# Section 3: Demographics
section_header("Churn by Demographics", "Gender, education, marital status")

col1, col2, col3 = st.columns(3)

with col1:
    if 'Gender' in df_filtered.columns:
        gender_stats = df_filtered.groupby('Gender')['churn'].agg(['sum', 'count', 'mean']).reset_index()
        gender_stats.columns = ['Gender', 'Churned', 'Total', 'Churn Rate']
        gender_stats['Churn Rate'] = gender_stats['Churn Rate'] * 100

        fig = go.Figure(data=[go.Bar(
            x=gender_stats['Gender'],
            y=gender_stats['Churn Rate'],
            marker_color=[COLORS['primary'], COLORS['green']],
            text=[f"{r:.1f}%" for r in gender_stats['Churn Rate']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        )])

        fig.update_layout(
            **CHART_LAYOUT,
            title=dict(text="By Gender", font=dict(size=14)),
            xaxis_title="",
            yaxis_title="Churn Rate (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'Education' in df_filtered.columns:
        edu_stats = df_filtered.groupby('Education')['churn'].agg(['sum', 'count', 'mean']).reset_index()
        edu_stats.columns = ['Education', 'Churned', 'Total', 'Churn Rate']
        edu_stats['Churn Rate'] = edu_stats['Churn Rate'] * 100

        # Order by education level
        edu_order = ['High School or Below', 'College', 'Bachelor', 'Master', 'Doctor']
        edu_stats['Order'] = edu_stats['Education'].apply(lambda x: edu_order.index(x) if x in edu_order else 99)
        edu_stats = edu_stats.sort_values('Order')

        fig = go.Figure(data=[go.Bar(
            x=edu_stats['Education'],
            y=edu_stats['Churn Rate'],
            marker_color=COLORS['primary'],
            text=[f"{r:.1f}%" for r in edu_stats['Churn Rate']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        )])

        fig.update_layout(
            **CHART_LAYOUT,
            title=dict(text="By Education", font=dict(size=14)),
            xaxis_title="",
            yaxis_title="Churn Rate (%)",
            height=300,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

with col3:
    if 'Marital Status' in df_filtered.columns:
        marital_stats = df_filtered.groupby('Marital Status')['churn'].agg(['sum', 'count', 'mean']).reset_index()
        marital_stats.columns = ['Status', 'Churned', 'Total', 'Churn Rate']
        marital_stats['Churn Rate'] = marital_stats['Churn Rate'] * 100

        fig = go.Figure(data=[go.Bar(
            x=marital_stats['Status'],
            y=marital_stats['Churn Rate'],
            marker_color=[COLORS['orange'], COLORS['primary'], COLORS['green']],
            text=[f"{r:.1f}%" for r in marital_stats['Churn Rate']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        )])

        fig.update_layout(
            **CHART_LAYOUT,
            title=dict(text="By Marital Status", font=dict(size=14)),
            xaxis_title="",
            yaxis_title="Churn Rate (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

gradient_divider()

# Section 4: Flight Behavior Comparison
section_header("Flight Behavior: Active vs Churned", "Distribution comparisons")

col1, col2 = st.columns(2)

with col1:
    # Total flights histogram
    fig = go.Figure()

    for churn_val, name in [(0, 'Active'), (1, 'Churned')]:
        data = df_filtered[df_filtered['churn'] == churn_val]['total_flights_24m']
        fig.add_trace(go.Histogram(
            x=data,
            name=name,
            marker_color=CHURN_COLORS[name],
            opacity=0.7,
            nbinsx=30
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Total Flights (24 months)", font=dict(size=14)),
        xaxis_title="Total Flights",
        yaxis_title="Count",
        barmode='overlay',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Flight trend YoY
    fig = go.Figure()

    for churn_val, name in [(0, 'Active'), (1, 'Churned')]:
        data = df_filtered[df_filtered['churn'] == churn_val]['flight_trend_yoy']
        fig.add_trace(go.Histogram(
            x=data,
            name=name,
            marker_color=CHURN_COLORS[name],
            opacity=0.7,
            nbinsx=30
        ))

    fig.add_vline(x=0, line_dash="dash", line_color=COLORS['gray'], annotation_text="No change")

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Flight Trend Year-over-Year", font=dict(size=14)),
        xaxis_title="Flight Change (2018 vs 2017)",
        yaxis_title="Count",
        barmode='overlay',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    # Points utilization
    fig = go.Figure()

    for churn_val, name in [(0, 'Active'), (1, 'Churned')]:
        data = df_filtered[df_filtered['churn'] == churn_val]['points_utilization']
        fig.add_trace(go.Histogram(
            x=data,
            name=name,
            marker_color=CHURN_COLORS[name],
            opacity=0.7,
            nbinsx=20
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Points Utilization", font=dict(size=14)),
        xaxis_title="Utilization Ratio",
        yaxis_title="Count",
        barmode='overlay',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Recency
    fig = go.Figure()

    for churn_val, name in [(0, 'Active'), (1, 'Churned')]:
        data = df_filtered[df_filtered['churn'] == churn_val]['recency_last_flight']
        fig.add_trace(go.Histogram(
            x=data,
            name=name,
            marker_color=CHURN_COLORS[name],
            opacity=0.7,
            nbinsx=20
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Recency of Last Flight", font=dict(size=14)),
        xaxis_title="Months Since Last Flight",
        yaxis_title="Count",
        barmode='overlay',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)
