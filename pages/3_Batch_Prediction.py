"""
Batch Prediction Page - CSV Upload and Bulk Scoring
Upload a CSV of customers to get predictions for all.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import io
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model_utils import batch_predict, get_feature_names, get_threshold
from src.feature_engineering import engineer_features, get_feature_columns
from src.ui_components import (
    load_css, section_header, gradient_divider, metric_card,
    COLORS, RISK_COLORS, CHART_LAYOUT
)

# Page config
st.set_page_config(page_title="Batch Prediction - SkyGuard", page_icon="F", layout="wide")
load_css()

# Header
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="color: #1D1D1F; font-size: 32px; font-weight: 700; margin: 0;">
        Batch Churn Prediction
    </h1>
    <p style="color: #6E6E73; font-size: 16px; margin: 5px 0 0 0;">
        Upload customer data for bulk risk scoring
    </p>
</div>
""", unsafe_allow_html=True)

# Create template download
@st.cache_data
def create_template():
    """Create a sample template CSV."""
    feature_cols = get_feature_columns()
    template_df = pd.DataFrame(columns=['Loyalty Number'] + feature_cols[:20])  # Simplified template

    # Add sample row
    sample_row = {'Loyalty Number': 100001}
    for col in feature_cols[:20]:
        sample_row[col] = 0.0
    template_df = pd.concat([template_df, pd.DataFrame([sample_row])], ignore_index=True)

    return template_df


# File upload section
st.markdown("""
<div style="
    background: white;
    border: 2px dashed #D2D2D7;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin-bottom: 30px;
">
    <p style="color: #1D1D1F; font-size: 18px; font-weight: 500; margin: 0 0 10px 0;">
        Upload Customer Data
    </p>
    <p style="color: #86868B; font-size: 14px; margin: 0;">
        Drag and drop a CSV file or click to browse
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=['csv'],
    help="Upload a CSV file with customer data. See template for required format.",
    label_visibility="collapsed"
)

# Template download
col1, col2 = st.columns([3, 1])
with col2:
    template_df = create_template()
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="Download Template",
        data=csv_template,
        file_name="batch_prediction_template.csv",
        mime="text/csv"
    )

with col1:
    st.markdown("""
    <p style="color: #86868B; font-size: 13px;">
        <b>Expected format:</b> CSV with Loyalty Number and customer features.
        Use pre-engineered features or raw data (flight activity + loyalty history will be processed automatically).
    </p>
    """, unsafe_allow_html=True)

gradient_divider()

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)

        st.markdown("### Processing Results")

        # Check if we have the required columns
        feature_names = get_feature_names()
        has_features = all(col in df.columns for col in feature_names)

        if not has_features:
            st.warning("Some features are missing. Adding default values for missing columns.")
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Processing customers...")
        progress_bar.progress(25)

        # Run batch prediction
        start_time = time.time()
        results_df = batch_predict(df)
        processing_time = time.time() - start_time

        progress_bar.progress(100)
        status_text.text(f"Processed {len(df):,} customers in {processing_time:.2f} seconds")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        high_risk = len(results_df[results_df['risk_level'] == 'High'])
        medium_risk = len(results_df[results_df['risk_level'] == 'Medium'])
        low_risk = len(results_df[results_df['risk_level'] == 'Low'])

        with col1:
            metric_card("Total Processed", f"{len(df):,}", icon="#")

        with col2:
            metric_card(
                "High Risk",
                f"{high_risk:,}",
                delta=f"{high_risk/len(df)*100:.1f}%",
                delta_color="red",
                icon="!"
            )

        with col3:
            metric_card(
                "Medium Risk",
                f"{medium_risk:,}",
                delta=f"{medium_risk/len(df)*100:.1f}%",
                delta_color="orange" if medium_risk > high_risk else "green",
                icon="~"
            )

        with col4:
            metric_card(
                "Low Risk",
                f"{low_risk:,}",
                delta=f"{low_risk/len(df)*100:.1f}%",
                delta_color="green",
                icon="+"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk distribution chart
        col1, col2 = st.columns([2, 1])

        with col1:
            section_header("Risk Score Distribution")

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=results_df['churn_probability'],
                nbinsx=30,
                marker_color=COLORS['primary'],
                opacity=0.8,
                hovertemplate='Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ))

            # Add threshold line
            threshold = get_threshold()
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=COLORS['red'],
                annotation_text=f"Threshold: {threshold:.2f}"
            )

            fig.update_layout(
                **CHART_LAYOUT,
                xaxis_title="Churn Probability",
                yaxis_title="Number of Customers",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_header("Risk Level Breakdown")

            risk_counts = results_df['risk_level'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.6,
                marker_colors=[RISK_COLORS.get(r, COLORS['gray']) for r in risk_counts.index],
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
            )])

            fig.update_layout(
                showlegend=False,
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        gradient_divider()

        # Preview table
        section_header("Results Preview", "Top 20 highest risk customers")

        # Select columns to display
        display_cols = ['Loyalty Number', 'churn_probability', 'risk_level', 'top_risk_factor']
        if 'CLV' in results_df.columns:
            display_cols.insert(2, 'CLV')
        if 'Loyalty Card' in results_df.columns:
            display_cols.insert(1, 'Loyalty Card')

        preview_df = results_df.nlargest(20, 'churn_probability')[display_cols].copy()

        # Format for display
        preview_df['churn_probability'] = preview_df['churn_probability'].apply(lambda x: f"{x:.1%}")
        if 'CLV' in preview_df.columns:
            preview_df['CLV'] = preview_df['CLV'].apply(lambda x: f"${x:,.2f}")

        # Style the dataframe
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Loyalty Number": st.column_config.NumberColumn("Customer ID"),
                "churn_probability": st.column_config.TextColumn("Churn Risk"),
                "risk_level": st.column_config.TextColumn("Risk Level"),
                "CLV": st.column_config.TextColumn("CLV"),
                "top_risk_factor": st.column_config.TextColumn("Top Risk Factor"),
            }
        )

        gradient_divider()

        # Download buttons
        section_header("Download Results")

        col1, col2, col3 = st.columns(3)

        # Full results
        with col1:
            csv_full = results_df.to_csv(index=False)
            st.download_button(
                label="Download Full Results",
                data=csv_full,
                file_name="churn_predictions_full.csv",
                mime="text/csv",
                use_container_width=True
            )

        # High risk only
        with col2:
            high_risk_df = results_df[results_df['risk_level'] == 'High']
            csv_high = high_risk_df.to_csv(index=False)
            st.download_button(
                label=f"Download High Risk Only ({len(high_risk_df):,})",
                data=csv_high,
                file_name="churn_predictions_high_risk.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Summary report
        with col3:
            summary_data = f"""Batch Prediction Summary Report
================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Customers Processed: {len(df):,}
Processing Time: {processing_time:.2f} seconds

Risk Distribution:
- High Risk (>60%): {high_risk:,} ({high_risk/len(df)*100:.1f}%)
- Medium Risk (30-60%): {medium_risk:,} ({medium_risk/len(df)*100:.1f}%)
- Low Risk (<30%): {low_risk:,} ({low_risk/len(df)*100:.1f}%)

Average Churn Probability: {results_df['churn_probability'].mean():.1%}
Median Churn Probability: {results_df['churn_probability'].median():.1%}

Threshold Used: {get_threshold():.2f}
"""
            st.download_button(
                label="Download Summary Report",
                data=summary_data,
                file_name="churn_predictions_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file matches the expected format. Download the template for reference.")

else:
    # Show instructions when no file is uploaded
    st.markdown("""
    <div style="
        background: rgba(0, 113, 227, 0.05);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    ">
        <p style="color: #0071E3; font-size: 16px; margin: 0;">
            Upload a CSV file to get started with batch predictions
        </p>
        <p style="color: #86868B; font-size: 14px; margin: 10px 0 0 0;">
            The system will automatically process your data and generate churn risk scores for each customer
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show sample output
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Sample Output Format")

    sample_output = pd.DataFrame({
        'Loyalty Number': [100001, 100002, 100003],
        'churn_probability': [0.82, 0.45, 0.15],
        'risk_level': ['High', 'Medium', 'Low'],
        'top_risk_factor': ['flight_trend_yoy', 'points_utilization', 'tenure_months']
    })

    st.dataframe(sample_output, use_container_width=True, hide_index=True)
